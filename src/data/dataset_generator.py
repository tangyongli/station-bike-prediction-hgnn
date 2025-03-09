

import numpy as np
import torch
from torch_geometric.data import HeteroData,Batch
import random
import time
from torch.utils.data import DataLoader

# ==================================================
#  HETEROGENEOUS GRAPH CONSTRUCTION
# ==================================================
def create_hetero_data_arrays(gdf,num_poi_categories):
    # if isinstance(gdf, gpd.GeoDataFrame):
    #     print("crs",gdf.crs)
    # else:
    #     gdf['geometry']=gdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    #     gdf=gpd.GeoDataFrame(gdf,geometry='geometry',crs='epsg:4326')
    
    num_stations=gdf['station_name'].nunique()
    ## after filtering, original index of df will change, such as from 0,1,2,3 to 3,5,8. reset_index() to get back
    gdf=gdf.query('day<="2022-12-01"').drop_duplicates(subset='station_name').reset_index(drop=True)  # Reset index to 0, 1, 2, 
  
    
    # --- Generate POI Category Counts for each station ---
    # 1. Station Features: [POI counts] (STATIC)

    poi_counts=gdf[['work_education', 'residential', 'commercial_retail',
                                  'food', 'Recreation and Leisure', 'Health and Medicine',
                                  'Art and Entertainment', 'Travel and Transportation']].values.astype(np.float32)


    # 2. POI Category Features: ONE-HOT ENCODING (STATIC)
    features_poi_category = np.eye(num_poi_categories).astype(np.float32)

    # --- Edge Indices
    # c ('station','located in','census tract')
    # Create a mapping from GEOID to census zone index
    unique_geoids=gdf['GEOID'].unique()
    census_features = []  # Shape: [num_census_zones, num_census_features]
    for geoid in unique_geoids:
        zone_data = gdf[gdf['GEOID'] == geoid][['popdensity', 'per_income','vehicleratio']].values
        census_features.append(zone_data)
    census_features=np.concatenate(census_features)

    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(unique_geoids)}
    edge_source = [] 
    edge_target = []  
    for station_idx in range(len(gdf)):
        geoid = gdf.iloc[station_idx]['GEOID']
        census_zone_idx = geoid_to_idx[geoid]
        # print(census_zone_idx)
        edge_source.append(station_idx)
        edge_target.append(census_zone_idx)
         
    # a) ('station', 'adjacent_to', 'station') - Connect stations within a radius
    edge_list_adjacent = []
    adjacency_threshold_m = 1000
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            geometry_i = gdf.loc[i]['geometry'] 
            geometry_j = gdf.loc[j]['geometry']
            distanceij = geometry_i.distance(geometry_j)
            # distanceij=bikeflow_gdf.head(1)['geometry'].distance(bikeflow_gdf.head(1)['geometry'])
            # print(distanceij)
            if  distanceij < adjacency_threshold_m:
                edge_list_adjacent.append([i, j])
    edge_index_adjacent = torch.tensor(np.array(edge_list_adjacent).T, dtype=torch.long)
    print(edge_list_adjacent) ;print('edge_index_adjacent',edge_index_adjacent.shape)
    
    # b) ('station', 'near_poi_category', 'poi_category') -
    edge_list_has_poi = []
    for station_index in range(num_stations):
        for poi_cat_index in range(num_poi_categories):
            if poi_counts[station_index, poi_cat_index] > 0:
                edge_list_has_poi.append([station_index, poi_cat_index])
    edge_index_has_poi = torch.tensor(np.array(edge_list_has_poi).T, dtype=torch.long)
    
    data = HeteroData()
    data['station_poi'].x = torch.tensor(poi_counts)
    data['census_zone'].x = torch.tensor(census_features, dtype=torch.float32)
    data['poi_category'].x = torch.tensor(features_poi_category)  
    data['station', 'adjacent_to', 'station'].edge_index = edge_index_adjacent
    data['station', 'near_poi_category', 'poi_category'].edge_index = edge_index_has_poi
    data['station', 'located_in', 'census_zone'].edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
    # if store data on cuda, accquire it on cpu. data['station_poi'].x.cpu().numpy(); data['station_poi'].x return one tensor
 
    return data


 
# ==================================================
#  Sequence creation
# ==================================================
def create_sequences_and_dataset_arrays(node_dynamicfeatures_optimized,sequence_length,prediction_length):
    '''
    Preparing dynamic data with shape: (seq_length,num_station,num_features)
    '''
    # # 1. flows features
    flows =node_dynamicfeatures_optimized[...,0:1].astype(np.float32)
    # print('flows',flows.shape)
    # weather
    weather_features = node_dynamicfeatures_optimized[...,1:6].astype(np.float32)
    hours_sequence =node_dynamicfeatures_optimized[...,6:7].astype(int)
    hours_one_hot = np.eye(24)[hours_sequence].astype(np.float32) #
    hours_one_hot = hours_one_hot.reshape(*hours_sequence.shape[:2], 24)
    days_sequence = (node_dynamicfeatures_optimized[...,7:8]).astype(int)
    days_one_hot = np.eye(7)[days_sequence].astype(np.float32) 
    days_one_hot = days_one_hot.reshape(*days_sequence.shape[:2], 7)
    # print('days_one_hot',days_one_hot.shape) # (5833, 1615, 1, 7)
    sequences=[]

    for i in range(flows.shape[0]-sequence_length-prediction_length):
        
        seq_features = flows[i:i+sequence_length]
       
        hourofday=hours_one_hot[i:i+sequence_length]
        dayofweek=days_one_hot[i:i+sequence_length]
        weather=weather_features[i:i+sequence_length]
        target = flows[i+sequence_length:i+sequence_length+prediction_length,...,0]
        sequences.append((seq_features, hourofday,dayofweek,weather,target))
    return sequences



# ==================================================
#  Dataset creating for model training and evualation
# ==================================================

class BikeFlowDataset(torch.utils.data.Dataset): 
    '''
    Min-Max Normalization 
    Integration of sequence and heterogeneous graph data 
    
    '''
    def __init__(self, sequences, hetero_data_metadata, is_train=True, scaler=None):
        self.sequences = sequences
        self.hetero_data_metadata = hetero_data_metadata
        self.scaler = scaler
        if is_train and self.scaler is None:  # Only calculate scaler on first init (training)
            self.scaler = self.calculate_scaler()

    def calculate_scaler(self):
        scaler = {}
        all_flows = np.concatenate([seq[0] for seq in self.sequences], axis=0)  
        all_weather = np.concatenate([seq[3] for seq in self.sequences], axis=0)  
        all_targets = np.concatenate([seq[4] for seq in self.sequences], axis=0)  
        # print(all_census.min(axis=(0)),all_census.max(axis=(0)))

        # Flows (feature 0 in station)
        scaler['flows'] = {'min': all_flows.min(), 'max': all_flows.max()}
        # POI counts (features 1-8 in station, from hetero_data)
        poi_counts = self.hetero_data_metadata ['station_poi'].x.numpy()
        scaler['poi'] = {
            'min': poi_counts.min(axis=0),  # [8]
            'max': poi_counts.max(axis=0)   # [8]
        }
        # Weather (5 features)
        scaler['weather'] = {
            'min': all_weather.min(axis=(0, 1)),  # [5]
            'max': all_weather.max(axis=(0, 1))   # [5]
        }
        # census (5 features)
        all_census = self.hetero_data_metadata ['census_zone'].x.numpy()
        scaler['census_zone'] = {
            'min': all_census.min(axis=(0)),  # [5]
            'max': all_census.max(axis=(0))   # [5]
        }
        # print(self.scaler['census']['min'],self.scaler['census']['max'])
        
        # Targets (flows)
        scaler['target'] = {'min': all_targets.min(), 'max': all_targets.max()}
        return scaler

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self,idx):
       
        seq_flows, hour_of_day, day_of_week, seq_weather, target = self.sequences[idx]
        # print(seq_flows.shape,hour_of_day.shape,day_of_week.shape,seq_weather.shape,target.shape)(12, 1615, 1) (12, 1615, 24) (12, 1615, 7) (12, 1615, 5) (3, 1615)

        
        x_dict = {}
        # --- Static features (Normalize) ---
        # print(self.scaler['poi']['min'].shape)
        x_dict['station_poi'] = (self.hetero_data_metadata['station_poi'].x.numpy() - self.scaler['poi']['min']) / (
                    self.scaler['poi']['max'] - self.scaler['poi']['min'] + 1e-8)
        
        x_dict['poi_category'] = self.hetero_data_metadata['poi_category'].x.numpy()
        
        x_dict['census_zone'] = (self.hetero_data_metadata['census_zone'].x.numpy() - self.scaler['census_zone']['min']) /(
            self.scaler['census_zone']['max'] - self.scaler['census_zone']['min'] + 1e-8
        )
        
        # print(self.hetero_data_metadata['census_zone'].x.shape,x_dict['census_zone'].shape,self.scaler['census']['min'].shape)
        # --- Dynamic features (Normalize) ---
        
        x_dict['station_flows'] = (seq_flows - self.scaler['flows']['min']) / (
            self.scaler['flows']['max'] - self.scaler['flows']['min'] + 1e-8)
        
        x_dict['weather'] = (seq_weather- self.scaler['weather']['min']) / (
                    self.scaler['weather']['max'] - self.scaler['weather']['min'] + 1e-8)
        
        # print(self.hetero_data_metadata['census_zone'].x.cpu().numpy().shape,x_dict['weather'].shape,self.scaler['weather']['min'].shape)
        x_dict['hour_of_day']=hour_of_day
        x_dict['day_of_week']=day_of_week
        
        target = (target - self.scaler['target']['min']) / (
                    self.scaler['target']['max'] - self.scaler['target']['min'] + 1e-8)
        
        # --- Create the combined data dictionary ---
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #only torch tensor dataset can use 'to(device)'
        x_dict={key:torch.from_numpy(value).to(device) for key,value in x_dict.items()}
        edge_index_dict={key:torch.from_numpy(value.numpy()).to(device) for key,value in self.hetero_data_metadata.edge_index_dict.items()}
        data_dict = {
                    'features': x_dict,
                    'edge_indices': edge_index_dict,  
                }
        return data_dict, target

def custom_collate(batch):
    # 'batch' is a list of tuples: (data_dict, target)
    batch_dict = {
        'features': {},
        'edge_indices': batch[0][0]['edge_indices'],  # Access edge_indices from the first data_dict
    }
    targets = []
    # Collect and concatenate node features
    for feature_name in batch[0][0]['features'].keys():  # Access features from the first data_dict
        print(feature_name)
        batch_dict['features'][feature_name] = torch.stack([item[0]['features'][feature_name] for item in batch])

    # Collect targets (they are separate now)
    targets = [item[1] for item in batch]
    targets = torch.cat(targets, dim=0) # Concatenate targets

    return batch_dict, targets

def create_data_loaders(sequences, hetero_data_example, batch_size):
    """
    [sequence data shape: (batch_size,seq_length,num_station,num_features)]
    [static data shape: (batch_size,num_station,num_features)]
    """
    train_ratio = 0.7
    val_ratio = 0.15
    train_size = int(len(sequences) * train_ratio)
    val_size = int(len(sequences) * val_ratio)
    random.shuffle(sequences)  # Shuffle before splitting

    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]
    # min,max information-train_dataset.scaler['flows']['max']
    train_dataset = BikeFlowDataset(train_sequences, hetero_data_example, is_train=True)
    val_dataset = BikeFlowDataset(val_sequences, hetero_data_example, is_train=False, scaler=train_dataset.scaler)
    test_dataset = BikeFlowDataset(test_sequences, hetero_data_example, is_train=False, scaler=train_dataset.scaler)
    ## batch, shuffle, parallel loading dataset 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader











 
