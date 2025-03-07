

import numpy as np
import torch
from torch_geometric.data import HeteroData
import random
from torch.utils.data import DataLoader

# ==================================================
#  HETEROGENEOUS GRAPH CONSTRUCTION
# ==================================================
def create_hetero_data_arrays(gdf, num_poi_categories):
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
    try:
         print(geoid_to_idx['36061018000'])
         print(geoid_to_idx.keys())
    except KeyError: # Handle cases where a timestep might be missing in grouped data
            raise ValueError(f"No data found for time step")
   
    edge_source = [] 
    edge_target = []  
    for station_idx in range(len(gdf)):
        geoid = gdf.iloc[station_idx]['GEOID']
        census_zone_idx = geoid_to_idx[geoid]
        # print(census_zone_idx)
        edge_source.append(station_idx)
        edge_target.append(census_zone_idx)
         
    # a) ('station', 'adjacent_to', 'station') - Example: Connect stations within a radius
    edge_list_adjacent = []
    # adjacency_distance_matrixall=np.load('../Dataset/2022-citibike-tripdata/processed_data/adjacency_distance_matrixall.npy')
    # edge_index_adjacent = torch.tensor(adjacency_distance_matrixall.T, dtype=torch.long)
    # adjacency_threshold_m = 1000
    # for i in range(num_stations):
    #     for j in range(i + 1, num_stations):
    #         geometry_i = gdf.loc[i]['geometry'] 
    #         geometry_j = gdf.loc[j]['geometry']
    #         distanceij = geometry_i.distance(geometry_j)
    #         # distanceij=bikeflow_gdf.head(1)['geometry'].distance(bikeflow_gdf.head(1)['geometry'])
    #         # print(distanceij)
    #         if  distanceij < adjacency_threshold_m:
    #             edge_list_adjacent.append([i, j])
    # edge_index_adjacent = torch.tensor(np.array(edge_list_adjacent).T, dtype=torch.long)
    # print(edge_list_adjacent) ;print('edge_index_adjacent',edge_index_adjacent.shape)
    
    # b) ('station', 'near_poi_category', 'poi_category') - Example: Random connections to POI categories
    edge_list_has_poi = []
    for station_index in range(num_stations):
        for poi_cat_index in range(num_poi_categories):
            if poi_counts[station_index, poi_cat_index] > 0:
                edge_list_has_poi.append([station_index, poi_cat_index])
    edge_index_has_poi = torch.tensor(np.array(edge_list_has_poi).T, dtype=torch.long)
    
    data = HeteroData()
    data['station'].x = torch.tensor(poi_counts)
    data['census_zone'].x = torch.tensor(census_features, dtype=torch.float32)
    data['poi_category'].x = torch.tensor(features_poi_category)  
    # data['station', 'adjacent_to', 'station'].edge_index = edge_index_adjacent
    data['station', 'near_poi_category', 'poi_category'].edge_index = edge_index_has_poi
    data['station', 'located_in', 'census_zone'].edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
 
    return data


 
# ==================================================
#  Sequence creation
# ==================================================
def create_sequences_and_dataset_arrays(node_dynamicfeatures_optimized,node_staticfeatures_optimized,sequence_length,prediction_length):

    # 1. flows features
    flows =node_dynamicfeatures_optimized[...,0:1].astype(np.float32)
    # print('flows',flows.shape)
    # weather
    weather_features = node_dynamicfeatures_optimized[...,1:6].astype(np.float32)
    hours_sequence =node_dynamicfeatures_optimized[...,6:7].astype(int)
    hours_one_hot = np.eye(24)[hours_sequence].astype(np.float32) # Shape (num_timesteps, 1, num_hours_of_day)
    hours_one_hot = hours_one_hot.reshape(*hours_sequence.shape[:2], 24)
    days_sequence = (node_dynamicfeatures_optimized[...,7:8]).astype(int)
    # days_sequence = np.random.randint(1, 7, size=days_sequence.shape)
    # print(np.unique(node_dynamicfeatures_optimized[...,7:8]))
    # print(days_sequence.shape) # (5833, 1615, 17)
    days_one_hot = np.eye(7)[days_sequence].astype(np.float32) 
    days_one_hot = days_one_hot.reshape(*days_sequence.shape[:2], 7)
    # print('days_one_hot',days_one_hot.shape) # (5833, 1615, 1, 7)

    # census data
    census_features=node_staticfeatures_optimized[...,8:].astype(np.float32).squeeze(0)
    sequences = []
    # print(flows.shape[0] - sequence_length-prediction_length)

    for i in range(flows.shape[0]-sequence_length-prediction_length):
        
        seq_features = flows[i:i+sequence_length]
       
        hourofday=hours_one_hot[i:i+sequence_length]
        dayofweek=days_one_hot[i:i+sequence_length]
        weather=weather_features[i:i+sequence_length]
        target = flows[i+sequence_length:i+sequence_length+prediction_length,...,0]
        sequences.append((seq_features, hourofday,dayofweek,weather,census_features,target))
    return sequences



# ==================================================
#  Dataset creating for model training and evualation
# ==================================================
class BikeFlowDataset(torch.utils.data.Dataset): 
    # 混合在空间上异质的数据x[flows and poi features]和时间上变化(空间上不变的)数据
    def __init__(self, sequences, data_metadata,is_train=True, scaler=None):
        self.sequences = sequences
        self.data_metadata = data_metadata
        
    # Compute min/max for each feature from training data
        if is_train:
            self.scaler = {}
            all_flows = np.concatenate([seq[0] for seq in sequences], axis=0)  # [N, 500, 1]
            all_weather = np.concatenate([seq[3] for seq in sequences], axis=0)  # [N, 500, 5]
            all_census = np.concatenate([seq[4] for seq in sequences], axis=0)  # [N, 500, 5]
            all_targets = np.concatenate([seq[5] for seq in sequences], axis=0)  # [N, 500]
            # print('all_census',all_flows.shape,all_census.shape,all_census[0,0],all_weather[0,0])
            # print(all_census.min(axis=(0)),all_census.max(axis=(0)))
          

            # Flows (feature 0 in station)
            self.scaler['flows'] = {'min': all_flows.min(), 'max': all_flows.max()}
            # POI counts (features 1-8 in station, from hetero_data)
            poi_counts = data_metadata['station'].x.cpu().numpy() # [500, 8]
            self.scaler['poi'] = {
                'min': poi_counts.min(axis=0),  # [8]
                'max': poi_counts.max(axis=0)   # [8]
            }
            # Weather (5 features)
            self.scaler['weather'] = {
                'min': all_weather.min(axis=(0, 1)),  # [5]
                'max': all_weather.max(axis=(0, 1))   # [5]
            }
            # census (5 features)
            self.scaler['census'] = {
                'min': all_census.min(axis=(0)),  # [5]
                'max': all_census.max(axis=(0))   # [5]
            }
            # print(self.scaler['census']['min'],self.scaler['census']['max'])
            
            # Targets (flows)
            self.scaler['target'] = {'min': all_targets.min(), 'max': all_targets.max()}
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for validation dataset (is_train=False)")
            self.scaler = scaler  # Reuse scaler from training dataset
          
            
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        
        features= self.sequences[idx]
        seq_flows,seq_hours,seq_days,weather,census,target=features[0],features[1],features[2],features[3],features[4],features[5]
        # print('seq_hours',seq_hours.shape)
        # Normalize flows (time-varying)
        seq_flows = (seq_flows - self.scaler['flows']['min']) / (self.scaler['flows']['max'] - self.scaler['flows']['min'] + 1e-8)
    
       # Normalize weather (per feature)
        weather= (weather - self.scaler['weather']['min']) / (self.scaler['weather']['max'] - self.scaler['weather']['min'] + 1e-8)
        weather = torch.tensor(weather, dtype=torch.float32)
        
        # Normalize weather (per feature)
        census= (census - self.scaler['census']['min']) / (self.scaler['census']['max'] - self.scaler['census']['min'] + 1e-8)
        census = torch.tensor(census, dtype=torch.float32)

        # Normalize target
        target= (target - self.scaler['target']['min']) / \
                      (self.scaler['target']['max'] - self.scaler['target']['min'] + 1e-8)
                      
        # Normalize POI counts (static, per feature)
        static_poi_counts = self.data_metadata['station'].x # [500, 8]
      
        # print("Station shape:",static_poi_counts.shape, "Device:", static_poi_counts.device,self.scaler['poi']['min'].device)
        static_poi_counts = (static_poi_counts.cpu().numpy() - self.scaler['poi']['min']) / (self.scaler['poi']['max'] - self.scaler['poi']['min'] + 1e-8)
        static_poi_counts=torch.tensor(static_poi_counts)
       
        # Combine normalized flows and POI counts
        # station_features = torch.cat([torch.tensor(seq_flows, dtype=torch.float32).to(device), 
        #                               poi_norm.repeat(seq_flows.shape[0], 1, 1)], dim=2)  # [12, 500, 9]

        # print(self.data_metadata['poi_category'].x.shape) # (8,8)
        
        x_dict = {
            # 'station': torch.tensor(seq_flows).to(device),
            'station': {
                'flows':  torch.tensor(seq_flows),  # [12, 1807, 1] for 'adjacent_to'
                'poi': static_poi_counts # [12, 1807, 8] for 'near_poi_category' repeat(seq_flows.shape[0], 1, 1)
            },
            'hour_of_day': torch.tensor(seq_hours),  # [12, 500, 24] 
            'day_of_week': torch.tensor(seq_days),   # [12, 500, 7] 
            'weather': torch.tensor(weather),  # [12, 500, 5] 
            'census':torch.tensor(census),
            'poi_category': self.data_metadata['poi_category'].x.unsqueeze(0), # Static POI, repeat for sequence length
        }
        

        return x_dict, torch.tensor(target)



# --- Collate Function for DataLoader 最终traindataset中的特征---
def collate_fn(batch):
    # batch is a list of tuples: [(x_dict_1, target_1), (x_dict_2, target_2), ...]
    batch_size = len(batch)
    x_dict_batched = {}
    targets_batched = []
    # Initialize lists to store batched tensors for each node type
    stationflow_features_list = []
    stationpoi_features_list = []
    poi_category_features_list = []
    hour_of_day_features_list = []
    day_of_week_features_list = []
    hour_of_weather_list=[]
    census_list=[]

    for x_dict, target in batch:
        stationflow_features_list.append(x_dict['station']['flows']) # Collect station features
        stationpoi_features_list.append(x_dict['station']['poi']) # Collect station features
        poi_category_features_list.append(x_dict['poi_category'])
        hour_of_day_features_list.append(x_dict['hour_of_day'])
        day_of_week_features_list.append(x_dict['day_of_week'])
        hour_of_weather_list.append(x_dict['weather'])
        targets_batched.append(target)
        census_list.append(x_dict['census'])
    # Stack features along batch dimension (dim=0)
    x_dict_batched['stationflow'] = torch.stack(stationflow_features_list, dim=0) # Shape: (batch_size, sequence_length, num_stations, feature_dim)
    x_dict_batched['stationpoi'] = torch.stack(stationpoi_features_list, dim=0) # Shape: (batch_size, sequence_length, num_stations, feature_dim)
    x_dict_batched['poi_category'] = torch.stack(poi_category_features_list, dim=0) # Shape: (batch_size, num_poi_categories, feature_dim)
    x_dict_batched['hour_of_day'] = torch.stack(hour_of_day_features_list, dim=0)
    x_dict_batched['day_of_week'] = torch.stack(day_of_week_features_list, dim=0)
    x_dict_batched['weather'] = torch.stack(hour_of_weather_list, dim=0)
    x_dict_batched['census'] = torch.stack(census_list, dim=0)
    targets_batched = torch.stack(targets_batched, dim=0) # Shape: (batch_size, num_stations)
    return x_dict_batched, targets_batched

def create_data_loaders(sequences, hetero_data_example, batch_size):
    """Split sequences and prepare PyTorch DataLoaders."""
    

    train_ratio = 0.7
    val_ratio = 0.15
    train_size = int(len(sequences) * train_ratio)
    val_size = int(len(sequences) * val_ratio)
    random.shuffle(sequences)  # Shuffle before splitting

    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]

    train_dataset = BikeFlowDataset(train_sequences, hetero_data_example, is_train=True)
    val_dataset = BikeFlowDataset(val_sequences, hetero_data_example, is_train=False, scaler=train_dataset.scaler)
    test_dataset = BikeFlowDataset(test_sequences, hetero_data_example, is_train=False, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader








 
