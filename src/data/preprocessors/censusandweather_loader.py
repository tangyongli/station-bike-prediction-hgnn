#%%
import geopandas as gpd
import pandas as pd
import numpy as np


def readcensus():
    pop=pd.read_csv('../../../Data/raw/census/ACSDT5Y2022.B01003-Data.csv',skiprows=1)[['Geography','Estimate!!Total']]
    income=pd.read_csv('../../../Data/raw/census/ACSDT5Y2022.B19013-Data.csv',skiprows=1)[['Geography','Estimate!!Median household income in the past 12 months (in 2022 inflation-adjusted dollars)']]
    transporting=pd.read_csv('../../../Data/raw/census/ACSST5Y2022.S0802-Data.csv',skiprows=1)[['Geography','Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over in households!!VEHICLES AVAILABLE!!No vehicle available']]
    soceoc=pop.merge(income,on='Geography',how='left').merge(transporting,on='Geography',how='left').assign(
        GEOID=lambda df_: df_.Geography.str.split('US').str[-1])\
    .drop(columns='Geography').replace('-', np.nan).dropna()
    # print(soceoc.columns)
    soceoc.columns=['pop','income','novehicle','GEOID']
    soceoc['novehicle']=soceoc['novehicle'].astype('float')
    soceoc['vehicleratio']=100-soceoc['novehicle']
    soceoc['income']=soceoc['income'].replace('250,000+','250000').astype('float')
    # print(transporting.shape,pop.shape,income.shape)
    return soceoc
    


def spatialjoinbikestationwithCensus(station,tractspath):
    censusdata=readcensus()
    station1=station.drop_duplicates(subset=['station_name']).drop(columns=['GEOID','Shape_Area'],errors='ignore')
   
    # '../Data/raw/nyct2020_25a/nyct2020.shp'
    tracts=gpd.read_file(tractspath)
    print(tracts.columns)
    if station1.crs!=tracts.crs:
        print(station1.crs,tracts.crs)
        station1=station1.to_crs(tracts.crs)
    bikestationJoin=gpd.sjoin(station1,tracts[['GEOID','Shape_Area','geometry']],how='left')
    print(bikestationJoin.columns)
    
    bikestation=station.merge(bikestationJoin[['station_name','GEOID','Shape_Area']],on='station_name',how='left')\
    .merge(censusdata[['GEOID','pop','income','vehicleratio']],on='GEOID',how='left')
    bikestation['popdensity']=bikestation['pop']/(bikestation['Shape_Area']/1e6)
    bikestation['per_income']=bikestation['income']/(bikestation['pop'])
    bikestation=bikestation[bikestation['income'].notna()]
    
    return bikestation
#%%
# bike_flow_gdf=gpd.read_file('../../../Data/processed_data/bikegdfbeforemodel.gpkg').query('day>"2022-04-01"').to_crs('EPSG:2263')
# #%%
# bike_flow_gdf1 = spatialjoinbikestationwithCensus(bike_flow_gdf, '../../../Data/raw/nyct2020_25a/nyct2020.shp')
bike_flow_gdf1=gpd.read_file('Data/processed_data/bikegdfbeforemodeladdweathercensus.gpkg')


# %%

# %%
import numpy as np
import torch
from torch_geometric.data import HeteroData,Batch
import random
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
    print('edge_index_adjacent',edge_index_adjacent.shape)
    
    # b) ('station', 'near_poi_category', 'poi_category') - Example: Random connections to POI categories
    edge_list_has_poi = []
    for station_index in range(num_stations):
        for poi_cat_index in range(num_poi_categories):
            if poi_counts[station_index, poi_cat_index] > 0:
                edge_list_has_poi.append([station_index, poi_cat_index])
    edge_index_has_poi = torch.tensor(np.array(edge_list_has_poi).T, dtype=torch.long)
    

    # census data
    # census_features=node_staticfeatures_optimized[...,8:].astype(np.float32).squeeze(0)
    
    data = HeteroData()
    data['station_poi'].x = torch.tensor(poi_counts)
    data['census_zone'].x = torch.tensor(census_features, dtype=torch.float32)
    data['poi_category'].x = torch.tensor(features_poi_category)  
    data['station', 'adjacent_to', 'station'].edge_index = edge_index_adjacent
    data['station', 'near_poi_category', 'poi_category'].edge_index = edge_index_has_poi
    data['station', 'located_in', 'census_zone'].edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
 
    return data

# ==================================================
#  Sequence creation
# ==================================================
def create_sequences_and_dataset_arrays(node_dynamicfeatures_optimized,sequence_length,prediction_length):
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

    # # census data

    # print(flows.shape[0] - sequence_length-prediction_length)

    for i in range(flows.shape[0]-sequence_length-prediction_length):
        
        flow_features = flows[i:i+sequence_length]
       
        hourofday=hours_one_hot[i:i+sequence_length]
        dayofweek=days_one_hot[i:i+sequence_length]
        weather=weather_features[i:i+sequence_length]
        target = flows[i+sequence_length:i+sequence_length+prediction_length,...,0]
    
        sequences.append((flow_features, hourofday,dayofweek,weather,target))
    return sequences
#%%
class BikeFlowDataset(torch.utils.data.Dataset): 
    def __init__(self, sequences, hetero_data_metadata, is_train=True, scaler=None):
        self.sequences = sequences
        self.hetero_data_metadata = hetero_data_metadata
        self.scaler = scaler
        if is_train and self.scaler is None:  # Only calculate scaler on first init (training)
            self.scaler = self.calculate_scaler()

    def calculate_scaler(self):
        scaler = {}
        all_flows = np.concatenate([seq[0] for seq in sequences], axis=0)  # [N, 500, 1]
        all_weather = np.concatenate([seq[3] for seq in sequences], axis=0)  # [N, 500, 5]
        all_targets = np.concatenate([seq[4] for seq in sequences], axis=0)  # [N, 500]
        # print(all_census.min(axis=(0)),all_census.max(axis=(0)))

        # Flows (feature 0 in station)
        scaler['flows'] = {'min': all_flows.min(), 'max': all_flows.max()}
        # POI counts (features 1-8 in station, from hetero_data)
        poi_counts = self.hetero_data_metadata ['station_poi'].x.cpu().numpy() # [500, 8]
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
        all_census = self.hetero_data_metadata ['census_zone'].x.cpu().numpy() # [500, 8]
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

    def __getitem__(self, idx):
       
        seq_flows = np.concatenate([seq[0] for seq in sequences], axis=0)  # [N, 500, 1]
        hour_of_day=np.concatenate([seq[1] for seq in sequences], axis=0)
        day_of_week=np.concatenate([seq[2] for seq in sequences], axis=0)
        seq_weather = np.concatenate([seq[3] for seq in sequences], axis=0)  # [N, 500, 5]
        target = np.concatenate([seq[4] for seq in sequences], axis=0)  # [N, 500]
        # print(seq_flows.shape,hour_of_day.shape,day_of_week.shape,seq_weather.shape,target.shape)
        
        x_dict = {}
        # --- Static features (Normalize) ---
        x_dict['station_poi'] = (self.hetero_data_metadata['station_poi'].x - self.scaler['poi']['min']) / (
                    self.scaler['poi']['max'] - self.scaler['poi']['min'] + 1e-8)
        
        x_dict['census_zone'] = (self.hetero_data_metadata['census_zone'].x - self.scaler['census_zone']['min']) /(
            self.scaler['census_zone']['max'] - self.scaler['census_zone']['min'] + 1e-8
        )
        
        # print(self.hetero_data_metadata['census_zone'].x.shape,x_dict['census_zone'].shape,self.scaler['census']['min'].shape)
        # --- Dynamic features (Normalize) ---
        
        x_dict['station_flows'] = (seq_flows - self.scaler['flows']['min']) / (
            self.scaler['flows']['max'] - self.scaler['flows']['min'] + 1e-8)
        
        x_dict['weather'] = (seq_weather- self.scaler['weather']['min']) / (
                    self.scaler['weather']['max'] - self.scaler['weather']['min'] + 1e-8)
        
        print(self.hetero_data_metadata['weather'].x.shape,x_dict['weather'].shape,self.scaler['weather']['min'].shape)
        x_dict['hour_of_day']=hour_of_day
        x_dict['day_of_week']=day_of_week
        # Normalize target
        target = (target - self.scaler['target']['min']) / (
                    self.scaler['target']['max'] - self.scaler['target']['min'] + 1e-8)
        x_dict['poi_category'] = self.hetero_data_metadata['poi_category'].x.cpu().numpy()
        
        # --- Create the combined data dictionary ---
        data_dict = {
                    'features': x_dict,
                    'edge_indices': self.hetero_data_metadata.edge_index_dict,  
                }
        return data_dict, target

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
    """Split sequences and prepare PyTorch DataLoaders."""
    

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# (5833, 1615, 8) (1, 1615, 11)
# node_staticfeatures_optimized=np.repeat(node_staticfeatures_optimized, node_dynamicfeatures_optimized.shape[0], axis=0)
# Concatenate along the feature dimension (axis -1, the last axis)
# node_features_optimized = np.concatenate([node_dynamicfeatures_optimized, node_staticfeatures_optimized], axis=-1)
#%%
node_dynamicfeatures_optimized=np.load('Data/processed_data/inputsarrayforTGNN_dynamicfeatures(flowsweatherhourday).npy')
# 2.2 Create HeteroData object
hetero_data_example = create_hetero_data_arrays(bike_flow_gdf1,8)
#%%
import pickle
def save_hetero_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
save_hetero_data(hetero_data_example,'Data/processed_data/hetero_data_addcensus1.pkl')  
hetero_data_example.metadata()
# %%
#2.3 Create sequences
sequences = create_sequences_and_dataset_arrays(node_dynamicfeatures_optimized,12,3)
# 2.4 Split into training and validation sets
train_dataset,val_dataset,test_dataset,train_dataloader, val_dataloader, test_dataloader=create_data_loaders(sequences, hetero_data_example, 64)

print('wwwww')
x_dict, target = val_dataset[0]
#%%

print(train_dataset.scaler['census_zone']['max'])
print("x_dict['station']['census'].shape:", x_dict['features']['station_flows'].shape)
print("x_dict['station']['poi'].shape:", x_dict['features']['station_poi'].shape)
print("x_dict['station']['poi'].shape:", x_dict['features']['census_zone'].shape)

print("x_dict['station']['weather'].shape:", x_dict['features']['weather'].shape)
print("x_dict['station']['hour_of_day'].shape:", x_dict['features']['hour_of_day'].shape)
print("x_dict['station']['day_of_week'].shape:", x_dict['features']['day_of_week'].shape)

print("target.shape:", target.shape)
print("x_dict device:", next(iter(x_dict.values())).device if isinstance(next(iter(x_dict.values())), torch.Tensor) else next(iter(next(iter(x_dict.values())).values())).device)
print("target device:", target.device)

# %%
