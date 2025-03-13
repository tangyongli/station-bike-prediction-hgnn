#%%
import numpy as np
import geopandas as gpd

def prepare_node_features_and_targets_optimized(stations_gdf,tag):
    """
    Optimized function to prepare node features and targets for H-TGNN or GRU.
    Return:
    Static array: (num_stations, num_staticfeatures)
    Dynamic array: (num_hourly_timesteps,num_stations, num_dynamicfeatures)

    """
    if stations_gdf['day of week'].max()==7:
        stations_gdf['day of week']= stations_gdf['day of week']-1
    station_names_list = stations_gdf['station_name'].unique()
    time_steps_list = stations_gdf['datetime'].unique()
    num_stations = len(station_names_list)
    num_time_steps = len(time_steps_list)
   
    
    # 1. Create Station Name to Index Mapping
    station_name_to_index = {name: index for index, name in enumerate(station_names_list)}
    # 2. Group stations_gdf by 'datetime'
    grouped_by_time = stations_gdf.groupby('datetime')
    ## POI,Census(pop_density,income,vehicle rate),station_fid
    ## enumerate迭代需要为一个list, list[0]后是一个值，需要增加[]
    if tag=='static':
        timestep=[time_steps_list[0]]
        node_features = np.zeros((1,num_stations, 12), dtype=np.float32)
    ## Weather,flows,days,hours,station_fid
    else:
        timestep=time_steps_list
        node_features = np.zeros((num_time_steps,num_stations, 9), dtype=np.float32)
   
    for t_idx, time_step in enumerate(timestep):
        try:
            hourly_gdf = grouped_by_time.get_group(time_step) 
        except KeyError: 
            raise ValueError(f"No data found for time step: {time_step}")
        # 3. Create a DataFrame indexed by 'station_name' for fast station lookup within the hour
        hourly_station_data_indexed = hourly_gdf.set_index('station_name')

        for station_name in station_names_list:
            s_idx = station_name_to_index[station_name]
            try:
                station_data_series = hourly_station_data_indexed.loc[station_name]
            except KeyError:
                raise ValueError(f"No data for station '{station_name}' at time step: {time_step}")

            ## start to write features from feature 0 to feature n
            feature_index = 0 
            if tag!='static':
                # flows features
                node_features[t_idx, s_idx, feature_index] = station_data_series['flows']
                feature_index += 1

                # Weather features
                node_features[t_idx, s_idx, feature_index] = station_data_series['temperature_2m (°C)']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['precipitation (mm)']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['cloud_cover (%)']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['visibility (m)']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['wind_speed_10m (km/h)']
                feature_index += 1

                # Time features 
                node_features[t_idx, s_idx, feature_index] = station_data_series['hour']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['day of week']
                feature_index += 1
                
                # station_index
                node_features[t_idx, s_idx, feature_index] = station_data_series['station_fid']
                feature_index += 1
                
            else: 
            #    # Static POI features
                node_features[t_idx, s_idx, feature_index] = station_data_series['work_education']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['residential']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['commercial_retail']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['food']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['Recreation and Leisure']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['Health and Medicine']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['Art and Entertainment']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['Travel and Transportation']
                feature_index += 1
    
                # Census data features
                node_features[t_idx, s_idx, feature_index] = station_data_series['popdensity']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['per_income']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['vehicleratio']
                feature_index += 1
                
                # station_index
                node_features[t_idx, s_idx, feature_index] = station_data_series['station_fid']
                print(station_data_series['station_fid'])
                feature_index += 1

    return node_features




