import numpy as np

def prepare_node_features_and_targets_optimized(stations_gdf,tag):
    """
    Optimized function to prepare node features and targets for T-GNN,

    """
    # stations_gdf=stations_gdf.drop_duplicates(subset='station_name')
    stations_gdf['day of week']= stations_gdf['day of week']-1
    station_names_list = stations_gdf['station_name'].unique()
    time_steps_list = stations_gdf['datetime'].unique()
    num_stations = len(station_names_list)
    num_time_steps = len(time_steps_list)
   
    
    # 1. Create Station Name to Index Mapping (Same as before - efficient lookup)
    station_name_to_index = {name: index for index, name in enumerate(station_names_list)}

    # 2. Group stations_gdf by 'datetime' for efficient timestep access
    grouped_by_time = stations_gdf.groupby('datetime')
    if tag=='static':
        timestep=[time_steps_list[0]]
        node_features = np.zeros((1,num_stations, 11), dtype=np.float32)
    else:
        timestep=time_steps_list
        node_features = np.zeros((num_time_steps,num_stations, 8), dtype=np.float32)
    
    #  迭代需要为一个list, get[0]后是一个值，需要增加[]
    for t_idx, time_step in enumerate(timestep):
      
        try:
            hourly_gdf = grouped_by_time.get_group(time_step) # Efficiently get data for this timestep
        except KeyError: # Handle cases where a timestep might be missing in grouped data
            raise ValueError(f"No data found for time step: {time_step}")

        # 3. Create a DataFrame indexed by 'station_name' for fast station lookup within the hour
        hourly_station_data_indexed = hourly_gdf.set_index('station_name')

        for station_name in station_names_list:
            s_idx = station_name_to_index[station_name]
            try:
                station_data_series = hourly_station_data_indexed.loc[station_name] # Fast lookup by station name
            except KeyError: # Handle cases where a station might be missing for a timestep
                raise ValueError(f"No data for station '{station_name}' at time step: {time_step}")

            feature_index = 0 # Keep track of feature index for direct array assignment
            
            if tag!='static':
                # flows features
                node_features[t_idx, s_idx, feature_index] = station_data_series['flows']
                feature_index += 1

                # Weather features (Directly assign values - no list appending)
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

                # Time features (Direct numerical values as in your feature dimension definition)
                node_features[t_idx, s_idx, feature_index] = station_data_series['hour']
                feature_index += 1
                node_features[t_idx, s_idx, feature_index] = station_data_series['day of week']
                feature_index += 1
            else:
               # Static POI features (Directly assign values)
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

    return node_features

