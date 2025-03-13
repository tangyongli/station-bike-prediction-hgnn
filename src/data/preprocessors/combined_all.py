
import numpy as np
import argparse
import os 
import rawflowloader 
import poi_loader 
import censusandweather_loader 
from nodefeatures_creation import prepare_node_features_and_targets_optimized

def process_bike_data(folder, tractspath,buffermeter, allfeaturesgdfpath):
    """
    Processes bike flow data, aggregates hourly flows, adds POI, weather, and census data.
    Build index for each bike station for 
    Returns:
        one gdf with shape:(num_timesteps*num_stations,num_features)
    """
    ## 1. Load raw bike flow dataset, and aggregate flow hourly, delete abnormal flows
    try:
        bikebymonth, bikestationloc = rawflowloader.readandcatcsvbymonth(folder)
        bike_flow_gdf = rawflowloader.aggrebyhour(bikebymonth,bikestationloc)
        print(bike_flow_gdf.loc[bike_flow_gdf['checkin_trips'].idxmax()])
        print(bike_flow_gdf.groupby('station_name')['flows'].sum().describe())
    except Exception as e:
        print(f"Error in raw data processing: {e}")
        return None

    ## 2. Fetch Poi data from json file using osmnx
    try:
        # Poi is static across all timesteps
        stations_unique = bike_flow_gdf.drop_duplicates(subset=['station_name'])
        if stations_unique.empty or 'station_name' not in stations_unique.columns:
            print("Error: No valid station data found for POI download.")
            return None
        # convert to projection coordinate before calculating distance
        stations_unique=stations_unique.to_crs('EPSG:2263')
        bikebystationbuffer = poi_loader.getosm_nyc(stations_unique, buffermeter)
        # Merge with time series data
        bike_flow_gdf = bike_flow_gdf.merge(bikebystationbuffer, suffixes=('', '_y'), on='station_name', how='left')
        columns_to_drop = [col for col in bike_flow_gdf.columns if col.endswith('_y')]
        bike_flow_gdf.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        print(bike_flow_gdf.sort_values(by=['station_name', 'day', 'hour']).head(2))
    except Exception as e:
        print(f"Error in POI data processing: {e}")
        return None

    ## 3. Add Weather data and census data
    try:
        bike_flow_gdf = censusandweather_loader.spatialjoinbikestationwithCensus(bike_flow_gdf, tractspath)
        bike_flow_gdf.to_parquet(allfeaturesgdfpath)
    except Exception as e:
        print(f"Error in census/weather data processing: {e}")
        return None
    
    return bike_flow_gdf


def main():
    """
    Main function to parse arguments and run the bike data processing.
    """
    parser = argparse.ArgumentParser(description="Process bike flow data.")
    parser.add_argument("--folder", type=str, default='Data/raw/2022-citibike-tripdata',help="Path to the folder containing raw bike data")
    parser.add_argument("--tractspath", type=str, default='Data/raw/nyct2020_25a/nyct2020.shp',help="Path to the census tracts shapefile")
    parser.add_argument("--allfeaturesgdfpath", type=str, default='Data/processed_data/bikefeaturesall.parquet')
    parser.add_argument("--buffermeter", type=int,default=100, help="Poi buffer distance of each station")
    
    parser.add_argument("--dynamicnodefeaturespath", type=str, default='Data/processed_data/inputsarrayforTGNN_dynamicfeatures(flowsweatherhourday).npy')
    parser.add_argument("--staticnodefeaturespath", type=str,default='Data/processed_data/inputsarrayforTGNN_staticfeatures(poicensus).npy')
    args = parser.parse_args()
    
    output_folder = os.path.dirname(args.allfeaturesgdfpath)
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder, exist_ok=True) 
            print(f"Created output directory: {output_folder}")
        except OSError as e:
            print(f"Error creating output directory {output_folder}: {e}")
            return

    processed_gdf = process_bike_data(args.folder, args.tractspath,args.buffermeter, args.allfeaturesgdfpath)
    if processed_gdf is not None:
        print(f"Bike data processing complete.  Data saved to {args.allfeaturesgdfpath}")
    else:
        print("Bike data processing failed.")
    ## Generate array features from gdf
    node_staticfeatures_optimized=prepare_node_features_and_targets_optimized(processed_gdf,'static')
    node_dynamicfeatures_optimized=prepare_node_features_and_targets_optimized(processed_gdf,'dynamic')
    np.save(args.dynamicnodefeaturespath,node_dynamicfeatures_optimized)
    np.save(args.staticnodefeaturespath,node_staticfeatures_optimized)


if __name__ == "__main__":
    main()
  