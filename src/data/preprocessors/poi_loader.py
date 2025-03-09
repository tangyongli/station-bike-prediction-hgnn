import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from pyproj import CRS
import json



filepath='Data/raw/poi/poi_categories_exact_column_format.json'
with open(filepath, 'r', encoding='utf-8') as f:  
            formatted_json_string = f.read()  
            poi_categories_data = json.loads(formatted_json_string)
            poi_categories_data = poi_categories_data['poi_categories_tags']



def getosm_nyc(stations_df,buffer_distance_meters):
    """
    Downloads POIs from OpenStreetMap for New York City based on a  JSON configuration.
    """
    city_name = "New York City, New York, USA"
    poi_categories_data = json.loads(formatted_json_string)['poi_categories_tags']
    combined_tags = {}
    category_tag_mapping={}

    for category, tags_list in poi_categories_data.items():
        for tag_dict in tags_list:
            for tag_key, tag_value in tag_dict.items():
                if tag_key not in combined_tags:
                    combined_tags[tag_key] = [] 

                if isinstance(tag_value, bool):
                    #if it is True no need add to list. just set combined_tags[tag_key]
                    if tag_value:
                         combined_tags[tag_key] = True
                elif isinstance(tag_value, str):
                    #check combined_tags[tag_key]  is not True, if True no need check list.
                    if combined_tags[tag_key] is not True:
                        if tag_value not in combined_tags[tag_key]:
                            combined_tags[tag_key].append(tag_value)
                            category_tag_mapping[(tag_key, tag_value)] = category  #add map

    # print(combined_tags)
    try:
        nyc_pois_gdf = ox.features_from_place(city_name, tags=combined_tags)
        if nyc_pois_gdf.empty:
            print("Warning: No POIs found in NYC for specified categories.")
            return
        print(f"Downloaded {len(nyc_pois_gdf)} POIs for {city_name}.")
        # print(nyc_pois_gdf)
    except Exception as e:
        print(f"Error downloading POIs for NYC: {e}")
    
    # --- Spatial Join and Counting ----
    
    # stations_df['geometry']=stations_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    # stations_df=gpd.GeoDataFrame(stations_df,geometry='geometry',crs='epsg:4326')
    # expected_crs=CRS('EPSG:2263')
    # if stations_df.crs!=expected_crs:
    #     stations_df=stations_df.to_crs('epsg:2263')
    #     # print(f"Stations CRS reprojected to: {stations_df.crs}")
    
    ## Build buffer for each station
    stations_df[f'buffer{buffer_distance_meters}'] = stations_df.geometry.buffer(buffer_distance_meters)
    stations_df=stations_df.drop(columns='geometry')
    stations_df.set_geometry(f'buffer{buffer_distance_meters}',inplace=True)
    stations_df=stations_df.to_crs('epsg:4326')
    
    ## Spatial join station buffer with all poi data in the NYC, only counting poi within station buffer
    try: 
        # Use 'intersects' for points within polygons.  Could also use 'within'.
        joined_gdf = stations_df.sjoin(nyc_pois_gdf, how="left", predicate="intersects")

        # print('joingdf:', joined_gdf.head(2))
        # 2. Count POIs per category, per station
        # Iterate through the original categories and create count columns
        for category in poi_categories_data.keys():
            joined_gdf[category] = 0  # Initialize count column
        # Efficient way to count POIs
        for index, row in joined_gdf.iterrows():
            for key, value in category_tag_mapping.items():
                if row[key[0]] == key[1]:
                    joined_gdf.loc[index, value] += 1
        # 3. Aggregate to get counts per station (groupby the original station index)
        station_counts = joined_gdf.groupby(level=0)[list(poi_categories_data.keys())].sum()
        # 4. Join the counts back to the original station_df
        #    Ensure we don't introduce duplicate columns.
        stations_df = stations_df.join(station_counts, lsuffix='_caller', rsuffix='_other')
        stations_df = stations_df.loc[:, ~stations_df.columns.duplicated()].copy()
        return stations_df

    except Exception as e: 
        print(f"Error during spatial join: {e}")
        return nyc_pois_gdf # Return the downloaded POI data if spatial join fails
   
