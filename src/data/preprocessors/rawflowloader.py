import os
import pandas as pd
import glob
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def readandcatcsvbymonth(folder):
    '''
    Read raw bike data;
    Process pickup and drop off time format to datetime objects;
    Calculate the median geographical location (latitude and longitude) for each station, using the locations from all trips associated with that station.
   
    '''
    csv_files = glob.glob(os.path.join(folder, '2022*.csv'))
  
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['month'] = file.split('-')[1][-2:]
        # print(df['month'])
        df_list.append(df)
    df = pd.concat(df_list)
    
    print('lendf',len(df),df.head(1))
    
    # Convert string columns to datetime objects
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df['ride_duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    df=df.query('180>ride_duration>0')

    # Extract day and hour for pickup time
    df['pickup_day'] = df['started_at'].dt.date  
    df['pickup_day']=pd.to_datetime(df['pickup_day'], format='%m/%d/%Y')
    df['pickup_hour'] = df['started_at'].dt.hour 

    # Extract day and hour for dropoff time
    df['dropoff_day'] = df['ended_at'].dt.date  
    df['dropoff_day']=pd.to_datetime(df['dropoff_day'], format='%m/%d/%Y')
    df['dropoff_hour'] = df['ended_at'].dt.hour 
    
    # Accquire the geographical locations of each station
    dfpickup=df[['start_station_name','start_lng','start_lat']].rename(columns={
        'start_station_name':'station_name',
        'start_lng':'lon',
        'start_lat':'lat'
    })
    dfdropoff=df[['end_station_name','end_lng','end_lat']].rename(columns={
        'end_station_name':'station_name',
        'end_lng':'lon',
        'end_lat':'lat'
    })
    dfstation_location=pd.concat([dfpickup,dfdropoff])
    dfstation_location['lon']=dfstation_location.groupby('station_name')['lon'].transform('median')
    dfstation_location['lat']=dfstation_location.groupby('station_name')['lat'].transform('median')
    dfstation_location.drop_duplicates(subset='station_name',inplace=True)

    
    return df,dfstation_location


def addrowfornoflow_hour(df):
    '''
    For any hour with no recorded flows, add a row with flow values set to zero.
    '''
    ## create one dataframe including complete hourly data from the start day to the end day
    min_date = df['day'].min()
    max_date = df['day'].max()
    df['datetime']=pd.to_datetime(df['day']) + pd.to_timedelta(df['hour'], unit='h')
    all_hours = pd.date_range(start=min_date.floor('D'), end=max_date.ceil('D'), freq='H')
    all_hours_df = pd.DataFrame({'datetime': all_hours})
    all_stations = df['station_name'].unique()
    all_stations = pd.merge(
            pd.DataFrame({'station_name': all_stations}),
            all_hours_df,
            how='cross')
    merged_df = pd.merge(
            all_stations,
            df,
            on=['station_name', 'datetime'], 
            how='left')

    ## Fill missing flow values with 0, assume missing flows represents zeros recordings
    merged_df['flows'] = merged_df['flows'].fillna(0)
    merged_df = merged_df.sort_values(by=['station_name', 'datetime']).reset_index(drop=True)
    merged_df['hour'] = merged_df['datetime'].dt.hour
    merged_df['day'] = merged_df['datetime'].dt.date
    merged_df['checkin_trips'] = merged_df['checkin_trips'].fillna(0)
    merged_df['checkout_trips'] = merged_df['checkout_trips'].fillna(0)
  

    return merged_df


## fill nan values[day of week, lon, lat]for no flows data within hour
def fill_na_with_group_value(series):
    first_valid = series.dropna().iloc[0] if not series.dropna().empty else np.nan 
    return series.fillna(first_valid)

def build_station_index(gdf):
    gdf_noduplicates=gdf.drop_duplicates(subset='station_name').reset_index(drop=True)
    gdf_noduplicates['station_fid'] = gdf_noduplicates.index
    gdf=gdf.merge(gdf_noduplicates[['station_name','station_fid']],on='station_name',how='left')
    print(gdf['station_fid'].nunique())
    return gdf

def aggrebyhour(df,gdf):
    ## Aggregate flows by station,day, hour, and bike_id
    df_checkin = df.groupby(['start_station_name','pickup_day', 'pickup_hour']).agg({
        # 'ride_duration':'mean',
        'ride_id':'size',
        'month':'first',
       
    }).reset_index().rename(columns={'start_station_name':'station_name',
                                     'ride_id':'checkin_trips',
                                 
                                     'pickup_day':'day',
                                     'pickup_hour':'hour',
                                     })
    df_checkout = df.groupby(['end_station_name','dropoff_day', 'dropoff_hour']).agg({
        'ride_id':'size',
    }).reset_index().rename(columns={'end_station_name':'station_name',
                                     'ride_id':'checkout_trips',
                                 
                                     'dropoff_day':'day',
                                     'dropoff_hour':'hour'
                                     })
    
    df=df_checkin.merge(df_checkout,on=['station_name','day','hour'],how='outer')
    df['checkin_trips']=df['checkin_trips'].fillna(0)
    df['checkout_trips']=df['checkout_trips'].fillna(0)
    df['flows']=df['checkin_trips']+df['checkout_trips']

    
    ## Convert the 'date_col' to datetime objects
    df['day'] = pd.to_datetime(df['day'])
    # Calculate the day of the week (Monday=0, Sunday=6)
    df['day of week'] = df['day'].dt.dayofweek
    df['day of week']=df['day of week'].astype(int)
    print(df['hour'].unique(),df['day of week'].unique())
    
    ## Add hour, day, flows data for no recording flows
    df=addrowfornoflow_hour(df)
    df=df.merge(gdf,on='station_name',how='left')
    
    ## Fill nan to zero if without data in this hour
    df['lon'] = df.groupby('station_name')['lon'].transform(fill_na_with_group_value)
    df['lat'] = df.groupby('station_name')['lat'].transform(fill_na_with_group_value)
    df['day of week'] = df.groupby('day')['day of week'].transform(fill_na_with_group_value)
    if isinstance(df, gpd.GeoDataFrame):
        print("crs",df.crs)
    else:
        df['geometry']=df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        df=gpd.GeoDataFrame(df,geometry='geometry',crs='epsg:4326')
    
    ## Build unique station_fid for retrieving geo-stations when predicting
    merged_df=build_station_index(merged_df)

    return df


