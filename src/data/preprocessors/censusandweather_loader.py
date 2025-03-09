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

