#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:24:44 2022

@author: yiannabekris

This takes shapefiles of the NERC regions and creates NumPy arrays
that can then be used as masks for data wrangling and plotting


"""

### Import packages
import geopandas as gpd
import pandas as pd
import numpy as np
from geo_funcs import shp_to_mask

# change the global options that Geopandas inherits from
pd.set_option('display.max_columns', None)


## Set the path to write the new files
write_path = "/Users/yiannabekris/Documents/energy_data/NERC/"

## Filename of input geojson file
nerc_json = gpd.read_file("/Users/yiannabekris/Documents/energy_data/NERC_Regions.geojson")


## Load latitude and longitude NumPy arrays
lat = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/tmax_lat_1979_2020.npy')
lon = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/tmax_lon_1979_2020.npy')


## These are the sub regions
nerc_sub = nerc_json[['NAME','geometry']]

## All NERC regions
nerc_regions = nerc_sub.dissolve()


## Loop through each region to get the geometries
for i, region in nerc_regions.iterrows():
    gdf_nerc = region
    name = nerc_regions['NAME'][i]
    namestr = name[-5:-1]
    if name[-5] == '(':
        namestr = name[-4:-1]
        

    # Create a dictionary from the Series to combine geometry and attribute data
    nerc_data = {'NAME': nerc_regions['NAME'][i],
                 'geometry': nerc_regions['geometry'][i]}
    
    # Create the GeoDataFrame
    nerc_gdf = gpd.GeoDataFrame(nerc_data)
    nerc_gdf.to_file(f'{write_path}{namestr}.shp')
    
    ## Save the new shapefiles and masks
    region_shp = f"{write_path}{namestr}.shp"
    region_mask = shp_to_mask(region_shp,lat,lon)
    
    lowercase_name = namestr.lower()
    np.save(f"/Users/yiannabekris/Documents/energy_data/nerc_masks/{lowercase_name}_mask.npy", region_mask)


    
