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
import numpy as np
from geo_funcs import shp_to_mask


## Read shapefiles
mro_shp = gpd.read_file('/Users/yiannabekris/Documents/energy_data/NERC/MRO.shp')
npcc_shp = gpd.read_file('/Users/yiannabekris/Documents/energy_data/NERC/NPCC.shp')
rfc_shp = gpd.read_file('/Users/yiannabekris/Documents/energy_data/NERC/RFC.shp')
serc_shp = gpd.read_file('/Users/yiannabekris/Documents/energy_data/NERC/SERC.shp')
tre_shp = gpd.read_file('/Users/yiannabekris/Documents/energy_data/NERC/TRE.shp')
wecc_shp = gpd.read_file('/Users/yiannabekris/Documents/energy_data/NERC/WECC.shp')

## These regions have subregions so this is to aggregate all subregions into one
serc_shp = serc_shp.dissolve()
mro_shp = mro_shp.dissolve()

## Save the aggregated regions into a new shapefile
serc_shp.to_file('/Users/yiannabekris/Documents/energy_data/NERC/SERC.shp')
mro_shp.to_file('/Users/yiannabekris/Documents/energy_data/NERC/MRO.shp')

## Merge/combine multiple shapefiles into one
nerc_all = gpd.pd.concat([mro_shp, npcc_shp, rfc_shp, serc_shp, tre_shp, wecc_shp])
 
#Export merged geodataframe into shapefile
nerc_all.to_file("/Users/yiannabekris/Documents/energy_data/NERC/nerc_no_sub.shp")


## Read shapefiles
spp_shp = gpd.read_file('/Users/yiannabekris/Documents/energy_data/NERC/MRO/SPP.shp')
mro_shp = gpd.read_file('/Users/yiannabekris/Documents/energy_data/NERC/MRO/MRO.shp')
 
## Merge/combine multiple shapefiles into one
mro_all = gpd.pd.concat([spp_shp, mro_shp])

## Export merged geodataframe into shapefile
mro_all.to_file("/Users/yiannabekris/Documents/energy_data/NERC/MRO.shp")

## Set the path to write the new files
write_path = "/Users/yiannabekris/Documents/energy_data/NERC/"

## Load latitude and longitude NumPy arrays
lat = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/tmax_lat_1979_2020.npy')
lon = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/tmax_lon_1979_2020.npy')

## Create the SERC shapefile
serc_shp = f"{write_path}SERC.shp"
serc_mask = shp_to_mask(serc_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/serc_mask.npy",serc_mask)

## Create the MRO shapefile
mro_shp = f"{write_path}MRO.shp"
mro_mask = shp_to_mask(mro_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/mro_mask.npy",mro_mask)

## Set write path to folder
write_path = "/Users/yiannabekris/Documents/energy_data/NERC"

## Filename of input geojson file
nerc_json = gpd.read_file("/Users/yiannabekris/Documents/energy_data/NERC_Regions.geojson")

## These are the sub regions
nerc_sub = nerc_json[['NAME','geometry']]

## All NERC regions
nerc_regions = nerc_sub.dissolve()

## Loop through each region to get the geometries
for region in nerc_regions:
      gdf_nerc = nerc_regions[region][0]
      print(gdf_nerc)
      # gdf_nerc.to_file(f'{write_path}_{nerc_regions[region]}.shp')  


## Read the new NERC shapefile with no subregions
nerc_regions = gpd.read_file("/Users/yiannabekris/Documents/energy_data/NERC/nerc_no_sub.shp")

## Loop through each region and get the string of the name
nerc_names = []
for name in nerc_regions['NAME']:
    namestr = name[-5:-1]
    if name[-5] == '(':
        namestr = name[-4:-1]
    nerc_names.append(namestr)
    
## Save the new shapefiles and masks
frcc_shp = f"{write_path}FRCC.shp"
frcc_mask = shp_to_mask(frcc_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/frcc_mask.npy",frcc_mask)

mro_shp = f"{write_path}MRO.shp"
mro_mask = shp_to_mask(mro_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/mro_mask.npy",mro_mask)

npcc_shp = f"{write_path}NPCC.shp"
npcc_mask = shp_to_mask(npcc_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/npcc_mask.npy",npcc_mask)

rfc_shp = f"{write_path}RFC.shp"
rfc_mask = shp_to_mask(rfc_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/rfc_mask.npy",rfc_mask)

serc_shp = f"{write_path}SERC.shp"
serc_mask = shp_to_mask(serc_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/serc_mask.npy",serc_mask)

spp_shp = f"{write_path}SPP.shp"
spp_mask = shp_to_mask(spp_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/spp_mask.npy",spp_mask)

tre_shp = f"{write_path}TRE.shp"
tre_mask = shp_to_mask(tre_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/tre_mask.npy",tre_mask)

wecc_shp = f"{write_path}WECC.shp"
wecc_mask = shp_to_mask(wecc_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/wecc_mask.npy",wecc_mask)

nerc_shp = r"/Users/yiannabekris/Documents/energy_data/NERC/nerc_no_sub.shp"
nerc_mask = shp_to_mask(nerc_shp,lat,lon)
np.save("/Users/yiannabekris/Documents/energy_data/nerc_masks/nerc_no_sub_mask.npy",nerc_mask)
