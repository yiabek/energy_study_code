#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:07:50 2022

@author: yiannabekris

This script molds the percent change data
into the shape of each NERC region and saves the data
as NumPy arrays which are plotted by figure2.py

"""

### Import packages
import pandas as pd
import numpy as np


### This takes the trend data and molds it into the shape of each NERC region
### This will then be used to make Figure 2
def map_data_create(trend_data, metric):
    
    trends = trend_data
    
    ## Load masks to subset by regions
    mro_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/mro_mask.npy")
    npcc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/npcc_mask.npy")
    rfc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/rfc_mask.npy")
    serc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/serc_mask.npy")
    tre_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/tre_mask.npy")
    wecc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/wecc_mask.npy")
    
    trends = trends.rename(columns={"NERC.Region": "NERC Region", "Event.Type": "Event Type"})
    
    ## Mask percent change by region
    mro_vals = trends.loc[(trends['NERC Region']=='MRO')]
    
    mro_jja_hot = mro_vals.loc[(mro_vals['Season']=='JJA')]
    
    mro_jja_hot_map = mro_jja_hot['Percent Change'].values * mro_mask
    
    npcc_vals = trends.loc[(trends['NERC Region']=='NPCC')]
    
    npcc_jja_hot = npcc_vals.loc[(npcc_vals['Season']=='JJA')]
    
    npcc_jja_hot_map = npcc_jja_hot['Percent Change'].values * npcc_mask
    
    npcc_vals = trends.loc[(trends['NERC Region']=='NPCC')]
    
    npcc_jja_hot = npcc_vals.loc[(npcc_vals['Season']=='JJA')]
    
    npcc_jja_hot_map = npcc_jja_hot['Percent Change'].values * npcc_mask
    
    rfc_vals = trends.loc[(trends['NERC Region']=='RFC')]
    
    rfc_jja_hot = rfc_vals.loc[(rfc_vals['Season']=='JJA')]
    
    rfc_jja_hot_map = rfc_jja_hot['Percent Change'].values * rfc_mask
    
    rfc_vals = trends.loc[(trends['NERC Region']=='RFC')]
    
    rfc_jja_hot = rfc_vals.loc[(rfc_vals['Season']=='JJA')]
    
    rfc_jja_hot_map = rfc_jja_hot['Percent Change'].values * rfc_mask
    
    serc_vals = trends.loc[(trends['NERC Region']=='SERC')]
    
    serc_jja_hot = serc_vals.loc[(serc_vals['Season']=='JJA')]
    
    serc_jja_hot_map = serc_jja_hot['Percent Change'].values * serc_mask
    
    tre_vals = trends.loc[(trends['NERC Region']=='TRE')]
    
    tre_jja_hot = tre_vals.loc[(tre_vals['Season']=='JJA')]
    
    tre_jja_hot_map = tre_jja_hot['Percent Change'].values * tre_mask
    
    wecc_vals = trends.loc[(trends['NERC Region']=='WECC')]
    
    wecc_jja_hot = wecc_vals.loc[(wecc_vals['Season']=='JJA')]
    
    wecc_jja_hot_map = wecc_jja_hot['Percent Change'].values * wecc_mask
    
    combined = npcc_jja_hot_map + mro_jja_hot_map + rfc_jja_hot_map + serc_jja_hot_map + tre_jja_hot_map + wecc_jja_hot_map
    
    combined[combined == 0] = np.nan
    
    np.save(f'/Users/yiannabekris/Documents/energy_data/nparrays/jja_hot_{metric}_1_5_80_trends.npy',combined)
    
    
    
    ### Cold events
    mro_vals = trends.loc[(trends['NERC Region']=='MRO')]
    
    mro_djf_cold = mro_vals.loc[(mro_vals['Season']=='DJF')]
    
    mro_djf_cold_map = mro_djf_cold['Percent Change'].values * mro_mask
    
    npcc_vals = trends.loc[(trends['NERC Region']=='NPCC')]
    
    npcc_djf_cold = npcc_vals.loc[(npcc_vals['Season']=='DJF')]
    
    npcc_djf_cold_map = npcc_djf_cold['Percent Change'].values * npcc_mask
    
    npcc_vals = trends.loc[(trends['NERC Region']=='NPCC')]
    
    npcc_djf_cold = npcc_vals.loc[(npcc_vals['Season']=='DJF')]
    
    npcc_djf_cold_map = npcc_djf_cold['Percent Change'].values * npcc_mask
    
    rfc_vals = trends.loc[(trends['NERC Region']=='RFC')]
    
    rfc_djf_cold = rfc_vals.loc[(rfc_vals['Season']=='DJF')]
    
    rfc_djf_cold_map = rfc_djf_cold['Percent Change'].values * rfc_mask
    
    rfc_vals = trends.loc[(trends['NERC Region']=='RFC')]
    
    rfc_djf_cold = rfc_vals.loc[(rfc_vals['Season']=='DJF')]
    
    rfc_djf_cold_map = rfc_djf_cold['Percent Change'].values * rfc_mask
    
    serc_vals = trends.loc[(trends['NERC Region']=='SERC')]
    
    serc_djf_cold = serc_vals.loc[(serc_vals['Season']=='DJF')]
    
    serc_djf_cold_map = serc_djf_cold['Percent Change'].values * serc_mask
    
    tre_vals = trends.loc[(trends['NERC Region']=='TRE')]
    
    tre_djf_cold = tre_vals.loc[(tre_vals['Season']=='DJF')]
    
    tre_djf_cold_map = tre_djf_cold['Percent Change'].values * tre_mask
    
    wecc_vals = trends.loc[(trends['NERC Region']=='WECC')]
    
    wecc_djf_cold = wecc_vals.loc[(wecc_vals['Season']=='DJF')]
    
    wecc_djf_cold_map = wecc_djf_cold['Percent Change'].values * wecc_mask
    
    combined2 = npcc_djf_cold_map + mro_djf_cold_map + rfc_djf_cold_map + serc_djf_cold_map + tre_djf_cold_map + wecc_djf_cold_map
    
    combined2[combined2 == 0] = np.nan
    
    np.save(f'/Users/yiannabekris/Documents/energy_data/nparrays/djf_cold_{metric}_1_5_80_trends.npy',combined2)
    
    print(f"Finished {metric}!")
    
## Bring in data and create data for mapping
freq_metric = 'frequency'
extent_metric = 'mean_extent'
duration_metric = 'mean_duration'
ci_metric = 'mean_ci'

## Full path to CSV folder
folder_path = "/Users/yiannabekris/Documents/energy_data/csv"

## Trend and percent change data
freq_trends = pd.read_csv(f'{folder_path}/{freq_metric}_trends_1_5_80.csv')
extent_trends = pd.read_csv(f'{folder_path}/{extent_metric}_trends_1_5_80.csv')
duration_trends = pd.read_csv(f'{folder_path}/{duration_metric}_trends_1_5_80.csv')
ci_trends = pd.read_csv(f'{folder_path}/{ci_metric}_trends_1_5_80.csv')

## Run these through the function
map_data_create(freq_trends, freq_metric)
map_data_create(extent_trends, extent_metric)
map_data_create(duration_trends, duration_metric)
map_data_create(ci_trends, ci_metric)



    
    
