#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:07:50 2023

@author: yiannabekris
"""

## Import packages
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as pe
import matplotlib as mpl

## Set font to Arial
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['savefig.dpi']=300

## Import subsetting script
from geo_funcs import geo_idx

## Figure filename as a string
fig_filename = "/Users/yiannabekris/Documents/energy_data/figures/png/sp_wp_clima_1991_2020_arial_contour_14w.png"

## Load NumPy arrays with energy potential data
scf = np.load("/Users/yiannabekris/Documents/energy_data/ssrd.nosync/solar_capacity_factor_1980_2021.npy")
wp = np.load("/Users/yiannabekris/Documents/energy_data/ws.nosync/wind_power_density_windmaxdaily_1980_2021.npy")

## Load latitude data for plotting
lat = np.load("/Users/yiannabekris/Documents/energy_data/nparrays/tmax_lat_1979_2020.npy")

## Load mask to include only NERC regions
nerc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/nerc_no_sub_mask.npy")

## Netcdf file for reading in longitude
netcdf_filename = '/Users/yiannabekris/Documents/energy_data/ssrd.nosync/ssrd_hourly_ERA5_historical_an-sfc_20180601_20180630.nc'

## Create date dataframe to subset by dates and
## to exclude leap years
dates = pd.date_range(start='1/1/1980', end='12/31/2022')
dates = dates[~((dates.month == 2) & (dates.day == 29))]
date_df = pd.DataFrame(dates)
date_df['month'] = pd.to_datetime(date_df.iloc[:,0]).dt.month
date_df['year'] = pd.to_datetime(date_df.iloc[:,0]).dt.year

## Find indices
date_idxs = date_df.index.values

## Designate the year range for the climatology period
year_range = [*range(1991, 2021)]

## Find indices of DJF and JJA
djf_dates = date_df[(date_df.month.isin([1,2,12])) & (date_df.year.isin(year_range))]
jja_dates = date_df[(date_df.month.isin([6,7,8])) & (date_df.year.isin(year_range))]

## Extract indices
djf_idxs = djf_dates.index.values
jja_idxs = jja_dates.index.values

## Subset to DJF and JJA
scf_djf = scf[:,:,djf_idxs]
wp_djf = wp[:,:,djf_idxs]
scf_jja = scf[:,:,jja_idxs]
wp_jja = wp[:,:,jja_idxs]


## Delineate colormaps for each subplot
cmaps = ['Purples','YlOrBr','Purples','YlOrBr']


## Title list for plotting
titles = ['Wind Power per Turbine (kW) 1991-2020', 
          'Solar Capacity Factor 1991-2020',
          '',
          '']

## Date list for plotting
season_list = ['DJF','DJF',
       'JJA','JJA']

## Take the mean of each array and put it in a list for plotting
data_list = [np.nanmean(wp_djf, axis=2), np.nanmean(scf_djf, axis=2), 
              np.nanmean(wp_jja, axis=2), np.nanmean(scf_jja, axis=2)]

## Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(
                               category='cultural',
                               name='admin_1_states_provinces_lines',
                               scale='50m',
                               facecolor='none'
                               )

## Load NERC region shapefile and extract geometry for drawing borders
nerc_regions = gpd.read_file("/Users/yiannabekris/Documents/energy_data/NERC.nosync/nerc_no_sub.shp")
nerc_geometry = nerc_regions['geometry']

## Set figure size
fig = plt.figure(figsize=(20, 14))

## Set figure resolution
# fig.set_dpi(resolution)

### ========== Loop through numpy arrays and map ========== ###
for ind in np.arange(0,len(data_list)):

    ## Define projection and boundaries
    crs = ccrs.AlbersEqualArea(central_latitude	=35, central_longitude=-100)
    
    ## Subplot and axes
    subplot = ind + 1
    ax = fig.add_subplot(2, 2, subplot, projection=crs)

    ## This can be converted into a `proj4` string/dict compatible with GeoPandas
    crs_proj4 = crs.proj4_init
    nerc_ae = nerc_regions.to_crs(crs_proj4)
    new_geometries = [crs.project_geometry(ii, src_crs=crs)
                  for ii in nerc_ae['geometry'].values]

    # fig, ax = plt.subplots(figsize=(20, 20),subplot_kw={'projection': crs})
    
    ## Set the extent of the axes
    ax.set_extent([-122, -73, 24.5, 50], ccrs.PlateCarree())

    ## Read in longitude directly from nc file
    var_netcdf = Dataset(netcdf_filename, "r")
    lon = np.asarray(var_netcdf.variables['longitude'][:], dtype="float")


    ## Conver longitude from 0 to 360 to -180 to 180 
    if lon[lon>180].size>0:
        lon[lon>180]=lon[lon>180]-360

    ## Subset longitude from file to the data's longitude
    east_lon_bnd = -65
    west_lon_bnd = -125

    ## Find indices and then mask array by indices    
    east_lon_idx = geo_idx(east_lon_bnd,lon)
    west_lon_idx = geo_idx(west_lon_bnd,lon)

    ## Subset longitude
    lon = lon[west_lon_idx:east_lon_idx]
    
    ## Data from data list
    data_plt = data_list[ind] 

    ## Mask plot with NERC region mask
    data_plt[nerc_mask==0] = np.nan
    
    ## Transpose
    data_plt = np.transpose(data_plt, (1,0))


    ### ------------ Plotting ------------ ###
    ## Wind potential maps
    if ind == 0 or ind == 2:
        levels = np.arange(0, 1700, 200)
        p1=ax.contourf(lon, lat, data_plt, cmap=cmaps[ind],
                        levels=levels, extend="max",
                        linewidth=0, rasterized=True, transform=ccrs.PlateCarree())

    ## Solar potential maps   
    elif ind == 1 or ind == 3:
        levels = np.arange(0.0, 0.375, 0.025)
        p1=ax.contourf(lon, lat, data_plt, cmap=cmaps[ind],
                        levels=levels, extend="max",
                        linewidth=0, rasterized=True, transform=ccrs.PlateCarree())
       
        
    ## Draw features -- borders, coastlines, lakes, states, and NERC regions
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=3)
    ax.add_feature(cartopy.feature.COASTLINE, zorder=1)
    ax.add_feature(cartopy.feature.LAKES, zorder=1, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='black', linewidth=2, 
                   path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
    ax.add_geometries(new_geometries, facecolor='None', linewidth=4, edgecolor='black', 
                      path_effects=[pe.Stroke(linewidth=6, foreground='w'), pe.Normal()], crs=crs)
    

    ## Add the season in the lower left corner
    ax.text(0.02, 0.02, season_list[ind], fontsize=30, transform=ax.transAxes, weight='bold', color='black')
    
    
    ## If subplot is on top of plot, draw a title
    if ind < 2:
        ax.set_title(titles[ind], color="black", fontdict={'fontsize': 30, 'fontweight': 'bold'})
    
    ## If subplot is at bottom of plot, draw colorbars
    if ind > 1:
        cax = inset_axes(ax, width="100%", height="14%", loc='lower center', bbox_to_anchor=(0, -0.05, 1, 0.3),
                          bbox_transform=ax.transAxes, borderpad=0)
        cbar = fig.colorbar(p1, cax=cax, orientation='horizontal', extend='both')
        cbar.ax.tick_params(labelsize=17)


## Tight layout for saving
fig.tight_layout(pad=0.6)

## Save figure
plt.savefig(fig_filename, bbox_inches='tight', pad_inches=0.4)
