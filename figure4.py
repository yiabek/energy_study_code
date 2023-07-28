#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:28:08 2022

@author: yiannabekris

This script reads in data created by figure2_data.py on the
changes over time (as a percent) and outputs an 8-panel plot
(2 seasons X 4 characteristics [Extent, Duration, Frequency, Cumulative Intensity])

"""

## Import packages for plotting
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

## Suppress warnings due to deprecated matplotlib functions 
import warnings
warnings.filterwarnings("ignore")

## Set font to Arial and resolution to 300 DPI
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['savefig.dpi'] = 300

## Import function for finding indices of latitudes and longitudes
from geo_funcs import geo_idx


### ========== MidPointNormalize ========== ###
### Used to choose midpoint on colorbar in plots
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
    
### ========== make_annotations ========== ###
### Annotation function
## data is the data for the plot
## masks are the NERC region masks
## sig_csv is a CSV which contains the results of the linear models
## metric is which metric (Extent, Duration, Frequency, Cunmulative Intensity)
## seas is the season
## lat and lon are NumPy arrays of latitude and longitude
def make_annotations(data, masks, sig_csv, metric, seas, lat, lon):
    
    ## Strings of NERC regions
    regions = ['MRO', 'NPCC', 'RFC', 'SERC', 'TRE', 'WECC']
    
    ## Create empty lists for annotations
    trend_list = []
    xy_list = []
    
    ## Go through each mask of each NERC region and calculate the centroid for the annotation
    for mask in masks:
    
        inds_x, inds_y = lon[np.where(mask == 1)[0]], lat[np.where(mask == 1)[1]]
    
        centroid = round(np.nansum(inds_x)/len(inds_x)), round(np.nansum(inds_y)/len(inds_y))
    
        lon_ind = geo_idx(centroid[0], lon)
        lat_ind = geo_idx(centroid[1], lat)
        
        trend_str = str(round(data[lat_ind, lon_ind], 1))
        
        ## Append the trend string and centroid to each lists
        trend_list.append(trend_str)
        xy_list.append(centroid)
      
    ## Loop through each region and find the significance
    sig_list = []
    for ind in range(0, len(regions)):
        
        ## Extrect the significance from the CSV
        sig = sig_csv.loc[(sig_csv['NERC.Region']==regions[ind])
                          & (sig_csv['Metric'] == metric)
                          & (sig_csv['Season']==seas)]
        
        ## Bold the annotation of the p-value if it is <= 0.1
        if sig.iloc[0]['p.value']>0.1:
            sig_list.append('medium')
        elif sig.iloc[0]['p.value']<=0.1:
            sig_list.append('bold')
        
        
    return trend_list, xy_list, sig_list

    

### ========== Plotting ========== ###
## Figure filename and resolution setting
fig_filename = '/Users/yiannabekris/Documents/energy_data/figures/mean_char_1980-2021_1_5_80_pval.png'
resolution = 96 ## Can also set resolution here

## Load shapefile arrays to subset by regions
mro_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/mro_mask.npy")
npcc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/npcc_mask.npy")
rfc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/rfc_mask.npy")
serc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/serc_mask.npy")
tre_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/tre_mask.npy")
wecc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/wecc_mask.npy")

## Load latitude and longitude NumPy arrays
lat = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/tmax_lat_1979_2020.npy')
lon = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/tmax_lon_1979_2020.npy')

## NERC region masks
mask_list = [mro_mask, npcc_mask, rfc_mask, serc_mask, tre_mask, wecc_mask]

## Read the table with p-values and trends
sig_table = pd.read_csv('/Users/yiannabekris/Documents/energy_data/csv/mean_sig_table_1_5_80.csv')

## Load the NetCDF file to open for plotting
netcdf_filename = '/Users/yiannabekris/Documents/energy_data/Tmax/Tmax_daily_ERA5_historical_an-sfc_20170601_20170630_0UTC.nc'

## NERC mask for plotting
nerc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/nerc_no_sub_mask.npy")

## Heatwave trends
freq_hot_jja = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/jja_hot_frequency_1_5_80_trends.npy')
intensity_hot_jja = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/jja_hot_mean_hsci_1_5_80_trends.npy')
extent_hot_jja = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/jja_hot_mean_extent_1_5_80_trends.npy')
duration_hot_jja = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/jja_hot_mean_duration_1_5_80_trends.npy')

## Coldwave trends
freq_cold_djf = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/djf_cold_frequency_1_5_80_trends.npy')
intensity_cold_djf = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/djf_cold_mean_hsci_1_5_80_trends.npy')
extent_cold_djf = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/djf_cold_mean_extent_1_5_80_trends.npy')
duration_cold_djf = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/djf_cold_mean_duration_1_5_80_trends.npy')

## Colormap list for plotting
cmap = 'seismic'


## Title list for plotting
titles = ['(a) Cold Frequency', '(b) Cold Extent',
        '(c) Cold Duration', '(d) Cold CI', 
        '(e) Hot Frequency', '(f) Hot Extent',
        '(g) Hot Duration', '(h) Hot CI']

metric_list=['Frequency', 'Extent', 'Duration', 'HSCI',
             'Frequency', 'Extent', 'Duration', 'HSCI']

seas_list=['DJF', 'DJF', 'DJF', 'DJF',
           'JJA', 'JJA', 'JJA', 'JJA']



labels=['(a)','(b)','(c)','(d)',
       '(e)','(f)','(g)','(h)']


## arrays for pltting
data_list = [freq_cold_djf, extent_cold_djf, duration_cold_djf, intensity_cold_djf, 
             freq_hot_jja, extent_hot_jja, duration_hot_jja, intensity_hot_jja]


## NERC outline
nerc_regions = gpd.read_file("/Users/yiannabekris/Documents/energy_data/NERC/nerc_no_sub.shp")
nerc_geometry = nerc_regions['geometry']


## Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(
category='cultural',
name='admin_1_states_provinces_lines',
scale='50m',
facecolor='none')


## Set figure size
fig = plt.figure(figsize=(16, 7))

## Set figure resolution
fig.set_dpi(resolution)

levels = np.arange(-100, 110, 10)

### ========== Loop through numpy arrays and map ========== ###
for ind in np.arange(0,len(data_list)):
    
    
    # ax=fig.add_subplot(int('33'+str(ind+1)),projection=cartopy.crs.PlateCarree())
    crs = ccrs.AlbersEqualArea(central_latitude	=35, central_longitude=-100)
    subplot = ind + 1
    ax = fig.add_subplot(2,4,subplot,projection=crs)
    
    # This can be converted into a `proj4` string/dict compatible with GeoPandas
    crs_proj4 = crs.proj4_init
    nerc_ae = nerc_regions.to_crs(crs_proj4)
    new_geometries = [crs.project_geometry(ii, src_crs=crs)
                  for ii in nerc_ae['geometry'].values]

    # fig, ax = plt.subplots(figsize=(20, 20),subplot_kw={'projection': crs})
    ax.set_extent([-122, -73, 24.5, 50], ccrs.PlateCarree())

    ### Read in longitude directly netCDF file
    var_netcdf = Dataset(netcdf_filename, "r")
    lon = np.asarray(var_netcdf.variables['longitude'][:], dtype="float")


    ## Convert longitude from 0 to 360 to -180 to 180 
    if lon[lon>180].size>0:
        lon[lon>180] = lon[lon>180]-360

    ## Subset lon from file to the data's longitude
    east_lon_bnd = -65
    west_lon_bnd = -125

    ## Find indices and then mask array by indices    
    east_lon_idx = geo_idx(east_lon_bnd,lon)
    west_lon_idx = geo_idx(west_lon_bnd,lon)

    ## Subset longitude to data
    lon = lon[west_lon_idx:east_lon_idx]

    ## Extract data from data list, mask with the NERC region, and transpose for plotting
    data_plt = data_list[ind]
    data_plt[nerc_mask==0] = np.nan
    data_plt = np.transpose(data_plt,(1,0))

    
    ## Make the annotations
    trend_list, xy_list, font_weights = make_annotations(
                                                         data_plt, mask_list, sig_table,
                                                         metric_list[ind], seas_list[ind], lat, lon
                                                         )
       
    ## Plotting
    ## Cold Frequency
    if ind==0:
        # levels = np.arange(-60,-15,5)
        p1=ax.contourf(lon, lat, data_plt, cmap=cmap,levels=levels,
                        norm=MidpointNormalize(midpoint=0),
                        extend='both',
                        linewidth=0, rasterized=True, transform=ccrs.PlateCarree())
        
        ## Make annotations
        for idx in range(len(trend_list)):
            ax.annotate(trend_list[idx], xy_list[idx],
                transform=ccrs.PlateCarree(), ha='center', 
                size=14, fontweight=font_weights[idx],
                bbox=dict(boxstyle="square,pad=0.3",
                fc="whitesmoke", ec="black", alpha=0.77, lw=2))
        
    ## Hot Frequency    
    elif ind==4:
        p1=ax.contourf(lon, lat, data_plt, cmap=cmap, levels=levels,
                        norm=MidpointNormalize(midpoint=0), 
                        extend='both',
                        linewidth=0, rasterized=True,transform=ccrs.PlateCarree())
        
        ## Make annotations
        for idx in range(len(trend_list)):
            ax.annotate(trend_list[idx], xy_list[idx],
                transform=ccrs.PlateCarree(), ha='center',
                size=14, fontweight=font_weights[idx],
                bbox=dict(boxstyle="square,pad=0.3",
                fc="whitesmoke", ec="black", alpha=0.77, lw=2))
        
    ## Cold Duration
    elif ind==1:
        p1=ax.contourf(lon, lat, data_plt, cmap=cmap,levels=levels,
                        norm=MidpointNormalize(midpoint=0), 
                        extend='both',
                        linewidth=0, rasterized=True,transform=ccrs.PlateCarree())
        
        ## Make annotations
        for idx in range(len(trend_list)):
            ax.annotate(trend_list[idx], xy_list[idx], 
                transform=ccrs.PlateCarree(), ha='center',
                size=14, fontweight=font_weights[idx],
                bbox=dict(boxstyle="square,pad=0.3",
                fc="whitesmoke", ec="black", alpha=0.77, lw=2))
    
    ## Hot Duration
    elif ind==5:
        p1=ax.contourf(lon, lat, data_plt, cmap=cmap,levels=levels,
                        norm=MidpointNormalize(midpoint=0),
                        extend='both',
                        linewidth=0, rasterized=True,transform=ccrs.PlateCarree())
        
        ## Make annotations
        for idx in range(len(trend_list)):
            ax.annotate(trend_list[idx], xy_list[idx],
                transform=ccrs.PlateCarree(), ha='center',
                size=14, fontweight=font_weights[idx],
                bbox=dict(boxstyle="square,pad=0.3",
                fc="whitesmoke", ec="black", alpha=0.77, lw=2))
        
    ## Cold Intensity
    elif ind==2:
        p1=ax.contourf(lon, lat, data_plt, cmap=cmap, levels=levels,
                        norm=MidpointNormalize(midpoint=0), 
                        extend='both',
                        linewidth=0, rasterized=True,transform=ccrs.PlateCarree())
        
        ## Make annotations
        for idx in range(len(trend_list)):
            ax.annotate(trend_list[idx], xy_list[idx],
                transform=ccrs.PlateCarree(), ha='center',
                size=14, fontweight=font_weights[idx],
                bbox=dict(boxstyle="square,pad=0.3",
                fc="whitesmoke", ec="black", alpha=0.77, lw=2))
    
    ## Hot Intensity
    elif ind==6:
        p1=ax.contourf(lon, lat, data_plt, cmap=cmap, levels=levels,
                        norm=MidpointNormalize(midpoint=0), 
                        extend='both',
                        linewidth=0, rasterized=True,transform=ccrs.PlateCarree())
        
        ## Make annotations
        for idx in range(len(trend_list)):
            ax.annotate(trend_list[idx], xy_list[idx],
                transform=ccrs.PlateCarree(), ha='center',
                size=14, fontweight=font_weights[idx],
                bbox=dict(boxstyle="square,pad=0.3",
                fc="whitesmoke", ec="black", alpha=0.77, lw=2))
       
    ## Cold Extent
    elif ind==3:
        p1=ax.contourf(lon, lat, data_plt, cmap=cmap, levels=levels,
                        norm=MidpointNormalize(midpoint=0), 
                        extend='both',
                        linewidth=0, rasterized=True,transform=ccrs.PlateCarree())
        
        ## Make annotations
        for idx in range(len(trend_list)):
            ax.annotate(trend_list[idx], xy_list[idx],      
                transform=ccrs.PlateCarree(), ha='center', 
                size=14, fontweight=font_weights[idx],
                bbox=dict(boxstyle="square,pad=0.3",
                fc="whitesmoke", ec="black", alpha=0.77, lw=2))
        
    ## Hot Extent
    elif ind==7:
        p1=ax.contourf(lon, lat, data_plt, cmap=cmap, levels=levels,
                        norm=MidpointNormalize(midpoint=0), 
                        extend='both',
                        linewidth=0, rasterized=True,transform=ccrs.PlateCarree())
        
        ## Make annotations
        for idx in range(len(trend_list)):
            ax.annotate(trend_list[idx], xy_list[idx], fontweight = font_weights[idx],
                transform=ccrs.PlateCarree(), ha='center', size=14,
            bbox=dict(boxstyle="square,pad=0.3",
                      fc="whitesmoke", ec="black", alpha=0.77, lw=2))
       

    ## Draw features
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='grey')
    ax.add_feature(cartopy.feature.COASTLINE, zorder=1)
    ax.add_feature(cartopy.feature.LAKES, zorder=1, linewidth=0.7, edgecolor='k', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='grey')
    
    ## Plot NERC region borders
    ax.add_geometries(new_geometries, facecolor='None', linewidth=2, edgecolor='black', 
                      path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()], crs=crs)
    
    ## Plot titles
    if ind < 4:
        ax.set_title(titles[ind], color="royalblue", fontdict={'fontsize': 19, 'fontweight': 'bold'})
    else:
        ax.set_title(titles[ind], color="indianred", fontdict={'fontsize': 19, 'fontweight': 'bold'})
        
    
    ## Draw one central colorbar
    if ind == 7:
        cax2 = fig.add_axes([0.2, 0.001, 0.6, 0.03])
        cbar = fig.colorbar(p1, cax=cax2, orientation='horizontal')
        cbar.ax.tick_params(labelsize=14)



## Tight layout for saving
fig.tight_layout(pad=0.5)

## Save figure
plt.savefig(fig_filename, bbox_inches='tight', pad_inches=0.4)



