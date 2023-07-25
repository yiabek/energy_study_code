#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 09:27:04 2022

@author: yiannabekris

This script outputs the coldwave portion of Figure 4 as a .png file.

"""

## Import packages
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib as mpl

## Set font to Arial and set DPI 
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['savefig.dpi'] = 300

## Import function for subsetting to latitude and longitude
from geo_funcs import geo_idx

### MidPointNormalize
## Used to choose midpoint on colorbar
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


### Load files
lat = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/tmax_lat_1980_2020.npy')
netcdf_filename = '/Users/yiannabekris/Documents/energy_data/Tmax/Tmax_daily_ERA5_historical_an-sfc_20170601_20170630_0UTC copy.nc'
temp_data = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/tmax_array_1980_2021.npy')
ws_data = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/ws_nanom15_1980_2021.npy')
ssrd_data = np.load("/Users/yiannabekris/Documents/energy_data/nparrays/ssrdsum_daily_W_anom15_1980_2021.npy")
nerc_mask = np.load("/Users/yiannabekris/Documents/energy_data/nerc_masks/nerc_no_sub_mask.npy")
anom15 = np.load("/Users/yiannabekris/Documents/energy_data/nparrays/tmax_anom15_1980_2021.npy")
hdd = np.load('/Users/yiannabekris/Documents/energy_data/nparrays/hdd_1980_2021.npy')

## Date for the image name
image_name = 'feb_15_2021'

## Date of the event
date_event = '2/15/21'

## Index for February 15th in the array
feb15ind = 46 - 366

## Colormaps for each panel
cmaps = ['coolwarm','RdYlBu_r','PuOr','RdGy_r']

## Titles for each panel
titles = ['(a) Standardized Temperature Anomaly', '(b) Heating Degree Days',
        '(c) Windspeed Anomaly', '(d) Solar Radiation Anomaly']

## List of data to be plotted, subset to February 15th
data_list = [anom15[:,:,feb15ind],
             hdd[:,:,feb15ind],
             ws_data[:,:,feb15ind],
             ssrd_data[:,:,feb15ind]]


## Load shapefile with NERC region geometry
nerc_regions = gpd.read_file("/Users/yiannabekris/Documents/energy_data/NERC/nerc_no_sub.shp")
nerc_geometry = nerc_regions['geometry']

# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(
category='cultural',
name='admin_1_states_provinces_lines',
scale='50m',
facecolor='none')

## Set figure size
fig = plt.figure(figsize=(10, 8))


### Loop through data list and plot each panel
for ind in np.arange(0,len(data_list)):
    
    ## Delineate central latitude and longitude and create projection
    crs = ccrs.AlbersEqualArea(central_latitude	=35, central_longitude=-100)
    
    ## Add subplot to figure
    subplot = ind + 1
    ax = fig.add_subplot(2, 2, subplot, projection=crs)
    
    ## This can be converted into a `proj4` string/dict compatible with GeoPandas
    crs_proj4 = crs.proj4_init
    nerc_ae = nerc_regions.to_crs(crs_proj4)
    new_geometries = [crs.project_geometry(ii, src_crs=crs)
                  for ii in nerc_ae['geometry'].values]

    ## Set the extent to the CONUS
    ax.set_extent([-122, -73, 24.5, 50], ccrs.PlateCarree())


    ## Read in longitude directly from netcdf file
    var_netcdf = Dataset(netcdf_filename, "r")
    lon = np.asarray(var_netcdf.variables['longitude'][:], dtype="float")

    ## This is in case longitude is 0 to 360 and not 180 to -180
    if lon[lon>180].size>0:
        lon[lon>180] = lon[lon>180]-360

    ## Subset lon from file to the data's longitude
    east_lon_bnd = -65
    west_lon_bnd = -125

    ## Find indices and then mask array by indices    
    east_lon_idx = geo_idx(east_lon_bnd, lon)
    west_lon_idx = geo_idx(west_lon_bnd, lon)

    ## Subset longitude
    lon = lon[west_lon_idx:east_lon_idx]

    ## Extract correct array from data_list
    data_plt = data_list[ind]
        
    ## Mask to NERC regions and transpose the data so it plots correctly
    data_plt[nerc_mask==0] = np.nan
    data_plt = np.transpose(data_plt, (1,0))

    ### Plotting
    ## Standardized Temperature anomalies 
    if ind==0:
        levels = np.arange(-3,3.5,0.5)
        p1=ax.contourf(lon, lat, data_plt, cmap=cmaps[ind], 
                       levels=levels,
                        norm=MidpointNormalize(midpoint=0), extend='both',
                        transform=ccrs.PlateCarree())
       
    ## Heating degree days
    elif ind==1:
        levels = np.arange(-30,35,5)
        p1=ax.contourf(lon, lat, data_plt, cmap=cmaps[ind], levels=levels,
                        # norm=MidpointNormalize(midpoint=0), 
                        extend='both',
                        transform=ccrs.PlateCarree())
       
    ## Windspeed anomalies     
    elif ind==2:
        levels=np.arange(-6,7,1)
        p1=ax.contourf(lon, lat, data_plt, cmap=cmaps[ind], levels=levels,
                        norm=MidpointNormalize(midpoint=0), 
                        extend='both',
                        transform=ccrs.PlateCarree())
        
    ## Solar radiation anomalies
    else:
        levels=np.arange(-60,70,10)
        p1=ax.contourf(lon, lat, data_plt, cmap=cmaps[ind], levels=levels,
                        norm=MidpointNormalize(midpoint=0), 
                        extend='both',
                        transform=ccrs.PlateCarree())
      
    ## Add features to each plot, including NERC region borders
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='grey')
    ax.add_feature(cartopy.feature.COASTLINE, zorder=1)
    ax.add_feature(cartopy.feature.LAKES, zorder=1, linewidth=0.7, edgecolor='k', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='grey')
    ax.set_title(titles[ind], color="royalblue", pad=6, fontdict={'fontsize': 15, 'fontweight': 'bold'})
    ax.add_geometries(new_geometries, facecolor='None', linewidth=2, edgecolor='black', 
                      path_effects=[pe.Stroke(linewidth=4, foreground='w'), pe.Normal()], crs=crs)


    ## Create individual colorbars per subplot
    axpos = ax.get_position()
    axpos0 = axpos.x0
    pos_x = axpos0 + axpos.width - 0.06
    cax = inset_axes(ax, width="100%", height="14%", loc='lower center', bbox_to_anchor=(0, -0.05, 1, 0.3),
                     bbox_transform=ax.transAxes, borderpad=-0.25)
    cbar = fig.colorbar(p1, cax=cax, orientation='horizontal', extend='both')
    cbar.ax.tick_params(labelsize=14)

## Adjust space between figures and save as .png
plt.subplots_adjust(hspace=0.2, wspace=0.05)
plt.savefig(f'/Users/yiannabekris/Documents/energy_data/figures/png/{image_name}_arial.png')














