#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:51:56 2022

@author: yiannabekris

This creates a csv which contains all temperature extreme
characteristics as well as energy proxy and energy potential metrics

"""

## Import packages
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr

## function to find lat and lon indices
def geo_idx(dd, dd_array):
   
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minimum value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minimum.
     https://stackoverflow.com/questions/33789379/netcdf-and-python-finding-the-closest-lon-lat-index-given-actual-lon-lat-values
     
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   
   return geo_idx


### conus_subset
### Subset latitude and longitude NumPy arrays to CONUS
def conus_subset(lat,lon):
    
    ## Convert longitudes from 0 to 360 to -180 to 180 
    if lon[lon>180].size>0:
        lon[lon>180] = lon[lon>180]-360

    ## Define latitude and longitude boundaries of CONUS
    north_lat_bnd = 50
    south_lat_bnd = 20
    east_lon_bnd = -65
    west_lon_bnd = -125

    ## Find indices and then mask array by indices    
    north_lat_idx = geo_idx(north_lat_bnd,lat)
    south_lat_idx = geo_idx(south_lat_bnd,lat)
    east_lon_idx = geo_idx(east_lon_bnd,lon)
    west_lon_idx = geo_idx(west_lon_bnd,lon)

    return north_lat_idx, south_lat_idx, east_lon_idx, west_lon_idx


## A 3D mask function
def mask_3d(array, newshape):
    
    array_3d = np.repeat(array[:,:,np.newaxis], newshape, axis=2) 
    
    return array_3d


## Flatten arrays and remove NaNs
def flat_nonan(data):
    
    data = data.flatten()
    # data = data[~np.isnan(data)]
    
    return data


# ======================================================================
# radius of Earth to get areas of each grid cell
def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    
    Credit to Luke Gloege
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r


# Calculates area of each grid cell
def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    
    Credit to Luke Gloege
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    
    total_area = xda.sum(['latitude','longitude'])
    
    np_area = xda.to_numpy()
    
    np_area = np_area  * -1
    
    total_area = total_area.to_numpy()
    
    
    return xda, total_area, np_area



### ========================================================
### Variables
yearbeg = 1980
yearend = 2021
monthsub=[1,2,3,4,5,6,7,8,9,10,11,12]
writepath='/data/singh/yianna/csv/annual_metrics_1_5_anom.csv'
monthstr='annual'


## Load shapefiles to subset by regions
mro_mask = np.load("/data/singh/yianna/nerc/mro_mask.npy")
npcc_mask = np.load("/data/singh/yianna/nerc/npcc_mask.npy")
rfc_mask = np.load("/data/singh/yianna/nerc/rfc_mask.npy")
serc_mask = np.load("/data/singh/yianna/nerc/serc_mask.npy")
tre_mask = np.load("/data/singh/yianna/nerc/tre_mask.npy")
wecc_mask = np.load("/data/singh/yianna/nerc/wecc_mask.npy")
#nerc_mask = np.load("/data/singh/yianna/nerc_masks/nerc_no_sub_mask.npy")


## Load arrays of temperature data
array = np.load(f'/data/singh/yianna/nparrays/tmax_array_{yearbeg}_{yearend}.npy')
anom15 = np.load(f'/data/singh/yianna/nparrays/tmax_anom15_{yearbeg}_{yearend}.npy')
lat = np.load(f'/data/singh/yianna/nparrays/tmax_lat_1979_{yearend}.npy')
lon = np.load(f'/data/singh/yianna/nparrays/tmax_lon_1979_{yearend}.npy')

## Load arrays of wind and SSRD data
windspeed = np.load(f'/data/singh/yianna/nparrays/ws_daily_array_annual_{yearbeg}_{yearend}.npy')
ssrd = np.load(f'/data/singh/yianna/nparrays/ssrdsum_daily_{yearbeg}_{yearend}.npy')
ws_anom15 = np.load(f'/data/singh/yianna/nparrays/ws_nanom15_{yearbeg}_{yearend}.npy')
ssrd_anom15 = np.load(f'/data/singh/yianna/nparrays/ssrdsum_nanom15_{yearbeg}_{yearend}.npy') 
wind_potential = np.load(f'/data/singh/yianna/nparrays/wind_power_density_windmaxdaily_{yearbeg}_{yearend}.npy')
solar_potential = np.load(f'/data/singh/yianna/nparrays/solar_capacity_factor_{yearbeg}_{yearend}.npy')
wp_anom15 = np.load(f'/data/singh/yianna/nparrays/wp_nanom15_{yearbeg}_{yearend}.npy')
sp_anom15 = np.load(f'/data/singh/yianna/nparrays/sp_nanom15_{yearbeg}_{yearend}.npy') 


## Create date dataframe
dates = pd.date_range(start='1/1/1980', end='12/31/2021')
datatime = pd.DataFrame(dates)


## Convert time data for filtering by desired season or time period
mo=np.array([int('{0.month:02d}'.format(t)) for t in list(datatime[0])])
yr=np.array([int('{0.year:04d}'.format(t)) for t in list(datatime[0])])
leapstr=np.array([t.strftime('%m-%d') for t in list(datatime[0])])

## Date as a string
date=np.array([t.strftime('%Y-%m-%d')for t in list(datatime[0])])

## Find year indices so month subset works properly
yearind = np.where(np.logical_and(yr>=yearbeg, yr<=yearend))[0]
mo=mo[yearind]
yr=yr[yearind]
date=date[yearind]

## Remove leap days if needed
leapstr=leapstr[yearind]
dateind=(leapstr != '02-29')
yr=yr[dateind]
mo=mo[dateind]
date=date[dateind]

## Subsetting data to season specified by "monthsub"
moinds=np.in1d(mo,monthsub)
moinds=(np.where(moinds)[0])
moinds=[int(indval) for indval in moinds]
yr=yr[moinds]
date=date[moinds]
array=array[:,:,moinds]
anom15=anom15[:,:,moinds]
windspeed=windspeed[:,:,moinds]
ssrd=ssrd[:,:,moinds]
ws_anom15=ws_anom15[:,:,moinds]
ssrd_anom15=ssrd_anom15[:,:,moinds]
wind_potential=wind_potential[:,:,moinds]
solar_potential=solar_potential[:,:,moinds]
wp_anom15=wp_anom15[:,:,moinds]
sp_anom15=sp_anom15[:,:,moinds]

              
## Heating degree days and extreme cold
cold_cond = [(array <= 18),(array > 18)]
cold_sel = [(18-array),0]
heat_deg_days = np.select(cold_cond, cold_sel)


## Cooling degree days and extreme heat
hot_cond = [(array >= 18),(array < 18)]
hot_sel = [(array-18),0]
cool_deg_days = np.select(hot_cond, hot_sel)

## General energy proxy
nonex_cond = [(array < 18),(array > 18), (array==0)]
nonex_sel = [(18-array),(array-18),0]
nonex_enprox = np.select(nonex_cond,nonex_sel)


## Subset lat and lon
north_lat_idx,south_lat_idx,east_lon_idx,west_lon_idx = conus_subset(lat, lon)


## Cooling degree days and extreme heat
hot_anom_thresh = 1.5
hot_loc = anom15 >= hot_anom_thresh

## Cooling degree days and extreme heat
cold_anom_thresh = -1.5
cold_loc = anom15 <= cold_anom_thresh

## Non-extremes
nonex_loc = (anom15 > cold_anom_thresh) & (anom15 < hot_anom_thresh)

## For seperating degree days by type
hdd_inds = np.isin(mo, [12, 1, 2]) # Heating degree days for DJF
cdd_inds = np.isin(mo, [6, 7, 8]) # Cooling degree days for JJA
enprox_inds = np.isin(mo, [3, 4, 5, 9, 10, 11]) # General degree das for MAM and SON

## Set to 0
heat_deg_days[:,:,~hdd_inds] = 0
cool_deg_days[:,:,~cdd_inds] = 0
nonex_enprox[:,:,~enprox_inds] = 0

## Add together
degree_days = heat_deg_days + cool_deg_days + nonex_enprox

### Load population data
## 2020 only
pop_file = r"/data/singh/yianna/population/gpw_v4_population_count_rev11_15_min.nc"

### Open population file
pop_dataset = Dataset(pop_file)
rawpop=pop_dataset.variables['Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 15 arc-minutes']
pop_arr = np.array(rawpop)
pop_arr[pop_arr < 0] = np.nan
pop_2020 = pop_arr[4,:,:]
poplat=np.asarray(pop_dataset['latitude'][:],dtype="float") 
poplon=np.asarray(pop_dataset['longitude'][:],dtype="float")


### Subset pop array to CONUS and transpose to (lat,lon)
n_ilat,s_ilat,e_ilon,w_ilon = conus_subset(poplat,poplon)
pop_2020 = np.squeeze(pop_2020[n_ilat:s_ilat,w_ilon:e_ilon])
pop_2020 = np.transpose(pop_2020,(1,0))
pop_2020_3d = (np.repeat(pop_2020[:,:,np.newaxis], 1, axis=2))


## Energy proxy multiplied by population
hdd_pop = degree_days * pop_2020_3d
cdd_pop = degree_days * pop_2020_3d
enprox_pop = degree_days * pop_2020_3d

## Shape for 3D mask
newshape = array.shape[2]

## Create masks 
mro_3d = mask_3d(mro_mask,newshape)
npcc_3d = mask_3d(npcc_mask,newshape)
rfc_3d = mask_3d(rfc_mask,newshape)
serc_3d = mask_3d(serc_mask,newshape)
tre_3d = mask_3d(tre_mask,newshape)
wecc_3d = mask_3d(wecc_mask,newshape)


mro_cdd_pop = cdd_pop.copy()
npcc_cdd_pop = cdd_pop.copy()
rfc_cdd_pop = cdd_pop.copy()
serc_cdd_pop = cdd_pop.copy()
tre_cdd_pop = cdd_pop.copy()
wecc_cdd_pop = cdd_pop.copy()

mro_cdd_pop[mro_3d==0] = np.nan
npcc_cdd_pop[npcc_3d==0] = np.nan
rfc_cdd_pop[rfc_3d==0] = np.nan
serc_cdd_pop[serc_3d==0] = np.nan
tre_cdd_pop[tre_3d==0] = np.nan
wecc_cdd_pop[wecc_3d==0] = np.nan

mro_hdd_pop = hdd_pop.copy()
npcc_hdd_pop = hdd_pop.copy()
rfc_hdd_pop = hdd_pop.copy()
serc_hdd_pop = hdd_pop.copy()
tre_hdd_pop = hdd_pop.copy()
wecc_hdd_pop = hdd_pop.copy()

mro_hdd_pop[mro_3d==0] = np.nan
npcc_hdd_pop[npcc_3d==0] = np.nan
rfc_hdd_pop[rfc_3d==0] = np.nan
serc_hdd_pop[serc_3d==0] = np.nan
tre_hdd_pop[tre_3d==0] = np.nan
wecc_hdd_pop[wecc_3d==0] = np.nan

mro_enprox_pop = enprox_pop.copy()
npcc_enprox_pop = enprox_pop.copy()
rfc_enprox_pop = enprox_pop.copy()
serc_enprox_pop = enprox_pop.copy()
tre_enprox_pop = enprox_pop.copy()
wecc_enprox_pop = enprox_pop.copy()

mro_enprox_pop[mro_3d==0] = np.nan
npcc_enprox_pop[npcc_3d==0] = np.nan
rfc_enprox_pop[rfc_3d==0] = np.nan
serc_enprox_pop[serc_3d==0] = np.nan
tre_enprox_pop[tre_3d==0] = np.nan
wecc_enprox_pop[wecc_3d==0] = np.nan

### Find area of each grid cell in m
xda, total_area, np_area = area_grid(lat,lon)

### Subset and reshape
# area_sub = np_area[north_lat_idx:south_lat_idx,west_lon_idx:east_lon_idx]
area_sub = np.transpose(np_area,(1,0))
area_km = area_sub/1000000
area_3d = np.repeat(area_sub[:,:,np.newaxis], array.shape[2], axis=2)

## Convert from sq m to sq km
area_km_3d = area_3d/1000000

mro_3d_km = area_km_3d.copy()
npcc_3d_km = area_km_3d.copy()
rfc_3d_km = area_km_3d.copy()
serc_3d_km = area_km_3d.copy()
tre_3d_km = area_km_3d.copy()
wecc_3d_km = area_km_3d.copy()
 
## Find extent of extreme events
## Need to be int
hot_loc_1s = hot_loc.astype(int)
cold_loc_1s = cold_loc.astype(int)
nonex_loc_1s = nonex_loc.astype(int)
area_hot = hot_loc_1s * area_km_3d
area_cold = cold_loc_1s * area_km_3d
area_nonex = nonex_loc_1s * area_km_3d

## Hot extent arrays
hot_area_mro = area_hot.copy()
hot_area_npcc = area_hot.copy()
hot_area_rfc = area_hot.copy()
hot_area_serc = area_hot.copy()
hot_area_tre = area_hot.copy()
hot_area_wecc = area_hot.copy()

## Mask hot extent arrays
hot_area_mro[mro_3d==0] = np.nan
hot_area_npcc[npcc_3d==0] = np.nan
hot_area_rfc[rfc_3d==0] = np.nan
hot_area_serc[serc_3d==0] = np.nan
hot_area_tre[tre_3d==0] = np.nan
hot_area_wecc[wecc_3d==0] = np.nan

## Cold extent arrays
cold_area_mro = area_cold.copy()
cold_area_npcc = area_cold.copy()
cold_area_rfc = area_cold.copy()
cold_area_serc = area_cold.copy()
cold_area_tre = area_cold.copy()
cold_area_wecc = area_cold.copy()

## Mask cold extent arrays
cold_area_mro[mro_3d==0] = np.nan
cold_area_npcc[npcc_3d==0] = np.nan
cold_area_rfc[rfc_3d==0] = np.nan
cold_area_serc[serc_3d==0] = np.nan
cold_area_tre[tre_3d==0] = np.nan
cold_area_wecc[wecc_3d==0] = np.nan

## Non-extreme extent arrays
nonex_area_mro = area_nonex.copy()
nonex_area_npcc = area_nonex.copy()
nonex_area_rfc = area_nonex.copy()
nonex_area_serc = area_nonex.copy()
nonex_area_tre = area_nonex.copy()
nonex_area_wecc = area_nonex.copy()

## Mask non-extreme arrays
nonex_area_mro[mro_3d==0] = np.nan
nonex_area_npcc[npcc_3d==0] = np.nan
nonex_area_rfc[rfc_3d==0] = np.nan
nonex_area_serc[serc_3d==0] = np.nan
nonex_area_tre[tre_3d==0] = np.nan
nonex_area_wecc[wecc_3d==0] = np.nan

## Calculate total areas of each NERC region
mro_hot_a = np.nansum(hot_area_mro, axis=(0,1))
npcc_hot_a = np.nansum(hot_area_npcc, axis=(0,1)) 
rfc_hot_a = np.nansum(hot_area_rfc, axis=(0,1)) 
serc_hot_a = np.nansum(hot_area_serc, axis=(0,1)) 
tre_hot_a = np.nansum(hot_area_tre, axis=(0,1))
wecc_hot_a = np.nansum(hot_area_wecc, axis=(0,1))


## Calculate total areas of each NERC region
mro_cold_a = np.nansum(cold_area_mro, axis=(0,1))
npcc_cold_a = np.nansum(cold_area_npcc, axis=(0,1))
rfc_cold_a = np.nansum(cold_area_rfc, axis=(0,1))
serc_cold_a = np.nansum(cold_area_serc, axis=(0,1))
tre_cold_a = np.nansum(cold_area_tre, axis=(0,1))
wecc_cold_a = np.nansum(cold_area_tre, axis=(0,1))


## Calculate total area for each NERC region
mro_tot_a = np.nansum(mro_mask * area_km)
npcc_tot_a = np.nansum(npcc_mask * area_km) 
rfc_tot_a = np.nansum(rfc_mask * area_km) 
serc_tot_a = np.nansum(serc_mask * area_km) 
tre_tot_a = np.nansum(tre_mask * area_km) 
wecc_tot_a = np.nansum(wecc_mask * area_km)


## Calculate weights for weighted anomalies
mro_weights = mro_3d / mro_tot_a
npcc_weights = npcc_3d / npcc_tot_a
rfc_weights = rfc_3d / rfc_tot_a
serc_weights = serc_3d / serc_tot_a
tre_weights = tre_3d / tre_tot_a
wecc_weights = wecc_3d / wecc_tot_a

## Create copies of arrays before masking 
mro_ws = windspeed.copy()
npcc_ws = windspeed.copy()
rfc_ws = windspeed.copy()
serc_ws = windspeed.copy()
tre_ws = windspeed.copy()
wecc_ws = windspeed.copy()

## Mask arrays
mro_ws[mro_3d==0] = np.nan
npcc_ws[npcc_3d==0] = np.nan
rfc_ws[rfc_3d==0] = np.nan
serc_ws[serc_3d==0] = np.nan
tre_ws[tre_3d==0] = np.nan
wecc_ws[wecc_3d==0] = np.nan

## Windspeed arrays by region
mro_ws = np.ma.array(mro_ws, mask=np.isnan(mro_ws))
npcc_ws = np.ma.array(npcc_ws, mask=np.isnan(npcc_ws))
rfc_ws = np.ma.array(rfc_ws, mask=np.isnan(rfc_ws))
serc_ws = np.ma.array(serc_ws, mask=np.isnan(serc_ws))
tre_ws = np.ma.array(tre_ws, mask=np.isnan(tre_ws))
wecc_ws = np.ma.array(wecc_ws, mask=np.isnan(wecc_ws))

## Calculate windspeed weighted average over each region
mro_ws = np.ma.average(mro_ws, axis=(0,1),weights=mro_weights)
npcc_ws = np.ma.average(npcc_ws, axis=(0,1),weights=npcc_weights)
rfc_ws = np.ma.average(rfc_ws, axis=(0,1),weights=rfc_weights)
serc_ws = np.ma.average(serc_ws, axis=(0,1),weights=serc_weights)
tre_ws = np.ma.average(tre_ws, axis=(0,1),weights=tre_weights)
wecc_ws = np.ma.average(wecc_ws, axis=(0,1),weights=wecc_weights)

## Repeat for CSV
mro_ws = np.tile(mro_ws, 3)
npcc_ws = np.tile(npcc_ws, 3)
rfc_ws = np.tile(rfc_ws, 3)
serc_ws = np.tile(serc_ws, 3)
tre_ws = np.tile(tre_ws, 3)
wecc_ws = np.tile(wecc_ws, 3)


## Create copies of arrays before masking
mro_ssrd = ssrd.copy()
npcc_ssrd = ssrd.copy()
rfc_ssrd = ssrd.copy()
serc_ssrd = ssrd.copy()
tre_ssrd = ssrd.copy()
wecc_ssrd = ssrd.copy()

## Mask SSRD  arrays
mro_ssrd[mro_3d==0] = np.nan
npcc_ssrd[npcc_3d==0] = np.nan
rfc_ssrd[rfc_3d==0] = np.nan
serc_ssrd[serc_3d==0] = np.nan
tre_ssrd[tre_3d==0] = np.nan
wecc_ssrd[wecc_3d==0] = np.nan


mro_ssrd = np.ma.array(mro_ssrd, mask=np.isnan(mro_ssrd))
npcc_ssrd = np.ma.array(npcc_ssrd, mask=np.isnan(npcc_ssrd))
rfc_ssrd = np.ma.array(rfc_ssrd, mask=np.isnan(rfc_ssrd))
serc_ssrd = np.ma.array(serc_ssrd, mask=np.isnan(serc_ssrd))
tre_ssrd = np.ma.array(tre_ssrd, mask=np.isnan(tre_ssrd))
wecc_ssrd = np.ma.array(wecc_ssrd, mask=np.isnan(wecc_ssrd))

## Compute average SSRD over region
mro_ssrd = np.ma.average(mro_ssrd, axis=(0,1), weights=mro_weights)
npcc_ssrd = np.ma.average(npcc_ssrd, axis=(0,1), weights=npcc_weights)
rfc_ssrd = np.ma.average(rfc_ssrd, axis=(0,1), weights=rfc_weights)
serc_ssrd = np.ma.average(serc_ssrd, axis=(0,1), weights=serc_weights)
tre_ssrd = np.ma.average(tre_ssrd, axis=(0,1), weights=tre_weights)
wecc_ssrd = np.ma.average(wecc_ssrd, axis=(0,1), weights=wecc_weights)

## Repeat for file
mro_ssrd = np.tile(mro_ssrd, 3)
npcc_ssrd = np.tile(npcc_ssrd, 3)
rfc_ssrd = np.tile(rfc_ssrd, 3)
serc_ssrd = np.tile(serc_ssrd, 3)
tre_ssrd = np.tile(tre_ssrd, 3)
wecc_ssrd = np.tile(wecc_ssrd, 3)

## Mask out non-heatwave locations
hot_anoms = anom15.copy()
cold_anoms = anom15.copy() 
nonex_anoms = anom15.copy()
hot_anoms = np.ma.array(hot_anoms, mask=~hot_loc)
cold_anoms = np.ma.array(cold_anoms, mask=~cold_loc)
nonex_anoms = np.ma.array(nonex_anoms, mask=~nonex_loc)

## Area weighted anomalies
## Subset by region first: hot
mro_hot_anoms = mro_3d * hot_anoms
npcc_hot_anoms = npcc_3d * hot_anoms
rfc_hot_anoms = rfc_3d * hot_anoms
serc_hot_anoms = serc_3d * hot_anoms
tre_hot_anoms = tre_3d * hot_anoms
wecc_hot_anoms = wecc_3d * hot_anoms


## Calculate weighted anomalies: hot
mro_whot_anoms = np.ma.average(mro_hot_anoms,axis=(0,1), weights=mro_weights)
npcc_whot_anoms = np.ma.average(npcc_hot_anoms,axis=(0,1), weights=npcc_weights)
rfc_whot_anoms = np.ma.average(rfc_hot_anoms,axis=(0,1), weights=rfc_weights)
serc_whot_anoms = np.ma.average(serc_hot_anoms,axis=(0,1), weights=serc_weights)
tre_whot_anoms = np.ma.average(tre_hot_anoms,axis=(0,1), weights=tre_weights)
wecc_whot_anoms = np.ma.average(wecc_hot_anoms,axis=(0,1), weights=wecc_weights)


## Subset by region first: cold
mro_cold_anoms = mro_3d * cold_anoms
npcc_cold_anoms = npcc_3d * cold_anoms
rfc_cold_anoms = rfc_3d * cold_anoms
serc_cold_anoms = serc_3d * cold_anoms
tre_cold_anoms = tre_3d * cold_anoms
wecc_cold_anoms = wecc_3d * cold_anoms


## Calculate weighted anomalies: cold
mro_wcold_anoms = np.ma.average(mro_cold_anoms,axis=(0,1), weights=mro_weights) * -1
npcc_wcold_anoms = np.ma.average(npcc_cold_anoms,axis=(0,1), weights=npcc_weights) * -1
rfc_wcold_anoms = np.ma.average(rfc_cold_anoms,axis=(0,1), weights=rfc_weights) * -1
serc_wcold_anoms = np.ma.average(serc_cold_anoms,axis=(0,1), weights=serc_weights) * -1
tre_wcold_anoms = np.ma.average(tre_cold_anoms,axis=(0,1), weights=tre_weights) * -1
wecc_wcold_anoms = np.ma.average(wecc_cold_anoms,axis=(0,1), weights=wecc_weights) * -1

### Subset by region first: non-extreme
mro_nonex_anoms = mro_3d * nonex_anoms
npcc_nonex_anoms = npcc_3d * nonex_anoms
rfc_nonex_anoms = rfc_3d * nonex_anoms
serc_nonex_anoms = serc_3d * nonex_anoms
tre_nonex_anoms = tre_3d * nonex_anoms
wecc_nonex_anoms = wecc_3d * nonex_anoms

### CI
mro_hot_ci = np.nansum((mro_hot_anoms * mro_3d_km), axis=(0,1))
npcc_hot_ci = np.nansum((npcc_hot_anoms * npcc_3d_km), axis=(0,1))
rfc_hot_ci = np.nansum((rfc_hot_anoms * rfc_3d_km), axis=(0,1))
serc_hot_ci = np.nansum((serc_hot_anoms * serc_3d_km), axis=(0,1))
tre_hot_ci = np.nansum((tre_hot_anoms * tre_3d_km), axis=(0,1))
wecc_hot_ci = np.nansum((wecc_hot_anoms * wecc_3d_km), axis=(0,1))

mro_cold_ci = np.nansum((mro_cold_anoms * mro_3d_km), axis=(0,1))
npcc_cold_ci = np.nansum((npcc_cold_anoms * npcc_3d_km), axis=(0,1))
rfc_cold_ci = np.nansum((rfc_cold_anoms * rfc_3d_km), axis=(0,1))
serc_cold_ci = np.nansum((serc_cold_anoms * serc_3d_km), axis=(0,1))
tre_cold_ci = np.nansum((tre_cold_anoms * tre_3d_km), axis=(0,1))
wecc_cold_ci = np.nansum((wecc_cold_anoms * wecc_3d_km), axis=(0,1))

mro_nonex_ci = np.nansum((mro_nonex_anoms * mro_3d_km), axis=(0,1))
npcc_nonex_ci = np.nansum((npcc_nonex_anoms * npcc_3d_km), axis=(0,1))
rfc_nonex_ci = np.nansum((rfc_nonex_anoms * rfc_3d_km), axis=(0,1))
serc_nonex_ci = np.nansum((serc_nonex_anoms * serc_3d_km), axis=(0,1))
tre_nonex_ci = np.nansum((tre_nonex_anoms * tre_3d_km), axis=(0,1))
wecc_nonex_ci = np.nansum((wecc_nonex_anoms * wecc_3d_km), axis=(0,1))

### Calculate weighted anomalies
mro_wnonex_anoms = np.ma.average(mro_nonex_anoms,axis=(0,1), weights=mro_weights)
npcc_wnonex_anoms = np.ma.average(npcc_nonex_anoms,axis=(0,1), weights=npcc_weights)
rfc_wnonex_anoms = np.ma.average(rfc_nonex_anoms,axis=(0,1), weights=rfc_weights)
serc_wnonex_anoms = np.ma.average(serc_nonex_anoms,axis=(0,1), weights=serc_weights)
tre_wnonex_anoms = np.ma.average(tre_nonex_anoms,axis=(0,1), weights=tre_weights)
wecc_wnonex_anoms = np.ma.average(wecc_nonex_anoms,axis=(0,1), weights=wecc_weights)

### Create copies of arrays before masking
mro_ws_anom15 = ws_anom15.copy()
npcc_ws_anom15 = ws_anom15.copy()
rfc_ws_anom15 = ws_anom15.copy()
serc_ws_anom15 = ws_anom15.copy()
tre_ws_anom15 = ws_anom15.copy()
wecc_ws_anom15 = ws_anom15.copy()

### Mask out non-heatwave locations
hot_ws_anoms = ws_anom15.copy()
cold_ws_anoms = ws_anom15.copy() 
nonex_ws_anoms = ws_anom15.copy()
hot_ws_anoms = np.ma.array(hot_ws_anoms)#, mask=~hot_loc)
cold_ws_anoms = np.ma.array(cold_ws_anoms)#, mask=~cold_loc)
nonex_ws_anoms = np.ma.array(nonex_ws_anoms)#, mask=~nonex_loc)

### Mask arrays
mro_ws_anom15[mro_3d==0] = np.nan
npcc_ws_anom15[npcc_3d==0] = np.nan
rfc_ws_anom15[rfc_3d==0] = np.nan
serc_ws_anom15[serc_3d==0] = np.nan
tre_ws_anom15[tre_3d==0] = np.nan
wecc_ws_anom15[wecc_3d==0] = np.nan


### Area weighted anomalies
### Subset by region first
mro_hot_ws_anoms = mro_3d * hot_ws_anoms
npcc_hot_ws_anoms = npcc_3d * hot_ws_anoms
rfc_hot_ws_anoms = rfc_3d * hot_ws_anoms
serc_hot_ws_anoms = serc_3d * hot_ws_anoms
tre_hot_ws_anoms = tre_3d * hot_ws_anoms
wecc_hot_ws_anoms = wecc_3d * hot_ws_anoms

### Calculate weighted anomalies
mro_whot_ws_anoms = np.ma.average(mro_hot_ws_anoms,axis=(0,1),weights=mro_weights)
npcc_whot_ws_anoms = np.ma.average(npcc_hot_ws_anoms,axis=(0,1),weights=npcc_weights)
rfc_whot_ws_anoms = np.ma.average(rfc_hot_ws_anoms,axis=(0,1),weights=rfc_weights)
serc_whot_ws_anoms = np.ma.average(serc_hot_ws_anoms,axis=(0,1),weights=serc_weights)
tre_whot_ws_anoms = np.ma.average(tre_hot_ws_anoms,axis=(0,1),weights=tre_weights)
wecc_whot_ws_anoms = np.ma.average(wecc_hot_ws_anoms,axis=(0,1),weights=wecc_weights)


### Area weighted anomalies
### Subset by region first
mro_cold_ws_anoms = mro_3d * cold_ws_anoms
npcc_cold_ws_anoms = npcc_3d * cold_ws_anoms
rfc_cold_ws_anoms = rfc_3d * cold_ws_anoms
serc_cold_ws_anoms = serc_3d * cold_ws_anoms
tre_cold_ws_anoms = tre_3d * cold_ws_anoms
wecc_cold_ws_anoms = wecc_3d * cold_ws_anoms

### Calculate weighted anomalies
mro_wcold_ws_anoms = np.ma.average(mro_cold_ws_anoms,axis=(0,1), weights=mro_weights)
npcc_wcold_ws_anoms = np.ma.average(npcc_cold_ws_anoms,axis=(0,1), weights=npcc_weights)
rfc_wcold_ws_anoms = np.ma.average(rfc_cold_ws_anoms,axis=(0,1), weights=rfc_weights)
serc_wcold_ws_anoms = np.ma.average(serc_cold_ws_anoms,axis=(0,1), weights=serc_weights)
tre_wcold_ws_anoms = np.ma.average(tre_cold_ws_anoms,axis=(0,1), weights=tre_weights)
wecc_wcold_ws_anoms = np.ma.average(wecc_cold_ws_anoms,axis=(0,1), weights=wecc_weights)

### Subset by region first
mro_nonex_ws_anoms = mro_3d * nonex_ws_anoms
npcc_nonex_ws_anoms = npcc_3d * nonex_ws_anoms
rfc_nonex_ws_anoms = rfc_3d * nonex_ws_anoms
serc_nonex_ws_anoms = serc_3d * nonex_ws_anoms
tre_nonex_ws_anoms = tre_3d * nonex_ws_anoms
wecc_nonex_ws_anoms = wecc_3d * nonex_ws_anoms

### Calculate weighted anomalies
mro_wnonex_ws_anoms = np.ma.average(mro_nonex_ws_anoms,axis=(0,1), weights=mro_weights)
npcc_wnonex_ws_anoms = np.ma.average(npcc_nonex_ws_anoms,axis=(0,1), weights=npcc_weights)
rfc_wnonex_ws_anoms = np.ma.average(rfc_nonex_ws_anoms,axis=(0,1), weights=rfc_weights)
serc_wnonex_ws_anoms = np.ma.average(serc_nonex_ws_anoms,axis=(0,1), weights=serc_weights)
tre_wnonex_ws_anoms = np.ma.average(tre_nonex_ws_anoms,axis=(0,1), weights=tre_weights)
wecc_wnonex_ws_anoms = np.ma.average(wecc_nonex_ws_anoms,axis=(0,1), weights=wecc_weights)



### Create copies of arrays before masking
mro_ssrd_anom15 = ssrd_anom15.copy()
npcc_ssrd_anom15 = ssrd_anom15.copy()
rfc_ssrd_anom15 = ssrd_anom15.copy()
serc_ssrd_anom15 = ssrd_anom15.copy()
tre_ssrd_anom15 = ssrd_anom15.copy()
wecc_ssrd_anom15 = ssrd_anom15.copy()

### Mask arrays
mro_ssrd_anom15[mro_3d==0] = np.nan
npcc_ssrd_anom15[npcc_3d==0] = np.nan
rfc_ssrd_anom15[rfc_3d==0] = np.nan
serc_ssrd_anom15[serc_3d==0] = np.nan
tre_ssrd_anom15[tre_3d==0] = np.nan
wecc_ssrd_anom15[wecc_3d==0] = np.nan

### Mask out non-heatwave locations
hot_ssrd_anoms = ssrd_anom15.copy()
cold_ssrd_anoms = ssrd_anom15.copy() 
nonex_ssrd_anoms = ssrd_anom15.copy()
hot_ssrd_anoms = np.ma.array(hot_ssrd_anoms)#, mask=~hot_loc)
cold_ssrd_anoms = np.ma.array(cold_ssrd_anoms)#, mask=~cold_loc)
nonex_ssrd_anoms = np.ma.array(nonex_ssrd_anoms)#, mask=~nonex_loc)


### Area weighted anomalies
### Subset by region first
mro_hot_ssrd_anoms = mro_3d * hot_ssrd_anoms
npcc_hot_ssrd_anoms = npcc_3d * hot_ssrd_anoms
rfc_hot_ssrd_anoms = rfc_3d * hot_ssrd_anoms
serc_hot_ssrd_anoms = serc_3d * hot_ssrd_anoms
tre_hot_ssrd_anoms = tre_3d * hot_ssrd_anoms
wecc_hot_ssrd_anoms = wecc_3d * hot_ssrd_anoms

### Calculate weighted anomalies
mro_whot_ssrd_anoms = np.ma.average(mro_hot_ssrd_anoms,axis=(0,1), weights=mro_weights)
npcc_whot_ssrd_anoms = np.ma.average(npcc_hot_ssrd_anoms,axis=(0,1), weights=npcc_weights)
rfc_whot_ssrd_anoms = np.ma.average(rfc_hot_ssrd_anoms,axis=(0,1), weights=rfc_weights)
serc_whot_ssrd_anoms = np.ma.average(serc_hot_ssrd_anoms,axis=(0,1), weights=serc_weights)
tre_whot_ssrd_anoms = np.ma.average(tre_hot_ssrd_anoms,axis=(0,1), weights=tre_weights)
wecc_whot_ssrd_anoms = np.ma.average(wecc_hot_ssrd_anoms,axis=(0,1), weights=wecc_weights)

### Area weighted anomalies
### Subset by region first
mro_cold_ssrd_anoms = mro_3d * cold_ssrd_anoms
npcc_cold_ssrd_anoms = npcc_3d * cold_ssrd_anoms
rfc_cold_ssrd_anoms = rfc_3d * cold_ssrd_anoms
serc_cold_ssrd_anoms = serc_3d * cold_ssrd_anoms
tre_cold_ssrd_anoms = tre_3d * cold_ssrd_anoms
wecc_cold_ssrd_anoms = wecc_3d * cold_ssrd_anoms

### Calculate weighted anomalies
mro_wcold_ssrd_anoms = np.ma.average(mro_cold_ssrd_anoms,axis=(0,1), weights=mro_weights)
npcc_wcold_ssrd_anoms = np.ma.average(npcc_cold_ssrd_anoms,axis=(0,1), weights=npcc_weights)
rfc_wcold_ssrd_anoms = np.ma.average(rfc_cold_ssrd_anoms,axis=(0,1), weights=rfc_weights)
serc_wcold_ssrd_anoms = np.ma.average(serc_cold_ssrd_anoms,axis=(0,1), weights=serc_weights)
tre_wcold_ssrd_anoms = np.ma.average(tre_cold_ssrd_anoms,axis=(0,1), weights=tre_weights)
wecc_wcold_ssrd_anoms = np.ma.average(wecc_cold_ssrd_anoms,axis=(0,1), weights=wecc_weights)

### Subset by region first
mro_nonex_ssrd_anoms = mro_3d * nonex_ssrd_anoms
npcc_nonex_ssrd_anoms = npcc_3d * nonex_ssrd_anoms
rfc_nonex_ssrd_anoms = rfc_3d * nonex_ssrd_anoms
serc_nonex_ssrd_anoms = serc_3d * nonex_ssrd_anoms
tre_nonex_ssrd_anoms = tre_3d * nonex_ssrd_anoms
wecc_nonex_ssrd_anoms = wecc_3d * nonex_ssrd_anoms

### Calculate weighted anomalies
mro_wnonex_ssrd_anoms = np.ma.average(mro_nonex_ssrd_anoms,axis=(0,1), weights=mro_weights)
npcc_wnonex_ssrd_anoms = np.ma.average(npcc_nonex_ssrd_anoms,axis=(0,1), weights=npcc_weights)
rfc_wnonex_ssrd_anoms = np.ma.average(rfc_nonex_ssrd_anoms,axis=(0,1), weights=rfc_weights)
serc_wnonex_ssrd_anoms = np.ma.average(serc_nonex_ssrd_anoms,axis=(0,1), weights=serc_weights)
tre_wnonex_ssrd_anoms = np.ma.average(tre_nonex_ssrd_anoms,axis=(0,1), weights=tre_weights)
wecc_wnonex_ssrd_anoms = np.ma.average(wecc_nonex_ssrd_anoms,axis=(0,1), weights=wecc_weights)

### Create copies of arrays before masking
mro_wp_anom15 = wp_anom15.copy()
npcc_wp_anom15 = wp_anom15.copy()
rfc_wp_anom15 = wp_anom15.copy()
serc_wp_anom15 = wp_anom15.copy()
tre_wp_anom15 = wp_anom15.copy()
wecc_wp_anom15 = wp_anom15.copy()

### Mask out non-heatwave locations
hot_wp_anoms = wp_anom15.copy()
cold_wp_anoms = wp_anom15.copy() 
nonex_wp_anoms = wp_anom15.copy()
hot_wp_anoms = np.ma.array(hot_wp_anoms)#, mask=~hot_loc)
cold_wp_anoms = np.ma.array(cold_wp_anoms)#, mask=~cold_loc)
nonex_wp_anoms = np.ma.array(nonex_wp_anoms)#, mask=~nonex_loc)

### Mask arrays
mro_wp_anom15[mro_3d==0] = np.nan
npcc_wp_anom15[npcc_3d==0] = np.nan
rfc_wp_anom15[rfc_3d==0] = np.nan
serc_wp_anom15[serc_3d==0] = np.nan
tre_wp_anom15[tre_3d==0] = np.nan
wecc_wp_anom15[wecc_3d==0] = np.nan


### Area weighted anomalies
### Subset by region first
mro_hot_wp_anoms = mro_3d * hot_wp_anoms
npcc_hot_wp_anoms = npcc_3d * hot_wp_anoms
rfc_hot_wp_anoms = rfc_3d * hot_wp_anoms
serc_hot_wp_anoms = serc_3d * hot_wp_anoms
tre_hot_wp_anoms = tre_3d * hot_wp_anoms
wecc_hot_wp_anoms = wecc_3d * hot_wp_anoms

### Calculate weighted anomalies
mro_whot_wp_anoms = np.ma.average(mro_hot_wp_anoms,axis=(0,1), weights=mro_weights)
npcc_whot_wp_anoms = np.ma.average(npcc_hot_wp_anoms,axis=(0,1), weights=npcc_weights)
rfc_whot_wp_anoms = np.ma.average(rfc_hot_wp_anoms,axis=(0,1), weights=rfc_weights)
serc_whot_wp_anoms = np.ma.average(serc_hot_wp_anoms,axis=(0,1), weights=serc_weights)
tre_whot_wp_anoms = np.ma.average(tre_hot_wp_anoms,axis=(0,1), weights=tre_weights)
wecc_whot_wp_anoms = np.ma.average(wecc_hot_wp_anoms,axis=(0,1), weights=wecc_weights)


### Area weighted anomalies
### Subset by region first
mro_cold_wp_anoms = mro_3d * cold_wp_anoms
npcc_cold_wp_anoms = npcc_3d * cold_wp_anoms
rfc_cold_wp_anoms = rfc_3d * cold_wp_anoms
serc_cold_wp_anoms = serc_3d * cold_wp_anoms
tre_cold_wp_anoms = tre_3d * cold_wp_anoms
wecc_cold_wp_anoms = wecc_3d * cold_wp_anoms


### Calculate weighted anomalies
mro_wcold_wp_anoms = np.ma.average(mro_cold_wp_anoms,axis=(0,1), weights=mro_weights)
npcc_wcold_wp_anoms = np.ma.average(npcc_cold_wp_anoms,axis=(0,1), weights=npcc_weights)
rfc_wcold_wp_anoms = np.ma.average(rfc_cold_wp_anoms,axis=(0,1), weights=rfc_weights)
serc_wcold_wp_anoms = np.ma.average(serc_cold_wp_anoms,axis=(0,1), weights=serc_weights)
tre_wcold_wp_anoms = np.ma.average(tre_cold_wp_anoms,axis=(0,1), weights=tre_weights)
wecc_wcold_wp_anoms = np.ma.average(wecc_cold_wp_anoms,axis=(0,1), weights=wecc_weights)

### Subset by region first
mro_nonex_wp_anoms = mro_3d * nonex_wp_anoms
npcc_nonex_wp_anoms = npcc_3d * nonex_wp_anoms
rfc_nonex_wp_anoms = rfc_3d * nonex_wp_anoms
serc_nonex_wp_anoms = serc_3d * nonex_wp_anoms
tre_nonex_wp_anoms = tre_3d * nonex_wp_anoms
wecc_nonex_wp_anoms = wecc_3d * nonex_wp_anoms

### Calculate weighted anomalies
mro_wnonex_wp_anoms = np.ma.average(mro_nonex_wp_anoms,axis=(0,1), weights=mro_weights)
npcc_wnonex_wp_anoms = np.ma.average(npcc_nonex_wp_anoms,axis=(0,1), weights=npcc_weights)
rfc_wnonex_wp_anoms = np.ma.average(rfc_nonex_wp_anoms,axis=(0,1), weights=rfc_weights)
serc_wnonex_wp_anoms = np.ma.average(serc_nonex_wp_anoms,axis=(0,1), weights=serc_weights)
tre_wnonex_wp_anoms = np.ma.average(tre_nonex_wp_anoms,axis=(0,1), weights=tre_weights)
wecc_wnonex_wp_anoms = np.ma.average(wecc_nonex_wp_anoms,axis=(0,1), weights=wecc_weights)


### Create copies of arrays before masking
mro_sp_anom15 = sp_anom15.copy()
npcc_sp_anom15 = sp_anom15.copy()
rfc_sp_anom15 = sp_anom15.copy()
serc_sp_anom15 = sp_anom15.copy()
tre_sp_anom15 = sp_anom15.copy()
wecc_sp_anom15 = sp_anom15.copy()

### Mask arrays
mro_sp_anom15[mro_3d==0] = np.nan
npcc_sp_anom15[npcc_3d==0] = np.nan
rfc_sp_anom15[rfc_3d==0] = np.nan
serc_sp_anom15[serc_3d==0] = np.nan
tre_sp_anom15[tre_3d==0] = np.nan
wecc_sp_anom15[wecc_3d==0] = np.nan

### Create masks for weighted averages
## Because this is for energy potential, it is the same
## for extreme events and non-extreme events
hot_sp_anoms = sp_anom15.copy()
cold_sp_anoms = sp_anom15.copy() 
nonex_sp_anoms = sp_anom15.copy()
hot_sp_anoms = np.ma.array(hot_sp_anoms)
cold_sp_anoms = np.ma.array(cold_sp_anoms)
nonex_sp_anoms = np.ma.array(nonex_sp_anoms)


### Area weighted anomalies
## Subset by region first
mro_hot_sp_anoms = mro_3d * hot_sp_anoms
npcc_hot_sp_anoms = npcc_3d * hot_sp_anoms
rfc_hot_sp_anoms = rfc_3d * hot_sp_anoms
serc_hot_sp_anoms = serc_3d * hot_sp_anoms
tre_hot_sp_anoms = tre_3d * hot_sp_anoms
wecc_hot_sp_anoms = wecc_3d * hot_sp_anoms

## Calculate weighted anomalies
mro_whot_sp_anoms = np.ma.average(mro_hot_sp_anoms,axis=(0,1), weights=mro_weights)
npcc_whot_sp_anoms = np.ma.average(npcc_hot_sp_anoms,axis=(0,1), weights=npcc_weights)
rfc_whot_sp_anoms = np.ma.average(rfc_hot_sp_anoms,axis=(0,1), weights=rfc_weights)
serc_whot_sp_anoms = np.ma.average(serc_hot_sp_anoms,axis=(0,1), weights=serc_weights)
tre_whot_sp_anoms = np.ma.average(tre_hot_sp_anoms,axis=(0,1), weights=tre_weights)
wecc_whot_sp_anoms = np.ma.average(wecc_hot_sp_anoms,axis=(0,1), weights=wecc_weights)

### Area weighted anomalies
## Subset by region first
mro_cold_sp_anoms = mro_3d * cold_sp_anoms
npcc_cold_sp_anoms = npcc_3d * cold_sp_anoms
rfc_cold_sp_anoms = rfc_3d * cold_sp_anoms
serc_cold_sp_anoms = serc_3d * cold_sp_anoms
tre_cold_sp_anoms = tre_3d * cold_sp_anoms
wecc_cold_sp_anoms = wecc_3d * cold_sp_anoms

### Calculate weighted anomalies
mro_wcold_sp_anoms = np.ma.average(mro_cold_sp_anoms,axis=(0,1), weights=mro_weights)
npcc_wcold_sp_anoms = np.ma.average(npcc_cold_sp_anoms,axis=(0,1), weights=npcc_weights)
rfc_wcold_sp_anoms = np.ma.average(rfc_cold_sp_anoms,axis=(0,1), weights=rfc_weights)
serc_wcold_sp_anoms = np.ma.average(serc_cold_sp_anoms,axis=(0,1), weights=serc_weights)
tre_wcold_sp_anoms = np.ma.average(tre_cold_sp_anoms,axis=(0,1), weights=tre_weights)
wecc_wcold_sp_anoms = np.ma.average(wecc_cold_sp_anoms,axis=(0,1), weights=wecc_weights)

### Subset by region first
mro_nonex_sp_anoms = mro_3d * nonex_sp_anoms
npcc_nonex_sp_anoms = npcc_3d * nonex_sp_anoms
rfc_nonex_sp_anoms = rfc_3d * nonex_sp_anoms
serc_nonex_sp_anoms = serc_3d * nonex_sp_anoms
tre_nonex_sp_anoms = tre_3d * nonex_sp_anoms
wecc_nonex_sp_anoms = wecc_3d * nonex_sp_anoms

### Calculate weighted anomalies
mro_wnonex_sp_anoms = np.ma.average(mro_nonex_sp_anoms,axis=(0,1), weights=mro_weights)
npcc_wnonex_sp_anoms = np.ma.average(npcc_nonex_sp_anoms,axis=(0,1), weights=npcc_weights)
rfc_wnonex_sp_anoms = np.ma.average(rfc_nonex_sp_anoms,axis=(0,1), weights=rfc_weights)
serc_wnonex_sp_anoms = np.ma.average(serc_nonex_sp_anoms,axis=(0,1), weights=serc_weights)
tre_wnonex_sp_anoms = np.ma.average(tre_nonex_sp_anoms,axis=(0,1), weights=tre_weights)
wecc_wnonex_sp_anoms = np.ma.average(wecc_nonex_sp_anoms,axis=(0,1), weights=wecc_weights)

### Create masks for weighted averages
## Because this is for energy potential, it is the same
## for extreme events and non-extreme events
hot_sp = solar_potential.copy()
cold_sp = solar_potential.copy() 
nonex_sp = solar_potential.copy()
hot_sp = np.ma.array(hot_sp)
cold_sp = np.ma.array(cold_sp)
nonex_sp = np.ma.array(nonex_sp)

### Area weighted anomalies
## Subset by region first
mro_hot_sp = mro_3d * hot_sp
npcc_hot_sp = npcc_3d * hot_sp
rfc_hot_sp = rfc_3d * hot_sp
serc_hot_sp = serc_3d * hot_sp
tre_hot_sp = tre_3d * hot_sp
wecc_hot_sp = wecc_3d * hot_sp

### Calculate weighted anomalies
mro_whot_sp = np.ma.average(mro_hot_sp,axis=(0,1), weights=mro_weights)
npcc_whot_sp = np.ma.average(npcc_hot_sp,axis=(0,1), weights=npcc_weights)
rfc_whot_sp = np.ma.average(rfc_hot_sp,axis=(0,1), weights=rfc_weights)
serc_whot_sp = np.ma.average(serc_hot_sp,axis=(0,1), weights=serc_weights)
tre_whot_sp = np.ma.average(tre_hot_sp,axis=(0,1), weights=tre_weights)
wecc_whot_sp = np.ma.average(wecc_hot_sp,axis=(0,1), weights=wecc_weights)

### Area weighted anomalies
## Subset by region first
mro_cold_sp = mro_3d * cold_sp
npcc_cold_sp = npcc_3d * cold_sp
rfc_cold_sp = rfc_3d * cold_sp
serc_cold_sp = serc_3d * cold_sp
tre_cold_sp = tre_3d * cold_sp
wecc_cold_sp = wecc_3d * cold_sp

### Calculate weighted anomalies
mro_wcold_sp = np.ma.average(mro_cold_sp,axis=(0,1), weights=mro_weights)
npcc_wcold_sp = np.ma.average(npcc_cold_sp,axis=(0,1), weights=npcc_weights)
rfc_wcold_sp = np.ma.average(rfc_cold_sp,axis=(0,1), weights=rfc_weights)
serc_wcold_sp = np.ma.average(serc_cold_sp,axis=(0,1), weights=serc_weights)
tre_wcold_sp = np.ma.average(tre_cold_sp,axis=(0,1), weights=tre_weights)
wecc_wcold_sp = np.ma.average(wecc_cold_sp,axis=(0,1), weights=wecc_weights)

### Subset by region first
mro_nonex_sp = mro_3d * nonex_sp
npcc_nonex_sp = npcc_3d * nonex_sp
rfc_nonex_sp = rfc_3d * nonex_sp
serc_nonex_sp = serc_3d * nonex_sp
tre_nonex_sp = tre_3d * nonex_sp
wecc_nonex_sp = wecc_3d * nonex_sp

### Calculate weighted anomalies
mro_wnonex_sp = np.ma.average(mro_nonex_sp,axis=(0,1), weights=mro_weights)
npcc_wnonex_sp = np.ma.average(npcc_nonex_sp,axis=(0,1), weights=npcc_weights)
rfc_wnonex_sp = np.ma.average(rfc_nonex_sp,axis=(0,1), weights=rfc_weights)
serc_wnonex_sp = np.ma.average(serc_nonex_sp,axis=(0,1), weights=serc_weights)
tre_wnonex_sp = np.ma.average(tre_nonex_sp,axis=(0,1), weights=tre_weights)
wecc_wnonex_sp = np.ma.average(wecc_nonex_sp,axis=(0,1), weights=wecc_weights)


### Create masks for weighted averages
## Because this is for energy potential, it is the same
## for extreme events and non-extreme events
hot_wp = wind_potential.copy()
cold_wp = wind_potential.copy() 
nonex_wp = wind_potential.copy()
hot_wp = np.ma.array(hot_wp)
cold_wp = np.ma.array(cold_wp)
nonex_wp = np.ma.array(nonex_wp)


### Area weighted anomalies
## Subset by region first
mro_hot_wp = mro_3d * hot_wp
npcc_hot_wp = npcc_3d * hot_wp
rfc_hot_wp = rfc_3d * hot_wp
serc_hot_wp = serc_3d * hot_wp
tre_hot_wp = tre_3d * hot_wp
wecc_hot_wp = wecc_3d * hot_wp

### Calculate weighted anomalies
mro_whot_wp = np.ma.average(mro_hot_wp,axis=(0,1), weights=mro_weights)
npcc_whot_wp = np.ma.average(npcc_hot_wp,axis=(0,1), weights=npcc_weights)
rfc_whot_wp = np.ma.average(rfc_hot_wp,axis=(0,1), weights=rfc_weights)
serc_whot_wp = np.ma.average(serc_hot_wp,axis=(0,1), weights=serc_weights)
tre_whot_wp = np.ma.average(tre_hot_wp,axis=(0,1), weights=tre_weights)
wecc_whot_wp = np.ma.average(wecc_hot_wp,axis=(0,1), weights=wecc_weights)

### Area weighted anomalies
## Subset by region first
mro_cold_wp = mro_3d * cold_wp
npcc_cold_wp = npcc_3d * cold_wp
rfc_cold_wp = rfc_3d * cold_wp
serc_cold_wp = serc_3d * cold_wp
tre_cold_wp = tre_3d * cold_wp
wecc_cold_wp = wecc_3d * cold_wp

### Calculate weighted anomalies
mro_wcold_wp = np.ma.average(mro_cold_wp,axis=(0,1), weights=mro_weights)
npcc_wcold_wp = np.ma.average(npcc_cold_wp,axis=(0,1), weights=npcc_weights)
rfc_wcold_wp = np.ma.average(rfc_cold_wp,axis=(0,1), weights=rfc_weights)
serc_wcold_wp = np.ma.average(serc_cold_wp,axis=(0,1), weights=serc_weights)
tre_wcold_wp = np.ma.average(tre_cold_wp,axis=(0,1), weights=tre_weights)
wecc_wcold_wp = np.ma.average(wecc_cold_wp,axis=(0,1), weights=wecc_weights)

### Subset by region first
mro_nonex_wp = mro_3d * nonex_wp
npcc_nonex_wp = npcc_3d * nonex_wp
rfc_nonex_wp = rfc_3d * nonex_wp
serc_nonex_wp = serc_3d * nonex_wp
tre_nonex_wp = tre_3d * nonex_wp
wecc_nonex_wp = wecc_3d * nonex_wp

### Calculate weighted anomalies
mro_wnonex_wp = np.ma.average(mro_nonex_wp,axis=(0,1), weights=mro_weights)
npcc_wnonex_wp = np.ma.average(npcc_nonex_wp,axis=(0,1), weights=npcc_weights)
rfc_wnonex_wp = np.ma.average(rfc_nonex_wp,axis=(0,1), weights=rfc_weights)
serc_wnonex_wp = np.ma.average(serc_nonex_wp,axis=(0,1), weights=serc_weights)
tre_wnonex_wp = np.ma.average(tre_nonex_wp,axis=(0,1), weights=tre_weights)
wecc_wnonex_wp = np.ma.average(wecc_nonex_wp,axis=(0,1), weights=wecc_weights)


### Find extents for each day
## Extent arrays for heatwaves
mro_hot_area = np.nansum(hot_area_mro, axis=(0,1)) 
npcc_hot_area = np.nansum(hot_area_npcc, axis=(0,1))
rfc_hot_area = np.nansum(hot_area_rfc, axis=(0,1))
serc_hot_area = np.nansum(hot_area_serc, axis=(0,1))
tre_hot_area = np.nansum(hot_area_tre, axis=(0,1))
wecc_hot_area = np.nansum(hot_area_wecc, axis=(0,1))

### Extent arrays for coldwaves
mro_cold_area = np.nansum(cold_area_mro, axis=(0,1))
npcc_cold_area = np.nansum(cold_area_npcc, axis=(0,1))
rfc_cold_area = np.nansum(cold_area_rfc, axis=(0,1))
serc_cold_area = np.nansum(cold_area_serc, axis=(0,1))
tre_cold_area = np.nansum(cold_area_tre, axis=(0,1))
wecc_cold_area = np.nansum(cold_area_wecc, axis=(0,1))

### Extent arrays for nonexwaves
mro_nonex_area = np.nansum(nonex_area_mro, axis=(0,1))
npcc_nonex_area = np.nansum(nonex_area_npcc, axis=(0,1))
rfc_nonex_area = np.nansum(nonex_area_rfc, axis=(0,1))
serc_nonex_area = np.nansum(nonex_area_serc, axis=(0,1))
tre_nonex_area = np.nansum(nonex_area_tre, axis=(0,1))
wecc_nonex_area = np.nansum(nonex_area_wecc, axis=(0,1))

### Percentages
mro_hot_ex = (mro_hot_area / mro_tot_a) * 100
npcc_hot_ex = (npcc_hot_area / npcc_tot_a) * 100
rfc_hot_ex = (rfc_hot_area / rfc_tot_a) * 100
serc_hot_ex = (serc_hot_area / serc_tot_a) * 100
tre_hot_ex = (tre_hot_area / tre_tot_a) * 100
wecc_hot_ex = (wecc_hot_area / wecc_tot_a) * 100

### Extent arrays for coldwaves
mro_cold_ex = (mro_cold_area / mro_tot_a) * 100
npcc_cold_ex = (npcc_cold_area / npcc_tot_a) * 100
rfc_cold_ex = (rfc_cold_area / rfc_tot_a) * 100
serc_cold_ex = (serc_cold_area / serc_tot_a) * 100
tre_cold_ex = (tre_cold_area / tre_tot_a) * 100
wecc_cold_ex = (wecc_cold_area / wecc_tot_a) * 100

### Extent arrays for nonexwaves
mro_nonex_ex = (mro_nonex_area / mro_tot_a) * 100
npcc_nonex_ex = (npcc_nonex_area / npcc_tot_a) * 100
rfc_nonex_ex = (rfc_nonex_area / rfc_tot_a) * 100
serc_nonex_ex = (serc_hot_ex / serc_tot_a) * 100
tre_nonex_ex = (tre_hot_ex / tre_tot_a) * 100
wecc_nonex_ex = (wecc_hot_ex / wecc_tot_a) * 100

### Energy proxies
## Heating Degree Days weighted by population
mro_hdd = np.nansum(mro_hdd_pop, axis=(0,1))
npcc_hdd = np.nansum(npcc_hdd_pop, axis=(0,1))
rfc_hdd = np.nansum(rfc_hdd_pop, axis=(0,1))
serc_hdd = np.nansum(serc_hdd_pop, axis=(0,1))
tre_hdd = np.nansum(tre_hdd_pop, axis=(0,1))
wecc_hdd = np.nansum(wecc_hdd_pop, axis=(0,1))

## Cooling Degree Days weighted by population
mro_cdd = np.nansum(mro_cdd_pop, axis=(0,1))
npcc_cdd = np.nansum(npcc_cdd_pop, axis=(0,1))
rfc_cdd = np.nansum(rfc_cdd_pop, axis=(0,1))
serc_cdd = np.nansum(serc_cdd_pop, axis=(0,1))
tre_cdd = np.nansum(tre_cdd_pop, axis=(0,1))
wecc_cdd = np.nansum(wecc_cdd_pop, axis=(0,1))

## All Degree Days weighted by population
mro_enprox = np.nansum(mro_enprox_pop, axis=(0,1))
npcc_enprox = np.nansum(npcc_enprox_pop, axis=(0,1))
rfc_enprox = np.nansum(rfc_enprox_pop, axis=(0,1))
serc_enprox = np.nansum(serc_enprox_pop, axis=(0,1))
tre_enprox = np.nansum(tre_enprox_pop, axis=(0,1))
wecc_enprox = np.nansum(wecc_enprox_pop, axis=(0,1))

## Create year arrays
mro_yr = yr.copy()
npcc_yr = yr.copy()
rfc_yr = yr.copy()
serc_yr = yr.copy()
tre_yr = yr.copy()
wecc_yr = yr.copy()

## Tile them by 3 for each event type
mro_year = np.tile(mro_yr, 3)
npcc_year = np.tile(npcc_yr, 3)
rfc_year = np.tile(rfc_yr, 3)
serc_year = np.tile(serc_yr, 3)
tre_year = np.tile(tre_yr, 3)
wecc_year = np.tile(wecc_yr, 3)

## Create array of NERC region names
nerc_names = np.array([['MRO'], ['NPCC'], ['RFC'],
                         ['SERC'], ['TRE'], ['WECC']])

## Shape the array so it follows the number of regions and the 3 types of events
nerc_regions = np.squeeze(np.repeat(nerc_names, mro_cdd.shape[0] * 3, axis=0))

## Create an array of the 3 types of events
waves = np.array([['Hot'],['Cold'],['Non-Extreme']])

## Mold the event type array into the correct shape
events = np.squeeze(np.repeat(waves, mro_cdd.shape[0], axis=0))
event_type = np.tile(events, 6)

## Tile the dates so they follow the number of NERC regions and event types
dates = np.tile(date, 18)

area = [mro_tot_a, npcc_tot_a, rfc_tot_a, serc_tot_a, tre_tot_a, wecc_tot_a]

area = np.repeat(area, len(mro_year))
 
### Create pandas dataframe ("long and skinny") for plotting
## Stack arrays and if there are any 0 values set them to NaN (as these are the masked grid cells)
intensity = np.concatenate((mro_whot_anoms, mro_wcold_anoms, mro_wnonex_anoms,
                            npcc_whot_anoms, npcc_wcold_anoms, npcc_wnonex_anoms,
                            rfc_whot_anoms, rfc_wcold_anoms, rfc_wnonex_anoms,
                            serc_whot_anoms, serc_wcold_anoms, serc_wnonex_anoms,
                            tre_whot_anoms, tre_wcold_anoms, tre_wnonex_anoms,
                            wecc_whot_anoms, wecc_wcold_anoms, wecc_wnonex_anoms))

intensity[intensity == 0] = np.nan

extent = np.concatenate((mro_hot_ex, mro_cold_ex, mro_nonex_ex,
                          npcc_hot_ex, npcc_cold_ex, npcc_nonex_ex,
                          rfc_hot_ex, rfc_cold_ex, rfc_nonex_ex,
                          serc_hot_ex, serc_cold_ex, serc_nonex_ex, 
                          tre_hot_ex, tre_cold_ex, tre_nonex_ex, 
                          wecc_hot_ex, wecc_cold_ex, wecc_nonex_ex))

extent[extent==0] = np.nan

event_area = np.concatenate((mro_hot_area, mro_cold_area, mro_nonex_area,
                          npcc_hot_area, npcc_cold_area, npcc_nonex_area,
                          rfc_hot_area, rfc_cold_area, rfc_nonex_area,
                          serc_hot_area, serc_cold_area, serc_nonex_area, 
                          tre_hot_area, tre_cold_area, tre_nonex_area, 
                          wecc_hot_area, wecc_cold_area, wecc_nonex_area))

CI = np.concatenate((mro_hot_ci, mro_cold_ci, mro_nonex_ci,
                            npcc_hot_ci, npcc_cold_ci, npcc_nonex_ci,
                            rfc_hot_ci, rfc_cold_ci, rfc_nonex_ci,
                            serc_hot_ci, serc_cold_ci, serc_nonex_ci,
                            tre_hot_ci, tre_cold_ci, tre_nonex_ci,
                            wecc_hot_ci, wecc_cold_ci, wecc_nonex_ci))

energy_proxy = np.concatenate((mro_cdd, mro_hdd, mro_enprox,
                                npcc_cdd, npcc_hdd, npcc_enprox,
                                rfc_cdd, rfc_hdd, rfc_enprox,
                                serc_cdd, serc_hdd, serc_enprox,
                                tre_cdd, tre_hdd, tre_enprox, 
                                wecc_cdd, wecc_hdd, wecc_enprox))

energy_proxy[energy_proxy==0] = np.nan

windspeed_array = np.concatenate((mro_ws, npcc_ws,
                            rfc_ws, serc_ws,
                            tre_ws, wecc_ws))

ssrd_array = np.concatenate((mro_ssrd, npcc_ssrd,
                             rfc_ssrd, serc_ssrd,
                             tre_ssrd, wecc_ssrd))

year = np.concatenate((mro_year, npcc_year, rfc_year,
                        serc_year, tre_year, wecc_year))

windspeed_anoms = np.concatenate((mro_whot_ws_anoms, mro_wcold_ws_anoms, mro_wnonex_ws_anoms,
                                  npcc_whot_ws_anoms, npcc_wcold_ws_anoms, npcc_wnonex_ws_anoms,
                                  rfc_whot_ws_anoms, rfc_wcold_ws_anoms, rfc_wnonex_ws_anoms, 
                                  serc_whot_ws_anoms, serc_wcold_ws_anoms, serc_wnonex_ws_anoms,
                                  tre_whot_ws_anoms, tre_wcold_ws_anoms, serc_wnonex_ws_anoms, 
                                  wecc_whot_ws_anoms, wecc_wcold_ws_anoms, serc_wnonex_ws_anoms))

ssrd_anoms = np.concatenate((mro_whot_ssrd_anoms, mro_wcold_ssrd_anoms, mro_wnonex_ssrd_anoms,
                                  npcc_whot_ssrd_anoms, npcc_wcold_ssrd_anoms, npcc_wnonex_ssrd_anoms,
                                  rfc_whot_ssrd_anoms, rfc_wcold_ssrd_anoms, rfc_wnonex_ssrd_anoms, 
                                  serc_whot_ssrd_anoms, serc_wcold_ssrd_anoms, serc_wnonex_ssrd_anoms,
                                  tre_whot_ssrd_anoms, tre_wcold_ssrd_anoms, serc_wnonex_ssrd_anoms, 
                                  wecc_whot_ssrd_anoms, wecc_wcold_ssrd_anoms, serc_wnonex_ssrd_anoms))

windpower_anoms = np.concatenate((mro_whot_wp_anoms, mro_wcold_wp_anoms, mro_wnonex_wp_anoms,
                                  npcc_whot_wp_anoms, npcc_wcold_wp_anoms, npcc_wnonex_wp_anoms,
                                  rfc_whot_wp_anoms, rfc_wcold_wp_anoms, rfc_wnonex_wp_anoms, 
                                  serc_whot_wp_anoms, serc_wcold_wp_anoms, serc_wnonex_wp_anoms,
                                  tre_whot_wp_anoms, tre_wcold_wp_anoms, serc_wnonex_wp_anoms, 
                                  wecc_whot_wp_anoms, wecc_wcold_wp_anoms, serc_wnonex_wp_anoms))

solarpower_anoms = np.concatenate((mro_whot_sp_anoms, mro_wcold_sp_anoms, mro_wnonex_sp_anoms,
                                  npcc_whot_sp_anoms, npcc_wcold_sp_anoms, npcc_wnonex_sp_anoms,
                                  rfc_whot_sp_anoms, rfc_wcold_sp_anoms, rfc_wnonex_sp_anoms, 
                                  serc_whot_sp_anoms, serc_wcold_sp_anoms, serc_wnonex_sp_anoms,
                                  tre_whot_sp_anoms, tre_wcold_sp_anoms, serc_wnonex_sp_anoms, 
                                  wecc_whot_sp_anoms, wecc_wcold_sp_anoms, serc_wnonex_sp_anoms))

windpotential = np.concatenate((mro_whot_wp, mro_wcold_wp, mro_wnonex_wp,
                                  npcc_whot_wp, npcc_wcold_wp, npcc_wnonex_wp,
                                  rfc_whot_wp, rfc_wcold_wp, rfc_wnonex_wp, 
                                  serc_whot_wp, serc_wcold_wp, serc_wnonex_wp,
                                  tre_whot_wp, tre_wcold_wp, serc_wnonex_wp, 
                                  wecc_whot_wp, wecc_wcold_wp, serc_wnonex_wp))

solarpotential = np.concatenate((mro_whot_sp, mro_wcold_sp, mro_wnonex_sp,
                                  npcc_whot_sp, npcc_wcold_sp, npcc_wnonex_sp,
                                  rfc_whot_sp, rfc_wcold_sp, rfc_wnonex_sp, 
                                  serc_whot_sp, serc_wcold_sp, serc_wnonex_sp,
                                  tre_whot_sp, tre_wcold_sp, serc_wnonex_sp, 
                                  wecc_whot_sp, wecc_wcold_sp, serc_wnonex_sp))


## Create the plotting dataframe
plotting_df = pd.DataFrame({'NERC Region' : nerc_regions, 'Event Type' : event_type,
                            'Intensity' : intensity, 'Extent' : extent, 'Event Area': event_area,
                            'CI': CI,'Energy Proxy' : energy_proxy, 'Average Max Windspeed' : windspeed_array,
                            'Mean Cumulative SSRD' : ssrd_array, 'Wind Potential Intensity' : windpower_anoms,
                            'Solar Potential Intensity' : solarpower_anoms, 'Wind Potential' : windpotential,
                            'Solar Potential' : solarpotential, 'Year' : year, 'Date':dates, 'Total Area': area})

## Save the plotting dataframe to a CSV
plotting_df.to_csv(writepath)

