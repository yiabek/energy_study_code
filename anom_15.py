#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" anom_15.py
This script outputs the following metrics for extreme hot and cold:
    1) data subset by the years specified (leap years removed)
    2) anomalies calculated for each day of the year using a 15-day climatology from 1991-2020
    3) extreme hot or cold locations in space and time, defined as one standard deviation above or below the mean
    4) cooling or heating degree days
    5) an array of datetime data
    5) latitude and longitude subset to the CONUS
    
Credit to Dr. Arielle Catalano for developing the original anomaly calculation script
from which the aanomaly_15_calc function is based on.

@author: yiannabekris
"""
## Import packages
import numpy as np
import time
from netCDF4 import Dataset,num2date
import glob

## Function to find lat and lon indices
def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minimum value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minimum.
     From: 
         https://stackoverflow.com/questions/33789379/netcdf-and-python-finding-the-closest-lon-lat-index-given-actual-lon-lat-values
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   
   return geo_idx
    

### conus_subset
def conus_subset(lat,lon):
    """ 
    Subsets to latitude and longitude extracted from
    netCDF files
    """
    ## Delineate the bounding coordinates
    north_lat_bnd = 50
    south_lat_bnd = 20
    east_lon_bnd = -65
    west_lon_bnd = -125

    ## Find indices and then mask array by indices    
    north_lat_idx = geo_idx(north_lat_bnd,lat)
    south_lat_idx = geo_idx(south_lat_bnd,lat)
    east_lon_idx = geo_idx(east_lon_bnd,lon)
    west_lon_idx = geo_idx(west_lon_bnd,lon)

    return north_lat_idx,south_lat_idx,east_lon_idx,west_lon_idx


""" anomaly_15_calc

This function calculates anomalies for each day with the climatology
a period spanning 7 days before and 7 days after the specified day.
The reference period is for 1991-2020 so files containing that data must be
input in order for this function to work as intended.

inputs: 
files = netCDF files (must be a list, can be from glob)
lon_var = the name of the longitude variable (string)
lat_var = the name of the latitude variable (string)
field_var = the name of the data variable (string)
time_var = the name of the time variable (string)
yearbeg = the start year (integer)
yearend = the end year (integer

outputs (all NumPy arrays):
var_data = the data subset to the CONUS 
anom15 = the standardized anomalies
lat = NumPy array of latitude subset to CONUS
lon = NumPy array of longitude subset to CONUS
 
"""
def anomaly_15_calc(files,lon_var,lat_var,field_var,time_var,yearbeg,yearend):
    ### Open one netCDF file to extract shape
    var_netcdf=Dataset(files[0],"r")
    var_data = np.asarray(var_netcdf.variables[field_var][0],dtype="float")
    datearray=var_netcdf.variables[time_var][:]
    timeunits=var_netcdf.variables[time_var].units
    #varunits=var_netcdf.variables[field_var].units
    lat=np.asarray(var_netcdf[lat_var][:],dtype="float")
    lon=np.asarray(var_netcdf[lon_var][:],dtype="float")    
    var_netcdf.close()

    ### Initiating arrays for multi-file datasets
    var_data=np.empty([0,var_data.shape[0],var_data.shape[1]])
    datatime=np.empty(0, dtype='datetime64')

    ### Loop through files and extract variables
    for f in range(len(files)):
        file = files[f]
        data = Dataset(file)
        var = np.array(data[field_var])
        datearray=data.variables[time_var][:]
        timeunits=data.variables[time_var].units
        datetemp=np.array([num2date(t,units=timeunits) for t in datearray]) #daily data
        datatime=np.concatenate([datatime,datetemp])
        var_data = np.concatenate([var_data,var], axis=0)
        data.close()    

    # extract years for subsetting 
    yr=np.array([int('{0.year:04d}'.format(t)) for t in list(datatime)])
    leapstr=np.array([t.strftime('%m-%d') for t in list(datatime)])
    date=np.array([t.strftime('%Y-%m-%d')for t in list(datatime)])
    anom_leapstr = leapstr.copy()
       
    ### Fix longitude
    if lon[lon>180].size>0:
        lon[lon>180]=lon[lon>180]-360

    ### Reshape data to [lon, lat, time] dimensions for code to run properly
    if len(var_data.shape) == 4:
       var_data=np.squeeze(var_data)
    if var_data.shape == (len(lon),len(lat),len(datatime)):
       var_data=var_data
    elif var_data.shape == (len(lat),len(datatime),len(lon)):
       var_data=np.transpose(var_data,(2,0,1))
    elif var_data.shape == (len(lon),len(datatime),len(lat)):
       var_data=np.transpose(var_data,(0,2,1))
    elif var_data.shape == (len(datatime),len(lon),len(lat)):
       var_data=np.transpose(var_data,(1,2,0))
    elif var_data.shape == (len(lat),len(lon),len(datatime)):
       var_data=np.transpose(var_data,(1,0,2))
    elif var_data.shape == (len(datatime),len(lat),len(lon)):
       var_data=np.transpose(var_data,(2,1,0))
     
    ### Subset data to CONUS    
    north_lat_idx,south_lat_idx,east_lon_idx,west_lon_idx=conus_subset(lat,lon)
    lat = lat[north_lat_idx:south_lat_idx]
    lon = lon[west_lon_idx:east_lon_idx]
    
    var_data = var_data[west_lon_idx:east_lon_idx,north_lat_idx:south_lat_idx,:]
       
    ## Remove duplicates
    filler,drop_duplicate_dates = np.unique(datatime,return_index=True)
    datatime = datatime[drop_duplicate_dates]
    var_data = var_data[:,:,drop_duplicate_dates]
    anom_data = var_data.copy()

    ## Find year indices
    yearind = np.where(np.logical_and(yr>=yearbeg, yr<=yearend))[0]
    leapstr = leapstr[yearind]
    var_data=var_data[:,:,yearind]
    date=date[yearind]
    
    ## Subset to time range specified by "yearbeg,yearend" values 
    anom_yearind = np.where(np.logical_and(yr>=1991, yr<=2020))[0]
    anom_leapstr = anom_leapstr[anom_yearind]
    anom_data = anom_data[:,:,anom_yearind]
    

    ### Remove leap days if needed
    dateind=(leapstr != '02-29')
    anom_dateind=(anom_leapstr != '02-29')
    leapstr=leapstr[dateind]
    date=date[dateind]
    anom_leapstr=anom_leapstr[anom_dateind]
    var_data=var_data[:,:,dateind]
    anom_data=anom_data[:,:,anom_dateind]

    ### Convert Celsius to Kelvin
    #if varunits == 'K' or varunits == 'Kelvin':
    var_data=var_data-273.15
    anom_data=anom_data-273.15
        

    ### Find unique days for calculating anomalies   
    days_uniq = np.unique(anom_leapstr)

    ### Find unique days
    dayinds = [np.where(anom_leapstr == dd)[0] for dd in days_uniq]
    beginds = [x-7 for x in dayinds]
    endinds = [x+8 for x in dayinds]
    
    ### Initiate array for days at their index positions
    days_list = []
    for ind in np.arange(0, len(dayinds)): 
        begind = [x for x in beginds[ind]]
        endind = [x for x in endinds[ind]]            
        days = [np.arange(begind[x],endind[x]) for x in np.arange(0,len(begind))]
        
        ### Replace indices at end of dataset
        if ind >= len(dayinds)-6:
            replacevalues = len(dayinds)-ind-8 
            repinds = days[-1][replacevalues:-1]
            newvals = range(0,len(repinds)+1)
            np.place(days[-1],days[-1]>=anom_data.shape[2],newvals)
        
        ### Replace indices at end of dataset if needed    
        else:
            np.place(days[-1],days[-1]>=anom_data.shape[2],0)
        
        ### Append to list of indices
        days_list.append(days)


    ### Initiate empty array for calculating anomalies    
    ### Find unique days for calculating anomalies   
    days_uniq = np.unique(leapstr)
    anom_15 = np.empty(var_data.shape)
    dayinds = [np.where(leapstr == dd)[0] for dd in days_uniq]
    for ind in np.arange(0, len(dayinds)):
        clim_15 = np.mean(anom_data[:,:,days_list[ind]], axis=(2,3))
        clim_15 = clim_15.reshape(clim_15.shape[0], clim_15.shape[1], 1)
        std_15 = np.std(anom_data[:,:,days_list[ind]], axis=(2,3))
        std_15 = std_15.reshape(std_15.shape[0], std_15.shape[1], 1)
        anom_15[:, :, dayinds[ind]] = (var_data[:, :, dayinds[ind]] - clim_15)/std_15
            
    print("...Computed!")
    
    return var_data, anom_15, lat, lon
