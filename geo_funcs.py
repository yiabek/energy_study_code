#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yiannabekris
"""

import numpy as np
from osgeo import gdal, ogr
from netCDF4 import Dataset
import pandas as pd

def mask_3d(array,newshape):
    array_3d = np.repeat(array[:,:,np.newaxis], newshape, axis=2) 
    return array_3d

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

### flatten 3d array and remove NaNs
def flat_nonan(array):
    array_flat = array.flatten()
    array_flat_nonan = array_flat[~np.isnan(array_flat)]
    
    return array_flat_nonan
    

### conus_mask
### Mask out water (if needed) and subset by region
def conus_subset(lat,lon):
    ### Fix longitudes so values range from -180 to 180 
    if lon[lon>180].size>0:
        lon[lon>180]=lon[lon>180]-360

    north_lat_bnd = 50
    south_lat_bnd = 20
    east_lon_bnd = -65
    west_lon_bnd = -125

    # Find indices and then mask array by indices    
    north_lat_idx = geo_idx(north_lat_bnd,lat)
    south_lat_idx = geo_idx(south_lat_bnd,lat)
    east_lon_idx = geo_idx(east_lon_bnd,lon)
    west_lon_idx = geo_idx(west_lon_bnd,lon)

    return north_lat_idx,south_lat_idx,east_lon_idx,west_lon_idx


### ----------------------------------------------------------------------
### This function creates a numpy array of 1s and 0s from a shapefile
### that can then be used to mask another numpy array
### shapefile is the shapefile
### lat is a numpy array of latitude
### lon is a numpy array of longitude
### If a multidimensional array is desired, repeat the dimension
### on the new axis
### Modified from: 
### https://gis.stackexchange.com/questions/16837/turning-shapefile-into-mask-and-calculating-mean
### Which is modified from the gdal documentation. 
def shp_to_mask(shapefile, lat, lon):
    
    ## Define extent based on lat and lon
    xmin,ymin,xmax,ymax=[min(lon),min(lat),max(lon),max(lat)] 
    
    ## Cols and rows (lons and lats)
    ncols,nrows=lon.shape[0],lat.shape[0] 
    maskvalue = 1

    ## Find resolution
    xres=(xmax-xmin)/float(ncols)
    yres=(ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)

    ## Open the shapefile
    src_ds = ogr.Open(shapefile)
    src_lyr=src_ds.GetLayer()

    dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal.GDT_Byte)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(0) #initialise raster with zeros
    dst_rb.SetNoDataValue(0)
    dst_ds.SetGeoTransform(geotransform)

    err = gdal.RasterizeLayer(dst_ds, [1], src_lyr, burn_values=[maskvalue])
    
    dst_ds.FlushCache()

    ## Get raster and transpose
    mask_arr=dst_ds.GetRasterBand(1).ReadAsArray()
    mask_arr=np.transpose(mask_arr,(1,0))
    
    return mask_arr
    

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

    








