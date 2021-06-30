#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 08:30:24 2020

@author: adag
"""
import xarray as xr
import numpy as np
from cftime import DatetimeNoLeap
from itertools import product
import warnings
warnings.simplefilter('ignore')


def consistent_naming(ds):
    '''The naming of the cmorized CMIP6 data is unfortunalely not consistent.
    This function is a decorator examining the names of the latitude and longitude dimensions 
    and renames them.  Checks here ensure that the names supplied via the xarray object dims 
    are changed to be synonymous with subset algorithm dimensions.  
    
    Parameters
    ----------
    ds : xarray.DataSet 

    Returns
    -------
    ds : xarray.DataSet with renamed coords, dims to keep consistency
    '''
    if 'latitude' in ds.coords and 'lat' not in ds.coords:
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude_bnds' in ds.variables:
        ds = ds.rename({'latitude_bnds':'lat_bnds','longitude_bnds':'lon_bnds'})
    if 'height' in ds.coords:
        ds = ds.drop('height')
    if 'nbnd' in ds.dims:
        ds = ds.rename({'nbnd':'bnds'})
    if 'nbounds' in ds.dims:
        ds = ds.rename({'nbounds':'bnds'})
    if 'lat_bnds' not in ds.coords and 'lat_bnds' not in ds.variables:
        ds = make_bounds(ds)
    return ds

def make_bounds(ds):
    '''
    Create latitude and longitude boundaries 
    Can be missing in cmorized files for some models (e.g. in IPSL-CM6A-LR)
    Parameters
    ----------
    ds : xarray.DataSet with latitude and longitude values, but missing boundaies

    Returns
    -------
    ds_out : xarray.DataSet with boundary values for latitude and longitude'''
    if  ds.lat.diff(dim='lat').values[0]>0:
        lat_b = np.concatenate((np.array([-90], float), 0.5*ds.lat.diff(dim='lat').values + ds.lat.values[:-1], np.array([90], float)))
        lat_b = np.reshape(np.concatenate([lat_b[:-1],lat_b[1:]]),[2,len(ds.lat.values)]).T
    else:
        lat_b = np.concatenate((np.array([90], float), 0.5*ds.lat.diff(dim='lat').values + ds.lat.values[:-1], np.array([-90], float)))
        lat_b = np.reshape(np.concatenate([lat_b[:-1],lat_b[1:]]),[2,len(ds.lat.values)]).T
    lon_b = np.concatenate((np.array([ds.lon[0].values - 0.5*ds.lon.diff(dim='lon').values[0]]),
                            0.5*ds.lon.diff(dim='lon').values + ds.lon.values[:-1],
                            np.array([ds.lon[-1].values + 0.5*ds.lon.diff(dim='lon').values[-1]])))
    #if lon_b[-1]>360:
    #   lon_b[-1] = 360 - lon_b[-1]
    lon_b = np.reshape(np.concatenate([lon_b[:-1],lon_b[1:]]),[2,len(ds.lon.values)]).T
    lon_b = xr.DataArray(lon_b, dims=('lon','bnds'),coords={'lon':ds.lon}).to_dataset(name='lon_bnds')
    lat_b = xr.DataArray(lat_b, dims=('lat','bnds'),coords={'lat':ds.lat}).to_dataset(name='lat_bnds')
    lon_b.attrs['long_name']='Longitude'
    lon_b.attrs['units']='degrees_east'
    lon_b.attrs['axis']='X'
    lon_b.attrs['bounds']='lon_bnds'
    lon_b.attrs['standard_name']='longitude'
    lat_b.attrs['long_name']='Latitude'
    lat_b.attrs['units']='degrees_north'
    lat_b.attrs['axis']='Y'
    lat_b.attrs['bounds']='lat_bnds'
    lat_b.attrs['standard_name']='latitude'
    ds_out = xr.merge([ds, lat_b,lon_b])
    return ds_out
    

def global_mean(ds):
    '''Calculates globally averaged values

    Parameters
    ----------
    ds : xarray.DaraArray i.e. ds[var]

    Returns
    -------
    ds_out :  xarray.DaraArray with globally averaged values
    '''
    # to include functionality for subsets or regional averages:
    if 'time' in ds.dims:
        weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat))*ds.notnull().mean(dim=('lon','time'))
    else:
        weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat))*ds.notnull().mean(dim=('lon'))
    ds_out = (ds.mean(dim='lon')*weights).sum(dim='lat')/weights.sum()
    if 'long_name'  in ds.attrs:
        ds_out.attrs['long_name']= 'Globally averaged ' + ds.long_name
    if 'units'  in ds.attrs:
        ds_out.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_out.attrs['standard_name']=ds.standard_name
    return ds_out

def yearly_avg(ds):
    ''' Calulates timeseries over yearly averages from timeseries of monthly means
    The weighted average considers that each month has a different number of days.

    Parameters
    ----------
    ds : xarray.DaraArray i.e. ds[var]

    Returns
    -------
    ds_weighted : xarray.DaraArray with yearly averaged values
    '''
    month_length = ds.time.dt.days_in_month
    # Calculate the weights by grouping by 'time.season'.
    #if ds.notnull().any():
    #    month_length = month_length.where(ds.notnull())
    #    weights = month_length.groupby('time.year') / month_length.groupby('time.year').sum()
    #else:
    weights = month_length.groupby('time.year') / month_length.groupby('time.year').sum()
    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.year').sum().values,
                               np.ones(len(np.unique(ds.time.dt.year))))
    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby('time.year').sum(dim='time')
    if 'long_name'  in ds.attrs:
        ds_weighted.attrs['long_name']= 'Annually averaged ' + ds.long_name
    if 'units' in ds.attrs:
        ds_weighted.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_weighted.attrs['standard_name']=ds.standard_name
    return ds_weighted

def single_monthly_mean(ds, monthnr):
    ''' Calulates timeseries for one particular month - don't need weighting

    Parameters
    ----------
    ds : xarray.DaraArray i.e. ds[var]
    monthnr: int , nr of month 0 = January, 1 = February etc.
    Returns
    -------
    ds_weighted : xarray.DaraArray with yearly averaged values
    '''
    ds = ds.sel(time = ds.time.dt.month == monthnr)
    return ds

def seasonal_avg_timeseries(ds, var=''):
    '''Calulates timeseries over seasonal averages from timeseries of monthly means
    The weighted average considers that each month has a different number of days.
    Using 'QS-DEC' frequency will split the data into consecutive three-month periods, 
    anchored at December 1st. 
    I.e. the first value will contain only the avg value over January and February 
    and the last value only the December monthly averaged value
    
    Parameters
    ----------
    ds : xarray.DaraArray i.e.  ds[var]
        
    Returns
    -------
    ds_out: xarray.DataSet with 4 timeseries (one for each season DJF, MAM, JJA, SON)
            note that if you want to include the output in an other dataset, e.g. dr,
            you should use xr.merge(), e.g.
            dr = xr.merge([dr, seasonal_avg_timeseries(dr[var], var)])
    '''    
    month_length = ds.time.dt.days_in_month
    #if ds.notnull().any():
    #    sesavg = ((ds * month_length).resample(time='QS-DEC').sum() / 
    #          month_length.resample(time='QS-DEC').sum())
    #else:
    sesavg = ((ds * month_length).resample(time='QS-DEC').sum() /
              month_length.where(ds.notnull()).resample(time='QS-DEC').sum())
    djf = sesavg[0::4].to_dataset(name = var + 'DJF').rename({'time':'timeDJF'})
    mam = sesavg[1::4].to_dataset(name = var +'MAM').rename({'time':'timeMAM'})
    jja = sesavg[2::4].to_dataset(name = var +'JJA').rename({'time':'timeJJA'})
    son = sesavg[3::4].to_dataset(name = var +'SON').rename({'time':'timeSON'})
    ds_out = xr.merge([djf, mam, jja, son])
    ds_out.attrs['long_name']= 'Seasonally averaged ' + ds.long_name
    ds_out.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_out.attrs['standard_name']=ds.standard_name
    return ds_out
    
def seasonal_avg(ds):
    '''Calculates seasonal averages from timeseries of monthly means
    The time dimension is reduced to 4 seasons: 
        * season   (season) object 'DJF' 'JJA' 'MAM' 'SON'
    The weighted average considers that each month has a different number of days.
    
    Parameters
    ----------
    ds : xarray.DaraArray i.e.  ds[var]
        
    Returns
    -------
    ds_weighted : xarray.DaraArray 
    '''
    month_length = ds.time.dt.days_in_month
    # Calculate the weights by grouping by 'time.season'.
    #if ds.notnull().any():
    #    month_length = month_length.where(ds.notnull())
    #    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()
    #else:
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()
    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))
    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby('time.season').sum(dim='time')
    ds_weighted.attrs['long_name']= 'Seasonally averaged ' + ds.long_name
    ds_weighted.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_weighted.attrs['standard_name']=ds.standard_name
    return ds_weighted


def fix_time(ds, yr0=1850):
    """
    If there are problems with the calender used in the cmorized files (e.g. GFDL-ESM4)
    This function will overwrite the time array such that (all) other functions can be used
    
    """
    yr = np.int(ds.time.shape[0]/12) 
    yr1 = yr + yr0
    dates = [DatetimeNoLeap(year, month, 16) for year, month in 
             product(range(yr0, yr1), range(1, 13)) ]
    bounds_a = [DatetimeNoLeap(year, month, 1) for year, month in 
             product(range(yr0, yr1), range(1, 13)) ]
    bounds_b = bounds_a[1:]
    bounds_b.append(DatetimeNoLeap(yr1, 1, 1))
    bounds = np.reshape(np.concatenate([bounds_a, bounds_b]),[ds.time.shape[0], 2])
    ds = ds.assign_coords(time = dates)
    # set attributes
    ds['time'].attrs['bounds'] = 'time_bnds'
    ds['time'].attrs['axis'] = 'T'
    ds['time'].attrs['long_name']= 'time'
    ds['time'].attrs['standard_name']='time'
    #ds['time'].attrs['cell_methods'] = 'time: mean'
    ds['time'].attrs['calendar']='noleap'
    ds['time'].attrs['units']='days since %04d-01-16 00:00'%yr0
    ds['time_bnds'] = xr.DataArray(bounds, dims=('time','bnds'))
    ds['time_bnds'].attrs['axis'] = 'T'
    ds['time_bnds'].attrs['long_name']= 'time bounds'
    ds['time_bnds'].attrs['standard_name']='time_bnds'
    return ds

def fix_time_yr(ds, month = 6, yr0=1850):
    """
    If there are problems with the calender used in the cmorized files (e.g. GFDL-ESM4)
    This function will overwrite the time array such that (all) other functions can be used
    
    """
    yr = np.int(ds.time.shape[0])
    yr1 = yr + yr0
    dates = [DatetimeNoLeap(year, month , 16) for year in range(yr0, yr1) ]
    bounds_a = [DatetimeNoLeap(year, month, 1) for year in  range(yr0, yr1)]
    bounds_b = bounds_a[1:]
    bounds_b.append(DatetimeNoLeap(yr1, 1, 1))
    bounds = np.reshape(np.concatenate([bounds_a, bounds_b]),[ds.time.shape[0], 2])
    ds = ds.assign_coords(time = dates)
    # set attributes
    ds['time'].attrs['calendar']='noleap'
    ds['time'].attrs['units']='days since %04d-%02d-16 00:00'%(yr0,month)
    ds['time'].attrs['bounds']='time_bnds'
    ds['time'].attrs['axis']='T'
    ds['time'].attrs['long_name']='time'
    ds['time'].attrs['standard_name']='time'
    ds['time_bnds'] = xr.DataArray(bounds, dims=('time','bnds'))
    ds['time_bnds'].attrs['axis'] = 'T'
    ds['time_bnds'].attrs['long_name']= 'time bounds'
    ds['time_bnds'].attrs['standard_name']='time_bnds'
    return ds



def mask_region(ds, lat_low=-90, lat_high=90, lon_low=0, lon_high=360):
    '''Subtract data from a confined region
    Note, for the atmosphere the longitude values go from 0 -> 360.
    Also after regridding
    This is not the case for ice and ocean variables for which some models
    use -180 -> 180
    
    Parameters
    ----------
    ds : xarray.DataArray or xarray.DataSet
    lat_low : int or float, lower latitude boudary. The default is -90.
    lat_high : int or float, lower latitude boudary. The default is 90.
    lon_low :  int or float, East boudary. The default is 0.
    lon_high : int or float, West boudary. The default is 360.
    
    Returns
    -------
    ds_out : xarray.DataArray or xarray.DataSet with data only for the selected region
    
    Then it is still possible to use other functions e.g. global_mean(ds) to 
    get an averaged value for the confined region 
    '''
    ds_out = ds.where(ds.lat>=lat_low).where(ds.lat<=lat_high)
    if lon_high>lon_low:
        ds_out = ds_out.where(ds_out.lon>=lon_low).where(ds_out.lon<=lon_high)
    else:
        boole = (ds_out.lon.values <= lon_high) | (ds_out.lon >= lon_low)
        ds_out = ds_out.sel(lon=ds_out.lon.values[boole])
    ds_out.attrs['long_name']= 'Regional subset of ' + ds.long_name
    ds_out.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_out.attrs['standard_name']=ds.standard_name
    return ds_out

def define_area(ds):
    '''Estimates the area of each grid cell 
    Parameters
    ----------
    ds : xarray.DataSet with grid information (lat, lat_bnds, lon, lon_bnds)
        
    Returns
    -------
    ds_out : xarray.DataArray with grid area in m

    '''
    if 'bnds' in ds.dims:
        lat_b = xr.concat([ds.lat_bnds.isel(bnds=0),ds.lat_bnds.isel(lat=-1).isel(bnds=1)],dim='lat')
        dlat = lat_b.diff(dim = 'lat').assign_coords(lat = ds.lat)
        lon_b = xr.concat([ds.lon_bnds.isel(bnds=0),ds.lon_bnds.isel(lon=-1).isel(bnds=1)],dim='lon')
        dlon = lon_b.diff(dim = 'lon').assign_coords(lon = ds.lon)
        dx, dy = dll_dist(dlon, dlat, ds.lon, ds.lat)
    elif 'x_b' in ds.dims:
        lon = ds.lon.isel(y=0).rename({'x':'lon'}).drop('lat')
        lat = ds.lat.isel(x=0).rename({'y':'lat'}).drop('lon')
        dlon = ds.lon_b.isel(y_b=0).diff(dim = 'x_b').drop('lon_b').drop('lat_b').rename({'x_b':'lon'}).assign_coords(lon = lon)
        dlat = ds.lat_b.isel(x_b=0).diff(dim = 'y_b').drop('lon_b').drop('lat_b').rename({'y_b':'lat'}).assign_coords(lat = lat)
        dx, dy = dll_dist(dlon, dlat,lon, lat)
    else:
        raise Exception('Bounds for lat and lon missing in file. Will fix')
    
    ds_out = (dy*dx).T
    ds_out.attrs['long_name']='Grid-Cell Area for Atmospheric Grid Variables'
    ds_out.attrs['name']='area'
    ds_out.attrs['units']='m2'
    return ds_out

