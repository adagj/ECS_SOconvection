#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ada Gjermundsen
year: 2019 - 2021
This script is used to calculate the observed (HadISST) sea surface temperature trends 
over the years 1960 - 2014
The result is used in FIGURE 5
"""
import sys
sys.path.insert(1, '/scratch/adagj/CMIP6/CLIMSENS/CMIP6_UTILS')
import CMIP6_ATMOS_UTILS as atmos
import numpy as np
from cftime import DatetimeNoLeap
from itertools import product
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)
from scipy.stats import linregress


def linearfit(ds):
    t = linregress(np.arange(0,len(ds)),ds)
    return t

def make_linear_trends_sst(ds):
     print('In SST trend function')
     if 'year' in ds.dims:
         ds = ds.rename({'year':'time'})
     dims = ds.dims
     trends = np.ones_like(ds.isel(time=0).drop('time'))
     tempdata = ds.values
     for yi in range(0, len(ds.lat.values)):
          print(yi)
          for xi in range(0, len(ds.lon.values)):
              trends[yi,xi]=(10*linearfit(tempdata[:,yi,xi]).slope)
     print('Trend calculations completed')
     dsout = xr.DataArray(trends,dims=('lat','lon'), coords = {'lat':ds.lat, 'lon':ds.lon}, name='trend')    
     dsout.attrs['long_name']= 'SST trend 1960 - 2014'
     dsout.attrs['standard_name']= 'SST trend'
     dsout.attrs['units']= 'degree C pr decade'
     return dsout

def fix_time(ds, yr0=1850):
    """
    If there are problems with the calender used in the cmorized files (e.g. GFDL-ESM4)
    This function will overwrite the time array such that (all) other functions can be used
    
    """
    yr = np.int(ds.time.shape[0]/12)
    yr1 = yr + yr0
    dates = [DatetimeNoLeap(year, month, 16) for year, month in
             product(range(yr0, yr1), range(1, 13)) ]
    ds = ds.assign_coords(time = dates)
    # set attributes
    ds['time'].attrs['bounds'] = 'time_bnds'
    ds['time'].attrs['axis'] = 'T'
    ds['time'].attrs['long_name']= 'time'
    ds['time'].attrs['standard_name']='time'
    return ds


def make_avg_sst_hist(var, inpath, filenames, startyr, outpath, outfile, areaavg=False):
        ds = xr.open_mfdataset(inpath + filenames , combine='nested',concat_dim='time', chunks={"time": 12})
        ds = atmos.consistent_naming(ds)
        ds = fix_time(ds, startyr)
        if 'HadISST' in inpath.split('/') or 'HadSST3' in inpath.split('/'):
            ds = ds[var]
            ds = ds.where(ds>-10) # set some large negative non-physical values to nan
            ds = atmos.yearly_avg(ds)
        else:
            ds = atmos.yearly_avg(ds[var])
        ds = ds.sel(year = slice(1960,2014))
        ds = make_linear_trends_sst(ds.squeeze())
        ds.to_dataset(name=var+'_trend').to_netcdf(outpath + var +'_' + outfile, compute=False)
        with ProgressBar():
            result = ds.compute()
  
    
if __name__ == '__main__':
    outpath = 'path_to_outdata/'
    make_avg_sst_hist(var='sst', inpath= 'path_to_obsdata/HadISST/', filenames = 'HadISST_sst_187001_201912.nc', startyr = 1870,    
                      outpath = outpath, outfile='HadISST_historical_1960_2014_trend.nc', areaavg=False)

