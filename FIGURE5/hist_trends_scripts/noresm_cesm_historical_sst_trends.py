#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ada Gjermundsen
year: 2019 - 2021
This script is used to calculate the sea surface temperature trends in CESM2 and NorESM2 (LM and MM) historical CMIP experiments
calculated as the ensemble mean over three members (r1i1p1f1, r2i1p1f1, r3i1p1f1) and over the years 1960 - 2014
The result is used in FIGURE 5
"""

import sys
sys.path.insert(1, '/scratch/adagj/CMIP6/CLIMSENS/CMIP6_UTILS')
import CMIP6_ATMOS_UTILS as atmos
import CMIP6_SEAICE_UTILS as ocean
from read_modeldata_cmip6 import ecs_models_cmip6, make_filelist_cmip6, Modelinfo
import numpy as np
import xesmf as xe
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)
from scipy.stats import linregress

def make_bounds(modelname, ds):
    '''
    Parameters
    ----------
    modelname: str, name of model. NorESM2: The ocean/sea-ice grid of NorESM2 is a tripolar grid with 360 and 384 unique grid cells in i- and j-direction, respectively. Due to the way variables are staggered in the ocean model, an additional j-row is required explaining the 385 grid cells in the j-direction for the ocean grid. The row with j=385 is a duplicate of the row with j=384, but with reverse i-index.
    ds : xarray.DataSet, with model grid information for the data which need to be regridded
                         on (i,j,vertices) format where the 4 vertices give grid corner information

    Returns
    -------
    ds_in :  xarray.DataSet, with 2D model grid information for the data which need to be regridded
    '''
    print('In make bounds')
    ny,nx = ds.lat.shape
    print('ny = %i'%ny)
    print('nx = %i'%nx)
    if modelname in ['NorESM2-LM', 'NorESM2-MM'] and 'siconc' in list(ds.keys()):
        #drop the last row with j=385 of area when dealing with the sea ice variables.
        ny = ny -1
    if 'lat' in ds.lon.coords:
        lat_model = ds.lat.isel(j=slice(0,ny)).rename({'i':'x','j':'y'}).drop('lon').drop('lat')
        lon_model = ds.lon.isel(j=slice(0,ny)).rename({'i':'x','j':'y'}).drop('lon').drop('lat')
    else:
        lat_model = ds.lat.isel(j=slice(0,ny)).rename({'i':'x','j':'y'})
        lon_model = ds.lon.isel(j=slice(0,ny)).rename({'i':'x','j':'y'})
    print('This goes well')
    lon_b_model = xr.concat([ds.vertices_longitude.isel(vertices=0), ds.vertices_longitude.isel(vertices=1,i=-1)],dim='i')
    lon_b_model = xr.concat([lon_b_model,xr.concat([ds.vertices_longitude.isel(vertices=3,j=-1), ds.vertices_longitude.isel(vertices=2,j=-1,i=-1)],dim='i')],dim='j')
    lat_b_model = xr.concat([ds.vertices_latitude.isel(vertices=0), ds.vertices_latitude.isel(vertices=1,i=-1)],dim='i')
    lat_b_model = xr.concat([lat_b_model,xr.concat([ds.vertices_latitude.isel(vertices=3,j=-1), ds.vertices_latitude.isel(vertices=2,j=-1,i=-1)],dim='i')],dim='j')
    print('This also goes well')
    if 'lat' in lon_b_model.coords:
        lon_b_model = lon_b_model.isel(j=slice(0,ny+1)).rename('lon_b').rename({'j':'y_b','i':'x_b'}).drop('lon').drop('lat')
        lat_b_model = lat_b_model.isel(j=slice(0,ny+1)).rename('lat_b').rename({'j':'y_b','i':'x_b'}).drop('lon').drop('lat')
    else:
        lon_b_model = lon_b_model.isel(j=slice(0,ny+1)).rename('lon_b').rename({'j':'y_b','i':'x_b'})
        lat_b_model = lat_b_model.isel(j=slice(0,ny+1)).rename('lat_b').rename({'j':'y_b','i':'x_b'})
    lat_b_model = lat_b_model.where(lat_b_model<90.0,90.0)
    lat_b_model = lat_b_model.where(lat_b_model>-90.0,-90.0)
    return xr.merge([lon_model,lat_model,lon_b_model,lat_b_model])

def make_regridder(modelname, ds, outgrid, grid_weight_path, regrid_mode = 'conservative', reuse_weights=False, periodic = False):
    ''' The first step of the regridding routine!
    There is an important reason why the regridding is broken into two steps
    (making the regridder and perform regridding). For high-resolution grids, 
    making the regridder (i.e. “computing regridding weights”, explained later) 
    is quite computationally expensive, 
    but performing regridding on data (“applying regridding weights”) is still pretty fast.
    
    Parameters
    ----------
    modelname: str, name of model. NorESM2 has one less gridpoint for cice compared to blom. 
                                  CanESM5 has flipped corner 0 and 2 compared to all the other models
    ds : xarray.DataSet, with model grid information for the data which need to be regridded 
    outgrid : xarray.DataSet, with output grid information which the data will be regridded to
    regrid_mode : str,  ['bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s']
    grid_weight_path : str, path to where the  regridder weight file will be stored
    reuse_weights :  bool, set to True to read existing weights from disk.
                           set to False if new weights need to be calculated
    Returns
    -------
    regridder : xarray.DataSet with regridder weight file information
    '''
    ds_in = make_bounds(modelname, ds)
    regridder = xe.Regridder(ds_in, outgrid, regrid_mode,
                             filename=grid_weight_path+'model_to_'+regrid_mode+'.nc',
                             reuse_weights=reuse_weights, ignore_degenerate=True)
    return regridder

def regrid_file(ds, var, regridder, outgrid, areao=None, area=None):
    '''Second step of the regridding routine!
    
    Parameters
    ----------
    ds : xarray.DataSet, with model grid information for the data which need to be regridded 
    var : srt, name of varible
    regridder: xarray.Dataset with regridding weights to be used for the regridding
                
    Returns
    -------
    dr : xarray.DataArray with regridded variable data and lon from 0-360
    '''
    print('In regrid file')
    da = ds[var]
    da = da.where(da!=0)
    if areao is not None:
        tmp = da.isel(year=0).squeeze()
        glb_mean =  (tmp*areao).sum(dim=('i','j'))/(areao*tmp.notnull()).sum(dim=('i','j'))
        print('Global mean value BEFORE regridding:%f '%np.round(glb_mean.values,4))
    dr = regridder(da) # need DataArray
    print('Regridding completed\n')
    dr = dr.where(dr!=0)
    dr = dr.rename({'x': 'lon', 'y':'lat'})
    dr = dr.to_dataset(name = var)
    dr[var].attrs['long_name']='Sea Water Potential Density'
    dr[var].attrs['units']= 'kg/m3'
    dr[var].attrs['standard_name']='sea_water_potential_density'
    dr[var].attrs['description']='Regridded density to lat, lon grid'
    if area is not None:
        tmp = dr[var].isel(year=0).squeeze()
        glb_mean_dr = (tmp*area).sum(dim=('lat','lon'))/(area*tmp.notnull()).sum(dim=('lat','lon'))
        print('Global mean value AFTER regridding:%f'%np.round(glb_mean_dr.values,4))
    return dr


def make_outgrid(ds):
    # make an outgrid used for regridding to be identical to the atmospheric grid used by the model 
    lat_b = ds.lat_bnds.values[:,0]
    lat_b = np.append(lat_b,  ds.lat_bnds.values[-1,-1])
    lat_b[0]=-90
    lat_b[-1]=90
    lon_b = ds.lon_bnds.values[:,0]
    lon_b = np.append(lon_b, ds.lon_bnds.values[-1,-1])
    ds_out = xr.Dataset({ 'lat': (['y'],ds.lat.values),
                          'lon': (['x'],ds.lon.values),
                          'lat_b': (['y_b'], lat_b),
                          'lon_b': (['x_b'], lon_b)
                           })
    return ds_out

def regrid_latlon(modelname, dsm, dam, var):
    # regrid files
    # use the NorESM2-MM atmospheric grid since it is 1x1, so closer to the grid resolution used in BLOM
    area = xr.open_dataset('path_to_grid_data/areacella_fx_NorESM2-MM_piControl_r1i1p1f1_gn.nc')
    outgrid = make_outgrid(area)
    grid_weight_path = 'path_to_dumpfiles/'
    # the areacello grid is identical for NorESM2-LM and NorESM2-MM 
    areao = xr.open_dataset('path_to_ocean_grid_data/areacello_Ofx_NorESM2-LM_piControl_r1i1p1f1_gn.nc')
    areao = areao.rename({'latitude':'lat', 'longitude':'lon'})
    regridder = make_regridder(modelname, dsm, outgrid, grid_weight_path, regrid_mode = 'nearest_s2d', reuse_weights=False, periodic = False)
    ds = regrid_file(dam.to_dataset(name=var), var, regridder, outgrid, areao=areao.areacello, area = area.areacella)
    da = ds[var]
    return da

def linearfit(ds):
    t = linregress(np.arange(0,len(ds)),ds)
    return t

def make_linear_trends_sst(ds):
     print('In trend function')
     trends = np.ones_like(ds.isel(year=0).drop('year'))
     tempdata = ds.values
     for yi in range(0, len(ds.lat.values)):
          for xi in range(0, len(ds.lon.values)):
              trends[yi,xi]=(10*linearfit(tempdata[:,yi,xi]).slope)
     print('Trend calculations completed')
     dsout = xr.DataArray(trends,dims=('lat','lon'), coords = {'lat':ds.lat, 'lon':ds.lon}, name='trend')
     dsout.attrs['long_name']= 'SST trend 1960 - 2014'
     dsout.attrs['standard_name']= 'SST trend'
     dsout.attrs['units']= 'degree C pr decade'
     return dsout

def make_hist_sst_trends(models, var, outpath, areaavg=False):
    print('HISTORICAL SST TREND CALCULATIONS: \n')
    for modelname,expinfo in models.items():
        print(modelname)
        dsout = None
        n=1
        for member in ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1']:
            modelctrl = Modelinfo(name = modelname, institute = expinfo['institute'], expid = 'historical', realm = 'Omon',
                      realiz=member, grid_atmos = expinfo['grid_label_atmos'][0], grid_ocean = expinfo['grid_label_ocean'], branchtime_year=expinfo['branch_yr'])
            if modelname in ['NorESM2-LM', 'NorESM2-MM']:
                make_filelist_cmip6(modelctrl, var,  component = 'ocean', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
            else:
                make_filelist_cmip6(modelctrl, var,  component = 'ocean')
            if modelctrl.filenames:
                if len(modelctrl.filenames)>1:
                    tas =  xr.open_mfdataset(modelctrl.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
                else:
                    tas =  xr.open_dataset(modelctrl.filenames[0], chunks={"time":12})
                print('%s loaded for model: %s, experiment: %s, ensamble member: %s. Lenght of simulation: %.1f years'%(var,modelctrl.name, modelctrl.expid, member, len(tas[var].time.values)/12))
                tas = ocean.consistent_naming(tas)
                tasctrl = atmos.fix_time(tas, 1850)
                tasctrl = atmos.yearly_avg(tasctrl[var])
                tasctrl = tasctrl.sel(year = slice(1960,2014))
                if modelname in ['NorESM2-LM', 'NorESM2-MM']:
                    tasctrl = regrid_latlon(modelname, tas, tasctrl, var=var) 
                if isinstance(dsout,xr.Dataset):
                    dsout = xr.concat([dsout, tasctrl.to_dataset(name=var)],dim='member')
                else: 
                    dsout = tasctrl.to_dataset(name=var)
                n = n + 1
                print(dsout)
            else:
                print('%s not loaded for model %s, experiment: piControl. Skipping model! Please check!'%(var,modelname))
                continue
            del modelctrl,tasctrl
        dsout = dsout[var].mean(dim='member')
        dsout = make_linear_trends_sst(dsout.squeeze()) 
        dsout = dsout.to_dataset(name=var+'_trend').to_netcdf(outpath + var +'_' + modelname + '_historical_1960_2014_ssttrend.nc', compute=False)
        with ProgressBar():
            result = dsout.compute()
        del dsout

    
if __name__ == '__main__':
    outpath = 'path_to_outdata/'
    models = ecs_models_cmip6()
    models = {'NorESM2-LM':models['NorESM2-LM'], 'NorESM2-MM':models['NorESM2-MM'], 'CESM2': models['CESM2']}
    for var in [ 'tos']:
        make_hist_sst_trends(models, var, outpath, True)
    
