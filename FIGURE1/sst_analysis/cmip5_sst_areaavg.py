#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ADA GJERMUNDSEN
YEARS: 2019 - 2021
CMIP5 model analysis
This script is used to calculate the area averaged sea surface temperature 
Globally and south of 35S
The averaging period is the last 30 years of the 150 years long simulations
For piControl, the time period is selected to the corresponding period as in the abrupt-4xCO2 simulation
"""

import sys
sys.path.insert(1, '/path_to_cmip_utils/')
import CMIP6_ATMOS_UTILS as atmos # this script also applies to CMIP5 
from read_model_data_cmip5 import ecs_models_cmip5, make_filelist_cmip5, Modelinfo, Ofx_files
import glob
import copy
import numpy as np
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)


def consistent_naming(ds):
    # setup model grid
    if 'height' in ds.coords:
        ds = ds.drop('height')
    if 'nav_lat' in ds.coords:
        ds = ds.rename({'nav_lon':'lon','nav_lat':'lat'})
    if 'bounds_nav_lat' in ds.variables:
        ds = ds.rename({'bounds_nav_lat':'vertices_latitude','bounds_nav_lon':'vertices_longitude'})
    if 'olevel' in ds.coords:
        ds = ds.rename({'olevel':'lev'})
    if 'olevel_bounds' in ds.variables:
        ds = ds.rename({'olevel_bounds':'lev_bnds'})
    if 'latitude' in ds.coords and 'lat' not in ds.coords:
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'vertex' in ds.dims and 'lat_bnds' in ds.variables:
        ds = ds.rename({'lat_bnds':'vertices_latitude','lon_bnds':'vertices_longitude'})
    if 'bounds_lat' in ds.variables:
        ds = ds.rename({'bounds_lat':'vertices_latitude','bounds_lon':'vertices_longitude'})
    if 'vertices' in ds.dims and 'lat_bnds' in ds.variables:
        ds = ds.rename({'lat_bnds':'vertices_latitude','lon_bnds':'vertices_longitude'})
    if 'nlat' in ds.dims:
        ds = ds.rename({'nlat':'j','nlon':'i'})
    if 'ni' in ds.dims:
        ds = ds.rename({'ni':'i','nj':'j'})
    if 'x' in ds.dims:
        ds = ds.rename({'x':'i','y':'j'})
    if 'vertex' in ds.dims:
        ds = ds.rename({'vertex':'vertices'})
    if 'nvertex' in ds.dims:
        ds = ds.rename({'nvertex':'vertices'})
    if 'axis_nbounds' in ds.dims:
        ds = ds.rename({'axis_nbounds':'bnds'})
    if 'd2' in ds.dims:
        ds = ds.rename({'d2':'bnds'})
    if 'nv' in ds.dims:
        ds = ds.rename({'nv':'bnds'})
    if 'lev_bounds' in ds.variables:
        ds = ds.rename({'lev_bounds':'lev_bnds'})
    if 'time_bounds' in ds.variables:
        ds = ds.rename({'time_bounds':'time_bnds'})
    if 'bound' in ds.dims:
        ds = ds.rename({'bound':'bnds'})
    if 'rlat' in ds.dims:
        ds = ds.rename({'rlat':'j','rlon':'i'})
    return ds

def read_tosga(model):
    modelga = copy.deepcopy(model)
    daga = None
    if model.name in ['NorESM1-ME', 'NorESM1-M']:
        make_filelist_cmip5(modelga, var = 'tosga', time_frequency='mon', component = 'ocean', path_to_data = '/projects/NS9034K/CMIP5/output1/')
    else:
        make_filelist_cmip5(modelga, var = 'tosga',  time_frequency='mon', component = 'ocean', path_to_data = '/projects/NS9252K/ESGF/cmip5/output1')
    if modelga.filenames:
        if len(modelga.filenames)>1:
            dsga =  xr.open_mfdataset(modelga.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
        else:
            dsga =  xr.open_dataset(modelga.filenames[0], chunks={"time":12})
        print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%('tosga',modelga.name, modelga.expid,  len(dsga['tosga'].time.values)/12))
        dsga = consistent_naming(dsga)
        if modelga.name == 'HadGEM2-ES':
            # HadGEM2-ES provides files which cover December - November (instead of January - December)
            dsga = dsga.isel(time=slice(1,12*151+1))
        dsga = atmos.fix_time(dsga, 1)
        daga = atmos.yearly_avg(dsga['tosga'])
        # if piControl -> start year at 4xCO2 branch time year to get the corresponding period
        if modelga.expid in ['piControl']:
            daga = daga.isel(year=slice(modelga.branchtime_year, None))
        daga = daga.isel(year = slice(0,150)).squeeze()
        if 'lat' in daga.coords:
            print('\nLat/lon variables not needed in tosga')
            daga = daga.drop('lat').drop('lon')
            print('Dropping lat/lon')
        print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%('tosga',modelga.name, modelga.expid,  len(daga.year.values)))
        daga = daga.to_dataset(name = 'tosga')
    return daga

def make_mask(model, da):
    ofx = glob.glob(Ofx_files('sftof', model.name))
    if model.name not in ['GISS-E2-H','GISS-E2-R','CSIRO-Mk3-6-0']:
        lsm = xr.open_mfdataset(ofx)
        lsm = consistent_naming(lsm)
        lsm = lsm.sftof
        if np.any(lsm.values > 1):
            print('Values given in precentage. Convert to fraction')
            lsm = 1e-2*lsm
        mask = lsm
    else:
        tmp = da.isel(year=0).squeeze()
        tmp = tmp.where(tmp!=0)
        mask = xr.where(xr.ufuncs.isnan(tmp),np.nan,1)
    return mask

def calc_areaavg(model, ds, da, areacello, lat_lim):
    mask = make_mask(model, da)
    areacello = areacello*mask
    area = areacello*xr.ones_like(da)
    if 'i' in area.dims:
        if 'year' in area.dims:
            area = area.transpose("year", "j","i")
        else:
            area = area.transpos("j","i")
    else:
        if 'year' in area.dims:
            area = area.transpose("year", "lat","lon")
        else:
            area = area.transpose( "lat","lon")
            print('Areacello from model: %s'%model.name)
    if 'year' in area.dims:
        area = area.isel(year=0)
    print('Load areacello:')
    area = area.load()
    print('Areacello loaded')
    area = area.where(area.lat <= lat_lim)
    print('Avg area for lat lim:%.1f: %.4e'%(lat_lim,area.sum()))
    if 'i' in da.dims:
        da = da.transpose("year","j","i")
        areaavg = ((da*area).sum(dim=("j","i")))/(area.sum())
    else:
        da = da.transpose("year","lat","lon")
        areaavg = ((da*area).sum(dim=("lat","lon")))/(area.sum())
    return areaavg, area

def make_tosavg_dataset(model, ds, da, area, lat_lim, name = None, volname = None):
    volavg, volcell = calc_areaavg(model, ds, da, area, lat_lim=lat_lim)
    volavg = volavg.to_dataset(name = 'sst_' + name)
    if volname:
        volcell = volcell.to_dataset(name = 'areacello_' + volname)
        volmerge = xr.merge([volavg, volcell])
    else:
        volmerge = volavg
    return volmerge

def make_areaavg(model, ds, da,  var,  outpath):
    print('In areaavg')
    if model.name not in ['GISS-E2-H','GISS-E2-R']:
        area = xr.open_dataset(Ofx_files('areacello', model.name))
    else:
        # the GISS models provide SST on the atmospheric grid
        area = xr.open_dataset('/projects/NS9252K/ESGF/cmip5/output1/NASA-GISS/'+ model.name+'/piControl/fx/atmos/fx/r0i0p0/v20160511/areacella/areacella_fx_' + model.name +'_piControl_r0i0p0.nc')
        area = area.rename({'areacella':'areacello'})
    area = consistent_naming(area)
    area = area.areacello.load()
    ## calculate dataset to be written
    tosglb = make_tosavg_dataset(model, ds, da, area, lat_lim=90, name = 'glb', volname = None)#'glb')
    tos35S = make_tosavg_dataset(model, ds, da, area, lat_lim=-35, name = '35S', volname = None)
    tosga = read_tosga(model)
    if tosga:
        tosavg = xr.merge([tosglb, tos35S,  tosga])
    else:
        tosavg = xr.merge([tosglb, tos35S])
    tosavg = tosavg.to_netcdf(outpath + var +'_areaavg_' + model.realm +'_' + model.name + '_' + model.expid + '_' + model.realiz + '_121-150_yravg.nc', compute=False)
    with ProgressBar():
        result = tosavg.compute()

def make_avg_sst(models, var, outpath):
    for modelname,expinfo in models.items():
        print(modelname)
        for exp in ['piControl', 'abrupt4xCO2']:
            if exp in ['piControl']:
                  model =  Modelinfo(name = modelname, institute = expinfo['institute'], expid = 'piControl', realm = 'Omon',
                  realiz='r1i1p1', version=expinfo['versions']['pic_ocean'], branchtime_year=expinfo['branch_yr'])
            if exp in ['abrupt4xCO2']:
                  model =  Modelinfo(name = modelname, institute = expinfo['institute'], expid = 'abrupt4xCO2', realm = 'Omon',
                  realiz='r1i1p1', version=expinfo['versions']['a4xco2_ocean'], branchtime_year=expinfo['branch_yr'])
            if modelname in ['NorESM1-ME', 'NorESM1-M']:
                make_filelist_cmip5(model, var, time_frequency='mon', component = 'ocean', path_to_data = '/projects/NS9034K/CMIP5/output1/')
            else:
                make_filelist_cmip5(model, var, time_frequency='mon', component = 'ocean', path_to_data = '/projects/NS9252K/ESGF/cmip5/output1')
            if model.filenames:
                if len(model.filenames)>1:
                    if modelname in ['FGOALS-g2']:
                        ds = xr.open_mfdataset(model.filenames, decode_times=False, combine='nested', concat_dim='time', chunks={"time":12}, parallel=True )
                    else:
                        ds =  xr.open_mfdataset(model.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
                else:
                    ds =  xr.open_dataset(model.filenames[0], chunks={"time":12})
                print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%(var,model.name, model.expid,  len(ds[var].time.values)/12))

                ds = consistent_naming(ds)
                if modelname == 'HadGEM2-ES':
                    ds = ds.isel(time=slice(1,12*151+1))
                ds = atmos.fix_time(ds, 1)
                da = atmos.yearly_avg(ds[var])

                # if piControl -> start year at 4xCO2 branch time year to get the corresponding period
                if model.expid in ['piControl']:
                    da = da.isel(year=slice(model.branchtime_year,None))
                da = da.isel(year = slice(0,150))
                print(da)
                print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%(var,model.name, model.expid,  len(da.year.values)))
                make_areaavg(model, ds, da,  var, outpath)
            else:
                print('%s not loaded for model %s, experiment: piControl. Skipping model! Please check!'%(var,modelname))
                continue
            del model

if __name__ == '__main__':
    outpath = '/scratch/adagj/CMIP5/modeldrift/'
    models = ecs_models_cmip5()
    models, realiz = ecs_models_cmip5()
    for var in [ 'tos']:
        make_avg_sst(models, var, outpath)
