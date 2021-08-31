#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ADA GJERMUNDSEN
YEARS: 2019 - 2021

CMIP5 model analysis

This script is used to calculate the volume averaged ocean temperature 
Globally, south of 50S and south of 50S and below 1500 m
The averaging period is the last 30 years of the 150 years long simulations
For piControl, the time period is selected to the corresponding period as in the abrupt-4xCO2 simulation
"""

import sys
sys.path.insert(1, '/path_to_cmip_utils/CMIP6_UTILS')
import CMIP6_ATMOS_UTILS as atmos # this script also applies to CMIP5 
from read_model_data_cmip5 import ecs_models_cmip5, make_filelist_cmip5, Modelinfo, Ofx_files
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


def read_thetaog(model):
    modelga = model
    daga = None
    if model.name in ['NorESM1-ME','NorESM1-M']:
        make_filelist_cmip5(model, 'thetaoga', time_frequency='mon', component = 'ocean', path_to_data = '/projects/NS9034K/CMIP5/output1/')
    else:
        make_filelist_cmip5(model, 'thetaoga', time_frequency='mon', component = 'ocean', path_to_data = '/projects/NS9252K/ESGF/cmip5/output1')
    if modelga.filenames:
        if len(modelga.filenames)>1:
            if model.name in ['FGOALS-g2']:
                dsga = xr.open_mfdataset(modelga.filenames, combine='nested', concat_dim='time', decode_times=False, parallel=True, chunks={"time":12})
            else:
                dsga =  xr.open_mfdataset(modelga.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
        else:
            dsga =  xr.open_dataset(modelga.filenames[0], chunks={"time":12})
        print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%('thetaoga',modelga.name, modelga.expid,  len(dsga['thetaoga'].time.values)/12))

        dsga = consistent_naming(dsga)
        if model.name == 'HadGEM2-ES':
            # HadGEM2-ES files cover in December-November instead of January-December 
            dsga = dsga.isel(time=slice(1,12*151+1))
        dsga = atmos.fix_time(dsga, 1)
        daga = atmos.yearly_avg(dsga['thetaoga'])
        # indecies start at 0 -> model.branchtime_year
        if modelga.expid in ['piControl']:
            daga = daga.isel(year=slice(modelga.branchtime_year, None))
        daga = daga.isel(year = slice(0,150)).squeeze() # select firt 150 years
        if 'lat' in daga.coords:
            print('\nLat/lon variables not needed in thetaoga')
            daga = daga.drop('lat').drop('lon')
            print('Dropping lat/lon')
        print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%('thetaoga',modelga.name, modelga.expid,  len(daga.year.values)))
        daga = daga.to_dataset(name = 'thetaoga')
    return daga


def make_lev_bnds(ds):
    if ds.lev[-1].values<4000:
        raise Exception('Please check depth coordinate')
    if ds.lev[-1].values>400000:
        #depth given in cm
        ds = ds.assign_coords(lev = ds.lev.values/100)
    lev_bnds = np.array([ds.lev.values[:-1] + 0.5*ds.lev.diff(dim='lev').values])
    lev_bnds = np.append(np.array([0.]), lev_bnds)
    lev_bnds = np.append( lev_bnds, np.array([ds.lev[-1].values + 0.5*ds.lev.diff(dim='lev').values[-1]]))
    lev_bnds = np.reshape(np.concatenate([lev_bnds[:-1],lev_bnds[1:]]),[2,len(ds.lev.values)]).T
    lev_bnds = xr.DataArray(lev_bnds, dims=('lev','bnds'),coords={'lev':ds.lev}).to_dataset(name='lev_bnds')
    lev_bnds.attrs['long_name']='ocean depth coordinate boundaries'
    lev_bnds.attrs['units']='m'
    lev_bnds.attrs['axis']='Z'
    lev_bnds.attrs['bounds']='lev_bnds'
    lev_bnds.attrs['standard_name']='depth bounds'
    ds_out = xr.merge([ds, lev_bnds])
    return ds_out


def make_volcello(ds, da, areacello, deptho, depth_lim):
    print('In make_volcello():')
    #print('ds')
    #print(ds)
    if 'lev_bnds' in ds.variables:
        print('Use level boundaries from ds')
        if 'time' in ds.lev_bnds.coords:
            lev_bnds = ds.lev_bnds.isel(time=0).drop('time')
        else:
            lev_bnds = ds.lev_bnds
        if lev_bnds.isel(bnds=0)[-1].values<4000:
            raise Exception('Please check depth coordinate')
        if lev_bnds.isel(bnds=0)[-1].values>400000:
            #depth given in cm. Rewrite to meters 
            lev_bnds = xr.DataArray(lev_bnds.values/100, dims=('lev','bnds'), coords={'lev':ds.lev})
    else:
        print('Level boundaries not found in ds. Calculate lev_bnds by make_lev_bnds function')
        lev_bnds = make_lev_bnds(ds)
        lev_bnds = lev_bnds.lev_bnds
    print('Thetao yearly averaged:')
    print(da)
    print('Level boundaries:')
    print(lev_bnds)

    bnd1 = xr.ones_like(da)*lev_bnds.isel(bnds=1)
    bnd1 = xr.where(bnd1<=deptho, bnd1, deptho)
    bnd0 = xr.ones_like(da)*lev_bnds.isel(bnds=0)
    bnd0 = xr.where(bnd0<=deptho, bnd0,deptho)
    tmp = da.isel(year=0).isel(lev=0).squeeze()
    tmp = tmp.where(tmp!=0)
    mask = xr.where(xr.ufuncs.isnan(tmp),np.nan,1)
    areacello = areacello*mask
    area = areacello*xr.ones_like(da)
    volcello = area*(xr.where(bnd1>=depth_lim, bnd1, depth_lim) - xr.where(bnd0>= depth_lim,bnd0,depth_lim))
    print('Integrated ocean volume for depth lim:%.1f is %.4e\n'%(depth_lim, volcello.isel(year=0).sum()))
    return volcello


def calc_volavg(model, ds, da, area, depth_lim, lat_lim):
    if model.name in ['FGOALS-g2']:
        deptho = xr.open_dataset('/projects/NS9252K/ESGF/cmip5/output1/LASG-CESS/FGOALS-g2/piControl/fx/ocean/fx/r0i0p0/v20130314/deptho/deptho_fx_FGOALS-g2_piControl_r0i0p0.nc')
        deptho = deptho.load()
        deptho = consistent_naming(deptho)
        volcello = make_volcello(ds, da, area, deptho.deptho, depth_lim)
    else:
        deptho = xr.open_dataset(Ofx_files('deptho', model.name))
        deptho = deptho.load()
        deptho = consistent_naming(deptho)
        volcello = make_volcello(ds, da, area, deptho.deptho, depth_lim)
    if 'i' in volcello.dims:
        volcello = volcello.transpose("year","lev", "j","i")
    else:
        volcello = volcello.transpose("year","lev", "lat","lon")
    volcello = volcello.isel(year=0)
    volcello = volcello.load()
    volcello = volcello.where(volcello.lat <= lat_lim)
    if 'i' in da.dims:
        volavg = ((da*volcello).sum(dim=("lev", "j","i")))/(volcello.sum())
    else:
        volavg = ((da*volcello).sum(dim=("lev", "lat","lon")))/(volcello.sum())
    return volavg, volcello

def make_volavg_dataset(model, ds, da, area, depth_lim, lat_lim, name = None, volname = None):
    volavg, volcell = calc_volavg(model, ds, da, area, depth_lim=depth_lim, lat_lim=lat_lim)
    volavg = volavg.to_dataset(name = 'ocntemp_' + name)
    if volname:
        volcell = volcell.to_dataset(name = 'volcello_' + volname)
        volmerge = xr.merge([volavg, volcell])
    else:
        volmerge = volavg
    return volmerge

def make_volavg(model, ds, da,  var,  outpath):
    area = xr.open_dataset(Ofx_files('areacello', model.name))
    area = consistent_naming(area)
    area = area.areacello.load()
    ## calculate dataset to be written
    volglb = make_volavg_dataset(model, ds, da, area, depth_lim=0, lat_lim=90, name = 'glb', volname = None)
    vol50S = make_volavg_dataset(model, ds, da, area, depth_lim=0, lat_lim=-50, name = '50S', volname = None)
    vol50S1500 =  make_volavg_dataset(model, ds, da, area, depth_lim=1500, lat_lim=-50, name = '50S_1500m_bottom', volname = None)
    thetaoga = read_thetaog(model)
    if thetaoga:
        print(thetaoga)
        volavg = xr.merge([volglb, vol50S, vol50S1500, thetaoga])
    else:
        volavg = xr.merge([volglb, vol50S, vol50S1500])
    volavg = volavg.to_netcdf(outpath + var +'_volumeavg_' + model.realm +'_' + model.name + '_' + model.expid + '_' + model.realiz + '_121-150_yravg.nc', compute=False)
    with ProgressBar():
        result = volavg.compute()

def make_avg_hist(models, var, outpath):
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
                    # HadGEM2-ES provides files which cover December - November (instead of January - December)
                    ds = ds.isel(time=slice(1,12*151+1))
                # Set all model years to start at year 1     
                ds = atmos.fix_time(ds, 1)
                # some models use cm as depth coordinate. Make sure to use meters:
                if ds.lev[-1].values<4000:
                    raise Exception('Please check depth coordinate')
                if ds.lev[-1].values>400000:
                    #depth given in cm
                    ds = ds.assign_coords(lev = ds.lev.values/100)
                da = atmos.yearly_avg(ds[var])

                # if piControl -> start year at 4xCO2 branch time year to get the corresponding period
                if model.expid in ['piControl']:
                    da = da.isel(year=slice(model.branchtime_year, None))
                # select first 150 years
                da = da.isel(year = slice(0,150))
                print(da)
                print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%(var,model.name, model.expid,  len(da.year.values)))
                make_volavg(model, ds, da,  var, outpath)
            else:
                print('%s not loaded for model %s, experiment: piControl. Skipping model! Please check!'%(var,modelname))
                continue
            del model

if __name__ == '__main__':
    outpath = 'path_to_outdata_folder/'
    models = ecs_models_cmip5()
    models, realiz = ecs_models_cmip5()
    for var in [ 'thetao']:
        make_avg_hist(models, var, outpath)


