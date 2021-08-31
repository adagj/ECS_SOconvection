#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ADA GJERMUNDSEN
YEARS: 2019 - 2021
CMIP6 model analysis
This script is used to calculate the area averaged sea surface temperature 
Globally and south of 35S
The averaging period is the last 30 years of the 150 years long simulations
For piControl, the time period is selected to the corresponding period as in the abrupt-4xCO2 simulation
"""
import sys
sys.path.insert(1, '/scratch/adagj/CMIP6/CLIMSENS/CMIP6_UTILS')
import CMIP6_ATMOS_UTILS as atmos
import CMIP6_SEAICE_UTILS as ocean
from read_modeldata_cmip6 import ecs_models_cmip6, make_filelist_cmip6, Modelinfo, Ofx_files
import glob
import copy
import numpy as np
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)


def read_tosga(model):
    modelga = model
    daga = None
    if model.name in ['NorESM2-LM', 'NorESM2-MM']:
        make_filelist_cmip6(modelga, var = 'tosga',  component = 'ocean', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
    else:
        make_filelist_cmip6(modelga, var = 'tosga',  component = 'ocean')
    if modelga.filenames:
        if len(modelga.filenames)>1:
            dsga =  xr.open_mfdataset(modelga.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
        else:
            dsga =  xr.open_dataset(modelga.filenames[0], chunks={"time":12})
        print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%('tosga',modelga.name, modelga.expid,  len(dsga['tosga'].time.values)/12))

        dsga = ocean.consistent_naming(dsga)
        dsga = atmos.fix_time(dsga, 1)
        daga = atmos.yearly_avg(dsga['tosga'])
        # if piControl -> start year at 4xCO2 branch time year to get the corresponding period
        if modelga.expid in ['piControl']:
            daga = daga.isel(year=slice(modelga.branchtime_year,None))
        daga = daga.isel(year = slice(0,150)).squeeze()
        if 'lat' in daga.coords:
            print('\nLat/lon variables not needed in tosga')
        #    print(daga)
            daga = daga.drop('lat').drop('lon')
            print('Dropping lat/lon')
        #    print(daga)
        print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%('tosga',modelga.name, modelga.expid,  len(daga.year.values)))
        daga = daga.to_dataset(name = 'tosga')
    return daga

def make_mask(model, da):
    if model.name in ['NorESM2-LM', 'NorESM2-MM']:
        Ofx_files(model, var='sftof', path_to_data = '/projects/NS9034K/CMIP6/')
    else:
        Ofx_files(model, var='sftof')
    ofx = sorted(glob.glob(model.ofxfile))
    if ofx and model.name not in ['GISS-E2-2-G']:
        lsm = xr.open_dataset(model.ofxfile)
        lsm = ocean.consistent_naming(lsm)
        lsm = lsm.sftof
        if np.any(lsm.values > 1):
            print('Values given in precentage. Convert to fraction')
            lsm = 1e-2*lsm
        mask = lsm
        if model.name in ['UKESM1-0-LL'] and model.expid in ['abrupt-4xCO2']:
            mask = mask.assign_coords(lat=da.lat)
            mask = mask.assign_coords(lon=da.lon)
    else:
        tmp = da.isel(year=0).squeeze()
        tmp = tmp.where(tmp!=0)
        mask = xr.where(xr.ufuncs.isnan(tmp),np.nan,1)
    return mask

def calc_areaavg(model, ds, da, areacello, lat_lim):
    mask = make_mask(model, da)
    print('In calc_areaavg():')
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
    #print(model.path)
    path = model.path.rsplit(model.expid)[0]
    #print(path)
    if model.name in ['NorESM2-LM', 'NorESM2-MM']:
        area = xr.open_dataset(path  +  'piControl/' + model.realiz +'/Ofx/areacello/gn/latest/' + 'areacello_Ofx_' + model.name + '_piControl_' +  model.realiz + '_gn.nc')
    elif model.name in ['CAMS-CSM1-0']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/CAMS/CAMS-CSM1-0/1pctCO2/r2i1p1f1/Ofx/areacello/gn/v20190830/areacello_Ofx_CAMS-CSM1-0_1pctCO2_r2i1p1f1_gn.nc')
    elif model.name in ['BCC-ESM1', 'BCC-CSM2-MR']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/BCC/BCC-ESM1/1pctCO2/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_BCC-ESM1_1pctCO2_r1i1p1f1_gn.nc')
    elif model.name in['EC-Earth3-Veg']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3-Veg/piControl/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_EC-Earth3-Veg_piControl_r1i1p1f1_gn.nc')
    elif model.name in['EC-Earth3']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/piControl/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_EC-Earth3_piControl_r1i1p1f1_gn.nc')
    elif model.name in ['FGOALS-f3-L']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/CAS/FGOALS-f3-L/historical/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_FGOALS-f3-L_historical_r1i1p1f1_gn.nc')
    elif model.name in ['NESM3']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/PMIP/NUIST/NESM3/lig127k/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_NESM3_lig127k_r1i1p1f1_gn.nc')
    elif model.name in ['GISS-E2-1-H']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/NASA-GISS/GISS-E2-1-H/piControl/r1i1p1f1/fx/areacella/gn/latest/areacella_fx_GISS-E2-1-H_piControl_r1i1p1f1_gn.nc')
        area = area.rename({'areacella':'areacello'})
    elif model.name in ['GISS-E2-2-G', 'GISS-E2-1-G']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/NASA-GISS/GISS-E2-1-G/piControl/r1i1p3f1/fx/areacella/gn/latest/areacella_fx_GISS-E2-1-G_piControl_r1i1p3f1_gn.nc')
        area = area.rename({'areacella':'areacello'})
    elif model.name in ['HadGEM3-GC31-LL']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc')
    elif model.name in ['HadGEM3-GC31-MM']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/piControl/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn.nc')
    elif model.name in ['MRI-ESM2-0']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/MRI/MRI-ESM2-0/abrupt-4xCO2/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_MRI-ESM2-0_abrupt-4xCO2_r1i1p1f1_gn.nc')
    elif model.name in ['E3SM-1-0']:
        area = xr.open_dataset('/projects//NS9252K/ESGF/CMIP6/CMIP/E3SM-Project/E3SM-1-0/piControl/r1i1p1f1/fx/areacella/gr/latest/areacella_fx_E3SM-1-0_piControl_r1i1p1f1_gr.nc')
        area = area.rename({'areacella':'areacello'})
    else:
        area = xr.open_dataset(path  +  'piControl/' + model.realiz +'/Ofx/areacello/'
                      + model.gridlabel + '/' + 'latest/' + 'areacello_Ofx_' + model.name + '_piControl_' +  model.realiz + '_'
                      + model.gridlabel +'.nc')
    area = ocean.consistent_naming(area)
    area = area.areacello.load()
    ## calculate dataset to be written
    tosglb = make_tosavg_dataset(model, ds, da, area, lat_lim=90, name = 'glb', volname = None)#'glb')
    tos35S = make_tosavg_dataset(model, ds, da, area, lat_lim=-35, name = '35S', volname = None)
    tosga =  read_tosga(model)
    if tosga:
        tosavg = xr.merge([tosglb,  tos35S,  tosga])
    else:
        tosavg = xr.merge([tosglb, tos35S])
    tosavg = tosavg.to_netcdf(outpath + var +'_areaavg_' + model.realm +'_' + model.name + '_' + model.expid + '_' + model.realiz + '_121-150_yravg.nc', compute=False)
    with ProgressBar():
        result = tosavg.compute()


def make_avg_sst(models, var, outpath):
    #######RTMTOA##########################
    print('TOS CALCULATIONS: \n')
    for modelname,expinfo in models.items():
        print(modelname)
        for exp in ['abrupt-4xCO2', 'piControl']:
            if exp == 'abrupt-4xCO2' and modelname in ['EC-Earth3','HadGEM3-GC31-LL','HadGEM3-GC31-MM']:
                model = Modelinfo(name = modelname, institute = expinfo['institute'], expid = exp, realm = 'Omon',
                                  realiz=expinfo['variant_labels'][1], grid_atmos = expinfo['grid_label_atmos'][0], grid_ocean = expinfo['grid_label_ocean'], branchtime_year=expinfo['branch_yr'])
            else:
                model = Modelinfo(name = modelname, institute = expinfo['institute'], expid = exp, realm = 'Omon',
                                  realiz=expinfo['variant_labels'][0], grid_atmos = expinfo['grid_label_atmos'][0], grid_ocean = expinfo['grid_label_ocean'], branchtime_year=expinfo['branch_yr'])
            if modelname in ['NorESM2-LM', 'NorESM2-MM']:
                make_filelist_cmip6(model, var,  component = 'ocean', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
            else:
                make_filelist_cmip6(model, var,  component = 'ocean')
            if model.filenames:
                if len(model.filenames)>1:
                    ds =  xr.open_mfdataset(model.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":1})
                else:
                    ds =  xr.open_dataset(model.filenames[0], chunks={"time":1})
                print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%(var,model.name, model.expid,  len(ds[var].time.values)/12))

                ds = ocean.consistent_naming(ds)
                ds = atmos.fix_time(ds, 1)
                da = atmos.yearly_avg(ds[var])

                # if piControl -> start year at 4xCO2 branch time year to get the corresponding period
                if model.expid in ['piControl']:
                    da = da.isel(year=slice(model.branchtime_year,-1))
                da = da.isel(year = slice(0,150))
                print(da)
                print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%(var,model.name, model.expid,  len(da.year.values)))
                make_areaavg(model, ds, da,  var, outpath)
            else:
                print('%s not loaded for model %s, experiment: piControl. Skipping model! Please check!'%(var,modelname))
                continue
            del model

if __name__ == '__main__':
    outpath = 'outdata_folder/'
    models = ecs_models_cmip6()
    for var in [ 'tos']:
        make_avg_sst(models, var, outpath)
