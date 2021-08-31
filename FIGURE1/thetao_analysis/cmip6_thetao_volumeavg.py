#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ADA GJERMUNDSEN
YEARS: 2019 - 2021

CMIP6 model analysis

This script is used to calculate the volume averaged ocean temperature 
Globally, south of 50S and south of 50S and below 1500 m
The averaging period is the last 30 years of the 150 years long simulations
For piControl, the time period is selected to the corresponding period as in the abrupt-4xCO2 simulation

"""
import sys
sys.path.insert(1, '/path_to_utils_scripts/CMIP6_UTILS')
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


def make_thetaog(model):
    modelga = model
    daga = None
    if model.name in ['NorESM2-LM', 'NorESM2-MM']:
        make_filelist_cmip6(modelga, var = 'thetaoga',  component = 'ocean', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
    else:
        make_filelist_cmip6(modelga, var = 'thetaoga',  component = 'ocean')
    if modelga.filenames:
        if len(modelga.filenames)>1:
            dsga =  xr.open_mfdataset(modelga.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
        else:
            dsga =  xr.open_dataset(modelga.filenames[0], chunks={"time":12})
        print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%('thetaoga',modelga.name, modelga.expid,  len(dsga['thetaoga'].time.values)/12))

        dsga = ocean.consistent_naming(dsga)
        dsga = atmos.fix_time(dsga, 1)
        daga = atmos.yearly_avg(dsga['thetaoga'])
        # indecies start at 0 -> modelctrl.branchtime_year-1
        if modelga.expid in ['piControl']:
            daga = daga.isel(year=slice(modelga.branchtime_year,-1))
        daga = daga.isel(year = slice(0,150)).squeeze()
        if 'lat' in daga.coords:
            print('\nLat/lon variables not needed in thetaoga')
        #    print(daga)
            daga = daga.drop('lat').drop('lon')
            print('Dropping lat/lon')
        #    print(daga)
        print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%('thetaoga',modelga.name, modelga.expid,  len(daga.year.values)))
        daga = daga.to_dataset(name = 'thetaoga')
    return daga


def make_pbo(model):
    print('In make_pbo')
    dapbo = None
    modelpbo = copy.deepcopy(model)
    modelpbo.expid='piControl'
    if model.name in ['NorESM2-LM', 'NorESM2-MM']:
        make_filelist_cmip6(modelpbo, 'pbo',  component = 'ocean', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
    else:
        make_filelist_cmip6(modelpbo, 'pbo',  component = 'ocean')
    if modelpbo.filenames:
        if len(model.filenames)>1:
            dspbo =  xr.open_mfdataset(modelpbo.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
        else:
            dspbo =  xr.open_dataset(modelpbo.filenames[0], chunks={"time":12})
        print('%s loaded for model: %s, experiment: %s. Lenght of simulation: %.1f years'%('pbo',modelpbo.name, modelpbo.expid,  len(dspbo['pbo'].time.values)/12))
        dspbo = ocean.consistent_naming(dspbo)
        dspbo = atmos.fix_time(dspbo, 1)
        dapbo = atmos.yearly_avg(dspbo['pbo'])
        print('PBO data array:')
        print(dapbo)
        if modelpbo.expid in ['piControl']:
            dapbo = dapbo.isel(year=slice(modelpbo.branchtime_year, None))
        dapbo = dapbo.isel(year = slice(0,150))
        dapbo = dapbo.mean(dim='year').squeeze()
    return dapbo

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

def estimate_volcello_from_bottompressure(ds, da, pbo,  areacello, mask,  depth_lim):
    # use for models E3SM-1-0, IPSL-CM6A-LR, INM-CM4-8, INM-CM5-0
    rho = 1036 #ocean mean density
    g = 9.8    # gravity 
    if 'lev_bnds' in ds.variables:
        print('Use level boundaries from ds')
        lev_bnds = ds.lev_bnds.isel(time=0).drop('time')
        if lev_bnds.isel(bnds=0)[-1].values<4000:
            raise Exception('Please check depth coordinate')
        if lev_bnds.isel(bnds=0)[-1].values>400000:
            #depth given in cm
            lev_bnds = xr.DataArray(lev_bnds.values/100, dims=('lev','bnds'), coords={'lev':ds.lev})
    else:
        print('Level boundaries not found in ds. Calculate lev_bnds by make_lev_bnds function')
        lev_bnds = make_lev_bnds(ds)
    depthpb = pbo.load()/(rho*g)
    print('\nIn estimate_volcello_from_bottompressure, da:')
    print(da)
    print('\nlev_bnds:')
    print(lev_bnds)
    bnd1 = xr.ones_like(da)*lev_bnds.isel(bnds=1)
    bnd1 = xr.where(bnd1<=depthpb, bnd1, depthpb)
    bnd0 = xr.ones_like(da)*lev_bnds.isel(bnds=0)
    bnd0 = xr.where(bnd0<=depthpb, bnd0,depthpb)
    areacello = areacello*mask
    area = areacello*xr.ones_like(da)
    volcello = area*(xr.where(bnd1>=depth_lim, bnd1, depth_lim) - xr.where(bnd0>= depth_lim,bnd0,depth_lim))
    print('Integrated ocean volume for depth lim:%.1f is %.4e\n'%(depth_lim, volcello.isel(year=0).sum()))
    return volcello

def make_volcello(ds, da, areacello, mask, deptho, depth_lim):
    print('In make_volcello():')
    if 'lev_bnds' in ds.variables:
        print('Use level boundaries from ds')
        lev_bnds = ds.lev_bnds
        print(lev_bnds.coords)
        if 'time' in lev_bnds.coords:
            print('remove time in lev_bnds:')
            lev_bnds = ds.lev_bnds.isel(time=0).drop('time')
        if lev_bnds.isel(bnds=0)[-1].values<4000:
            raise Exception('Please check depth coordinate')
        if lev_bnds.isel(bnds=0)[-1].values>400000:
            #depth given in cm
            lev_bnds = xr.DataArray(lev_bnds.values/100, dims=('lev','bnds'), coords={'lev':ds.lev})
    else:
        print('Level boundaries not found in ds. Calculate lev_bnds by make_lev_bnds function')
        lev_bnds = make_lev_bnds(ds)
    bnd1 = xr.ones_like(da)*lev_bnds.isel(bnds=1)
    bnd1 = xr.where(bnd1<=deptho, bnd1, deptho)
    bnd0 = xr.ones_like(da)*lev_bnds.isel(bnds=0)
    bnd0 = xr.where(bnd0<=deptho, bnd0,deptho)
    areacello = areacello*mask
    area = areacello*xr.ones_like(da)
    volcello = area*(xr.where(bnd1>=depth_lim, bnd1, depth_lim) - xr.where(bnd0>= depth_lim,bnd0,depth_lim))
    print('Integrated ocean volume for depth lim:%.1f is %.4e\n'%(depth_lim, volcello.isel(year=0).sum()))
    return volcello

def make_mask(model, da):
    if model.name in ['NorESM2-LM', 'NorESM2-MM']:
        Ofx_files(model, var='sftof', path_to_data = '/projects/NS9034K/CMIP6/')
    else:
        Ofx_files(model, var='sftof')
    ofx = sorted(glob.glob(model.ofxfile))
    print('In mask function')
    if ofx:
        lsm = xr.open_dataset(model.ofxfile)
        lsm = ocean.consistent_naming(lsm)
        lsm = lsm.sftof
        #print(lsm.values[:,100])
        if np.any(lsm.values > 1):
            print('Values given in precentage. Convert to fraction')
            lsm = 1e-2*lsm
        mask = lsm
        if model.name in ['UKESM1-0-LL'] and model.expid in ['abrupt-4xCO2']:
            mask = mask.assign_coords(lat=da.lat)
            mask = mask.assign_coords(lon=da.lon)
    else:
        tmp = da.isel(year=0).isel(lev=0).squeeze()
        tmp = tmp.where(tmp!=0)
        mask = xr.where(xr.ufuncs.isnan(tmp),np.nan,1)
    return mask


def calc_volavg(model, ds, da, area, depth_lim, lat_lim):
    mask = make_mask(model, da)
    if model.name in ['MRI-ESM2-0', 'E3SM-1-0', 'IPSL-CM6A-LR', 'INM-CM4-8', 'INM-CM5-0']:
        pbo = make_pbo(model)
        print('In calc_volavg, dpo calculated in make_pbo:')
        print(pbo)
        volcello = estimate_volcello_from_bottompressure(ds, da, pbo,  area, mask, depth_lim)
    else:
        if model.name in ['NorESM2-LM', 'NorESM2-MM']:
            Ofx_files(model, var='deptho', path_to_data = '/projects/NS9034K/CMIP6/')
        else:
            Ofx_files(model, var='deptho')
        deptho = xr.open_dataset(model.ofxfile)
        deptho = ocean.consistent_naming(deptho)
        volcello = make_volcello(ds, da, area, mask, deptho.deptho, depth_lim)
    if 'i' in volcello.dims:
        if 'year' in volcello.dims:
            volcello = volcello.transpose("year","lev", "j","i")
        else:
            volcello = volcello.transpos("lev", "j","i")
    else:
        if 'year' in volcello.dims:
            volcello = volcello.transpose("year","lev", "lat","lon")
        else:
            volcello.transpose("lev", "lat","lon")
            print('Volcello from model: %s'%model.name)
    if 'year' in volcello.dims:
        volcello = volcello.isel(year=0)
    print('Load volcell:')
    volcello = volcello.load()
    print('Volcello loaded')
    ## you have to make sure that volcello and da have the same dimensions but
    ## also the same bounds. Several models have shifted lon in da and volcello
    print('\n problem with crash here:')
    print(volcello)
    volcello = volcello.where(volcello.lat <= lat_lim)
    if 'i' in da.dims:
        da = da.transpose("year","lev", "j","i")
        volavg = ((da*volcello).sum(dim=("lev", "j","i")))/(volcello.sum())
    else:
        da = da.transpose("year","lev", "lat","lon")
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
    print('In areaavg')
    path = model.path.rsplit(model.expid)[0]
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
    elif model.name in ['GISS-E2-2-G']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/NASA-GISS/GISS-E2-1-G/piControl/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_GISS-E2-1-G_piControl_r1i1p1f1_gn.nc')
    elif model.name in ['HadGEM3-GC31-LL']:
        area = xr.open_dataset('/projects/NS9252K/ESGF/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc')
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
    volglb = make_volavg_dataset(model, ds, da, area, depth_lim=0, lat_lim=90, name = 'glb', volname = None)
    vol50S = make_volavg_dataset(model, ds, da, area, depth_lim=0, lat_lim=-50, name = '50S', volname = None)
    vol50S1500 =  make_volavg_dataset(model, ds, da, area, depth_lim=1500, lat_lim=-50, name = '50S_1500m_bottom', volname = None)
    thetaoga = make_thetaog(model)
    if thetaoga:
        volavg = xr.merge([volglb, vol50S, vol50S1500, thetaoga])
    else:
        volavg = xr.merge([volglb, vol50S, vol50S1500])
    print(volavg)
    volavg = volavg.to_netcdf(outpath + var +'_volumeavg_' + model.realm +'_' + model.name + '_' + model.expid + '_' + model.realiz + '_121-150_yravg.nc', compute=False)
    with ProgressBar():
        result = volavg.compute()


def make_avg_hist(models, var, outpath):
    #######RTMTOA##########################
    print('THETAO CALCULATIONS: \n')
    for modelname,expinfo in models.items():
        if modelname not in [ 'CMCC-CM2-SR5']:
            continue
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
                # some models use cm as depth coordinate. Make sure to use meters:
                if ds.lev[-1].values<4000:
                    raise Exception('Please check depth coordinate')
                if ds.lev[-1].values>400000:
                    #depth given in cm
                    ds = ds.assign_coords(lev = ds.lev.values/100)
                da = atmos.yearly_avg(ds[var])

                # # if piControl -> start year at 4xCO2 branch time year to get the corresponding period
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
    models = ecs_models_cmip6()
    for var in [ 'thetao']:
        make_avg_hist(models, var, outpath)

