#!/usr/bin/env pyhon3
# -*- coding: utf-8 -*-
"""
@author: ADA GJERMUNDSEN
YEARS: 2019 - 2021
CMIP6 model analysis
This script is used to calculate effective climate sensitivity EffCS
For piControl, the time period is selected to the corresponding period as in the abrupt-4xCO2 simulation
"""
import sys
sys.path.insert(1, 'path_to_cmip_utils/')
import CMIP6_ATMOS_UTILS as atmos
from read_modeldata_cmip6 import ecs_models_cmip6, make_filelist_cmip6, Modelinfo
import numpy as np
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression



def linearfit(ds):
    t = linregress(np.arange(0,ds.shape[0]),ds.values)
    val = t.slope*np.arange(0,ds.shape[0]) + t.intercept
    return val

def read_radfiles(modelobj):
        if modelobj.name in ['NorESM2-LM', 'NorESM2-MM']:
            make_filelist_cmip6(modelobj, 'rsdt',  component = 'atmos', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
        else:
            make_filelist_cmip6(modelobj, 'rsdt',  component = 'atmos')
        if modelobj.filenames:
            if len(modelobj.filenames)>1:
                rsdt =  xr.open_mfdataset(modelobj.filenames, combine='nested', concat_dim='time')
            else:
                rsdt =  xr.open_dataset(modelobj.filenames[0])
            print('rsdt loaded for model: %s, %s. Lenght of simulation: %.1f years'%(modelobj.name, modelobj.expid, len(rsdt.rsdt.time.values)/12))
            rsdt = atmos.consistent_naming(rsdt)
            if modelobj.name == 'HadGEM2-ES':
                rsdt = rsdt.isel(time=slice(1,12*151+1))
            rsdt = atmos.fix_time(rsdt,1)
            rsdt = atmos.yearly_avg(atmos.global_mean(rsdt.rsdt))
            if modelobj.expid == 'piControl':
                rsdt = rsdt.isel(year=slice(modelobj.branchtime_year, None))
        else:
             print('File rsdt missing for Model %s in experiment %s'%(modelobj.name, modelobj.expid))
             rsdt = None
        if modelobj.name in ['NorESM2-LM', 'NorESM2-MM']:
            make_filelist_cmip6(modelobj, 'rsut',  component = 'atmos', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
        else:
            make_filelist_cmip6(modelobj, 'rsut',  component = 'atmos')
        if modelobj.filenames:
            if len(modelobj.filenames)>1:
                rsut =  xr.open_mfdataset(modelobj.filenames, combine='nested', concat_dim='time')
            else:
                rsut =  xr.open_dataset(modelobj.filenames[0])
            print('rsut loaded for model: %s , %s. Lenght of simulation: %.1f years'%(modelobj.name, modelobj.expid, len(rsut.rsut.time.values)/12))
            rsut = atmos.consistent_naming(rsut)
            if modelobj.name == 'HadGEM2-ES':
                rsut = rsut.isel(time=slice(1,12*151+1))
            rsut = atmos.fix_time(rsut,1)
            rsut = atmos.yearly_avg(atmos.global_mean(rsut.rsut))
            if modelobj.expid == 'piControl':
                rsut = rsut.isel(year=slice(modelobj.branchtime_year, None))
        else:
             print('File rsut missing for Model %s in experiment %s'%(modelobj.name, modelobj.expid))
             rsut = None
        if modelobj.name in ['NorESM2-LM', 'NorESM2-MM']:
             make_filelist_cmip6(modelobj, 'rlut',  component = 'atmos', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
        else:
            make_filelist_cmip6(modelobj, 'rlut',  component = 'atmos')
        if modelobj.filenames:
            if len(modelobj.filenames)>1:
                rlut =  xr.open_mfdataset(modelobj.filenames, combine='nested', concat_dim='time')
            else:
                rlut =  xr.open_dataset(modelobj.filenames[0])
            print('rlut loaded for model: %s ,%s. Lenght of simulation: %.1f years'%(modelobj.name, modelobj.expid, len(rlut.rlut.time.values)/12))
            rlut = atmos.consistent_naming(rlut)
            if modelobj.name == 'HadGEM2-ES':
                rlut = rlut.isel(time=slice(1,12*151+1))
            rlut = atmos.fix_time(rlut,1)
            rlut = atmos.yearly_avg(atmos.global_mean(rlut.rlut))
            if modelobj.expid == 'piControl':
                rlut = rlut.isel(year=slice(modelobj.branchtime_year, None))
        else:
             print('File rlut missing for Model %s in experiment %s'%(modelobj.name, modelobj.expid))
             rlut = None
        return rsdt, rsut, rlut

def calculate_ecs_values(models, startyr=0,  endyr=150):
    #######RTMTOA##########################
    print('ECS CALCULATIONS: \n')
    ecsvalues={}
    for modelname,expinfo in models.items():
        print(modelname)
        modelctrl = Modelinfo(name = modelname, institute = expinfo['institute'], expid = 'piControl', realm = 'Amon',
                  realiz=expinfo['variant_labels'][0], grid_atmos = expinfo['grid_label_atmos'][0], grid_ocean = expinfo['grid_label_ocean'], branchtime_year=expinfo['branch_yr'])
        if modelname in ['NorESM2-LM', 'NorESM2-MM']:
            make_filelist_cmip6(modelctrl, 'tas',  component = 'atmos', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
        else:
            make_filelist_cmip6(modelctrl, 'tas',  component = 'atmos')
        if modelctrl.filenames:
            if len(modelctrl.filenames)>1:
                tasctrl =  xr.open_mfdataset(modelctrl.filenames, combine='nested', concat_dim='time')
            else:
                tasctrl =  xr.open_dataset(modelctrl.filenames[0])
            print('tas loaded for model: %s, piControl . Lenght of simulation: %.1f years'%(modelctrl.name, len(tasctrl.tas.time.values)/12))
            tasctrl = atmos.consistent_naming(tasctrl)
            tasctrl = atmos.fix_time(tasctrl, 1)
            tasctrl = atmos.yearly_avg(atmos.global_mean(tasctrl.tas))
            tasctrl = tasctrl.isel(year=slice(modelctrl.branchtime_year, None))
        else:
            print('tas not loaded for model %s, piControl. Skipping model! Please check!'%modelname)
            continue
        if modelname in ['EC-Earth3','HadGEM3-GC31-LL','HadGEM3-GC31-MM']:
            model4xco2 = Modelinfo(name = modelname, institute = expinfo['institute'], expid = 'abrupt-4xCO2', realm = 'Amon',
                  realiz=expinfo['variant_labels'][1], grid_atmos = expinfo['grid_label_atmos'][0], grid_ocean = expinfo['grid_label_ocean'], branchtime_year=expinfo['branch_yr'])
        else:
            model4xco2 = Modelinfo(name = modelname, institute = expinfo['institute'], expid = 'abrupt-4xCO2', realm = 'Amon',
                  realiz=expinfo['variant_labels'][0], grid_atmos = expinfo['grid_label_atmos'][0], grid_ocean = expinfo['grid_label_ocean'], branchtime_year=expinfo['branch_yr'])
        if modelname in [ 'NorESM2-LM', 'NorESM2-MM']:
            make_filelist_cmip6(model4xco2, 'tas', component = 'atmos', path_to_data = '/projects/NS9034K/CMIP6/')
        else:
            make_filelist_cmip6(model4xco2, 'tas',  component = 'atmos')
        if model4xco2.filenames:
            if len(model4xco2.filenames)>1:
                tascase =  xr.open_mfdataset(model4xco2.filenames, combine='nested', concat_dim='time')
            else:
                tascase =  xr.open_dataset(model4xco2.filenames[0])
            print('tas loaded for model: %s , abrupt4xCO2. . Lenght of simulation: %.1f years'%(model4xco2.name, len(tascase.tas.time.values)/12))
            tascase = atmos.consistent_naming(tascase)
            tascase = atmos.fix_time(tascase,1)
            tascase = atmos.yearly_avg(atmos.global_mean(tascase.tas))
        else:
            print('tas not loaded for model %s, abrupt4xCO2. Skipping model! Please check!'%modelname)
            continue

        rsdt, rsut, rlut = read_radfiles(modelctrl)
        radctrl = rsdt - rsut - rlut
        rsdt, rsut, rlut = read_radfiles(model4xco2)
        radcase = rsdt - rsut - rlut
        drad = radcase.values[startyr:endyr] - linearfit(radctrl[startyr:endyr])
        dtas = tascase.values[startyr:endyr] - linearfit(tasctrl[startyr:endyr])
        linmodel =  LinearRegression().fit(dtas.reshape((-1, 1)), drad)
        ecsvalues[modelname] = str(-np.round(linmodel.intercept_/linmodel.coef_[0]/2,2))
        print('EffCS from %s : %s K'%(modelname,ecsvalues[modelname]))
        del model4xco2, modelctrl
    return ecsvalues

if __name__ == '__main__':
    models = ecs_models_cmip6()
    ecs_values = calculate_ecs_values(models,startyr=0,  endyr=150)
    np.save('outdata/ecs_cmip6_150yrs.npy', ecs_values)
