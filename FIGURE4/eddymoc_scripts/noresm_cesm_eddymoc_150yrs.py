#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ada Gjermundsen
year: 2019 - 2021
This script is used to calculate the eddy-induced overturning in CESM2 and NorESM2 (LM and MM) south of 50S
for the CMIP experiments piControl and abrupt-4xCO2 after 150 
the average time is 30 years
The result is used in FIGURE 4
"""

import sys
sys.path.insert(1, '/scratch/adagj/CMIP6/CLIMSENS/CMIP6_UTILS')
import CMIP6_ATMOS_UTILS as atmos
import CMIP6_SEAICE_UTILS as ocean
from read_modeldata_cmip6 import ecs_models_cmip6, make_filelist_cmip6, Modelinfo
import numpy as np
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)

def make_attributes(da, var, expid):
    da.attrs['long_name']='Global Ocean Meridional Overturning Mass Streamfunction Due to Parameterized Mesoscale Advection'
    da.attrs['name']='eddymoc'
    da.attrs['units']='kg s-1'
    da.attrs['standard_name']='global_ocean_meridional_overturning_mass_streamfunction_due_to_parameterized_mesoscale_eddy_advection'
    da.attrs['expid']=expid
    ds = da.to_dataset(name = var)
    return ds

def extract_global_moc(modelname, da, dac, var):
    if 'sector' in da.coords:
        da = da.drop('sector')
    if 'sector' in dac.coords:
        dac = dac.drop('sector')
    da = da.isel(basin=-1)
    dac = dac.isel(basin=-1)
    return da, dac
                                                                                                                      
def make_reference_slice(model, ds, var, endyr):
    ds = ocean.consistent_naming(ds)
    ds = atmos.fix_time(ds, 1)
    return ds

def make_yearly_avg(model, ds, var, endyr):
    da = atmos.yearly_avg(ds[var])
    if model.expid in ['piControl']:
        da = da.isel(year=slice(model.branchtime_year+endyr-30, model.branchtime_year+endyr))
    else:
        da = da.isel(year=slice(endyr-30, endyr))
    da = da.mean(dim='year')
    return da

def make_modelobj(modelname, expinfo, expid='piControl'):
    model = Modelinfo(name = modelname, institute = expinfo['institute'], expid = expid, realm = 'Omon',
                 realiz=expinfo['variant_labels'][0], grid_atmos = expinfo['grid_label_atmos'][0], grid_ocean = expinfo['grid_label_ocean'], branchtime_year=expinfo['branch_yr'])
    return model

def read_files(model, var):
    if model.name in ['NorESM2-LM', 'NorESM2-MM']:
        make_filelist_cmip6(model, var,  component = 'ocean', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
    else:
        make_filelist_cmip6(model, var,  component = 'ocean')
    print(model.filenames)
    if model.filenames:
        if len(model.filenames)>1:
            ds =  xr.open_mfdataset(model.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":1})
        else:
            ds =  xr.open_dataset(model.filenames[0], chunks={"time":1})
        print('%s loaded for model: %s, experiment: piControl . Lenght of simulation: %.1f years'%(var,model.name, len(ds[var].time.values)/12))
    else:
        print('%s not loaded for model %s, experiment: piControl. Skipping model! Please check!'%(var,model.name))
    return ds

def make_last_30yrs_avg(models, var, outpath, endyr=150):
    print('global eddy moc: \n')
    for modelname,expinfo in models.items():
        print(modelname)
        if var in ['msftmzsmpa'] and modelname in ['NorESM2-LM']:
            continue
        modelctrl = make_modelobj(modelname, expinfo, expid='piControl')
        dsc = read_files(modelctrl, var)
        dsc = make_reference_slice(modelctrl, dsc, var, endyr)
        model4xco2 = make_modelobj(modelname, expinfo, expid='abrupt-4xCO2')
        ds = read_files(model4xco2, var)
        ds = make_reference_slice(model4xco2, ds, var, endyr)
        ds, dsc = extract_global_moc(modelname, ds, dsc, var)
        da = make_yearly_avg(model4xco2, ds, var, endyr)
        dac = make_yearly_avg(modelctrl, dsc, var, endyr)
        dsout_ctrl = make_attributes(dac, var, 'piControl')
        dsout_case = make_attributes(da, var, 'abrupt-4xCO2')
        print(dsout_ctrl)
        print(dsout_case)
        dsout_ctrl = dsout_ctrl.to_netcdf(outpath + var +'_' + modelctrl.realm +'_' + modelctrl.name + '_' + modelctrl.expid + '_' + modelctrl.realiz + '_'+str(endyr) + '_30yravg.nc', compute=False)
        dsout_case = dsout_case.to_netcdf(outpath + var +'_' + model4xco2.realm +'_' + model4xco2.name + '_' + model4xco2.expid + '_' + model4xco2.realiz + '_'+str(endyr) + '_30yravg.nc', compute=False)
        with ProgressBar():
            result = dsout_ctrl.compute()
            result = dsout_case.compute()
        del model4xco2, modelctrl, dsc, ds, dac, da, dsout_ctrl, dsout_case


if __name__ == '__main__':
    outpath = 'path_to_outdata/' 
    models = ecs_models_cmip6()
    models = {'NorESM2-LM':models['NorESM2-LM'], 'CESM2':models['CESM2']}
    for var in ['msftmzsmpa', 'msftmzmpa']:
        make_last_30yrs_avg(models, var=var, outpath=outpath, endyr=150)
   
