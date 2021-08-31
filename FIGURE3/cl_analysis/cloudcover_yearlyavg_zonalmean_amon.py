#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ADA GJERMUNDSEN
YEARS: 2019 - 2021
CMIP6 models: NorESM2-LM, CESM2
This script is used to calculate zonal mean cloud cover, south of 35S
For piControl, the time period is selected to the corresponding period as in the abrupt-4xCO2 simulation
"""

import sys
sys.path.insert(1, 'path_to_cmip_utils/')
import CMIP6_ATMOS_UTILS as atmos
from read_modeldata_cmip6 import ecs_models_cmip6, make_filelist_cmip6, Modelinfo
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)

def make_avg_hist(models, var, outpath):
    #######RTMTOA##########################
    print('CLOUD COVER ANALYSIS: \n')
   
    for modelname,expinfo in models.items():
        print(modelname)
        for exp in ['piControl', 'abrupt-4xCO2']:
            model = Modelinfo(name = modelname, institute = expinfo['institute'], expid = exp, realm = 'Amon',
                                  realiz=expinfo['variant_labels'][-1], grid_atmos = expinfo['grid_label_atmos'][0], grid_ocean = expinfo['grid_label_ocean'], branchtime_year=expinfo['branch_yr'])
            if modelname in ['NorESM2-LM', 'NorESM2-MM']:
                make_filelist_cmip6(model, var,  component = 'atmos', activity_id='CMIP',path_to_data = '/projects/NS9034K/CMIP6/')
            else:
                make_filelist_cmip6(model, var,  component = 'atmos')
            if model.filenames:
                if len(model.filenames)>1:
                    ds =  xr.open_mfdataset(model.filenames, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
                else:
                    ds =  xr.open_dataset(model.filenames[0])
                # models use different name conventions (e.g. lat vs latitude). Overwrite to have the same names for dimensions and coordinates 
                ds = atmos.consistent_naming(ds)
                # models use different calenders, overwrite to get the same and the same start year. Easier to make comparisons
                ds = atmos.fix_time(ds, 1)
                da = atmos.yearly_avg(ds[var])
                # to get the corresponding years in the piControl simulation:
                if model.expid in ['piControl']:
                    da = da.isel(year=slice(model.branchtime_year,None))
                # use isel instead of sel to get the first 500 years after branch time from parent
                da = da.isel(year = slice(0,500))
                da = da.mean(dim='lon').squeeze()
                print('Calculate 35 S')
                wgts = xr.ufuncs.cos(xr.ufuncs.deg2rad(da.lat))
                wgts = wgts.where(wgts.lat<=-35)
                da = da.where(da.lat<=-35)
                da = (wgts*da).sum(dim='lat')/wgts.sum(dim='lat')
                dsout = da.to_dataset(name = var + '_35S')
                dsout = dsout.to_netcdf(outpath + var +'_90S_35S_' + modelname + '_' +  exp +  '.nc', compute=False)
                with ProgressBar():
                    result = dsout.compute()
            del model

if __name__ == '__main__':
    outpath = 'path_to_outdata/'
    models = ecs_models_cmip6()
    models ={'NorESM2-LM':models['NorESM2-LM'], 'CESM2':models['CESM2']}
    for var in ['cl']:
        make_avg_hist(models, var, outpath)
