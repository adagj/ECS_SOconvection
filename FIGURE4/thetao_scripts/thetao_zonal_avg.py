#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ada Gjermundsen
year: 2019 - 2021
This script is used to calculate the averaged ocean temperature in CESM2 and NorESM2 (LM and MM) south of 50S
for the CMIP experiments piControl and abrupt-4xCO2 after 150 and 500 years
the average time is 30 years
The result is used in FIGURE 4
"""
import sys
sys.path.insert(1, 'path_to_cmip_utils/CMIP6_UTILS')
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
    ny,nx = ds.lat.shape
    if modelname in ['NorESM2-LM', 'NorESM2-MM'] and 'siconc' in list(ds.keys()):
        #drop the last row with j=385 of area when dealing with the sea ice variables.
        ny = ny -1
    lat_model = ds.lat.isel(j=slice(0,ny)).rename({'i':'x','j':'y'}).drop('lon').drop('lat')
    lon_model = ds.lon.isel(j=slice(0,ny)).rename({'i':'x','j':'y'}).drop('lon').drop('lat')
    lon_b_model = xr.concat([ds.vertices_longitude.isel(vertices=0), ds.vertices_longitude.isel(vertices=1,i=-1)],dim='i')
    lon_b_model = xr.concat([lon_b_model,xr.concat([ds.vertices_longitude.isel(vertices=3,j=-1), ds.vertices_longitude.isel(vertices=2,j=-1,i=-1)],dim='i')],dim='j')
    lat_b_model = xr.concat([ds.vertices_latitude.isel(vertices=0), ds.vertices_latitude.isel(vertices=1,i=-1)],dim='i')
    lat_b_model = xr.concat([lat_b_model,xr.concat([ds.vertices_latitude.isel(vertices=3,j=-1), ds.vertices_latitude.isel(vertices=2,j=-1,i=-1)],dim='i')],dim='j')

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
    ''' The first step of the regridding routine
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
    '''Second step of the regridding routine
    
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
    print('Variable: %s \n'%var)
    print(ds)
    if areao is not None:
        tmp = ds[var].isel(lev=0).squeeze()
        glb_mean =  (tmp*areao).sum(dim=('i','j'))/(areao*tmp.notnull()).sum(dim=('i','j'))
        print('Global mean value BEFORE regridding:%f '%np.round(glb_mean.values,4))
    dr = regridder(ds[var]) # need DataArray
    print('Regridding completed\n')
    dr = dr.where(dr!=0)
    dr = dr.rename({'x': 'lon', 'y':'lat'})
    dr = dr.to_dataset(name = var)
    dr[var].attrs['long_name']='Sea Water Potential Density'
    dr[var].attrs['units']= 'kg/m3'
    dr[var].attrs['standard_name']='sea_water_potential_density'
    dr[var].attrs['description']='Regridded density to lat, lon grid'
    if area is not None:
        tmp = dr[var].isel(lev=0).squeeze()
        glb_mean_dr = (tmp*area).sum(dim=('lat','lon'))/(area*tmp.notnull()).sum(dim=('lat','lon'))
        print('Global mean value AFTER regridding:%f'%np.round(glb_mean_dr.values,4))
    return dr


def make_outgrid(ds):
    # make an outgrid used for regridding to be identical to the atmospheric grid used by the model... not sure that is necessary 
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


def make_modelobj(modelname, expinfo, expid='piControl'):
    ogr = ['gr']
    model = Modelinfo(name = modelname, institute = expinfo['institute'], expid = expid, realm = 'Omon',
                 realiz=expinfo['variant_labels'][0], grid_atmos = expinfo['grid_label_atmos'][0], grid_ocean = ogr, branchtime_year=expinfo['branch_yr'])
    model.grid_ocean=ogr
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


def regrid_latlon(model, dsm, dam, dacm, var):
    # regrid files
    # use the NorESM2-MM atmospheric grid since it is 1x1, so closer to the grid resolution used in BLOM
    area = xr.open_dataset('path_to_grid_files/areacella_fx_NorESM2-MM_piControl_r1i1p1f1_gn.nc')
    outgrid = make_outgrid(area)
    grid_weight_path = 'path_to_dumpfiles/dump/'
    areao = xr.open_dataset('path_to_grid_files/areacello_Ofx_NorESM2-LM_piControl_r1i1p1f1_gn.nc')
    areao = areao.rename({'latitude':'lat', 'longitude':'lon'})
    regridder = make_regridder(model.name, dsm, outgrid, grid_weight_path, regrid_mode = 'nearest_s2d', reuse_weights=False, periodic = False)
    dsc = regrid_file(dacm.to_dataset(name = var), var, regridder, outgrid, areao=areao.areacello, area = area.areacella)
    dac = dsc[var]
    ds = regrid_file(dam.to_dataset(name=var), var, regridder, outgrid, areao=areao.areacello, area = area.areacella)
    da = ds[var]
    return da, dac


def make_reference_slice(model, ds, var, endyr):
    ds = ocean.consistent_naming(ds)
    ds = atmos.fix_time(ds, 1)
    da = atmos.yearly_avg(ds[var])
    if model.expid in ['piControl']:
        da = da.isel(year=slice(model.branchtime_year+endyr-30, model.branchtime_year+endyr))
    else:
        da = da.isel(year=slice(endyr-30, endyr))
    da = da.mean(dim='year')
    return ds, da


def make_zonalmeans(da, var):
    # set all land points to Nan
    da = da.where(da != 0)
    print('Calculate Zonal mean')
    da_a = da.mean(dim='lon', skipna=True)
    print('Calculate area averaged 50-90S')
    wgts = xr.ufuncs.cos(xr.ufuncs.deg2rad(da_a.lat))
    da_b = (wgts.where(wgts.lat<=-50)*da_a.where(wgts.lat<=-50)).sum(dim='lat')/wgts.where(wgts.lat<=-50).sum(dim='lat')
    da_b = da_b.squeeze()
    print('Calculate area averaged 35-90S')
    da_c = (wgts.where(wgts.lat<=-35)*da_a.where(wgts.lat<=-35)).sum(dim='lat')/wgts.where(wgts.lat<=-35).sum(dim='lat')
    da_c = da_c.squeeze()
    ds_out = xr.merge([da_a.to_dataset(name=var), da_b.to_dataset(name = var + '_50S') , da_c.to_dataset(name = var + '_35S')])
    return ds_out


def make_regional_avg(models, outpath, endyr=150):
    for modelname,expinfo in models.items():
        print(modelname)
        if endyr > 150 and modelname in ['NorESM2-MM']:
        # NorESM2-MM abrupt-4xCO2 was only run for 150 years 
            continue
        var = 'thetao'
        modelctrl = make_modelobj(modelname, expinfo, expid='piControl')
        dsc = read_files(modelctrl, var)        
        t_dsc, t_dac = make_reference_slice(modelctrl, dsc, var, endyr)
        model4xco2 = make_modelobj(modelname, expinfo, expid='abrupt-4xCO2')
        ds = read_files(model4xco2, var)        
        t_ds, t_da = make_reference_slice(model4xco2, ds, var, endyr)
 
        if modelname in ['NorESM2-LM','NorESM2-MM']:
            # regrid files
            # use the NorESM2-MM atmospheric grid since it is 1x1, so closer to the grid resolution used in BLOM
            t_da, t_dac = regrid_latlon(model4xco2, t_ds.isel(time=0).drop('time'), t_da, t_dac, var)
        dsout_ctrl = make_zonalmeans(t_dac, var)
        dsout_case = make_zonalmeans(t_da, var)
        dsout_ctrl = dsout_ctrl.to_netcdf(outpath + var +'_' + modelctrl.realm +'_' + modelctrl.name + '_' + modelctrl.expid + '_' + modelctrl.realiz + '_'+str(endyr) + '_30yravg.nc', compute=False)
        dsout_case = dsout_case.to_netcdf(outpath + var +'_' + model4xco2.realm +'_' + model4xco2.name + '_' + model4xco2.expid + '_' + model4xco2.realiz + '_'+str(endyr) + '_30yravg.nc', compute=False)
        with ProgressBar():
            result = dsout_ctrl.compute()
            result = dsout_case.compute()
        del model4xco2, modelctrl, dsc, ds, t_dsc, t_ds, t_dac, t_da, dsout_ctrl, dsout_case

    
if __name__ == '__main__':
    outpath = 'path_to_outdata/' 
    models = ecs_models_cmip6()
    models = { 'NorESM2-LM':models['NorESM2-LM'], 'CESM2':models['CESM2'], 'NorESM2-MM':models['NorESM2-MM']}
    for endyr in [150, 500]: 
        make_regional_avg(models, outpath=outpath, endyr=endyr)
    
