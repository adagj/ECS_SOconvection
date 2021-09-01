#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ada Gjermundsen
year: 2019 - 2021
This script is used to calculate the initial (5 years) sea surface temperature response in CESM2 and NorESM2-LM 
forced by 4xCO2 and by freshwater forcing in the Weddell Sea and the surrounding area
The result is used in FIGURE 5
"""
import sys
sys.path.insert(1, 'path_to_cmip_utils/CMIP6_UTILS')
import CMIP6_ATMOS_UTILS as atmos
import CMIP6_SEAICE_UTILS as ocean
from read_modeldata_cmip6 import ecs_models_cmip6, make_filelist_cmip6, Modelinfo
import glob
import numpy as np
import xesmf as xe
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)

def get_noresm_raw(modelname, expid, var, syr, eyr):
    path= 'path_to_noresm_raw_data/noresm/cases'
    ofxpath = path + '/' + expid  + '/' + 'ocn/hist/'
    filenames = sorted(glob.glob(ofxpath + expid + '.micom.hm.*.nc'))
    xn = filenames[0]
    print(np.int(xn.split('.')[-2][:4])) 
    # only need time period 160101 - 160512
    filenames = list(filter(lambda x: syr<= np.int(x.split('.')[-2][:4]) <= eyr, filenames))
    if len(filenames)>1:
        print('Ocean files for model: %s, experiment: %s'%(modelname, expid))
    else:
        raise Exception('Ofx file: %s missing for model: %s'%(var,modelname))
    return filenames

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
    print(ds)
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
    da = ds[var]
    da = da.where(da!=0)
    if areao is not None:
        tmp = da.squeeze()
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
        tmp = dr[var].squeeze()
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
    ogr = 'gr'
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


def regrid_latlon(modelname, dsm, dam, dacm, var, areao=None):
    # regrid files
    # use the NorESM2-MM atmospheric grid since it is 1x1, so closer to the grid resolution used in BLOM
    area = xr.open_dataset('path_to_grid_files/areacella_fx_NorESM2-MM_piControl_r1i1p1f1_gn.nc')
    outgrid = make_outgrid(area)
    grid_weight_path = 'path_to_dump_files/'
    regrid_mode = 'nearest_s2d'
    if areao is None:
        areao = xr.open_dataset('path_to_grid_files/areacello_Ofx_NorESM2-LM_piControl_r1i1p1f1_gn.nc')
        areao = areao.rename({'latitude':'lat', 'longitude':'lon'})
        areao = areao.areacello
    regridder = make_regridder(modelname, dsm, outgrid, grid_weight_path, regrid_mode = 'nearest_s2d', reuse_weights=False, periodic = False)
    if isinstance(dacm,xr.Dataset) or isinstance(dacm, xr.DataArray):
        dsc = regrid_file(dacm.to_dataset(name = var), var, regridder, outgrid, areao=areao, area = area.areacella)
    else:
        dsc = 1
    ds = regrid_file(dam.to_dataset(name=var), var, regridder, outgrid, areao=areao, area = area.areacella)
    return ds, dsc


def make_reference_slice(model, ds, var, endyr):
    ds = ocean.consistent_naming(ds)
    print('\nIn reference slice')
    print(ds.time)
    ds = atmos.fix_time(ds, 1)
    da = atmos.yearly_avg(ds[var])
    if model.expid in ['piControl']:
        da = da.isel(year=slice(model.branchtime_year+endyr-5, model.branchtime_year+endyr))
    else:
        da = da.isel(year=slice(endyr-5, endyr))
    print('\nIn reference slice')
    print(da.year)
    da = da.mean(dim='year')
    return ds, da

def initial_sst_response(models, outpath, endyr=5):
    for modelname,expinfo in models.items():
        print('Initial sst response for model %s:'%modelname)
        if modelname in ['NorESM2-MM','NorESM2-LM', 'CESM2']:
            var = 'tos'
            #piControl
            modelctrl = make_modelobj(modelname, expinfo, expid='piControl')
            dsc = read_files(modelctrl, var)
            t_dsc, t_dac = make_reference_slice(modelctrl, dsc, var, endyr)
            #abrupt-4xCO2
            model4xco2 = make_modelobj(modelname, expinfo, expid='abrupt-4xCO2')
            ds = read_files(model4xco2, var)
            t_ds, t_da = make_reference_slice(model4xco2, ds, var, endyr)
            areao = None
        if modelname in ['NorESM2-hosing']:
            var = 'sst'
            # piControl - the hosing experience was branched off in year 1601, so the piControl branchtime in 1601
            filelist = get_noresm_raw(modelname, expinfo[1], var, syr = 1601, eyr = 1605)
            dsc = xr.open_mfdataset(filelist, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
            print(dsc.time)
            dsc = atmos.fix_time(dsc, 1)
            dac = atmos.yearly_avg(dsc[var])
            dac = dac.isel(year=slice(endyr-5, endyr))
            print(dac.year)
            t_dac = dac.mean(dim='year')
            t_dac = t_dac.rename({'x':'i','y':'j'})
            # hosing experiment
            filelist = get_noresm_raw(modelname, expinfo[0], var, syr = 1601, eyr = 1605)
            ds = xr.open_mfdataset(filelist, combine='nested', concat_dim='time', parallel=True, chunks={"time":12})
            print(ds.time)
            ds = atmos.fix_time(ds, 1)
            da = atmos.yearly_avg(ds[var])
            da = da.isel(year=slice(endyr-5, endyr))
            print(da.year)
            t_da = da.mean(dim='year')
            t_da = t_da.rename({'x':'i','y':'j'})
            grid = xr.open_dataset('path_to_ocngrid/grid.nc')
            t_ds = grid['pclat'].to_dataset(name = 'vertices_latitude')
            t_ds['lat'] = grid['plat']
            t_ds['vertices_longitude'] = grid['pclon']
            t_ds['lon'] =  grid['plon']
            t_ds['areao']=grid['parea']
            t_ds = t_ds.rename({'x':'i','y':'j','nv':'vertices'})
            t_da = t_da.assign_coords(lat = t_ds['lat'])
            t_da = t_da.assign_coords(lon = t_ds['lon'])
            t_dac = t_dac.assign_coords(lat = t_ds['lat'])
            t_dac = t_dac.assign_coords(lon = t_ds['lon'])
            areao = t_ds.areao
            var='tos'
        if modelname in ['NorESM2-LM','NorESM2-MM', 'NorESM2-hosing']:
            # regrid files
            # use the NorESM2-MM atmospheric grid since it is 1x1, so closer to the grid resolution used in BLOM
            if 'time' in t_ds.dims:
                gridinfo= t_ds.isel(time=0).drop('time')
            else:
                gridinfo = t_ds
            tos_ds, tos_dsc = regrid_latlon(modelname, gridinfo, t_da, t_dac, var, areao = areao)
      
        if modelname in ['NorESM2-LM','NorESM2-MM', 'CESM2']:
            dsout_ctrl = tos_dsc.to_netcdf(outpath + var +'_' + modelctrl.realm +'_' + modelctrl.name + '_' + modelctrl.expid + '_' + modelctrl.realiz + '_'+str(endyr) + '_5yravg.nc', compute=False)
            dsout_case = tos_ds.to_netcdf(outpath + var +'_' + model4xco2.realm +'_' + model4xco2.name + '_' + model4xco2.expid + '_' + model4xco2.realiz + '_'+str(endyr) + '_5yravg.nc', compute=False)
            del model4xco2, modelctrl
        if modelname in ['NorESM2-hosing']:
            dsout_ctrl = tos_dsc.to_netcdf(outpath + var +'_Omon_NorESM2-LM_piControl-hosing_r1i1p1f1_'+str(endyr) + '_5yravg.nc', compute=False)
            dsout_case = tos_ds.to_netcdf(outpath + var +'_Omon_NorESM2-LM_hosing_r1i1p1f1_'+ str(endyr) + '_5yravg.nc', compute=False)

        with ProgressBar():
            result = dsout_ctrl.compute()
            result = dsout_case.compute()
        
        del  dsc, ds, t_ds, tos_dsc, tos_ds, dsout_case, dsout_ctrl

    
if __name__ == '__main__':
    outpath = 'path_to_outdata/' 
    models = ecs_models_cmip6()
    models = {'NorESM2-hosing': ['N1850_f19_tn14_20200220_hosing5','N1850_f19_tn14_20190621'], 'NorESM2-LM':models['NorESM2-LM'], 'CESM2':models['CESM2'], 'NorESM2-MM':models['NorESM2-MM']}
    for endyr in [5]: 
        make_last_30yrs_avg(models, outpath=outpath, endyr=endyr)
    
