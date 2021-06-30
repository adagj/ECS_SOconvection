#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 12:36:29 2020

@author: adag
"""

import xarray as xr
import xesmf as xe
import numpy as np
import glob
#import iris
import CMIP6_ATMOS_UTILS as atmos
import warnings
warnings.simplefilter('ignore')

def make_bounds(modelname, ds):
    '''
    Parameters
    ----------
    modelname: str, name of model. NorESM2 has one less gridpoint for cice compared to blom. 
                                  CanESM5 has flipped corner 0 and 2 compared to all the other models
    ds : xarray.DataSet, with model grid information for the data which need to be regridded
                         on (i,j,vertices) format where the 4 vertices give grid corner information

    Returns
    -------
    ds_in :  xarray.DataSet, with 2D model grid information for the data which need to be regridded
    '''
    ny,nx = ds.lat.shape
    if modelname in ['NorESM2-LM', 'NorESM2-MM'] and 'siconc' in list(ds.keys()):
        ny = ny -1
    if 'xh' in ds.dims:
        lat_model = ds.lat.isel(yh=slice(0,ny)).rename({'xh':'x','yh':'y'}).drop('lon').drop('lat')
        lon_model = ds.lon.isel(yh=slice(0,ny)).rename({'xh':'x','yh':'y'}).drop('lon').drop('lat')
    else:
        lat_model = ds.lat.isel(j=slice(0,ny)).rename({'i':'x','j':'y'}).drop('lon').drop('lat')
        lon_model = ds.lon.isel(j=slice(0,ny)).rename({'i':'x','j':'y'}).drop('lon').drop('lat')

    if modelname in ['CanESM5']:
        print(modelname)
        lat_b_model, lon_b_model = make_bnds_canesm5(ds)
    elif modelname in ['EC-Earth3', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR']:
        lat_b_model, lon_b_model = make_bounds_ec_earth(ds)
    elif modelname == 'FGOALS-f3-L':
        lat_b_model, lon_b_model = make_bounds_fgoals_f3_L(ds)
    else:
        print(modelname)
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
       
def make_bnds_canesm5(ds):
    '''
    Models like CnaESM5 and CMCC-CM2-SR5 have flipped grid corners 0 and 2
    compared to the other models
    
    Parameters
    ----------
    ds : xarray.DataSet with grid corner (boundary) information (i,j,vertices)
    
    Returns
    -------
    lat_b_model : xarray.DataArray with 2D (i,j) latitude grid boundary info
    lon_b_model : xarray.DataArray with 2D (i,j) longitude grid boundary infor

    '''
    lon_b_model = xr.concat([ds.vertices_longitude.isel(vertices=2), ds.vertices_longitude.isel(vertices=1,i=-1)],dim='i')
    lon_b_model = xr.concat([lon_b_model,xr.concat([ds.vertices_longitude.isel(vertices=3,j=-1),
                                                    ds.vertices_longitude.isel(vertices=0,j=-1,i=-1)],dim='i')],dim='j')
    lat_b_model = xr.concat([ds.vertices_latitude.isel(vertices=2), ds.vertices_latitude.isel(vertices=1,i=-1)],dim='i')
    lat_b_model = xr.concat([lat_b_model,xr.concat([ds.vertices_latitude.isel(vertices=3,j=-1),
                                                    ds.vertices_latitude.isel(vertices=0,j=-1,i=-1)],dim='i')],dim='j')
    return lat_b_model, lon_b_model


def make_bounds_ec_earth(ds):
    '''
    EC-Earth has a discontinuity along lon = 180
    this will probably be corrected in the future and then the regridder call will 
    fail and this fix can be removed
    
    Parameters
    ----------
    ds : xarray.DataSet with latitude and longitude values, but missing boundaies

    Returns
    -------
    ds_out : xarray.DataSet with boundary values for latitude and longitude'''
    tmp = ds.vertices_longitude.isel(vertices=0).values
    arr = np.where((tmp < 20) | (tmp > 300.))
    listOfCoordinates = list(zip(arr[0], arr[1]))
    for cord in listOfCoordinates:
        if cord[1]<150:
            if tmp[cord]<20:
                tmp[cord]=tmp[cord] + 180
            if tmp[cord]>300:
                tmp[cord]=tmp[cord]-180
    test = xr.DataArray(tmp, dims=('j','i'), coords={'i':ds.i, 'j':ds.j})
    lon_b_model = xr.concat([test, ds.vertices_longitude.isel(vertices=1,i=-1)],dim='i')
    lon_b_model = xr.concat([lon_b_model,xr.concat([ds.vertices_longitude.isel(vertices=3,j=-1), ds.vertices_longitude.isel(vertices=2,j=-1,i=-1)],dim='i')],dim='j')
    lat_b_model = xr.concat([ds.vertices_latitude.isel(vertices=0), ds.vertices_latitude.isel(vertices=1,i=-1)],dim='i')
    lat_b_model = xr.concat([lat_b_model,xr.concat([ds.vertices_latitude.isel(vertices=3,j=-1), ds.vertices_latitude.isel(vertices=2,j=-1,i=-1)],dim='i')],dim='j')
    return lat_b_model, lon_b_model
    
def make_bounds_fgoals_f3_L(ds):
    '''
    Create latitude and longitude boundaries 
    Can be missing in cmorized files for some models (e.g. in IPSL-CM6A-LR)
    Parameters
    ----------
    ds : xarray.DataSet with latitude and longitude values, but missing boundaies

    Returns
    -------
    ds_out : xarray.DataSet with boundary values for latitude and longitude'''
    ny, nx = ds.lat.shape
    
    lat_b =  np.zeros((ny+1,nx))
    lat_b[0,:] =  ds.lat.values[0,:] - 0.5*ds.lat[:2,:].diff(dim='j').values
    lat_b[1:-1,:] =ds.lat.values[1:,:] - 0.5*ds.lat.diff(dim='j').values 
    lat_b[-1,:] =  ds.lat.values[-1,:] + 0.5*ds.lat[-2:,:].diff(dim='j').values
    tmp = np.column_stack((lat_b[:,-1],lat_b,lat_b[:,0]))
    lat_b = 0.5*tmp[:,1:] + 0.5*tmp[:,0:-1]
    
    lon_b =  np.zeros((ny,nx+1))
    lon_b[:,0] =  ds.lon.values[:,0] - np.squeeze(0.5*ds.lon[:,:2].diff(dim='i').values)
    lon_b[:,1:-1] =ds.lon.values[:,1:] - 0.5*ds.lon.diff(dim='i').values 
    lon_b[:,-1] =  ds.lon.values[:,-1] + np.squeeze(0.5*ds.lon[:,-2:].diff(dim='i').values)
    tmp = np.row_stack((lon_b[0,:],lon_b,lon_b[-1,:]))
    lon_b = 0.5*tmp[1:,:] + 0.5*tmp[0:-1,:]
    
    lon_b_model = xr.DataArray(lon_b, dims=('j','i'))
    lat_b_model = xr.DataArray(lat_b, dims=('j','i'))
    
    return lat_b_model, lon_b_model
    

def make_regridder_cice_gr(modelname, ds, outgrid, grid_weight_path,
                           regrid_mode = 'conservative', reuse_weights=False, periodic = False):
    ''' The first step of the regridding routine!
    This function is used for already regridded data on (lat,lon) format

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
    if 'lat_bnds' not in ds.coords and 'lat_bnds' not in ds.variables:
        ds = atmos.make_bounds(ds)
    lon = ds.lon.rename({'lon':'x'})
    lat = ds.lat.rename({'lat':'y'})
    lat_b = xr.concat([ds.lat_bnds.isel(bnds=0), ds.lat_bnds.isel(lat=-1).isel(bnds=1)],dim='lat').rename('lat_b').rename({'lat':'y_b'})
    lon_b = xr.concat([ds.lon_bnds.isel(bnds=0), ds.lon_bnds.isel(lon=-1).isel(bnds=1)],dim='lon').rename('lon_b').rename({'lon':'x_b'})
    ds_in = xr.merge([lon,lat,lon_b,lat_b])

    regridder = xe.Regridder(ds_in, outgrid, regrid_mode,
                             filename=grid_weight_path+'SImodel_to_WOA_'+regrid_mode+'.nc',
                             reuse_weights=reuse_weights, periodic = False)
    return regridder


def make_regridder_cice(modelname, ds, outgrid, grid_weight_path, 
                        regrid_mode = 'conservative', reuse_weights=False, periodic = False):
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
                             filename=grid_weight_path+'SImodel_to_'+regrid_mode+'.nc',
                             reuse_weights=reuse_weights, ignore_degenerate=True)
    return regridder

def add_bounday_info(dr,var, outgrid):
    '''
    Add boundary values (arrays) and information to the regridded DataArray
    Parameters
    ----------
    dr : xr.DataArray, regridded file
    var : str, variable name
    outgrid : xr.DataSet, outgrid used for regridding

    Returns
    -------
    dr : xr.DataArray, regridded file with boundary information
    '''
    if len(dr.lon.shape)==1:
        dr = xr.DataArray(dr.values,  coords={'time':dr.time, 
                                          'lon':dr.lon.rename({'x':'lon'}),
                                          'lat':dr.lat.rename({'y':'lat'})},
                      dims=('time','lat','lon'), name=var)
        lat_b = outgrid.lat_b.values.astype(float)
    else:
        
        dr = xr.DataArray(dr.values,  coords={'time':dr.time, 
                                              'lon':dr.lon.isel(y=0).rename({'x':'lon'}),
                                              'lat':dr.lat.isel(x=0).rename({'y':'lat'})},
                          dims=('time','lat','lon'), name=var)
        lat_b = outgrid.lat_b.isel(x_b=0).values.astype(float)
    lat_b = np.reshape(np.concatenate([lat_b[:-1],lat_b[1:]]),[2,len(dr.lat.values)]).T
    lat_b = xr.DataArray(lat_b, dims=('lat','bnds'),coords={'lat':dr.lat}).to_dataset(name='lat_bnds')
    if np.any(dr.lon.values<0):
        # make sure all longitude values go from 0 - 360
        dr = dr.assign_coords(lon=(((dr.lon + 360) % 360))).sortby('lon')
        if len(outgrid.lon.shape)==1:
            tmp = outgrid.lon_b
        else:
            tmp = outgrid.lon_b.isel(y_b=0).drop('lat_b')
        tmp[np.where(tmp<-0)] = tmp[np.where(tmp<-0)]+361
        if 'lon_b' not in tmp.coords:
            tmp = tmp.to_dataset(name = 'lon_b')
        lon_b = tmp.sortby('lon_b').lon_b.values.astype(float)
    else:
        if len(outgrid.lon.shape)==1:
            lon_b = outgrid.lon_b.values.astype(float)
        else:
            lon_b = outgrid.lon_b.isel(y_b=0).values.astype(float)
    lon_b = np.reshape(np.concatenate([lon_b[:-1],lon_b[1:]]),[2,len(dr.lon.values)]).T
    lon_b = xr.DataArray(lon_b, dims=('lon','bnds'),coords={'lon':dr.lon}).to_dataset(name='lon_bnds')
    dr = xr.merge([dr, lat_b,lon_b])
    # set attributes:
    dr.lon.attrs['long_name']='Longitude'
    dr.lon.attrs['units']='degrees_east'
    dr.lon.attrs['axis']='X'
    dr.lon.attrs['bounds']='lon_bnds'
    dr.lon.attrs['standard_name']='longitude'
    dr.lat.attrs['long_name']='Latitude'
    dr.lat.attrs['units']='degrees_north'
    dr.lat.attrs['axis']='Y'
    dr.lat.attrs['bounds']='lat_bnds'
    dr.lat.attrs['standard_name']='latitude'
    return dr

def regrid_file(ds,var, regridder, outgrid, area=None):
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
    
    if area is not None:
        print('Ocean area: %f'%(area*xr.ufuncs.isfinite(ds[var].isel(time=0))).sum())
        glb_mean = cice_area_avg(ds[var].isel(time=0), area*xr.ufuncs.isfinite(ds[var].isel(time=0)))
        print('Global mean value BEFORE regridding:%f '%np.round(glb_mean.values,6))
        glb_sum = cice_area_sum(ds[var].isel(time=0), area*xr.ufuncs.isfinite(ds[var].isel(time=0)))
        print('Global sum value BEFORE regridding:%f\n '%np.round(glb_sum.values,6))
    dr = regridder(ds[var]) # need DataArray
    print('Regridding completed\n')
    if ds.source_id=='GFDL-CM4':
       fname='/scratch/adagj/CMIP6/CLIMSENS/SIOS/'+ds.source_id + 'tmpfile.nc'
       dr.to_netcdf(fname)
       dr = xr.open_dataset(fname) 
    dr =  add_bounday_info(dr,var, outgrid)
    dr[var].attrs['long_name']=ds[var].long_name
    dr[var].attrs['units']=ds[var].units
    dr[var].attrs['standard_name']=ds[var].standard_name
    ocnarea = atmos.calc_darea(dr.isel(time=0))*xr.ufuncs.isfinite(dr[var].isel(time=0))
    ocnarea = ocnarea.where(dr.lat>-77.5,0)
    print('Regridded Ocean area: %f'%ocnarea.sum())
    glb_mean = cice_area_avg(dr[var].isel(time=0), ocnarea)
    print('Global mean value AFTER regridding:%f'%np.round(glb_mean.values,4))
    glb_sum = cice_area_sum(dr[var].isel(time=0), ocnarea)
    print('Global sum value AFTER regridding:%f '%np.round(glb_sum.values,6))
    return dr

def cice_area_avg(ds, area):
    if np.any(ds.values>1):
        '''Convert from percentage to fraction'''
        ds = ds/100
    if 'lat' in ds.dims:
        ds_out = (ds*area).sum(dim=('lat','lon'))/area.sum(dim=('lat','lon'))
    elif 'i' in ds.dims:
        ds_out =  (ds*area).sum(dim=('i','j'))/area.sum(dim=('i','j'))
    else:
        raise Exception('Cannot calculate area avrage. Please check coordinates')
    ds_out.attrs['long_name']= 'Area averaged Sea-Ice Area fraction (Ocean-Grid)'
    ds_out.attrs['units']= 'fraction [0 - 1]'
    ds_out.attrs['standard_name']='sea_ice_area_fraction'
    return ds_out

def cice_area_sum(ds, area):
    if np.any(ds.values>1):
        '''Convert from percentage to fraction'''
        ds = ds/100
    if 'lat' in ds.dims:
        ds_out = (ds*area).sum(dim=('lat','lon'))/(1E6*(1000*1000))
    elif 'i' in ds.dims:
        ds_out =  (ds*area).sum(dim=('i','j'))/(1E6*(1000*1000))
    else:
        raise Exception('Cannot calculate area sum. Please check coordinates')
    ds_out.attrs['long_name']= 'Sea-Ice Area Sum (Ocean Grid)'
    ds_out.attrs['units']= '10^12 m^2'
    ds_out.attrs['standard_name']='sea_ice_area'
    return ds_out


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
    if 'nvertices' in ds.dims and 'lat_bnds' in ds.variables:
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
    if 'nvertices' in ds.dims:
        ds = ds.rename({'nvertices':'vertices'})
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
    return ds



