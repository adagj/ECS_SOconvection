#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YEAR: 2019 - 2021

@author: ADA GJERMUNDSEN

This script will reproduce FIGURE 4 in Gjermundsen et. al 2021
The data used for plotting is generated by scripts 
contained in the same folder as this (FIGURE4) 
"""

import xarray as xr
import warnings
warnings.simplefilter('ignore')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean.cm as cmo
from matplotlib import colors
from scipy.stats import linregress
import scipy.ndimage

def linearfit(ds):
    t = linregress(np.arange(0,ds.shape[0]),ds.values)
    val = t.slope*np.arange(0,ds.shape[0]) + t.intercept
    return val

if __name__ == '__main__':

    mm = .1/2.54  # millimeters in inches
    fig =  plt.figure(figsize=(180*mm, 185*mm))
    gs = fig.add_gridspec(8, 2*12)
    
#############################################################################################################
    path = '/home/adag/Documents/NorESM2/analysis/ECS/CMIP6data/Omon/temp_slices/zm2/'
    modellist={'NorESM2-MM':['thetao_Omon_NorESM2-MM_abrupt-4xCO2_r1i1p1f1_150_30yravg.nc',
                             'thetao_Omon_NorESM2-MM_piControl_r1i1p1f1_150_30yravg.nc'],
                'NorESM2-LM':['thetao_Omon_NorESM2-LM_abrupt-4xCO2_r1i1p1f1_150_30yravg.nc',
                             'thetao_Omon_NorESM2-LM_piControl_r1i1p1f1_150_30yravg.nc',
                             'thetao_Omon_NorESM2-LM_abrupt-4xCO2_r1i1p1f1_500_30yravg.nc',
                             'thetao_Omon_NorESM2-LM_piControl_r1i1p1f1_500_30yravg.nc'],
               'CESM2':['thetao_Omon_CESM2_abrupt-4xCO2_r1i1p1f1_150_30yravg.nc',
                        'thetao_Omon_CESM2_piControl_r1i1p1f1_150_30yravg.nc',
                       'thetao_Omon_CESM2_abrupt-4xCO2_r1i1p1f1_500_30yravg.nc',
                        'thetao_Omon_CESM2_piControl_r1i1p1f1_500_30yravg.nc']}
#%%##########################################################################
    pvlim=[-2, 4]
    cvlim=[-2, 8]
    reg = '_50S'
    n=-1
###################################################################################################################
    for expid,files in modellist.items():
        n=n+1
        ax = fig.add_subplot(gs[0:5, 2:5*2])
        ax.set_title(r"$\bf{a)}$" +' Temperature change (50°S - 90°S)', fontsize = 10)
        case = xr.open_dataset(path + files[0])
        case_ctrl = xr.open_dataset(path + files[1])
        change = case['thetao' + reg].squeeze()   - case_ctrl['thetao' + reg].squeeze() 
        if expid in ['CESM2', 'NorESM2-LM']:
            case500 = xr.open_mfdataset(path + files[2])
            case_ctrl500 = xr.open_mfdataset(path + files[3])
            change500 = case500['thetao' + reg].squeeze()   - case_ctrl500['thetao' + reg].squeeze() 
        if expid == 'CESM2':
            plt.plot(change.values, change.lev.values, color = 'tab:blue', 
                     label = expid+ '  (150 yrs)', linewidth = 2)
            plt.plot(change500.values, change500.lev.values, color = 'lightskyblue', linestyle='-', 
                     linewidth = 2, label = expid + '  (500 yrs)')
        elif expid == 'NorESM2-LM':
            plt.plot( change.values,change.lev.values,color = 'black', 
                     label = expid + '   (150 yrs)', linewidth = 2)
            plt.plot(change500.values, change.lev.values, color = 'grey', 
                     linewidth = 2, linestyle='-',label = expid + '   (500 yrs)')
        else:
            plt.plot( change.values, change.lev.values,color = 'tab:orange', 
                     label = expid + '  (150 yrs)', linewidth = 2)
        ax.invert_yaxis()
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10) 
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(10) 
        ax.set_xlabel('Temperature change [K]', fontsize=10)
        ax.set_ylabel('Depth [km]', fontsize=10)
        ax.set_ylim(5500,0)   
        ax.set_xlim(-.5,10)
        ax.set_yticklabels(['0', '1', '2', '3' ,'4','5'])
        plt.grid()
        plt.legend(loc='lower right',facecolor="white", framealpha=1, fontsize=8)
#%%################################################################################################################
   
    modellist={'NorESM2-LM':['thetao_Omon_NorESM2-LM_abrupt-4xCO2_r1i1p1f1_500_30yravg.nc',
                             'thetao_Omon_NorESM2-LM_piControl_r1i1p1f1_500_30yravg.nc'],
               'CESM2':['thetao_Omon_CESM2_abrupt-4xCO2_r1i1p1f1_500_30yravg.nc',
                        'thetao_Omon_CESM2_piControl_r1i1p1f1_500_30yravg.nc']}
    #%## GLOBAL
    var = 'thetao'
    cmap = cm.seismic
    cmap=cmo.balance
    cval=[-2,10]
    
    levels=np.arange(cval[0],cval[1]+1,.5)
    n=6
    nn=1
    dl=2
    for expid, file in modellist.items():

        if expid=='CESM2':
            ax = fig.add_subplot(gs[3:5, 2*n:-1])
        else: 
            ax = fig.add_subplot(gs[0:2, 2*n:-1])
        case = xr.open_mfdataset(path +file[0])
        case=case[var].squeeze()
        ctrl = xr.open_mfdataset(path  + file[1] )
        ctrl=ctrl[var].squeeze()
        casevals = case.values - ctrl.values
        x,y=np.meshgrid(case.lat.values,case.lev.values)
        if expid=='CESM2':
            ax.annotate('c)\n', xy=(-82,100) , fontsize=10, weight = 'bold')
        else:
            ax.annotate('b)\n', xy=(-82,100) , fontsize=10, weight = 'bold')
        plt.contourf(x,y,casevals,levels = levels , cmap=cmap, 
                     norm =  colors.DivergingNorm(vmin=cval[0], vcenter=0., vmax=cval[1]), extend='max')
        ax.invert_yaxis()
        if nn==2:
            cbar = plt.colorbar( ticks=levels[::4])
            cbar.set_label(label='Temperature change [K]', fontsize=10)
            cbar.ax.tick_params(labelsize=12) 
        if nn==1:
            ax.set_ylabel('Depth [km]', fontsize=10)
            ax.set_yticklabels(['0', '1', '2', '3' ,'4','5'])
        ax.set_ylim(5500,0)
        ax.set_title(expid +' (500 years)', fontsize=10)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10) 
        if nn==2:
            ax.set_yticklabels(['']*len(case.lev.values))
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
        ax.set_xlabel('Latitude [degrees north]', fontsize=10)
        ax.set_facecolor('gainsboro')
        ax.set_xlim(-83,83)

    cax = fig.add_axes([0.9, 0.55, 0.018, 0.25])
    cb = plt.colorbar(cax=cax,ticks=levels[::2], orientation='vertical' )
    cb.set_label(label='Temperature change [K]', fontsize=10)
    cb.ax.tick_params(labelsize=10) 
    
#################################################################################################################
    nn=0
    syear=121
    syear = syear -1
    nyears = 29
    ddepth = 1700
    ddlat = -50
    cmap = cmo.balance_r
    var = 'rho'
    mocval = [-7, 7]
    moclevels=np.arange(mocval[0],mocval[1]+1,1)
    cval=[-1,1]
    rholevels=np.arange(cval[0],cval[1],.1)
    n = {'piControl':1,'abrupt-4xCO2':0}
    var = 'msftmzmpa'
    expid_a = 'piControl'
    expid_b = 'abrupt-4xCO2'


    modellist={'CESM2':['rho_Omon_CESM2_abrupt-4xCO2_r1i1p1f1_150_pref_surface_mdeos.30yravg.nc',
                        'rho_Omon_CESM2_piControl_r1i1p1f1_150_pref_surface_mdeos.30yravg.nc','tab:blue'],
               'NorESM2-LM':['rho_Omon_NorESM2-LM_abrupt-4xCO2_r1i1p1f1_150_pref_surface.30yravg.nc',
                       'rho_Omon_NorESM2-LM_piControl_r1i1p1f1_150_pref_surface.30yravg.nc','k']}
    for model in [ 'CESM2','NorESM2-LM']:
        moc_a = xr.open_dataset(path + var + '_Omon_'+ model + '_' + expid_a+'_r1i1p1f1_150_30yravg.nc')
        moc_b = xr.open_dataset(path + var + '_Omon_'+ model + '_' + expid_b+'_r1i1p1f1_150_30yravg.nc')
        
        temp_a=xr.open_dataset(path + 'thetao_Omon_'+ model + '_' + expid_a+'_r1i1p1f1_150_30yravg.nc')
        temp_b=xr.open_dataset(path + 'thetao_Omon_'+ model + '_' + expid_b+'_r1i1p1f1_150_30yravg.nc')
        tempdiff =  temp_b['thetao'] - temp_a['thetao'] 
        if model in ['CESM2']:
             mocs_a = xr.open_dataset(path + 'msftmzsmpa' + '_Omon_'+ model + '_' + expid_a+'_r1i1p1f1_150_30yravg.nc')
             mocs_b = xr.open_dataset(path + 'msftmzsmpa' + '_Omon_'+ model + '_' + expid_b+'_r1i1p1f1_150_30yravg.nc')
        
        if model in ['CESM2']:
            moc_a = moc_a.assign_coords(lev = moc_a.lev.values/100.)
            moc_b = moc_b.assign_coords(lev = moc_b.lev.values/100.)
            mocs_a = mocs_a.assign_coords(lev = mocs_a.lev.values/100.)
            mocs_b = mocs_b.assign_coords(lev = mocs_b.lev.values/100.)
            moc = (mocs_b['msftmzsmpa'] - mocs_a['msftmzsmpa']) +(moc_b[var] - moc_a[var])
        else:
            moc = moc_b[var] - moc_a[var]
        moc = moc.sel(lat=slice(-84,ddlat))
        moc = moc.sel(lev=slice(0,ddepth+500))
        x, y = np.meshgrid(moc.lat, moc.lev.values/1000)
        z = moc.values/1e9
        
        ctrl_a = xr.open_dataset(path +  modellist[model][n[expid_a]])
        ctrl_b = xr.open_dataset(path +  modellist[model][n[expid_b]])
        
        ctrl_a = ctrl_a['rho']
        ctrl_a = ctrl_a.sel(lat=slice(-75.5,ddlat+5))
        ctrl_a = ctrl_a.sel(lev=slice(0,ddepth))
        
        ctrl_b = ctrl_b['rho']
        ctrl_b = ctrl_b.sel(lat=slice(-75.5,ddlat+5))
        ctrl_b = ctrl_b.sel(lev=slice(0,ddepth))
        ctrl = ctrl_b - ctrl_a
        dx, dy = np.meshgrid(ctrl.lat.values, ctrl.lev.values/1000)
        d = ctrl.values
        
        tempdiff = tempdiff.sel(lat=slice(-76.5,ddlat+5))
        tempdiff = tempdiff.sel(lev=slice(0,ddepth+210))
        xt, yt = np.meshgrid(tempdiff.lat.values, tempdiff.lev.values/1000)
        
        if model=='CESM2':
            ax = fig.add_subplot(gs[6:8, 2:12])
            ax.annotate('d)\n', xy=(-74,0) , fontsize=10, weight = 'bold')
        if model=='NorESM2-LM':
            ax = fig.add_subplot(gs[6:8, 13:-1])
            ax.annotate('e)\n', xy=(-74,0) , fontsize=10, weight = 'bold')
        cf = ax.contourf(x, y, z, levels=moclevels, cmap=cmo.balance,
                     norm =  colors.DivergingNorm(vmin=mocval[0], vcenter=0., vmax=mocval[1]), extend='both')
        kmd = 1e-3
        if model in ['CESM2']:
            CS = ax.contour(xt, yt, tempdiff.values, levels=np.arange(1,2), colors = 'red', linewidths = 2)
            CS = ax.contour(dx,dy, d, levels=rholevels,  colors = 'k', linewidths = .5)     
            arrowprops = dict(facecolor='royalblue', lw=1., edgecolor='k',
                                        alpha=0.7, width=3, headwidth=8)
            ax.annotate("", xy=(-64.9, kmd*1000), xytext=(-64.9, kmd*600), 
                        arrowprops=arrowprops)
            ax.annotate("", xy=(-60., kmd*600), xytext=(-60., kmd*1000), 
                        arrowprops=arrowprops)
            arrowprops = dict(facecolor='k', lw=.1, edgecolor='k', alpha=0.7, width=3, headwidth=8)
            ax.annotate("", xy=(-70., kmd*190), xytext=(-67.5, kmd*205), 
                        arrowprops=arrowprops)
            ax.annotate("", xy=(-58., kmd*600), xytext=(-56.1, kmd*720), 
                        arrowprops=arrowprops)
            ax.annotate("", xy=(-58., kmd*900), xytext=(-56.5, kmd*1020), 
                        arrowprops=arrowprops)
            
        else:
            tempdata = scipy.ndimage.zoom(tempdiff.values, 3)
            xt = scipy.ndimage.zoom(xt, 3)
            yt = scipy.ndimage.zoom(yt, 3)
            CS = ax.contour(xt, yt, tempdata, levels=np.arange(1,2), colors = 'red', linewidths = 2)
            CS = ax.contour(dx,dy, d, levels=rholevels,  colors = 'k', linewidths = .5) 
            arrowprops = dict(facecolor='royalblue', lw=1.,edgecolor='k',
                              alpha=0.7, width=3, headwidth=8)
            ax.annotate("", xy=(-60, kmd*1000), xytext=(-60, kmd*600), 
                        arrowprops=arrowprops)
            ax.annotate("", xy=(-52, kmd*600), xytext=(-52, kmd*1000),
                        arrowprops=arrowprops)
            arrowprops = dict(facecolor='k', lw=.1, edgecolor='k', alpha=0.7, width=3, headwidth=8)
            ax.annotate("", xy=(-70., kmd*670), xytext=(-67.5, kmd*670), 
                        arrowprops=arrowprops)
            ax.annotate("", xy=(-70., kmd*320), xytext=(-67.5, kmd*320), 
                        arrowprops=arrowprops)
            ax.annotate("", xy=(-62., kmd*1000), xytext=(-60.8, kmd*1150), 
                        arrowprops=arrowprops)
            ax.annotate("", xy=(-62.5, kmd*400), xytext=(-60.9, kmd*530), 
                        arrowprops=arrowprops)
            
        
        ax.set_title(model + ' (150 years)', fontsize=10)
        ax.set_xlabel('Latitude [degrees north]', fontsize=10)
        ax.set_xlim(-75,ddlat)
        ax.set_ylim(0,(ddepth+100)/1000)
        ax.invert_yaxis()
        if nn==0:
            ax.set_ylabel('Depth [km]', fontsize=10)
        if nn==1:
            ax.set_yticklabels(['']*len(moc.lev.values))
            cax = fig.add_axes([0.9, 0.08, 0.018, 0.2])
            cb = plt.colorbar(cf, cax=cax,ticks=moclevels[::2], orientation='vertical' )
            cb.set_label(label='Eddy-induced MOC [Sv]', fontsize=10)
            cb.ax.tick_params(labelsize=11) 
        nn = nn+1
    
   #
    plt.subplots_adjust(left=0.01, bottom=0.07, right=0.9, top=.95, wspace=0., hspace=0.)
    plt.savefig('path_to_save_figures_folder/' + 'FIGURE4.pdf')    
