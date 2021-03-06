#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YEAR: 2019 - 2021

@author: ADA GJERMUNDSEN

This script will reproduce FIGURE 3 in Gjermundsen et. al 2021
The data used for plotting is generated by scripts 
contained in the same folder as this (FIGURE3) 
"""
import xarray as xr
import warnings
warnings.simplefilter('ignore')
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from matplotlib import colors

if __name__ == '__main__':
#%############### CESM INCLUDED #############################################
    path = 'path_to_data'
    modellist={'NorESM2-LM':['NorESM2-LM_kernel_feedbacks_zm_150.nc', path, False,
                               'black','','-',2,0 ],
               'NorESM2-MM':['NorESM2-MM_kernel_feedbacks_zm_150.nc', path, False,
                               'tab:orange','grey','-',2,0 ],
                   'CESM2':['CESM2_kernel_feedbacks_zm_150.nc', path, False,
                              'tab:blue','grey','-',2,0 ]}
#%%
    fig = plt.figure(figsize=(9, 11))
    gs = fig.add_gridspec(13, 4)
    for expid, file in modellist.items():
        ax = fig.add_subplot(gs[7:10, :])
        path = file[1]
        case = xr.open_mfdataset(path + file[0])
        case = case['sw_cf_zm'].squeeze()
        ax.plot(case.lat.values,case.values,
                  color=file[3], linestyle = file[5], linewidth = file[6], label = expid+ '  150 years')
        ax.plot(case.lat.values, np.zeros(len(case.lat.values)),'k--', linewidth=.5)
        ax.set_ylabel('SW cloud feedback\n[W m$^{-2}$ K$^{-1}$]', fontsize=12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        ax.set_xlim(-83,83)
        ax.set_ylim(-5,5)
    ax.annotate('e)', xy=(-80,4.) , fontsize=12, weight = 'bold')
    plt.legend(loc='upper right', bbox_to_anchor = (1.33,1.03),
         facecolor="white", framealpha=1, fontsize=12)
    
#%  
   
    modellist={'CESM2 500 years':['CESM2_kernel_feedbacks_zm_500.nc', path, False,
                             'lightskyblue','grey','-',2,0 ],
        'NorESM2-LM 500 years':['NorESM2-LM_kernel_feedbacks_zm_500.nc', path, False,
                               'grey','','-',2,0 ] }
       
    ax = fig.add_subplot(gs[10:, :])
    ax.plot(case.lat.values,case.values,
                  color=file[3], linestyle = file[5], linewidth = file[6], label = expid + ' 150 years')
    for expid, file in modellist.items():
        path = file[1]
        case = xr.open_mfdataset(path + file[0])
        case=case['sw_cf_zm'].squeeze()
        ax.plot(case.lat.values,case.values,
                  color=file[3], linestyle = file[5], linewidth = file[6], label = expid)
        ax.plot(case.lat.values, np.zeros(len(case.lat.values)),'k--', linewidth=.5)
        ax.set_ylabel('SW cloud feedback\n[W m$^{-2}$ K$^{-1}$]', fontsize=12)
        ax.set_xlabel('Latitude [degrees north]', fontsize=12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        ax.set_xlim(-83,83)
        ax.set_ylim(-5,5)
    ax.annotate('f)', xy=(-80,4.) , fontsize=12, weight = 'bold')        
    plt.legend(loc='upper right', bbox_to_anchor = (1.33,1.03),
         facecolor="white", framealpha=1, fontsize=12)
    
#%%
    path = 'path_to_cloud_data/'
    modellist={ 
    'CESM2':['cl_90S_35S_CESM2_abrupt-4xCO2.nc',
                       'cl_90S_35S_CESM2_piControl.nc', path,
                              'blue','-',2],
             'NorESM2-LM':['cl_90S_35S_NorESM2-LM_abrupt-4xCO2.nc',
                          'cl_90S_35S_NorESM2-LM_piControl.nc' , path,
                               'black','-',2]}

#%%##########################################################################
    scenario='change'
    cvlim=[-16, 17]
    levels= np.arange(cvlim[0], cvlim[1],2)
    cm_cmap=cmo.balance
    ocn_cmap = cmo.amp
#%%############### CESM INCLUDED #############################################       
    
 
    n=0
    var = 'cl_35S'
    for expid, caseprops in modellist.items():
        if expid=='NorESM2-LM':
            ax = fig.add_subplot(gs[0:4, 2:4])
        else:
            ax = fig.add_subplot(gs[0:4, 0:2])
        case_abrupt = xr.open_dataset(path + caseprops[0])
        case_abrupt= case_abrupt[var].squeeze()    
        case_ctrl = xr.open_dataset(path + caseprops[1])
        case_ctrl = case_ctrl[var].squeeze()
        # due to 500 years branchtime, the years differ. 
        # Need to make new array
        case = xr.DataArray((case_abrupt.values - case_ctrl.values),
                            coords={'year': case_abrupt.year, 'lev':case_abrupt.lev},
                                         dims = ('year','lev'),
                                         name = var)
        if expid == 'CESM2':
            x,y=np.meshgrid(case.year.values, -case.lev.values)
        else:
            case = case.assign_coords(lev = 1000*case.lev)
            x,y=np.meshgrid(case.year.values, case.lev.values)
        plt.contourf(x,y,case.values.T,levels = levels, cmap=cm_cmap, 
                      norm =  colors.DivergingNorm(vmin=cvlim[0], vcenter=0., vmax=cvlim[1]),
                      extend="both")
        plt.clim(cvlim[0],cvlim[1])
        ax.set_xlabel('Time [years]', fontsize=12)
        if expid == 'NorESM2-LM':
            print('hey hey')
            ax.annotate('b)\n', xy=(10,50) , fontsize=12, weight = 'bold')
            ax.set_yticklabels(['']*len(case.lev.values))
            ax.set_ylabel('')
        else:
            ax.annotate('a)\n', xy=(10,50) , fontsize=12, weight = 'bold')
            ax.set_ylabel('Height [hPa]', fontsize=12)
        ax.invert_yaxis()
        ax.set_title(expid, fontsize=12)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        ax.set_xlabel('')
        ax.set_xticklabels(['']*len(case.year.values))
        ax.set_xlim(1,500)
       
        if expid == 'CESM2':
            cbar_ax = fig.add_axes([0.85, 0.695, 0.02, 0.25])
            cbar = plt.colorbar(cax=cbar_ax)
            cbar.set_label(label='Cloud cover change [%]', fontsize=12)
            cbar.ax.tick_params(labelsize=12)
   
        #################### OCEAN ##########################################
    path = 'path_to_ocean_data/'
    modellist={ 
    'CESM2':['tos_areaavg_Omon_CESM2_abrupt-4xCO2_r1i1p1f1_timeseries.nc',
                       'tos_areaavg_Omon_CESM2_piControl_r1i1p1f1_timeseries.nc', path,
                              'blue','-',2],
             'NorESM2-LM':['tos_areaavg_Omon_NorESM2-LM_abrupt-4xCO2_r1i1p1f1_timeseries.nc',
                          'tos_areaavg_Omon_NorESM2-LM_piControl_r1i1p1f1_timeseries.nc' , path,
                               'black','-',2]}   
    var = 'sst_35S'
    for expid, caseprops in modellist.items():
            
        if expid=='NorESM2-LM':
            ax = fig.add_subplot(gs[4:6, 2:4])
        else:
            ax = fig.add_subplot(gs[4:6, 0:2])
        case = xr.open_mfdataset(path + caseprops[0])
        case= case[var].squeeze()    
        case_ctrl = xr.open_mfdataset(path + caseprops[1])
        case_ctrl = case_ctrl[var].squeeze()
        if scenario=='change':
            case = xr.DataArray((case.values - case_ctrl.values) ,
                            coords={'year': case.year},
                                          dims = 'year',
                                          name = var)
            case.attrs = case_ctrl.attrs
            case.attrs['units'] = 'K'
            
        if expid == 'NorESM2-LM':
            plt.plot(case.year.values, case.values, 'k')
            ax.annotate('d)\n', xy=(10,9.5) , fontsize=12, weight = 'bold')
            ax.set_yticklabels(['']*len(case.values))
            ax.set_ylabel('')
        else:
            if expid == 'CESM2':
                plt.plot(case.year.values, case.values, 'tab:blue')
                ax.annotate('c)\n', xy=(10,9.5) , fontsize=12, weight = 'bold')
                ax.set_ylabel('SST anomaly [K]', fontsize=12)
        ax.set_xlabel('Time [years]', fontsize=12)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12) 
        ax.set_xlabel('Time [years]')
        ax.set_xlim(1,500)
        ax.set_ylim(0,10)
        plt.grid()
        n=n+1
    plt.subplots_adjust(left=0.12, bottom=0.05, right=0.78, top=.95, hspace=0.6)
    plt.savefig('path_to_save_figures_folder/' + 'FIGURE3.pdf')    
    
