#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YEAR: 2019 - 2021

@author: ADA GJERMUNDSEN

This script will reproduce FIGURE 2 in Gjermundsen et. al 2021
The data used for plotting is generated by scripts 
contained in the same folder as this (FIGURE2) 
"""
import xarray as xr
import warnings
warnings.simplefilter('ignore')
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
#%%#
    path = 'path_to_data/'
    modellist={'NorESM2-LM':['NorESM2-LM_kernel_feedbacks_timeseries.nc', path, False,
                               'black','','-',2,0 ],
               'NorESM2-MM':['NorESM2-MM_kernel_feedbacks_timeseries.nc', path, False,
                               'tab:orange','grey','-',2,0 ],
                   'CESM2':['CESM2_kernel_feedbacks_timeseries.nc', path, False,
                              'tab:blue','grey','-',2,0 ]}
#%%
    mm = .1/2.54  # millimeters in inches
    fig =  plt.figure(figsize=(180*mm, 185*mm))
    varlist={'planck_fb':['Planck','Global mean Planck feedback', '[W/m2 K]'],
             'lapserate_fb':['Lapse rate','Global mean lapse rate feedback', '[W/m2 K]'],
             'albedo_fb':['Albedo','Global mean albedo feedback', '[W/m2 K]'],
             'wv_fb':['Water vapor','Global mean water vapor feedback', '[W/m2 K]'],
             'tot_cf':['Tot Coud','Global mean tot cloud feedback', '[W/m2 K]'],
             'lw_cf':['LW Cloud','Global mean lw cloud feedback', '[W/m2 K]'],
             'sw_cf':['SW Cloud','Global mean sw cloud feedback', '[W/m2 K]']}

    fign=0
    gs = fig.add_gridspec(4, 1, hspace=0.7)
    ax = fig.add_subplot(gs[0,0]) 
    ax.set_title('Radiative feedbacks', fontsize=12)
    ax.set_ylabel('[W m$^{-2}$ K$^{-1}$]', fontsize=10)
    for expid, file in modellist.items():
        print(expid)
        lll=0
        fbsum=[]
        for var, varprops in varlist.items():
            path = file[1]
            case = xr.open_dataset(path + file[0])
            case = case[var]
            case = case.sel(year=slice(125,145)).mean(dim='year')
            ax.scatter(lll, case.values, s=100,
                       edgecolors=file[3], facecolors='none' )
            if var not in ['sw_cf','lw_cf']: # only use tot_cf for the cloud contribution to the net
                print(var)
                fbsum.append(case.values)
            lll=lll+1
        ax.scatter(lll,np.sum(np.array(fbsum)), s=100, edgecolors=file[3], facecolors='none', label=expid)
    ax.set_xticks(np.arange(0,8))
    ax.set_xticklabels(['Planck','LR','Albedo','WV','Tot Cloud','LW Cloud','SW Cloud','Net']) 
    ax.set_ylim(-4.5,2.5)
    ax.plot(np.full((len(np.arange(-4,3.,.5))),4.5), np.arange(-4,3.,.5),
             linestyle = '--', color = 'black', linewidth = 1)
    ax.plot(np.full((len(np.arange(-4,3.,.5))),6.5), np.arange(-4,3.,.5),
             linestyle = '--', color = 'black', linewidth = 1)
    ax.annotate(r"$\bf{a)}$"+ '\n', xy=(-.35, 1.98) , fontsize=10, weight = 'bold')
    ax.grid()
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10) 
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
    
#%%

    varlist={'sw_cf':['SW Cloud','SW cloud feedback', '[W/m2K]']}
    
    for i in [0, 2, 3]:
        if i==0:
            ax = fig.add_subplot(gs[1,0]) 
        elif i==2:
            ax = fig.add_subplot(gs[2,0]) 
        elif i==3:
            ax = fig.add_subplot(gs[3,0]) 
        for var, varprops in varlist.items():
            for expid, file in modellist.items():
                path = file[1]
                case = xr.open_mfdataset(path + file[0])
                casetemp = case.dtas
                case = case[var]
                if i==0:
                    print(expid)
                   
                    ax.plot(case.year.values,case.values,
                              color=file[3], linestyle = file[5], linewidth = file[6], label = expid)
                    ax.set_ylabel('[W m$^{-2}$ K$^{-1}$]', fontsize=10)
                    ax.set_title('SW cloud feedback', fontsize=12)
                    ax.set_xlim(0, 500)
                    ax.set_ylim(-.1,1.)
                    ax.set_xlabel('Time [years]', fontsize=10)
                    ax.annotate(r"$\bf{b)}$" + '\n', xy=(2.1,1.) , fontsize=12, weight = 'bold')
                elif i==2:
                    ax.plot(casetemp.year.values,casetemp.values,
                              color=file[3], linestyle = file[5], linewidth = file[6], label = expid)
                    ax.set_ylabel('[K]', fontsize=12)
                    ax.set_title('Surface (2m) air temperature change', fontsize=12)
                    ax.set_xlim(0, 500)
                    ax.set_ylim(2,10.1)
                    ax.set_xlabel('Time [years]', fontsize=10)
                    ax.annotate(r"$\bf{c)}$" + '\n', xy=(2.1,9.5) , fontsize=10, weight = 'bold')
                else:
                    ax.plot(casetemp.values, case.values,
                              color=file[3], marker = 'o', linestyle = '', markersize=4,
                              label = expid, zorder = 0)
                    ax.scatter(casetemp.sel(year=slice(125,145)).mean(dim='year').values,
                               case.sel(year=slice(125,145)).mean(dim='year').values,
                               s=150, linewidth=1, edgecolors=file[3], facecolors='none', zorder =30)
                    ax.set_ylim(-.1,1.)
                    ax.set_xlim(2,10.1)
                    ax.set_title('SW cloud feedback', fontsize=12)
                    ax.set_ylabel('[W m$^{-2}$ K$^{-1}$]', fontsize=10)
                    ax.set_xlabel('Surface (2m) air temperature change [K]', fontsize=10)
                    ax.annotate(r"$\bf{d)}$" + '\n', xy=(2.1,1.) , fontsize=10, weight = 'bold')
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10) 
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
            ax.grid()
            if i ==3:
                ax.legend(fontsize=9)
    plt.subplots_adjust(left=0.12, bottom=.08, right=0.97, top=.95)
    plt.savefig('path_to_save_figures_folder/' + 'FIGURE2.pdf')    
