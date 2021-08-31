#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:08:18 2020

@author: adag
"""
import sys
sys.path.insert(1, '/scratch/adagj/CMIP6/CLIMSENS/CMIP6_UTILS')
import CMIP6_ATMOS_UTILS as atmos
import glob
import numpy as np
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from matplotlib import cm

def Ofx_files(self, var='deptho', path_to_data = '/projects/NS9252K/ESGF/CMIP6/'):
    models={'BCC-CSM2-MR': {'ofxpath':path_to_data + '/' + 'ScenarioMIP'  + '/' + self.institute + '/' + self.name + '/ssp370/' + self.realiz + '/Ofx/' + var + '/gn/latest/',
                             'ofxfile': var + '_Ofx_BCC-CSM2-MR_ssp370_r1i1p1f1_gn.nc'},
            'BCC-ESM1':{'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/' + self.name + '/1pctCO2/' + self.realiz + '/Ofx/' + var + '/gn/latest/',
                             'ofxfile': var + '_Ofx_BCC-ESM1_1pctCO2_r1i1p1f1_gn.nc'},
            'CAMS-CSM1-0':{'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/' + self.name + '/1pctCO2/r2i1p1f1/Ofx/' + var + '/gn/latest/',
                             'ofxfile': var + '_Ofx_CAMS-CSM1-0_1pctCO2_r2i1p1f1_gn.nc'},
            'E3SM-1-0':{'ofxpath':'/projects/NS9252K/ESGF/CMIP6/CMIP/NCAR/CESM2/piControl/r1i1p1f1/Ofx/'+var +'/gr/latest/', 'ofxfile':var + '_Ofx_CESM2_piControl_r1i1p1f1_gr.nc'},
            'EC-Earth3-Veg':{'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/' + self.name + '/historical/r5i1p1f1/Ofx/' + var + '/gn/latest/', 
                             'ofxfile': var + '_Ofx_EC-Earth3-Veg_historical_r5i1p1f1_gn.nc'},
            'FGOALS-f3-L':{'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/' + self.name + '/historical/r1i1p1f1/Ofx/' + var + '/gn/latest/', 
                             'ofxfile': var + '_Ofx_FGOALS-f3-L_historical_r1i1p1f1_gn.nc'},
            # The GFDL models share the same regridded grid (will only work for gr files)
            'GFDL-ESM4': {'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/GFDL-CM4/piControl/r1i1p1f1/Ofx/' + var + '/gr/latest/', 
                             'ofxfile': var + '_Ofx_GFDL-CM4_piControl_r1i1p1f1_gr.nc'},
            'GISS-E2-2-G':{'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/GISS-E2-1-G/piControl/' + self.realiz+ '/Ofx/' + var + '/gn/latest/', 
                             'ofxfile': var + '_Ofx_GISS-E2-1-G_piControl_' + self.realiz +'_gn.nc'},
            'INM-CM4-8':{'ofxpath':'', 'ofxfile':''},
            'INM-CM5-0':{'ofxpath':'', 'ofxfile':''},
            'IPSL-CM6A-LR':{'ofxpath':'', 'ofxfile':''},
            'MIROC6': {'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/' + self.name + '/abrupt-4xCO2/r1i1p1f1/Ofx/' + var + '/gn/latest/', 
                             'ofxfile': var + '_Ofx_' + self.name +'_abrupt-4xCO2_r1i1p1f1_gn.nc'},
            'MRI-ESM2-0':{'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/' + self.name + '/abrupt-4xCO2/r1i1p1f1/Ofx/' + var + '/gn/latest/', 
                             'ofxfile': var + '_Ofx_' + self.name +'_abrupt-4xCO2_r1i1p1f1_gn.nc'},
            'CMCC-ESM2': {'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/' + self.name + '/abrupt-4xCO2/r1i1p1f1/Ofx/' + var + '/gn/latest/',
                             'ofxfile': var + '_Ofx_' + self.name +'_abrupt-4xCO2_r1i1p1f1_gn.nc'},
            'HadGEM3-GC31-LL':{'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/' + self.name + '/piControl/r1i1p1f1/Ofx/' + var + '/gn/latest/',
                             'ofxfile': var + '_Ofx_' + self.name +'_piControl_r1i1p1f1_gn.nc'},
            'HadGEM3-GC31-MM':{'ofxpath':path_to_data + '/' + 'CMIP'  + '/' + self.institute + '/' + self.name + '/piControl/r1i1p1f1/Ofx/' + var + '/gn/latest/',
                             'ofxfile': var + '_Ofx_' + self.name +'_piControl_r1i1p1f1_gn.nc'},
            # NESM3 does not provide Ofx deptho, but uses the same ocean model as EC-Earth
            'NESM3':{'ofxpath':'/projects/NS9252K/ESGF/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3-Veg/historical/r5i1p1f1/Ofx/' + var + '/gn/latest/', 
                             'ofxfile': var + '_Ofx_EC-Earth3-Veg_historical_r5i1p1f1_gn.nc'}}

    if self.name in models.keys():
         if self.name in ['E3SM-1-0'] and var=='sftof':
             self.ofxfile = ''
         else:   
             self.ofxfile = models[self.name]['ofxpath'] + models[self.name]['ofxfile']
    elif self.name in ['NorESM2-LM', 'NorESM2-MM']:
         ofxpath = path_to_data + '/' + self.activity_id  + '/' + self.institute + '/' + self.name + '/piControl/' + self.realiz + '/Ofx/' + var + '/gn/latest/'
         self.ofxfile = ofxpath + var + '_Ofx_' + self.name + '_piControl_'+ self.realiz + '_gn.nc'
    elif self.name in ['EC-Earth3']:
         ofxpath = path_to_data + '/' + self.activity_id  + '/' + self.institute + '/' + self.name + '/piControl/r1i1p1f1/Ofx/' + var + '/' + self.gridlabel+ '/latest/'
         self.ofxfile = ofxpath + var + '_Ofx_' + self.name + '_piControl_r1i1p1f1_' + self.gridlabel +'.nc' 
    else:
         ofxpath = path_to_data + '/' + self.activity_id  + '/' + self.institute + '/' + self.name + '/piControl/' + self.realiz + '/Ofx/' + var + '/' + self.gridlabel+ '/latest/'
         self.ofxfile = ofxpath + var + '_Ofx_' + self.name + '_piControl_' + self.realiz + '_' + self.gridlabel +'.nc' 

def ecs_models_cmip6():
    models={'ACCESS-CM2':{'institute':'CSIRO-ARCCSS', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'ACCESS-ESM1-5':{'institute':'CSIRO', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'AWI-CM-1-1-MR':{'institute':'AWI', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':250},
          'BCC-CSM2-MR':{'institute':'BCC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'BCC-ESM1':{'institute':'BCC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'CMCC-CM2-SR5': {'institute':'CMCC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'CMCC-ESM2': {'institute':'CMCC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'CAMS-CSM1-0': {'institute':'CAMS', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':130},
          'CAS-ESM2-0':{'institute':'CAS', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'CanESM5':{'institute':'CCCma', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1','r1i1p2f1'],'branch_yr':0},
          'CESM2':{'institute':'NCAR', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn', 'gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':501},
          'CESM2-FV2':{'institute':'NCAR', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':320},
          'CESM2-WACCM':{'institute':'NCAR', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn', 'gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':69},
          'CESM2-WACCM-FV2':{'institute':'NCAR', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':300},
          'CNRM-CM6-1':{'institute':'CNRM-CERFACS', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gn','gr1'], 'variant_labels':['r1i1p1f2'],'branch_yr':0},
          'CNRM-CM6-1-HR':{'institute':'CNRM-CERFACS', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f2'],'branch_yr':0},
          'CNRM-ESM2-1':{'institute':'CNRM-CERFACS', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gn','gr1'], 'variant_labels':['r1i1p1f2'],'branch_yr':0},
          'E3SM-1-0':{'institute':'E3SM-Project', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':100},
    #     # EC-Earth3:  piControl: r1i1p1f1. abrupt-4xCO2: r3i1p1f1. 
    #     # abrupt-4xCO2 har different variant label from it's parent; r1i1p1f1
          'EC-Earth3':{'institute':'EC-Earth-Consortium', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1','r3i1p1f1'],'branch_yr':0},
          'EC-Earth3-AerChem':{'institute':'EC-Earth-Consortium', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'EC-Earth3-Veg':{'institute':'EC-Earth-Consortium', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'FGOALS-f3-L':{'institute':'CAS', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'FIO-ESM-2-0':{'institute':'FIO-QLNM', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'GFDL-CM4':{'institute':'NOAA-GFDL', 'grid_label_atmos':['gr1'],  'grid_label_ocean':['gn','gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':100},
          'GFDL-ESM4':{'institute':'NOAA-GFDL', 'grid_label_atmos':['gr1'],  'grid_label_ocean':['gn','gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':100},
          'GISS-E2-1-G':{'institute':'NASA-GISS', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p3f1','r1i1p1f1'],'branch_yr':0},
          'GISS-E2-1-H':{'institute':'NASA-GISS', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1','r1i1p3f1'],'branch_yr':0},
          'GISS-E2-2-G':{'institute':'NASA-GISS', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1','r1i1p1f1'],'branch_yr':0},
    #     #  HadGEM3-GC31-LL and HadGEM3-GC31-MM: piControl: r1i1p1f1. abrupt-4xCO2: r3i1p1f1. 
    #     # abrupt-4xCO2 har different variant label from it's parent; r1i1p1f1
          'HadGEM3-GC31-LL':{'institute':'MOHC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1','r1i1p1f3'],'branch_yr':0},
          'HadGEM3-GC31-MM':{'institute':'MOHC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1','r1i1p1f3'],'branch_yr':0},
          'INM-CM4-8':{'institute':'INM', 'grid_label_atmos':['gr1'],  'grid_label_ocean':['gr1'], 'variant_labels':['r1i1p1f1'],'branch_yr':97},
          'INM-CM5-0':{'institute':'INM', 'grid_label_atmos':['gr1'],  'grid_label_ocean':['gr1'], 'variant_labels':['r1i1p1f1'],'branch_yr':249},
          'IITM-ESM':{'institute':'CCCR-IITM', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'IPSL-CM6A-LR':{'institute':'IPSL', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':20},
          'KACE-1-0-G':{'institute':'NIMS-KMA', 'grid_label_atmos':['gr'],  'grid_label_ocean':['gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'MCM-UA-1-0':{'institute':'UA', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'MIROC6':{'institute':'MIROC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'MIROC-ES2L':{'institute':'MIROC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f2'],'branch_yr':0},
          'MPI-ESM1-2-HR':{'institute':'MPI-M', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'MPI-ESM1-2-LR':{'institute':'MPI-M', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'MPI-ESM-1-2-HAM':{'institute':'HAMMOZ-Consortium', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':100},
          'MRI-ESM2-0':{'institute':'MRI', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn','gr'], 'variant_labels':['r1i1p1f1','r1i2p1f1'],'branch_yr':0},
          'NESM3':{'institute':'NUIST', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':50},
          'NorESM2-LM':{'institute':'NCC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn','gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'NorESM2-MM':{'institute':'NCC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn','gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
          'SAM0-UNICON':{'institute':'SNU', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':273},
          'UKESM1-0-LL':{'institute':'MOHC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f2'],'branch_yr':0}}
    return models


def make_filelist_cmip6(self, var, component = 'atmos', activity_id='CMIP', path_to_data = '/projects/NS9252K/ESGF/CMIP6/'):
        '''
        This function can be used not only for atmosphere, but the idea is that it looks for the native grid first 
        and if that is not found it looks for gr grid. There is usually only one grid for the atmosphere
        For the ocean and sea-ice files, there are often both native and regridded files. Then it is better to use
        the make_filelist_ocean function and state which grid label is wanted

        Parameters
        ----------
        var :          str, name of variable
        time_frequency : str, avg time frequency of data. default is monthly averaged data: 'mon'
        path_to_data : str, path to the data folders. The default is '/projects/NS9252K/ESGF/cmip5/output1'.

        Returns
        -------
        Sets a list if filename(s) as an attribute of the model object

        '''
        import glob
        self.variable = var
        self.activity_id = activity_id
        if component == 'atmos':
            gridlabel = self.grid_label_atmos
        if component == 'ocean':
            if len(self.grid_label_ocean)>1:
                gridlabel = 'gr'
            if self.name in ['NorESM2-LM', 'NorESM2-MM'] and var not in  ['msftmz','hfbasin', 'hfbasinpadv','hfbasinpmadv','hfbasinpsmadv', 'hfbasinpmdiff', 'thetao', 'thetaoga']:
                gridlabel = 'gn'
            elif self.name in ['NorESM2-LM', 'NorESM2-MM'] and var == 'thetao':
                gridlabel = 'gr'
            elif self.name in ['NorESM2-LM', 'NorESM2-MM'] and var == 'thetaoga':
                gridlabel = 'gm'
                print(gridlabel)
            elif self.name in ['NorESM2-LM', 'NorESM2-MM'] and var in ['msftmz','hfbasin', 'hfbasinpadv','hfbasinpmadv','hfbasinpsmadv', 'hfbasinpmdiff']:
                gridlabel = 'grz'
            elif self.name in ['GFDL-CM4', 'GFDL-ESM4'] and var != 'thetaoga':
                gridlabel = 'gr'
            elif self.name in ['GFDL-CM4', 'GFDL-ESM4'] and var == 'thetaoga':
                gridlabel = 'gn'
            elif self.name in ['INM-CM4-8', 'INM-CM5-0'] and var == 'thetaoga':
                gridlabel = 'gr1'
            elif self.name in ['CESM2']:
                gridlabel = 'gn'
            elif self.name in ['HadGEM3-GC31-LL', 'HadGEM3-GC31-MM'] and var == 'thetaoga':
                gridlabel = 'gm'
            elif self.name[:4] == 'CNRM' and var in ['thetao','mlotst','msftyz']:
                gridlabel = 'gn'
            elif self.name == 'GISS-E2-1-H' and var =='thetao':
                gridlabel = 'gr'
            elif self.institute == 'MIROC' and var =='msftmz':
                gridlabel = 'gr'
            elif self.name == 'MRI-ESM2-0':
                gridlabel = 'gn'
                if var == 'thetaoga':
                    gridlabel = 'gm'
                if var =='msftmz':
                    gridlabel = 'gr2z'
                if var == 'msftyz':
                    gridlabel = 'gnz' 
            else:
                gridlabel = self.grid_label_ocean[0]
        self.gridlabel = gridlabel
        self.path = path_to_data + '/' + self.activity_id  + '/' + self.institute + '/' + self.name + '/' + self.expid + '/' + self.realiz + '/' + self.realm + '/' + self.variable + '/' + gridlabel+ '/' 
        #  Not all files are necessarily located in 'latest'
        versions = sorted(glob.glob(self.path +'*'))
        if versions:
            fnames = sorted(glob.glob(versions[0] +'/' + self.variable +'_' + self.realm +'_' + self.name + '_' + self.expid + '_' + self.realiz +'_' + gridlabel + '_*.nc'))
        else:
            fnames = []
        if len(versions)>1:
            for version in versions[1:]:
                files = sorted(glob.glob(version +'/' + self.variable +'_' + self.realm +'_' + self.name + '_' + self.expid + '_' + self.realiz +'_' + gridlabel + '_*.nc'))   
                for file in files:
                    if versions[0] +'/' +file.split('/')[-1] not in fnames:
                        fnames.append(file)
        if self.name == 'IPSL-CM6A-LR'and self.expid=='abrupt-4xCO2' and var in ['thetao', 'msftyz']:
           version = '/projects/NS9252K/ESGF/CMIP6//CMIP/IPSL/IPSL-CM6A-LR/abrupt-4xCO2/r1i1p1f1/Omon/' +var+ '/gn/v20190522'
           fnames = sorted(glob.glob(version +'/' + self.variable +'_' + self.realm +'_' + self.name + '_' + self.expid + '_' + self.realiz +'_' + gridlabel + '_*.nc'))
        if self.name == 'IPSL-CM6A-LR'and self.expid=='piControl' and var in ['msftyz']:
           version = '/projects/NS9252K/ESGF/CMIP6//CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Omon/' +var+ '/gn/v20200326' 
           fnames = sorted(glob.glob(version +'/' + self.variable +'_' + self.realm +'_' + self.name + '_' + self.expid + '_' + self.realiz +'_' + gridlabel + '_*.nc'))              
        if fnames:
           if self.name=='NorESM2-MM' and self.realiz == 'r3i1p1f1':
              # This file is an erroneous file and should not be included in the analysis. Year 1860 is included in the file covering 186001-186912 and already included
              fnames.remove('/projects/NS9034K/CMIP6//CMIP/NCC/NorESM2-MM/historical/r3i1p1f1/Omon/' + var + '/' +gridlabel +'/latest/' + var +'_Omon_NorESM2-MM_historical_r3i1p1f1_'+gridlabel+'_186001-186105.nc')
           if len(fnames)>1:
               fnames = sorted(fnames ,key=lambda x: extract_number(x))
               if self.name not in ['SAM0-UNICON']: 
                   checkConsecutive(fnames)
           self.filenames = fnames
           print('\n \n Final list of filenames:')
           print(self.filenames) 
        if not fnames:
           self.filenames = ''
        if not fnames:
            print('Variable %s not prestent in output folder for model %s\n'%(self.variable, self.name))
            #raise Exception

def extract_number(string):
    return string.split('_')[-1]

def extract_dates(string):
    return string.split('_')[-1].split('.')[0]


def checkConsecutive(fnames):
    sorteddates = [extract_dates(x) for x in fnames]
    for i in range(1,len(sorteddates)):
        if int(sorteddates[i].split('01-')[0]) != int(sorteddates[i-1].split('-')[1][:-2])+1:
            raise Exception('NOTE! The files are not in consecutive order. Please check directory')
            
     
class Modelinfo:
    '''
    Sets the details of the model experiment, including filenames
    '''
    
    def __init__(self, name, institute, expid, realm, 
                  realiz=['r1i1p1f1'], grid_atmos = 'gn', grid_ocean = 'gn', branchtime_year=0):
        '''

        Attributes
        ----------
        name :        str, name of the CMIP model - typically the Source ID
        institute :   str, Institution ID
        expid :       str, Experiment ID, e.g. piControl, abrupt-4xCO2
        realm :       str, which model domain and time frequency used, e.g. Amon, AERmon, Omon
        grid_labels : list, which grid resolutions are available. Modt useful for ocean and sea-ice variables. 
                      e.g. ['gn', 'gr'] gn: native grid, 'gr': regridded somehow - not obvious
        realiz :      str, variant labels saying something about which realization (ensemble member), initial conditions, forcings etc.
                      e.g. 'r1i1p1f1'
        version :     str, which version of the data is read. default is latest. 
        branchtime_year : int, when simulation was branched off from parent. 
                          Useful when anomalies are considered e.g. abrupt-4xCO2, historical 
                          then you only consider data from piCOntrol for the corresponding period i.e. piControl_data_array(time = branchtime_year:)
        
        '''
        self.name = name
        self.institute = institute
        self.expid = expid
        self.realm = realm
        self.realiz = realiz
        self.grid_label_atmos = grid_atmos
        self.grid_label_ocean = grid_ocean
        self.branchtime_year = branchtime_year

