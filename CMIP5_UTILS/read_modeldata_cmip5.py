#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:08:18 2020

@author: adag
"""
import glob
import numpy as np
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)

def Ofx_files(var, modelname):
    ofxfiles={'ACCESS1-0':'/projects/NS9252K/ESGF/cmip5/output1/CSIRO-BOM/ACCESS1-0/piControl/fx/ocean/fx/r0i0p0/v20130305/' + var + '/' + var + '_fx_ACCESS1-0_piControl_r0i0p0.nc',
            'ACCESS1-3':'/projects/NS9252K/ESGF/cmip5/output1/CSIRO-BOM/ACCESS1-3/piControl/fx/ocean/fx/r0i0p0/v20130305/' +var+'/' + var +'_fx_ACCESS1-3_piControl_r0i0p0.nc',
            'BNU-ESM':'/projects/NS9252K/ESGF/cmip5/output1/BNU/BNU-ESM/piControl/fx/ocean/fx/r0i0p0/latest/' + var + '/' + var + '_fx_BNU-ESM_piControl_r0i0p0.nc',
            'bcc-csm1-1':'/projects/NS9252K/ESGF/cmip5/output1/BCC/bcc-csm1-1/piControl/fx/ocean/fx/r0i0p0/v20130307/' + var + '/' + var + '_fx_bcc-csm1-1_piControl_r0i0p0.nc',
            'bcc-csm1-1-m':'/projects/NS9252K/ESGF/cmip5/output1/BCC/bcc-csm1-1-m/piControl/fx/ocean/fx/r0i0p0/v20130307/'  + var + '/' + var +  '_fx_bcc-csm1-1-m_piControl_r0i0p0.nc',
            'CCSM4':'/projects/NS9252K/ESGF/cmip5/output1/NCAR/CCSM4/piControl/fx/ocean/fx/r0i0p0/latest/' + var + '/' + var + '_fx_CCSM4_piControl_r0i0p0.nc',
            'CNRM-CM5':'/projects/NS9252K/ESGF/cmip5/output1/CNRM-CERFACS/CNRM-CM5/piControl/fx/ocean/fx/r0i0p0/v20130826/' + var + '/' + var + '_fx_CNRM-CM5_piControl_r0i0p0.nc',
            'CSIRO-Mk3-6-0':'/projects/NS9252K/ESGF/cmip5/output1/CSIRO-QCCCE/CSIRO-Mk3-6-0/piControl/fx/ocean/fx/r0i0p0/v1/' + var + '/' + var + '_fx_CSIRO-Mk3-6-0_piControl_r0i0p0.nc',
            'CanESM2':'/projects/NS9252K/ESGF/cmip5/output1/CCCma/CanESM2/piControl/fx/ocean/fx/r0i0p0/v20130119/' + var + '/' + var +'_fx_CanESM2_piControl_r0i0p0.nc',
            'EC-EARTH':'', # rsut, rlut and ocean variables missing
            'FGOALS-g2':'/projects/NS9252K/ESGF/cmip5/output1/LASG-CESS/FGOALS-g2/historical/fx/ocean/fx/r0i0p0/v1/'  + var + '/' + var + '_fx_FGOALS-g2_historical_r0i0p0.nc',
            'FGOALS-s2':'/projects/NS9252K/ESGF/cmip5/output1/LASG-IAP/FGOALS-s2/piControl/fx/ocean/fx/r0i0p0/v20161204/' + var + '/' + var + '_fx_FGOALS-s2_piControl_r0i0p0.nc',
            'GFDL-CM3':'/projects/NS9252K/ESGF/cmip5/output1/NOAA-GFDL/GFDL-CM3/piControl/fx/ocean/fx/r0i0p0/latest/'  + var + '/' + var + '_fx_GFDL-CM3_piControl_r0i0p0.nc',
            'GFDL-ESM2G':'/projects/NS9252K/ESGF/cmip5/output1/NOAA-GFDL/GFDL-ESM2G/piControl/fx/ocean/fx/r0i0p0/v20110601/' + var +'/' + var + '_fx_GFDL-ESM2G_piControl_r0i0p0.nc',
            'GFDL-ESM2M':'/projects/NS9252K/ESGF/cmip5/output1/NOAA-GFDL/GFDL-ESM2M/piControl/fx/ocean/fx/r0i0p0/v20130514/' + var + '/' + var + '_fx_GFDL-ESM2M_piControl_r0i0p0.nc',
            'GISS-E2-H':'/projects/NS9252K/ESGF/cmip5/output1/NASA-GISS/GISS-E2-H/piControl/fx/ocean/fx/r0i0p0/v20160511/' + var +'/' + var +'_fx_GISS-E2-H_piControl_r0i0p0.nc',
            'GISS-E2-R':'/projects/NS9252K/ESGF/cmip5/output1/NASA-GISS/GISS-E2-R/piControl/fx/ocean/fx/r0i0p0/v20160511/' + var + '/' + var + '_fx_GISS-E2-R_piControl_r0i0p0.nc',
            'HadGEM2-ES':'/projects/NS9252K/ESGF/cmip5/output1/MOHC/HadGEM2-ES/piControl/fx/ocean/fx/r0i0p0/v20130612/' + var +'/' + var +'_fx_HadGEM2-ES_piControl_r0i0p0.nc',
            # INM-CM4: inmcm4 is used in folder structure and in filenames
            'inmcm4':'/projects/NS9252K/ESGF/cmip5/output1/INM/inmcm4/esmHistorical/fx/ocean/fx/r0i0p0/v20110323/' + var +'/' + var +'_fx_inmcm4_esmHistorical_r0i0p0.nc',
            'IPSL-CM5A-LR':'/projects/NS9252K/ESGF/cmip5/output1/IPSL/IPSL-CM5A-LR/piControl/fx/ocean/fx/r0i0p0/v20120430/' + var + '/' + var + '_fx_IPSL-CM5A-LR_piControl_r0i0p0.nc',
            'IPSL-CM5A-MR':'/projects/NS9252K/ESGF/cmip5/output1/IPSL/IPSL-CM5A-MR/piControl/fx/ocean/fx/r0i0p0/v20120430/' + var + '/' + var +'_fx_IPSL-CM5A-MR_piControl_r0i0p0.nc',
            'IPSL-CM5B-LR':'/projects/NS9252K/ESGF/cmip5/output1/IPSL/IPSL-CM5B-LR/piControl/fx/ocean/fx/r0i0p0/v20120430/' + var + '/' + var + '_fx_IPSL-CM5B-LR_piControl_r0i0p0.nc',
            'MIROC-ESM':'/projects/NS9252K/ESGF/cmip5/output1/MIROC/MIROC-ESM/piControl/fx/ocean/fx/r0i0p0/v20120608/' + var + '/' + var + '_fx_MIROC-ESM_piControl_r0i0p0.nc',
            'MIROC5':'/projects/NS9252K/ESGF/cmip5/output1/MIROC/MIROC5/piControl/fx/ocean/fx/r0i0p0/v20110901/' + var + '/' + var + '_fx_MIROC5_piControl_r0i0p0.nc',
            'MPI-ESM-LR':'/projects/NS9252K/ESGF/cmip5/output1/MPI-M/MPI-ESM-LR/piControl/fx/ocean/fx/r0i0p0/v20111006/' + var + '/' + var + '_fx_MPI-ESM-LR_piControl_r0i0p0.nc',
            'MPI-ESM-MR':'/projects/NS9252K/ESGF/cmip5/output1/MPI-M/MPI-ESM-MR/piControl/fx/ocean/fx/r0i0p0/v20120503/' + var + '/' + var + '_fx_MPI-ESM-MR_piControl_r0i0p0.nc',
            'MPI-ESM-P':'/projects/NS9252K/ESGF/cmip5/output1/MPI-M/MPI-ESM-P/piControl/fx/ocean/fx/r0i0p0/v20120625/' + var + '/' + var + '_fx_MPI-ESM-P_piControl_r0i0p0.nc',
            'MRI-CGCM3':'/projects/NS9252K/ESGF/cmip5/output1/MRI/MRI-CGCM3/piControl/fx/ocean/fx/r0i0p0/v20110831/' + var +'/' + var + '_fx_MRI-CGCM3_piControl_r0i0p0.nc',
            'NorESM1-M':'/projects/NS9034K/CMIP5/output1/NCC/NorESM1-M/piControl/fx/ocean/fx/r0i0p0/latest/' + var + '/' + var +'_fx_NorESM1-M_piControl_r0i0p0.nc',
            'NorESM1-ME':'/projects/NS9034K/CMIP5/output1/NCC/NorESM1-ME/piControl/fx/ocean/fx/r0i0p0/latest/' + var +'/' + var +'_fx_NorESM1-ME_piControl_r0i0p0.nc'}
    return ofxfiles[modelname]
    
def ecs_models_cmip5():
    '''
    Some thoughts:
        - all models have the same relaization = r1i1p1
        - all models need a version (latest does not exist)
        - need to find branch time
        - need to manually download missing files - some models miss tos
    

    Returns
    -------
    models : TYPE
        DESCRIPTION.

    '''
    realization = 'r1i1p1' #used for all models
    models={'ACCESS1-0':{'institute':'CSIRO-BOM', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20130524', 'pic_ocean':'v20140305', 'a4xco2_atmos':'v20120727','a4xco2_ocean':'v2'}},
            'ACCESS1-3':{'institute':'CSIRO-BOM', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20130524', 'pic_ocean':'v20140305', 'a4xco2_atmos':'v20120413','a4xco2_ocean':'v2'}},
            'bcc-csm1-1':{'institute':'BCC', 'branch_yr':160,
                          'versions':{'pic_atmos':'v1', 'pic_ocean':'v20120202', 'a4xco2_atmos':'v1','a4xco2_ocean':'v20120202'}},
            'bcc-csm1-1-m':{'institute':'BCC', 'branch_yr':240,
                          'versions':{'pic_atmos':'v20120705', 'pic_ocean':'v20120705', 'a4xco2_atmos':'v20120910','a4xco2_ocean':'v20120910'}},
            'BNU-ESM':{'institute':'BNU', 'branch_yr':0,
                          'versions':{'pic_atmos':'v20120626', 'pic_ocean':'v20120504', 'a4xco2_atmos':'v20120626','a4xco2_ocean':'v20120503'}},
            'CNRM-CM5':{'institute':'CNRM-CERFACS', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20110701', 'pic_ocean':'v20121001', 'a4xco2_atmos':'v20110701','a4xco2_ocean':'v20130101'}},
            'CCSM4':{'institute':'NCAR', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20160829', 'pic_ocean':'v20121128', 'a4xco2_atmos':'v20120604','a4xco2_ocean':'v20140820'}},
            'CSIRO-Mk3-6-0':{'institute':'CSIRO-QCCCE', 'branch_yr':103,
                         'versions':{'pic_atmos':'v20120607', 'pic_ocean':'v20130205', 'a4xco2_atmos':'v20120323','a4xco2_ocean':'v1'}},
            'CSIRO-Mk3L-1-2':{'institute':'', 'branch_yr':0,
                         'versions':{'pic_atmos':'', 'pic_ocean':'', 'a4xco2_atmos':'','a4xco2_ocean':''}}, #*#
            'CanESM2':{'institute':'CCCma', 'branch_yr':306,
                         'versions':{'pic_atmos':'v20120623', 'pic_ocean':'v20120410', 'a4xco2_atmos':'v20111027','a4xco2_ocean':'v20111027'}}, #*#
            'EC-EARTH':{'institute':'ICHEC', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20120622', 'pic_ocean':'', 'a4xco2_atmos':'v20130212','a4xco2_ocean':'20120924'}}, # rsut, rlut and ocean variables missing
            'FGOALS-g2':{'institute':'LASG-CESS', 'branch_yr':290,
                         'versions':{'pic_atmos':'v20161204', 'pic_ocean':'latest', 'a4xco2_atmos':'v1','a4xco2_ocean':'latest'}},
            'FGOALS-s2':{'institute':'LASG-IAP', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20161204', 'pic_ocean':'v20130225', 'a4xco2_atmos':'','a4xco2_ocean':'v1'}}, 
            'GFDL-CM3': {'institute':'NOAA-GFDL', 'branch_yr':0,
                          'versions':{'pic_atmos':'v20120227', 'pic_ocean':'v20110601', 'a4xco2_atmos':'v20120227','a4xco2_ocean':'v20110601'}},
            'GFDL-ESM2G':{'institute':'NOAA-GFDL', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20120830', 'pic_ocean':'v20120820', 'a4xco2_atmos':'v20120830','a4xco2_ocean':'v20120820'}},
            'GFDL-ESM2M':{'institute':'NOAA-GFDL', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20130214', 'pic_ocean':'v20130226', 'a4xco2_atmos':'v20130214','a4xco2_ocean':'v20130226'}},
            'GISS-E2-H':{'institute':'NASA-GISS', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20160511', 'pic_ocean':'v20160511', 'a4xco2_atmos':'v20160505','a4xco2_ocean':'v20160505'}},
            'GISS-E2-R':{'institute':'NASA-GISS', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20161004', 'pic_ocean':'v20160930', 'a4xco2_atmos':'v20160919','a4xco2_ocean':'v20160919'}},
            'HadGEM2-ES':{'institute':'MOHC', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20130114', 'pic_ocean':'v20110928', 'a4xco2_atmos':'v20111129','a4xco2_ocean':'v20130107'}},
            # INM-CM4: inmcm4 is used in folder structure and in filenames
            'inmcm4':{'institute':'INM', 'branch_yr':240,
                         'versions':{'pic_atmos':'v20130207', 'pic_ocean':'v20110323', 'a4xco2_atmos':'v20130207','a4xco2_ocean':'v20110323'}},
            'IPSL-CM5A-LR':{'institute':'IPSL', 'branch_yr':50,
                         'versions':{'pic_atmos':'v20130506', 'pic_ocean':'v20111119', 'a4xco2_atmos':'v20190903','a4xco2_ocean':'v20130322'}},
            'IPSL-CM5A-MR':{'institute':'IPSL', 'branch_yr':50,
                         'versions':{'pic_atmos':'v20111119', 'pic_ocean':'v20111119', 'a4xco2_atmos':'v20120114','a4xco2_ocean':'v20120114'}},
            'IPSL-CM5B-LR':{'institute':'IPSL', 'branch_yr':20,
                         'versions':{'pic_atmos':'v20120114', 'pic_ocean':'v20120114', 'a4xco2_atmos':'v20120430','a4xco2_ocean':'v20120430'}},
            'MIROC-ESM':{'institute':'MIROC', 'branch_yr':80,
                         'versions':{'pic_atmos':'v20120710', 'pic_ocean':'v20130712', 'a4xco2_atmos':'v20120710','a4xco2_ocean':'v20131008'}},
            'MIROC5':{'institute':'MIROC', 'branch_yr':100,
                         'versions':{'pic_atmos':'v20161012', 'pic_ocean':'v20161012', 'a4xco2_atmos':'v20120710','a4xco2_ocean':'v20131009'}},
            'MPI-ESM-LR':{'institute':'MPI-M', 'branch_yr':30,
                         'versions':{'pic_atmos':'v20120602', 'pic_ocean':'v20120625', 'a4xco2_atmos':'v20120602','a4xco2_ocean':'v20120625'}},
            'MPI-ESM-MR':{'institute':'MPI-M', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20120602', 'pic_ocean':'v20120625', 'a4xco2_atmos':'v20120602','a4xco2_ocean':'v20120625'}},
            'MPI-ESM-P':{'institute':'MPI-M', 'branch_yr':16,
                         'versions':{'pic_atmos':'v20120602', 'pic_ocean':'v20120625', 'a4xco2_atmos':'v20120602','a4xco2_ocean':'v20120625'}},
            'MRI-CGCM3':{'institute':'MRI', 'branch_yr':40,
                         'versions':{'pic_atmos':'v20120701', 'pic_ocean':'v20120510', 'a4xco2_atmos':'v20120701','a4xco2_ocean':'v20120510'}},
            'NorESM1-M' :{'institute':'NCC', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20120412', 'pic_ocean':'v20110901', 'a4xco2_atmos':'v20120412','a4xco2_ocean':'v20110901'}},
            'NorESM1-ME':{'institute':'NCC', 'branch_yr':0,
                         'versions':{'pic_atmos':'v20130926', 'pic_ocean':'v20120225', 'a4xco2_atmos':'v20200617','a4xco2_ocean':'v20200617'}}}
    return models, realization

def make_filelist_cmip5(self, var,  time_frequency='mon', component = 'atmos', path_to_data = '/projects/NS9252K/ESGF/cmip5/output1'):
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
        self.variable = var
        self.component = component
        self.time_frequency = time_frequency
        self.path = path_to_data + '/' + self.institute + '/' + self.name + '/' + self.expid + '/' + self.time_frequency + '/' + self.component + '/' + self.realm + '/' + self.realiz +'/' + self.version + '/' + self.variable
        #print(self.path)
        fnames = sorted(glob.glob(self.path +'/' + self.variable +'_' + self.realm +'_' + self.name + '_' + self.expid + '_' + self.realiz + '_*.nc'))
        print(fnames)
        if fnames:
           if len(fnames)>1 and self.name not in ['HadGEM2-ES']:
               fnames = sorted(fnames ,key=lambda x: extract_number(x))
               checkConsecutive(fnames)
           self.filenames = fnames 
        if not fnames and self.variable in ['thetaoga', 'pbo']:
            versions = sorted(glob.glob(path_to_data + '/' + self.institute + '/' + self.name + '/' + self.expid + '/' + self.time_frequency + '/' + self.component + '/' + self.realm + '/' + self.realiz +'/*'))
            print('Versions:')
            print(versions)
            for version in versions:
                fnames = sorted(glob.glob(version +'/' + self.variable +'/' + self.variable +'_' + self.realm +'_' + self.name + '_' + self.expid + '_' + self.realiz + '_*.nc')) 
                if fnames:
                   self.filenames = fnames
                   break
        if not fnames:
           self.filenames = ''
        if not fnames:
            print('Variable %s not prestent in output folder for model %s\n'%(self.variable, self.name))

def extract_number(string):
    return string.split('_')[-1]

def extract_dates(string):
    return string.split('_')[-1].split('.')[0]


def checkConsecutive(fnames):
    sorteddates = [extract_dates(x) for x in fnames]
    for i in range(1,len(sorteddates)):
        if int(sorteddates[i].split('01-')[0]) != int(sorteddates[i-1].split('-')[1][:-2])+1:
            print(fnames)
            raise Exception('NOTE! The files are not in consecutive order. Please check directory')


class Modelinfo:
    '''
    Sets the details of the model experiment, including filenames
    '''
    
    def __init__(self, name, institute, expid, realm, 
                  realiz='r1i1p1', version='latest', branchtime_year=0):
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
                      default: 'r1i1p1'
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
        self.version = version
        self.branchtime_year = branchtime_year
