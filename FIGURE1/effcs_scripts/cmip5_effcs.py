@author: ADA GJERMUNDSEN
YEARS: 2019 - 2021
CMIP5 model analysis
This script is used to calculate effective climate sensitivity EffCS
For piControl, the time period is selected to the corresponding period as in the abrupt-4xCO2 simulation
"""
import sys
sys.path.insert(1, 'path_to_cmip_utils/')
import CMIP6_ATMOS_UTILS as atmos
from read_model_data_cmip5 import ecs_models_cmip5, make_filelist_cmip5, Modelinfo
import numpy as np
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression



def linearfit(ds):
    t = linregress(np.arange(0,ds.shape[0]),ds.values)
    val = t.slope*np.arange(0,ds.shape[0]) + t.intercept
    return val

def read_radfiles(modelobj):
        if modelobj.name == 'NorESM1-ME':
            make_filelist_cmip5(modelobj, 'rsdt', time_frequency='mon', component = 'atmos', path_to_data = '/projects/NS9034K/CMIP5/output1/')
        else:
            make_filelist_cmip5(modelobj, 'rsdt', time_frequency = 'mon', component = 'atmos')
        if modelobj.filenames:
            if len(modelobj.filenames)>1:
                rsdt =  xr.open_mfdataset(modelobj.filenames, combine='nested', concat_dim='time')
            else:
                rsdt =  xr.open_dataset(modelobj.filenames[0])
            print('rsdt loaded for model: %s, %s. Lenght of simulation: %.1f years'%(modelobj.name, modelobj.expid, len(rsdt.rsdt.time.values)/12))
            rsdt = atmos.consistent_naming(rsdt)
            if modelobj.name == 'HadGEM2-ES':
                rsdt = rsdt.isel(time=slice(1,12*151+1))
            rsdt = atmos.fix_time(rsdt,1)
            rsdt = atmos.yearly_avg(atmos.global_mean(rsdt.rsdt))
            if modelobj.expid == 'piControl':
                rsdt = rsdt.isel(year=slice(modelobj.branchtime_year,-1))
        else:
             raise Exception('File rsdt missing for Model %s in experiment %s'%(modelobj.name, modelobj.expid))
        if modelobj.name == 'NorESM1-ME':
            make_filelist_cmip5(modelobj, 'rsut', time_frequency='mon', component = 'atmos', path_to_data = '/projects/NS9034K/CMIP5/output1/')
        else:
            make_filelist_cmip5(modelobj, 'rsut', time_frequency = 'mon', component = 'atmos')
        if modelobj.filenames:
            if len(modelobj.filenames)>1:
                rsut =  xr.open_mfdataset(modelobj.filenames, combine='nested', concat_dim='time')
            else:
                rsut =  xr.open_dataset(modelobj.filenames[0])
            print('rsut loaded for model: %s , %s. Lenght of simulation: %.1f years'%(modelobj.name, modelobj.expid, len(rsut.rsut.time.values)/12))
            rsut = atmos.consistent_naming(rsut)
            if modelobj.name == 'HadGEM2-ES':
                rsut = rsut.isel(time=slice(1,12*151+1))
            rsut = atmos.fix_time(rsut,1)
            rsut = atmos.yearly_avg(atmos.global_mean(rsut.rsut))
            if modelobj.expid == 'piControl':
                rsut = rsut.isel(year=slice(modelobj.branchtime_year,-1))
        else:
             raise Exception('File rsut missing for Model %s in experiment %s'%(modelobj.name, modelobj.expid))
        if modelobj.name == 'NorESM1-ME':
            make_filelist_cmip5(modelobj, 'rlut', time_frequency='mon', component = 'atmos', path_to_data = '/projects/NS9034K/CMIP5/output1/')
        else:
            make_filelist_cmip5(modelobj, 'rlut', time_frequency = 'mon', component = 'atmos')
        if modelobj.filenames:
            if len(modelobj.filenames)>1:
                rlut =  xr.open_mfdataset(modelobj.filenames, combine='nested', concat_dim='time')
            else:
                rlut =  xr.open_dataset(modelobj.filenames[0])
            print('rlut loaded for model: %s ,%s. Lenght of simulation: %.1f years'%(modelobj.name, modelobj.expid, len(rlut.rlut.time.values)/12))
            rlut = atmos.consistent_naming(rlut)
            if modelobj.name == 'HadGEM2-ES':
                rlut = rlut.isel(time=slice(1,12*151+1))
            rlut = atmos.fix_time(rlut,1)
            rlut = atmos.yearly_avg(atmos.global_mean(rlut.rlut))
            if modelobj.expid == 'piControl':
                rlut = rlut.isel(year=slice(modelobj.branchtime_year,-1))
        else:
             raise Exception('File rlut missing for Model %s in experiment %s'%(modelobj.name, modelobj.expid))
        return rsdt, rsut, rlut

def calculate_ecs_values(models,startyr=0,  endyr=150):
    #######RTMTOA##########################
    print('ECS CALCULATIONS: \n')
    ecsvalues={}
    for modelname,expinfo in models.items():
        print(modelname)
        modelctrl = Modelinfo(name = modelname, institute = expinfo['institute'], expid = 'piControl', realm = 'Amon',
                  realiz='r1i1p1', version=expinfo['versions']['pic_atmos'], branchtime_year=expinfo['branch_yr'])
        if modelname == 'NorESM1-ME':
            make_filelist_cmip5(modelctrl, 'tas', time_frequency='mon', component = 'atmos', path_to_data = '/projects/NS9034K/CMIP5/output1/')
        else:
            make_filelist_cmip5(modelctrl, 'tas', time_frequency='mon', component = 'atmos', path_to_data = '/projects/NS9252K/ESGF/cmip5/output1')
        if modelctrl.filenames:
            if len(modelctrl.filenames)>1:
                tasctrl =  xr.open_mfdataset(modelctrl.filenames, combine='nested', concat_dim='time')
            else:
                tasctrl =  xr.open_dataset(modelctrl.filenames[0])
            print('tas loaded for model: %s, experiment: piControl . Lenght of simulation: %.1f years'%(modelctrl.name, len(tasctrl.tas.time.values)/12))
            tasctrl = atmos.consistent_naming(tasctrl)
            if modelname == 'HadGEM2-ES':
                tasctrl = tasctrl.isel(time=slice(1,12*151+1))
            tasctrl = atmos.fix_time(tasctrl, 1)
            tasctrl = atmos.yearly_avg(atmos.global_mean(tasctrl.tas))
            tasctrl = tasctrl.isel(year=slice(modelctrl.branchtime_year,-1))
        else:
            raise Exception('tas not loaded for model %s, experiment: piControl. Skipping model! Please check!'%modelname)
        model4xco2 = Modelinfo(name = modelname, institute = expinfo['institute'], expid = 'abrupt4xCO2', realm = 'Amon',
                  realiz='r1i1p1', version=expinfo['versions']['a4xco2_atmos'], branchtime_year=expinfo['branch_yr'])
        if modelname == 'NorESM1-ME':
            make_filelist_cmip5(model4xco2, 'tas', time_frequency='mon', component = 'atmos', path_to_data = '/projects/NS9034K/CMIP5/output1/')
        else:
            make_filelist_cmip5(model4xco2, 'tas', time_frequency='mon', component = 'atmos', path_to_data = '/projects/NS9252K/ESGF/cmip5/output1')
        if model4xco2.filenames:
            if len(model4xco2.filenames)>1:
                tascase =  xr.open_mfdataset(model4xco2.filenames, combine='nested', concat_dim='time')
            else:
                tascase =  xr.open_dataset(model4xco2.filenames[0])
            print('tas loaded for model: %s, experiment: abrupt4xCO2. . Lenght of simulation: %.1f years'%(model4xco2.name, len(tascase.tas.time.values)/12))
            tascase = atmos.consistent_naming(tascase)
            if modelname == 'HadGEM2-ES':
                tascase = tascase.isel(time=slice(1,12*151+1))
            tascase = atmos.fix_time(tascase,1)
            tascase = atmos.yearly_avg(atmos.global_mean(tascase.tas))
        else:
            raise Exception('tas not loaded for model %s, experiment: abrupt4xCO2. Skipping model! Please check!'%modelname)

        rsdt, rsut, rlut = read_radfiles(modelctrl)
        radctrl = rsdt - rsut - rlut
        rsdt, rsut, rlut = read_radfiles(model4xco2)
        radcase = rsdt - rsut - rlut

        drad = radcase.values[startyr:endyr] - linearfit(radctrl[startyr:endyr])
        dtas = tascase.values[startyr:endyr] - linearfit(tasctrl[startyr:endyr])
        linmodel =  LinearRegression().fit(dtas.reshape((-1, 1)), drad)
        ecsvalues[modelname] = str(-np.round(linmodel.intercept_/linmodel.coef_[0]/2,2))
        print('EffCS from %s : %s K'%(modelname,ecsvalues[modelname]))
        del model4xco2, modelctrl
    return ecsvalues

if __name__ == '__main__':
    models, realiz = ecs_models_cmip5()
    ecs_values = calculate_ecs_values(models,startyr=0,  endyr=150)
    np.save('outdata/ecs_cmip5_150yrs.npy', ecs_values)
