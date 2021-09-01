# ECS_SOconvection
Python scripts used for analysis and plotting in the article: Shutdown of Southern Ocean convection controls long-term greenhouse gas-induced warming

## FIGURE#X
One folder for each published figure. The folder contains analysis script and plotting rutines needed to reproduce the figure, except the radiative feedbacks for which we used the scripts of https://github.com/apendergrass/cam5-kernels. 

## Read CMIP data

### CMIP6_UTILS
Python functions used for analyzing CMIP5 and CMIP6 data. Functions used for calculating time, regional and global averages, extract sub-regions, regridding etc. Used for both CMIP5 and CMIP6 data. Some functions used in CMIP6_SEA_ICE_UTILS.py import functions from CMIP6_ATMOS_UTILS.py  

- **CMIP6_ATMOS_UTILS.py**: python functions used for atmsophere variables

- **CMIP6_SEAICE_UTILS.py**: python functions used for ocean and sea ice variables 

- For the regridding package xesmf to work properly, you need to make a new conda environment before installing the packages:

```
  (base)$ conda create -n xesmf_env
  (base)$ conda activate xesmf_env
  (xesmf_env)$ conda install -c anaconda xarray
  (xesmf_env)$ conda install -c conda-forge esmpy scipy dask netCDF4
  (xesmf_env)$ conda install -c conda-forge xesmf
```


see https://xesmf.readthedocs.io/en/latest/installation.html)

- **POP EOS**: For density calculations we use the pop tools provided here https://pop-tools.readthedocs.io/en/latest/api.html#pop_tools.eos:

```
  (base)$ conda create -n ocn_env
  (base)$ conda activate ocn_env
  (xesmf_env)$ conda install -c anaconda xarray
  (xesmf_env)$ conda install -c conda-forge esmpy scipy dask netCDF4 cftime 
  (xesmf_env)$ conda install -c conda-forge xesmf pop-tools
```
