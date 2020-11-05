import os
import random
import numpy as np
import pandas as pd
import xarray as xr

abs_dir = os.path.dirname(__file__)
data_path = os.path.join(abs_dir, "../../../data/")

def sample_beta_binomial(n, p, k, size=None):
    p = np.random.beta(k/(1-p), k/p, size=size)
    r = np.random.binomial(n, p)
    return r

def name2nis(name):
    """
    A function to convert the name of a Belgian municipality/arrondissement/province/etc. to its NIS code

    Parameters
    ----------
    name : str
        the name of the municipality/arrondissement/province/etc.

    Returns
    -------
    NIS : float
        the NIS code corresponding with the given name

    """
    # Load the list of name-NIS couples
    name_df=pd.read_csv(os.path.join(data_path, 'raw/GIS/NIS_name.csv'))
    pos_name = name_df['name'].values
    # Convert list of possible names to lowercase only
    pos_name_lower = [string.lower() for string in pos_name]
    name_df['name'] = pos_name_lower
    # Check if input is a string
    if not isinstance(name,str):
        raise TypeError(
            "name2nis input must be a string"
            )
    # Convert input to lowercase
    name = name.lower()
    # Search for a match and return NIS code
    if not name in pos_name_lower:
        raise ValueError(
                "No match for '{0}' found".format(name)
            )
    else:
        return name_df[name_df['name'] == name]['NIS'].values[0]

def read_coordinates_nis():
    """
    A function to extract from /data/interim/demographic/initN_arrond.csv the list of arrondissement NIS codes

    Returns
    -------
     NIS: list
        a list containing the NIS codes of the 43 Belgian arrondissements

    """

    initN_df=pd.read_csv(os.path.join(data_path, 'interim/demographic/initN_arrond.csv'))
    NIS = initN_df.NIS.values

    return NIS

def draw_sample_COVID19_SEIRD(param_dict,samples_dict,beta_only=False):
    """
    A function to draw parameter samples obtained with MCMC during model calibration and assign them to the parameter dictionary of the model.
    Tailor-made for the BIOMATH COVID-19 SEIRD model.

    Parameters
    ----------
    param_dict : dict
        Parameter dictionary of the BIOMATH COVID-19 model.
    
    samples_dict : dictionary
        Dictionary containing the MCMC samples of the BIOMATH COVID-19 model parameters: beta, l and tau.

    Returns
    -------
    param_dict : dict
        Parameter dictionary of the BIOMATH COVID-19 model.

    """
    
    param_dict['beta'] = np.random.choice(samples_dict['beta'],1,replace=False)
    if beta_only == False:
        idx,param_dict['l'] = random.choice(list(enumerate(samples_dict['l'])))
        param_dict['tau'] = samples_dict['tau'][idx]
        param_dict['prevention'] = samples_dict['prevention'][idx]

    return param_dict

def draw_sample_beta_COVID19_SEIRD(param_dict,samples_dict):
    """
    A function to draw parameter samples obtained with MCMC during model calibration and assign them to the parameter dictionary of the model.
    Tailor-made for the BIOMATH COVID-19 SEIRD model.

    Parameters
    ----------
    param_dict : dict
        Parameter dictionary of the BIOMATH COVID-19 model.
    
    samples_dict : dictionary
        Dictionary containing the MCMC samples of the BIOMATH COVID-19 model parameters: beta, l and tau.

    Returns
    ----------
    param_dict : dict
        Parameter dictionary of the BIOMATH COVID-19 model.

    """

    param_dict['beta'] = np.random.choice(samples_dict['beta'],1,replace=False)

    return param_dict