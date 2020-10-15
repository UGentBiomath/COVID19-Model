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

def draw_sample_COVID19_SEIRD(parameter_dictionary,samples_dict):
    """
    A function to draw parameter samples obtained with MCMC during model calibration and assign them to the parameter dictionary of the model.
    Tailor-made for the BIOMATH COVID-19 SEIRD model.

    Parameters
    ----------
    model : object
        BIOMATH model object
    
    samples_dict : dictionary
        Dictionary containing the samples of the sampled parameters: beta, l and tau.

    Returns
    ----------
    model : object
        BIOMATH model object

    """
    # Use posterior samples of fitted parameters
    parameter_dictionary['beta'] = np.random.choice(samples_dict['beta'],1,replace=False)
    idx,parameter_dictionary['l'] = random.choice(list(enumerate(samples_dict['l'])))
    parameter_dictionary['tau'] = samples_dict['tau'][idx]
    parameter_dictionary['prevention'] = samples_dict['prevention'][idx]
    return parameter_dictionary

def MC_sim(model,N,T,draw_function=None,*samples):
    """
    A function to perform N repeated simulations of T days with 'model'.
    Can optionally use samples draw using MCMC to perform the simulation.

    Parameters
    ----------
    model : object
        BIOMATH model object

    N : int
        Number of repeated simulations
    
    T : int
        length of simulation

    Optional parameters
    -------------------
    draw_function: function
        Function that draws MCMC parameters and assigns them to the model object.
    
    samples: dictionary
        Dictionary containing the sampled parameters. To be used by 'draw_function'.

    Returns
    ----------
    dataset : xarray dataset
        Contains an extra dimension "draws".

    """

    if draw_function:
        model = draw_function(model,samples[0])
    dataset = model.sim(T)
    for i in range(N-1):
        if draw_function:
            model = draw_function(model,samples[0])
        dataset = xr.concat([dataset, model.sim(T)], "draws")
        
    return dataset