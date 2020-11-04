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

def draw_sample_COVID19_SEIRD(parameter_dictionary,samples_dict, to_sample=['beta','l','tau','prevention']):
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
    if 'beta' in to_sample:
        parameter_dictionary['beta'] = np.random.choice(samples_dict['beta'],1,replace=False)
    if 'l' in to_sample:
        idx,parameter_dictionary['l'] = random.choice(list(enumerate(samples_dict['l'])))
        parameter_dictionary['tau'] = samples_dict['tau'][idx]
        if 'prevention' in to_sample:
            parameter_dictionary['prevention'] = samples_dict['prevention'][idx]
    return parameter_dictionary


def ramp_fun(Nc_old, Nc_new, t, tau_days, l, t_start):
    """
    t : timestamp
        current date
    tau : int
        number of days before measures start having an effect
    l : int
        number of additional days after the time delay until full compliance is reached
    """
    return Nc_old + (Nc_new-Nc_old)/l * (t-t_start-tau_days)/pd.Timedelta('1D')

def lockdown_func(t,param,policy0,policy1,l,tau,prevention,start_date):
    """
    Lockdown function handling t as datetime
    
    t : timestamp
        current date
    policy0 : matrix
        policy before lockdown (= no policy)
    policy1 : matrix
        policy during lockdown
    tau : int
        number of days before measures start having an effect
    l : int
        number of additional days after the time delay until full compliance is reached
    start_date : timestamp
        start date of the data
    """
    tau_days = pd.to_timedelta(tau, unit='D')
    l_days = pd.to_timedelta(l, unit='D')
    if t <= start_date + tau_days:
        return policy1
    elif start_date + tau_days < t <= start_date + tau_days + l_days:
        return ramp_fun(policy0, prevention*policy1, t, tau_days, l, start_date)
    else:
        return prevention*policy1
    
def policies_until_september(t,param,start_date,policy0,policy1,policy2,policy3,policy4,policy5,
                               policy6,policy7,policy8,policy9,l,tau,prevention):
    """
    t : timestamp
        current date
    policy0 : matrix
        policy before lockdown (= no policy)
    policy1 : matrix
        policy during lockdown
    policy 2: matrix
        reopening industry
    policy 3: matrix
        merging of two bubbels
    policy 4: matrix
        reopening of businesses
    policy 5: matrix
        partial reopening schools
    policy 6: matrix
        reopening schools, bars, restaurants
    policy 7: matrix
        school holidays, gatherings 15 people, cultural event
    policy 8: matrix
       ?
    policy 9: matrix
        opening schools
    tau : int
        number of days before measures start having an effect
    l : int
        number of additional days after the time delay until full compliance is reached
    start_date : timestamp
        start date of the data
    
    """

    tau_days = pd.to_timedelta(tau, unit='D')
    l_days = pd.to_timedelta(l, unit='D')
    t2 = pd.to_datetime('2020-05-04') # reopening industry
    t3 = pd.to_datetime('2020-05-06') # merging of two bubbels
    t4 = pd.to_datetime('2020-05-11') # reopening of businesses
    t5 = pd.to_datetime('2020-05-18') # partial reopening schools
    t6 = pd.to_datetime('2020-06-04') # reopening schools, bars, restaurants
    t7 = pd.to_datetime('2020-07-01') # school holidays, gatherings 15 people, cultural event
    t8 = pd.to_datetime('2020-07-31') # ?
    t9 = pd.to_datetime('2020-09-01') # opening schools
    
    if t <= start_date + tau_days:
        return policy0
    elif start_date + tau_days < t <= start_date + tau_days + l_days:
        return ramp_fun(policy0, prevention*policy1, t, tau_days, l, start_date)
    elif start_date + tau_days + l_days < t <= t1: 
        return prevention*policy1
    elif t2 < t <= t3:
        return prevention*policy2
    elif t3 < t <= t4:
        return prevention*policy3
    elif t4 < t <= t5:
        return prevention*policy4
    elif t5 < t <= t6:
        return prevention*policy5
    elif t6 < t <= t7:
        return prevention*policy6
    elif t7 < t <= t8:
        return prevention*policy7
    elif t8 < t <= t9:
        return prevention*policy8
    elif t9 < t:
        return prevention*policy9