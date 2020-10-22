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

def read_coordinates_nis(spatial='arr'):
    """
    A function to extract from /data/interim/demographic/initN_arrond.csv the list of arrondissement NIS codes

    Parameters
    ----------
    spatial : str
        choose geographical aggregation. Pick between 'arr', 'mun', 'prov', 'test'. Default is 'arr'.

    Returns
    -------
     NIS: list
        a list containing the NIS codes of the 43 Belgian arrondissements, 581 municipalities, 10 provinces (+ Brussels-Capital), or 3
        arrondissements in the 'test' case (Antwerp, Brussels, Gent)

    """

    initN_df=pd.read_csv(os.path.join(data_path, 'interim/demographic/initN_' + spatial + '.csv'), index_col=[0])
    NIS = initN_df.index.values

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
    -------
    model : object
        BIOMATH model object

    """
    # Use posterior samples of fitted parameters
    parameter_dictionary['beta'] = np.random.choice(samples_dict['beta'],1,replace=False)
    idx,parameter_dictionary['l'] = random.choice(list(enumerate(samples_dict['l'])))
    parameter_dictionary['tau'] = samples_dict['tau'][idx]
    parameter_dictionary['prevention'] = samples_dict['prevention'][idx]
    return parameter_dictionary

def social_policy_func(t,param,policy_time,policy1,policy2,tau,l):
    """
    Delayed ramp social policy function to implement a gradual change between policy1 and policy2.
    
    Parameters
    ----------
    t : int
        Time parameter. Runs simultaneously with simulation time
    param : 
        Currently obsolete parameter that may be used in a future stage
    policy_time : int
        Time in the simulation at which a new policy is imposed
    policy1 : float or int or list or matrix
        Value corresponding to the policy before t = policy_time (e.g. full mobility)
    policy2 : float or int or list or matrix (same dimensions as policy1)
        Value corresponding to the policy after t = policy_time (e.g. 50% mobility)
    tau : int
        Delayed ramp parameter: number of days before the new policy has any effect
    l : int
        Delayed ramp parameter: number of days after t = policy_time + tau the new policy reaches full effect (policy2)
        
    Return
    ------
    state : float or int or list or matrix
        Either policy1, policy2 or an intermediate state.
        
    """
    # Nothing changes before policy_time
    if t < policy_time:
        state = policy1
    # From t = policy time onward, the delayed ramp takes effect toward policy2
    else:
        # Time starting at policy_time
        tt = t-policy_time
        if tt <= tau:
            state = policy1
        if (tt > tau) & (tt <= tau + ll):
            intermediate = (policy2 - policy1) / ll * (tt - tau) + policy1
            state = intermediate
        if tt > tau + ll:
            state = policy2
    return state

def dens_dep(rho, xi=0.01):
    """
    A function used by Arenas et al. (2020) and justified by Hu et al. (2013) (https://pubmed.ncbi.nlm.nih.gov/23665296/)

    Parameters
    ----------
    rho : population density
    xi : scale parameter. Default value is 0.01 square kilometer

    Returns
    -------
    f : density dependence value, ranges between 1 and 2
    """

    f = 1 + ( 1 - np.exp(-xi * rho) )

    return f

def read_areas(spatial='arr'):
    """
    Reads full CSV with area per NIS code

    Parameters
    ----------
    spatial : str
        Choose between municipalities ('mun'), arrondissements ('arr'), provinces ('prov') or Antwerp-Brussel-Gent ('test').
        Defaults is 'arr'

    Returns
    -------
    areas : dictionary
        NIS codes are keys, values are population in square meters
    """

    areas_df = pd.read_csv(os.path.join(data_path, 'interim/demographic/area_' + spatial + '.csv'), index_col='NIS')
    areas = areas_df['area'].to_dict()

    return areas

def read_pops(spatial='arr'):
    """
    Reads initial population per age and per area

    Parameters
    ----------
    spatial : str
        choose geographical aggregation. Pick between 'arr', 'mun', 'prov', or 'test'. Default is 'arr'.

    Returns
    -------
    pops : dictionary
        NIS codes are keys, values are dictionaries. The inner dictionary has population classes as keys and
        population per age class and per NIS code as values.
        Age classes are [0,10), [10,20), ... , [80, 110)
    """

    pops_df = pd.read_csv(os.path.join(data_path, 'interim/demographic/initN_' + spatial + '.csv'), index_col='NIS')
    pops = pops_df.T.to_dict()

    return pops
