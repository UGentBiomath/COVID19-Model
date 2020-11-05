import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import zarr

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
        return policy0
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
       "second" wave 
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
    t1 = pd.to_datetime('2020-05-04') # reopening industry
    t2 = pd.to_datetime('2020-05-06') # merging of two bubbels
    t3 = pd.to_datetime('2020-05-11') # reopening of businesses
    t4 = pd.to_datetime('2020-05-18') # partial reopening schools
    t5 = pd.to_datetime('2020-06-04') # reopening schools, bars, restaurants
    t6 = pd.to_datetime('2020-07-01') # school holidays, gatherings 15 people, cultural event
    t7 = pd.to_datetime('2020-07-31') # "second" wave
    t8 = pd.to_datetime('2020-09-01') # opening schools
    
    if t <= start_date + tau_days:
        return policy0
    elif start_date + tau_days < t <= start_date + tau_days + l_days:
        return ramp_fun(policy0, prevention*policy1, t, tau_days, l, start_date)
    elif start_date + tau_days + l_days < t <= t1: 
        return prevention*policy1 # lockdown
    elif t1 < t <= t2:
        return prevention*policy2 # re-opening industry
    elif t2 < t <= t3:
        return prevention*policy3
    elif t3 < t <= t4:
        return prevention*policy4
    elif t4 < t <= t5:
        return prevention*policy5
    elif t5 < t <= t6:
        return prevention*policy6
    elif t6 < t <= t7:
        return prevention*policy7
    elif t7 < t <= t8:
        return prevention*policy8
    elif t8 < t:
        return prevention*policy9

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
        if (tt > tau) & (tt <= tau + l):
            intermediate = (policy2 - policy1) / l * (tt - tau) + policy1
            state = intermediate
        if tt > tau + l:
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

def save_sim(out, name, group, new=False, descr=None, verbose=True):
    """Save the output xarray of a simulation as a zarr file for future reference. Especially pertinent when simulation takes a long time to process, such that simulations may be saved in a database. Note that data variable values are saved as Disk arrays (but open_sim converts them back to Numpy arrays)
    
    Parameters
    ----------
    out : xarray object
        output of the base sim function: xarray with dimensions 'time' and 'draws', and possibly 'Nc' (age stratification) and 'place' (geographical stratification), and attributes "parameters" (which holds a dictionary)
    name : string
        Name under which to save the zarr directory. If no directory path is provided, the directory is saved in the current location.
    group : string
        Name of the group in the zarr file to save the particular model scenario under
    new : Boolean
        If True, creates a new zarr file with name provided in the function argument. False by default.
    descr : string
        Description of the particular simulation inside the group. This is saved as an attribute.
    verbose : Boolean
        Print under which file and group name the xarray data has been saved. True by default.
    """
    
    ### EXCEPTIONS
    
    # Check the zarr name format
    if (type(name) != str) or (name[-5:] != '.zarr'):
        raise Exception(f"The 'name' parameter value '{name}' is invalid. 'name' should be a string with a .zarr extension.")
    
    # Check whether the zarr directory exists
    if (new == False) and not os.path.isdir(name):
        raise Exception(f"The 'name' parameter value '{name}' does not exist. If you want to create a new zarr file with this name, include 'new=True' in the function arguments.")
    
    if (new == True) and os.path.isdir(name):
        raise Exception(f"Cannot create new zarr file with name '{name}' because it already exists. Either add a group to this zarr file, or create a new zarr directory with a different name.")
        
    if os.path.isdir(name + '/' + group):
        raise Exception(f"The group '{group}' already exists in the zarr directory '{name}'. Create a new group or a new zarr directory.")
    
    # Check description properties
    if not descr:
        descr = 'No description provided'
    if type(descr) != str:
        raise Exception(f"The description '{descr}' is invalid. The 'descr' parameter value should be a string.")
    
    ### CORE
    
    # Save parameters dictionary one level up (otherwise it cannot be saved in zarr format)
    out.attrs = out.attrs['parameters']
    
    # Add description attribute
    out.attrs['description'] = descr
    
    # Save to zarr under the right name/group
    out.to_zarr(name, group=group)
    
    # Include message if verbose (default)
    if verbose:
        if new:
            message = f"Saved simulation output in newly created zarr directory '{name}' under the group '{group}'"
        else:
            message = f"Added simulation output to zarr directory '{name} under the group '{group}'"
        print(message)
        print("Description:")
        print("------------")
        print(f"'{descr}'")
    
    
def open_sim(name, group, verbose=True):
    """Open the saved simulation output xarray and (optionally) display the main characteristics. Note: can only handle parameter attributes + description (additional attributes will be wrongly categorised under 'parameters'). Note: changes the order of the parameters in the dictionary
    
    Parameters
    ----------
    name : str
        directory and name of the main zarr file
    group : str
        name of the group within the zarr directory in which the xarray is saved
    verbose : Boolean
        Print description and dimensional information of the simulation. Default is True.
    """
    
    ### EXCEPTIONS
    
    # Check the zarr name format
    if (type(name) != str) or (name[-5:] != '.zarr'):
        raise Exception(f"The 'name' parameter value '{name}' is invalid. 'name' should be a string with a .zarr extension.")
    
    # Verify whether the file and group exists
    if not os.path.isdir(name + '/' + group):
        raise Exception(f"The group '{name}/{group}' does not exist. Check the 'name' and 'group' parameter values.")
    
    ### CORE
    
    # Open xarray from zarr
    out = xr.open_zarr(name, group=group)
    if 'description' in out.attrs.keys():
        # Copy and delete description from attributes
        descr = out.attrs.pop('description')

    # Repack parameters into a dictionary
    param_dict = dict({})
    for key in out.attrs.copy():
        param_dict[key] = out.attrs[key]
        out.attrs.pop(key)
    out.attrs['parameters'] = param_dict
    
    # Convert Disk arrays values of data variables back to Numpy arrays
    for var in out:
        out[var].values = np.array(out[var].values)
        
    if verbose:
        print(f"Opened simulation output that is saved in {name}/{group}")
        print('')
        print('Dimensions:')
        print('-----------')
        for key, value in out.dims.items(): print(f"{key}: {value}")
        print('')
        print('Description')
        print('-----------')
        if descr:
            print(f"'{descr}'")
        else:
            print("WARNING: No 'description' attribute in opened model output. Has it been saved properly, using the save_sim() function?")
            
    return out