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

def stratify_beta(beta_R, beta_U, beta_M, agg, RU_threshold=400, UM_threshold=4000):
    """
    Function that returns a spatially stratified infectivity parameter. IMPORTANT: this assumes that throughout the model, all NIS values are in order (e.g. 11000 to 93000). Currently hard-coded on threshold densities of 400/km2 and 4000/km2. Indices indicated in order of density.
    
    Input
    -----
    beta_R : float
        Infecitivity in rural areas
    beta_U : float
        Infectivity in urban areas
    beta_M : float
        Infectivity in metropolitan areas
    agg : str
        Aggregation level. Either 'prov', 'arr' or 'mun', for provinces, arrondissements or municipalities, respectively.
    RU_threshold : float
        Threshold population density to distinguish between rural and urbanised regions. Default: 400/km2
    UM_threshold : float
        Threshold population density to distinguish between urbanised and metropolitan regions. Default: 4000/km2

    Returns
    -------
    beta : np.array of floats
        Array with length fitting to aggregation level agg, and three degrees of freedom depending on beta_R, beta_U, beta_M
    """
    # Exceptions
    if agg not in ['prov', 'arr', 'mun']:
        raise Exception(f"Aggregation level {agg} not recognised. Choose between 'prov', 'arr' or 'mun'.")
    if (RU_threshold >= UM_threshold) or (RU_threshold < 0) or (UM_threshold < 0):
        raise Exception(f"RU_threshold ({RU_threshold}) must be smaller than UM_threshold ({UM_threshold}) and both values must be positive (units of people/km2).")

    # Load areas in ordered array in km2
    areas = (pd.read_csv(os.path.join(data_path, 'interim/demographic/area_' + agg + '.csv'))['area']/1e6).values
    # Load populations in ordered array
    pops = pd.read_csv(os.path.join(data_path, 'interim/demographic/initN_' + agg + '.csv'))['total'].values
    # Define densities
    dens = pops/areas

    # Initialise and fill beta array
#     beta = np.empty(len(dens))
    beta = np.array([])
    for i in range(len(dens)):
        if dens[i] < RU_threshold:
#             beta[i] = beta_R
            beta = np.append(beta, beta_R)
        elif RU_threshold <= dens[i] < UM_threshold:
#             beta[i] = beta_U
            beta = np.append(beta, beta_U)
        else:
#             beta[i] = beta_M
            beta = np.append(beta, beta_M)

    return beta

def initial_state(dist='bxl', agg='arr', number=1, age=-1):
    """
    Function determining the initial state of a model compartment. Note: currently only works with 9 age classes.
    
    Input
    -----
    dist: str or int
        Spatial distribution of the initial state. Choose between 'bxl' (Brussels-only, default), 'hom' (homogeneous), or 'data' (data-inspired), or choose a NIS code (5-digit int) corresponding to the aggregation level.
    agg: str
        Level of spatial aggregation. Choose between 'mun' (581 municipalities), 'arr' (43 arrondissements, default), or 'prov' (10+1 provinces)
    number: int
        Total number of people initialised in the compartment. 1 by default. Note that this generally needs to be changed if dist != 'bxl'
    age: int
        Integer larger than -1. If -1 (default), random ages are chosen (following demography). If a positive integer is chosen, the age class is the decade this age falls into (e.g. 43 is age class 4). Everything over 80 is read as age class 80+.

    Returns
    -------
    init: np.array containing integers
        The initial state with 11, 43 or 581 rows and 9 columns, representing the initial age and spatial distribution of people in a particular SEIR compartment.
    """
    
    # Raise exceptions if input is wrong
    if not isinstance(dist, int) and (dist not in ['bxl', 'hom', 'data']):
        raise Exception(f"Input dist={dist} is not recognised. Choose between 'bxl', 'hom' or 'data', or pick a NIS code (integer).")
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception(f"Aggregation level {agg} not recognised. Choose between 'prov', 'arr' or 'mun'.")
    if not ((number > 0) and float(number).is_integer()):
        raise Exception(f"Input number={number} is not acceptable. Choose a natural number.")
    if not ((-1 <= age) and float(number).is_integer()):
        raise Exception(f"Input age={age} is not acceptable. Choose an integer -1 (random) or positive (reduces to age decade).")

    # Turn age into decade
    age = age//10
    if age > 8:
        age = 8
        
    # Load population distribution. Fixed number of age classes (9)
    pops = read_pops(spatial=agg, return_matrix=True, drop_total=True)
        
    # Initialise matrix with proper dimensions.
    G = pops.shape[0]
    init = np.zeros((G,9))
    
    # Case for chosen NIS code
    if isinstance(dist,int):
        # Find coordinate of chosen NIS code
        gg=np.where(read_coordinates_nis(spatial=agg)==dist)[0][0]
        init[gg] = _initial_age_dist(number, age, pops[gg])
    
    # Case for Brussels
    elif dist=='bxl':
        # Find coordinate of bxl NIS code
        if agg in ['arr', 'prov']:
            gg=np.where(read_coordinates_nis(spatial=agg)==21000)[0][0]
        else:
            gg=np.where(read_coordinates_nis(spatial=agg)==21004)[0][0] # Choice is made for historical Brussels
        init[gg]= _initial_age_dist(number, age, pops[gg])
    
    # Case for data-inspired initial conditions, based on highest concentration in first peak
    # Note: the cases are spread almost equally (local population is only a secondary attention point)
    elif dist=='data':
        # Find coordinates of NIS codes for Alken, Sint-Truiden, Quévy
        coordinates_nis=read_coordinates_nis(spatial=agg)
        gg_array=[]
        if agg == 'arr':
            gg_array.append(np.where(coordinates_nis==73000)[0][0]) # arrondissement Tongeren (Alken)
            gg_array.append(np.where(coordinates_nis==71000)[0][0]) # arrondissement Hasselt (Sint-Truiden)
            gg_array.append(np.where(coordinates_nis==53000)[0][0]) # arrondissement Mons (Quévy)
        elif agg == 'prov':
            gg_array.append(np.where(coordinates_nis==70000)[0][0]) # province Limburg (Alken)
            gg_array.append(np.where(coordinates_nis==70000)[0][0]) # province Limburg (Sint-Truiden), count double
            gg_array.append(np.where(coordinates_nis==50000)[0][0]) # province Hainaut (Quévy)
        else:
            gg_array.append(np.where(coordinates_nis==73001)[0][0]) # municipality Alken
            gg_array.append(np.where(coordinates_nis==71053)[0][0]) # municipality Sint-Truiden
            gg_array.append(np.where(coordinates_nis==53084)[0][0]) # municipality Quévy
        pops_tot=np.array([pops[gg_array[i]].sum(axis=0) for i in range(3)])
        init_all_ages = np.array([number//3 for i in range(3)])
        for i in range(number%3):
            jj = np.where(pops_tot==np.sort(pops_tot)[-i-1])[0][0] # find index of highest populations
            init_all_ages[jj] += 1 # add remaining initial states to region with highest population first
        for i in range(3):
            init[gg_array[i]] += _initial_age_dist(init_all_ages[i], age, pops[gg_array[i]])
            
    # Case for homogeneous initial conditions: equal country-wide distribution
    # Note: the cases are spread almost equally (local population is only a secondary attention point)
    else:
        pops_tot = pops.sum(axis=1)
        init_all_ages = np.array([number//G for i in range(G)])
        pops_tot_sorted = np.sort(pops_tot)
        for i in range(number%G):
            jj = np.where(pops_tot==pops_tot_sorted[i-1])[0][0] # find index of highest populations
            init_all_ages[jj] += 1 # add remaining initial states to region with highest population first
        for i in range(G):
            init[i] += _initial_age_dist(init_all_ages[i], age, pops[i])
            
    return init

def _initial_age_dist(number, age, pop):
    """
    Help function for initial_state, for the distribution of the initial state over the 9 age classes.
    
    Input
    -----
    number: int
        Total number of people initialised in the compartment.
    age: int
        Integer ranging from -1 to 8. If -1, random ages are chosen (following demography). If 0-8 is chosen, the number corresponds to the age decade (e.g. 1 = ages 10-19)
    pop: np.array
        Contains population in the various age classes
        
    Returns
    -------
    init_per_age: np.array with integers of dimension 9
        The distribution of the people in a particular state in one particular region per age    
    """
    # Initialise age vector
    init_per_age = np.zeros(9)
    
    # Return vector with people in one particular age class
    if age > -1:
        init_per_age[int(age)] = number
    
    else:
        indices = list(range(0,9))
        probs = pop/pop.sum(axis=0)
        index_choices = np.random.choice(indices, p=probs, size=number)
        unique, counts = np.unique(index_choices, return_counts=True)
        index_dict = dict(zip(unique, counts))
        for key in index_dict:
            init_per_age[key] = index_dict[key]
    
    return init_per_age

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

def draw_sample_COVID19_SEIRD(parameter_dictionary,samples_dict, to_sample=['beta','l','tau','prevention']):
    """
    A function to draw parameter samples obtained with MCMC during model calibration and assign them to the parameter dictionary of the model.
    Tailor-made for the BIOMATH COVID-19 SEIRD model.

    Parameters
    ----------
    param_dict : dict
        Parameter dictionary of the BIOMATH COVID-19 model.
    
    samples_dict : dictionary
        Dictionary containing the MCMC samples of the BIOMATH COVID-19 model parameters: beta, l and tau.

    to_sample : list
        list of parameters to sample, default ['beta','l','tau','prevention']

    Returns
    -------
    param_dict : dict
        Parameter dictionary of the BIOMATH COVID-19 model.

    """
    
    if 'beta' in to_sample:
        parameter_dictionary['beta'] = np.random.choice(samples_dict['beta'],1,replace=False)
    if 'l' in to_sample:
        idx,parameter_dictionary['l'] = random.choice(list(enumerate(samples_dict['l'])))
        parameter_dictionary['tau'] = samples_dict['tau'][idx]
        if 'prevention' in to_sample:
            parameter_dictionary['prevention'] = samples_dict['prevention'][idx]
    return parameter_dictionary

def draw_sample_COVID19_SEIRD_google(param_dict,samples_dict,google=False):
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
    idx,param_dict['l'] = random.choice(list(enumerate(samples_dict['l'])))
    param_dict['tau'] = samples_dict['tau'][idx]    
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

def read_pops(spatial='arr',return_matrix=False,drop_total=False):
    """
    Reads initial population per age and per area

    Parameters
    ----------
    spatial : str
        choose geographical aggregation. Pick between 'arr', 'mun', 'prov', or 'test'. Default is 'arr'.
    return_matrix : boolean
        if True, return np.array instead of dictionary
    drop_total : boolean
        if True, drop the 10th column containing the sums of the rows (total population)

    Returns
    -------
    pops : dictionary
        NIS codes are keys, values are dictionaries. The inner dictionary has population classes as keys and
        population per age class and per NIS code as values.
        Age classes are [0,10), [10,20), ... , [80, 110)
    """

    pops_df = pd.read_csv(os.path.join(data_path, 'interim/demographic/initN_' + spatial + '.csv'), index_col='NIS')
    if drop_total:
        pops_df.drop(columns='total', inplace=True)
    if return_matrix:
        pops = pops_df.values
    else:
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
