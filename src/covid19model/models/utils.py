import numpy as np
import pandas as pd
import os

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

def read_coordinates_nis(name='arrond'):
    """
    A function to extract from /data/interim/demographic/initN_arrond.csv the list of arrondissement NIS codes
    
    Parameters
    ----------
    name : str
        choose geographical aggregation. Pick between 'arrond', 'municip', 'province'. Default is 'arrond'.

    Returns
    -------
     NIS: list
        a list containing the NIS codes of the 43 Belgian arrondissements

    """
    
    initN_df=pd.read_csv(os.path.join(data_path, 'interim/demographic/initN_' + name + '.csv'), index_col=[0])
    NIS = initN_df.index.values

    return NIS

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

def read_areas(name='arrond'):
    """
    Reads full CSV with area per NIS code
    
    Parameters
    ----------
    name : str
        Choose between municipalities ('municip'), arrondissements ('arrond') or provinces ('province'). Defaults is 'arrond'
    
    Returns
    -------
    areas : dictionary
        NIS codes are keys, values are population in square meters
    """
    
    areas_df = pd.read_csv(os.path.join(data_path, 'interim/demographic/area_' + name + '.csv'), index_col='NIS')
    areas = areas_df['area'].to_dict()
    
    return areas

def read_pops(name='arrond'):
    """
    Reads initial population per age and per area
    
    Parameters
    ----------
    name : str
        choose geographical aggregation. Pick between 'arrond', 'municip', 'province'. Default is 'arrond'.
        
    Returns
    -------
    pops : dictionary
        NIS codes are keys, values are dictionaries. The inner dictionary has population classes as keys and
        population per age class and per NIS code as values.
        Age classes are [0,10), [10,20), ... , [80, 110)
    """
    
    pops_df = pd.read_csv(os.path.join(data_path, 'interim/demographic/initN_' + name + '.csv'), index_col='NIS')
    pops = pops_df.T.to_dict()
    
    return pops