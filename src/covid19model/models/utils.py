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
