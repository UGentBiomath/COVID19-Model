import os
import pandas as pd
import numpy as np
import xarray as xr

# Set path to interim data folder
abs_dir = os.path.dirname(__file__)
par_interim_path = os.path.join(abs_dir, "../../../data/EPNM/interim/")

def get_sectoral_conversion_matrix(from_to):
    """
    Returns conversion matrices to more easily aggregate data from different sector classifications. F.i. converting from NACE 64 to WIOD 55 classification.

    Parameters
    ----------
    from_to : string
        Desired conversion matrix. Valid options are: 'NACE21_NACE10', 'NACE38_NACE21', 'NACE64_NACE38', 'WIOD55_NACE64'

    Returns
    -------
    conversion matrix : np.array

    Example use
    -----------
    mat = get_conversion_matrix('WIOD55_NACE64')
    """

    # Load dataframe containing matrices
    if from_to == 'NACE21_NACE10':
        return np.array(pd.read_excel(os.path.join(par_interim_path,"model_parameters/conversion_matrices.xlsx"), sheet_name = 'NACE 21 to NACE 10', header=[0], index_col=[0]).values)
    elif from_to == 'NACE38_NACE21':
        return np.array(pd.read_excel(os.path.join(par_interim_path,"model_parameters/conversion_matrices.xlsx"), sheet_name = 'NACE 38 to NACE 21', header=[0], index_col=[0]).values)
    elif from_to == 'NACE64_NACE38':
        return np.array(pd.read_excel(os.path.join(par_interim_path,"model_parameters/conversion_matrices.xlsx"), sheet_name = 'NACE 64 to NACE 38', header=[0], index_col=[0]).values)
    elif from_to == 'WIOD55_NACE64':
        return np.array(pd.read_excel(os.path.join(par_interim_path,"model_parameters/conversion_matrices.xlsx"), sheet_name = 'NACE 64 to WIOD 55', header=[0], index_col=[0]).values)
    else:
        raise ValueError(
                        "conversion matrix '{0}' not recognized \n"
                        "valid arguments are: 'NACE21_NACE10', 'NACE38_NACE21', 'NACE64_NACE38', 'WIOD55_NACE64'".format(from_to)
                    )

def get_sector_names(classification_name):
    """
    Returns the sector names of the desired classification.

    Input
    =====
    classification_name : string
        Desired classification. Valid options are: NACE64, NACE38, NACE21, NACE10, WIOD55

    Output
    ======
    names : list

    Example use
    ===========
    names = get_sector_names('NACE21')
    """

    if classification_name not in ['NACE64','NACE38','NACE21','NACE10','WIOD55']:
        raise ValueError(
                        "classification '{0}' not recognized \n"
                        "valid arguments are: 'NACE64', 'NACE38', 'NACE21', 'NACE10' and 'WIOD55'"
                    ).format(classification_name)
    else:
        return list(pd.read_excel(os.path.join(par_interim_path,"model_parameters/sector_labels_names.xlsx"), sheet_name = classification_name, header=[0], index_col=[0])['name'].values)

def get_sector_labels(classification_name):
    """
    Returns the sector labels of the desired classification.

    Input
    =====
    classification_name : string
        Desired classification. Valid options are: NACE64, NACE38, NACE21, NACE10, WIOD55

    Output
    ======
    labels : list

    Example use
    ===========
    labels = get_sector_labels('NACE21')
    """
    if classification_name not in ['NACE64','NACE38','NACE21','NACE10','WIOD55']:
        raise ValueError(
                        "classification '{0}' not recognized \n"
                        "valid arguments are: 'NACE64', 'NACE38', 'NACE21', 'NACE10' and 'WIOD55'"
                    ).format(classification_name)
    else:
        return list(pd.read_excel(os.path.join(par_interim_path,"model_parameters/sector_labels_names.xlsx"), sheet_name = classification_name, header=[0], index_col=[0]).index.values)

def aggregate_simulation(simulation_in, desired_agg):
    """ A function to convert a simulation of the EPNM on the NACE64 level to another economical classification
    
    Input
    =====
    simulation_in: xarray.DataArray
        Simulation result (NACE64 level). Obtained from a pySODM xarray. Dataset simulation result by using: xarray.Dataset[state_name]
    
    Output
    ======
    simulation_out: xarray.DataArray
        Simulation result
        
    Remarks
    =======
    The economic IO model does not support the use of draws because xarray.concat does not support 
    a concatenation on variables with repeated dimensions (stock matrix is 2D!)
    No support was implemented here for the dimension 'draws' 
    """

    if desired_agg == 'NACE38':
        simulation_out = xr.DataArray(np.matmul(simulation_in.values, np.transpose(get_sectoral_conversion_matrix('NACE64_NACE38'))),
                                      dims = ['date', 'NACE38'],
                                      coords = dict(NACE38=(['NACE38'], get_sector_labels('NACE38')),
                                                    date=simulation_in.coords['date']))
    elif desired_agg == 'NACE21':
        simulation_out = xr.DataArray(np.matmul(np.matmul(simulation_in.values, np.transpose(get_sectoral_conversion_matrix('NACE64_NACE38'))), np.transpose(get_sectoral_conversion_matrix('NACE38_NACE21'))),
                              dims = ['date', 'NACE21'],
                              coords = dict(NACE21=(['NACE21'], get_sector_labels('NACE21')),
                                            date=simulation_in.coords['date']))
    elif desired_agg == 'NACE10': 
        simulation_out = xr.DataArray(np.matmul(np.matmul(np.matmul(simulation_in.values, np.transpose(get_sectoral_conversion_matrix('NACE64_NACE38'))), np.transpose(get_sectoral_conversion_matrix('NACE38_NACE21'))), np.transpose(get_sectoral_conversion_matrix('NACE21_NACE10'))),
                              dims = ['date', 'NACE10'],
                              coords = dict(NACE10=(['NACE10'], get_sector_labels('NACE10')),
                                            date=simulation_in.coords['date']))
    else:
        raise ValueError("Valide desired aggregations are 'NACE38', 'NACE21', 'NACE10'")
    
    return simulation_out