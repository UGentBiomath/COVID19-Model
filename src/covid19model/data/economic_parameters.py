import os
import pandas as pd
import numpy as np
import xarray as xr

def get_economic_model_parameters():
    """
    Extracts and returns the parameters for the economic model

    This function returns a dictionary with all parameters needed to run the economic model.

    Returns
    -------

    pars_dict : dictionary
        contains the values of all economic parameters

        Parameters
        ------------
        IO: input-output matrix
        x_0 : sectoral output during business-as-usual
        c_0 : household demand during business-as-usual
        f_0 : other final demand during business-as-usual
        n : desired stock
        c_s : consumer demand shock during lockdown
        f_s : other final demand shock during lockdown
        l_0 : sectoral employees during business-as-usual
        l_s : sectoral employees during lockdown
        C : matrix of crictical inputs

    Example use
    -----------
    parameters = get_economic_model_parameters()
    """

    abs_dir = os.path.dirname(__file__)
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/economical/model_parameters")

    # Initialize parameters dictionary
    pars_dict = {}

    # IO_NACE64.csv
    df = pd.read_csv(os.path.join(par_interim_path,"IO_NACE64.csv"), sep=',',header=[0],index_col=[0])
    pars_dict['IO'] = df.values/365

    # Others.csv
    df = pd.read_csv(os.path.join(par_interim_path,"other_parameters.csv"), sep=',',header=[1],index_col=[0])
    pars_dict['x_0'] = np.array(df['Sectoral output (M€/y)'].values)/365
    pars_dict['O_j'] = np.array(df['Intermediate demand (M€/y)'].values)/365
    pars_dict['l_0'] = np.array(df['Labor compensation (M€/y)'].values)/365
    pars_dict['c_0'] = np.array(df['Household demand (M€/y)'].values)/365
    pars_dict['f_0'] = np.array(df['Total other demand (M€/y)'].values)/365
    pars_dict['n'] = np.expand_dims(np.array(df['Desired stock (days)'].values), axis=1)
    # shock vectors
    pars_dict['l_s'] = 1-np.array((df['Telework (%)'].values+df['Mix (%)'].values+df['Workplace (%)'].values)/100)
    pars_dict['c_s'] = -np.array(df['Consumer demand shock (%)'].values)/100
    pars_dict['f_s'] = -np.array(df['Other demand shock (%)'].values)/100
    pars_dict['on_site'] = np.array(df['On-site consumption (-)'].values)

    # IHS_critical_NACE64.csv
    df = pd.read_csv(os.path.join(par_interim_path,"IHS_critical_NACE64.csv"), sep=',',header=[0],index_col=[0])
    pars_dict['C'] = df.values

    # Derived variables
    # ~~~~~~~~~~~~~~~~~

    # Matrix of technical coefficients
    A = np.zeros([pars_dict['IO'].shape[0],pars_dict['IO'].shape[0]])
    for i in range(pars_dict['IO'].shape[0]):
        for j in range(pars_dict['IO'].shape[0]):
            A[i,j] = pars_dict['IO'][i,j]/pars_dict['x_0'][j]
    pars_dict['A'] = A

    # Stock matrix under business as usual
    S_0 = np.zeros([pars_dict['IO'].shape[0],pars_dict['IO'].shape[0]])
    for i in range(pars_dict['IO'].shape[0]):
        for j in range(pars_dict['IO'].shape[0]):
            S_0[i,j] = pars_dict['IO'][i,j]*pars_dict['n'][j]
    pars_dict['S_0'] = S_0

    return pars_dict

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

    # Define path to conversion matrices
    abs_dir = os.path.dirname(__file__)
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/economical/model_parameters")

    # Load dataframe containing matrices
    if from_to == 'NACE21_NACE10':
        return np.array(pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 21 to NACE 10', header=[0], index_col=[0]).values)
    elif from_to == 'NACE38_NACE21':
        return np.array(pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 38 to NACE 21', header=[0], index_col=[0]).values)
    elif from_to == 'NACE64_NACE38':
        return np.array(pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 64 to NACE 38', header=[0], index_col=[0]).values)
    elif from_to == 'WIOD55_NACE64':
        return np.array(pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 64 to WIOD 55', header=[0], index_col=[0]).values)
    else:
        raise ValueError(
                        "conversion matrix '{0}' not recognized \n"
                        "valid arguments are: 'NACE21_NACE10', 'NACE38_NACE21', 'NACE64_NACE38', 'WIOD55_NACE64'".format(from_to)
                    )

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
    labels = read_economic_labels('WIOD55')
    """
    
     # Define path to conversion matrices
    abs_dir = os.path.dirname(__file__)
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/economical/model_parameters")

    # Load dataframe containing matrices
    if classification_name == 'NACE64':
        return list(pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 64 to NACE 38', header=[0], index_col=[0]).columns.values)
    elif classification_name == 'NACE38':
        return list(pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 64 to NACE 38', header=[0], index_col=[0]).index.values)
    elif classification_name == 'NACE21':
        return list(pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 21 to NACE 10', header=[0], index_col=[0]).columns.values)
    elif classification_name == 'NACE10':
        return list(pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 21 to NACE 10', header=[0], index_col=[0]).index.values)
    elif classification_name == 'WIOD55':
        return list(pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'WIOD 55 to NACE 64', header=[0], index_col=[0]).columns.values)
    else:
        raise ValueError(
                        "conversion matrix '{0}' not recognized \n"
                        "valid arguments are: 'NACE64', 'NACE38', 'NACE21', 'NACE10', 'WIOD55"
                    )

def aggregate_simulation(simulation_in, desired_agg):
    """ A function to convert a simulation of the economic IO model on the NACE64 level to another classification
    
    Input
    =====
    simulation_in: xarray.DataArray
        Simulation result (NACE64 level). Obtained from a pySODM xarray.Dataset simulation result by using: xarray.Dataset[state_name]
    
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
        simulation_out = xr.DataArray(np.matmul(get_sectoral_conversion_matrix('NACE64_NACE38'), simulation_in.values),
                                      dims = ['NACE38', 'date'],
                                      coords = dict(NACE38=(['NACE38'], get_sector_labels('NACE38')),
                                                    date=simulation_in.coords['date']))
    elif desired_agg == 'NACE21':
        simulation_out = xr.DataArray(np.matmul(get_sectoral_conversion_matrix('NACE38_NACE21'), np.matmul(get_sectoral_conversion_matrix('NACE64_NACE38'), simulation_in.values)),
                              dims = ['NACE21', 'date'],
                              coords = dict(NACE38=(['NACE21'], get_sector_labels('NACE21')),
                                            date=simulation_in.coords['date']))
    elif desired_agg == 'NACE10':
        simulation_out = xr.DataArray(np.matmul(get_sectoral_conversion_matrix('NACE21_NACE10'),np.matmul(get_sectoral_conversion_matrix('NACE38_NACE21'), np.matmul(get_sectoral_conversion_matrix('NACE64_NACE38'), simulation_in.values))),
                              dims = ['NACE10', 'date'],
                              coords = dict(NACE38=(['NACE10'], get_sector_labels('NACE10')),
                                            date=simulation_in.coords['date']))
    else:
        raise ValueError("Valide desired aggregations are 'NACE38', 'NACE21', 'NACE10'")
    
    return simulation_out