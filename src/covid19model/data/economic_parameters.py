import os
import pandas as pd
import numpy as np

def get_economic_parameters():
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
    parameters = get_economic_parameters()
    """

    abs_dir = os.path.dirname(__file__)
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/economical/")

    # Initialize parameters dictionary
    pars_dict = {}

    # IO_NACE64.csv
    df = pd.read_csv(os.path.join(par_interim_path,"IO_NACE64.csv"), sep=',',header=[0],index_col=[0])
    pars_dict['IO'] = df.values/365

    # Others.csv
    df = pd.read_csv(os.path.join(par_interim_path,"others.csv"), sep=',',header=[1],index_col=[0])
    pars_dict['x_0'] = np.expand_dims(np.array(df['Sectoral output (M€/y)'].values), axis=1)/365
    pars_dict['O_j'] = np.expand_dims(np.array(df['Intermediate demand (M€/y)'].values), axis=1)/365
    pars_dict['l_0'] = np.expand_dims(np.array(df['Labor compensation (M€/y)'].values), axis=1)/365
    pars_dict['l_s'] = 1-np.expand_dims(np.array((df['Telework (%)'].values+df['Mix (%)'].values+df['Workplace (%)'].values)/100), axis = 1)
    pars_dict['c_0'] = np.expand_dims(np.array(df['Household demand (M€/y)'].values), axis=1)/365
    pars_dict['f_0'] = np.expand_dims(np.array(df['Total other demand (M€/y)'].values), axis=1)/365
    pars_dict['n'] = np.expand_dims(np.array(df['Desired stock (days)'].values), axis=1)
    pars_dict['c_s'] = -np.expand_dims(np.array(df['Consumer demand shock (%)'].values), axis=1)/100
    pars_dict['f_s'] = -np.expand_dims(np.array(df['Other demand shock (%)'].values), axis=1)/100
    #pars_dict['n_0'] = np.expand_dims(np.array(df['Employees (x1000)'].values), axis=1)*1000
    #pars_dict['n_s'] = np.expand_dims(np.array(df['Employees (x1000)'].values), axis=1)*1000*np.expand_dims(np.array((df['Telework (%)'].values+df['Mix (%)'].values+df['Workplace (%)'].values)/100), axis = 1)
    pars_dict['on_site'] = np.expand_dims(np.array(df['On-site consumption (-)'].values), axis=1)
    #pars_dict['TS_0'] = np.expand_dims(np.array(df['Taxes minus subsidies (M€/y)'].values), axis=1)
    #pars_dict['FC_0'] = np.expand_dims(np.array(df['Consumption fixed capital (M€/y)'].values), axis=1)
    #pars_dict['I_0'] = np.expand_dims(np.array(df['Imports (M€/y)'].values), axis=1)
    #pars_dict['OS_0'] = np.expand_dims(np.array(df['Operating surplus and mixed income (M€/y)'].values), axis=1)
    #pars_dict['GVA_0'] = np.expand_dims(np.array(df['Gross value added (M€/y)'].values), axis=1)
    #pars_dict['GDPi_0'] = np.expand_dims(np.array(df['Gross domestic product i (M€/y)'].values), axis=1)

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

    # Consumer preference vector
    pars_dict['theta_0'] = pars_dict['c_0']/sum(pars_dict['c_0'])

    return pars_dict

def get_conversion_matrix(from_to):
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
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/economical/")

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

def read_economic_labels(class_name):
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
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/economical/")

    # Load dataframe containing matrices
    if class_name == 'NACE64':
        return pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 64 to NACE 38', header=[0], index_col=[0]).columns.values
    elif class_name == 'NACE38':
        return pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 64 to NACE 38', header=[0], index_col=[0]).index.values
    elif class_name == 'NACE21':
        return pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 21 to NACE 10', header=[0], index_col=[0]).columns.values
    elif class_name == 'NACE10':
        return pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'NACE 21 to NACE 10', header=[0], index_col=[0]).index.values
    elif class_name == 'WIOD55':
        return pd.read_excel(os.path.join(par_interim_path,"conversion_matrices.xlsx"), sheet_name = 'WIOD 55 to NACE 64', header=[0], index_col=[0]).columns.values
    else:
        raise ValueError(
                        "conversion matrix '{0}' not recognized \n"
                        "valid arguments are: 'NACE64', 'NACE38', 'NACE21', 'NACE10', 'WIOD55".format(from_to)
                    )