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
    pars_dict['IO'] = df.values

    # Others.csv
    df = pd.read_csv(os.path.join(par_interim_path,"others.csv"), sep=',',header=[1],index_col=[0])
    pars_dict['x_0'] = np.expand_dims(np.array(df['Sectoral output (M€)'].values), axis=1)
    pars_dict['c_0'] = np.expand_dims(np.array(df['Household demand (M€)'].values), axis=1)
    pars_dict['f_0'] = np.expand_dims(np.array(df['Other demand (M€)'].values), axis=1)
    pars_dict['n'] = np.expand_dims(np.array(df['Desired stock (days)'].values), axis=1)
    pars_dict['c_s'] = np.expand_dims(np.array(df['Consumer demand shock (%)'].values), axis=1)
    pars_dict['f_s'] = np.expand_dims(np.array(df['Other demand shock (%)'].values), axis=1)
    pars_dict['l_0'] = np.expand_dims(np.array(df['Employees (x1000)'].values), axis=1)*1000
    pars_dict['l_s'] = np.expand_dims(np.array(df['Employees (x1000)'].values), axis=1)*1000*np.expand_dims(np.array((df['Telework (%)'].values+df['Mix (%)'].values+df['Workplace (%)'].values)/100), axis = 1)

    # IHS_critical_NACE64.csv
    df = pd.read_csv(os.path.join(par_interim_path,"IHS_critical_NACE64.csv"), sep=',',header=[0],index_col=[0])
    pars_dict['C'] = df.values

    return pars_dict
