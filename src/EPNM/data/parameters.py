import os
import numpy as np
import pandas as pd
from datetime import datetime
from EPNM.data.utils import get_sectoral_conversion_matrix,get_sector_labels

# Set path to interim data folder
abs_dir = os.path.dirname(__file__)
par_interim_path = os.path.join(abs_dir, "../../../data/EPNM/interim/")

def get_model_parameters(shocks='alleman'):
    """
    Extracts and returns the parameters for the economic model

    This function returns a dictionary with all parameters needed to run the economic model.

    Returns
    -------

    pars_dict : dictionary
        contains the values of all economic parameters

        Parameters
        ----------

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
        rho: Economic recovery time (0.6 quarters); influences income expectations of households
        delta_S: Household savings rate (delta_S = 1; households save all money they are not spending due to shock)
        L: Fraction of population believing in L-shaped economic recovery
        l_start_lockdown: Labor income before lockdown
        tau: Restock rate(days)
        gamma_F: Firing rate (days) 
        gamma_H: Hiring rate (days)

        Time-dependent parameters
        -------------------------

        zeta: Household income expectations
        epsilon_S: Labor supply shock vector 
        epsilon_D: Household demand shock vector
        epsilon_F: Exogeneous demand shock vector
        b: Fraction of compensated prepandemic labor income (actual parameter in model)
        b_s: Fraction of compensated prepandemic labor income (value under lockdown; used in TDPF of `b`)
        start_compensation: Start of government furloughing program
        end_compensation: End of government furloughing program

    Example use
    -----------
    parameters = get_model_parameters()
    """

    # Initialize parameters dictionary
    pars_dict = {}

    # Input-Ouput matrix
    # ~~~~~~~~~~~~~~~~~~

    # IO_NACE64.csv
    df = pd.read_csv(os.path.join(par_interim_path,"model_parameters/IO_NACE64.csv"), sep=',',header=[0],index_col=[0])
    pars_dict['IO'] = df.values/365
    # others.csv
    df = pd.read_csv(os.path.join(par_interim_path,"model_parameters/other_parameters.csv"), sep=',',header=[0],index_col=[0])
    pars_dict['x_0'] = np.array(df['Sectoral output (M€/y)'].values)/365
    pars_dict['O_j'] = np.array(df['Intermediate demand (M€/y)'].values)/365
    pars_dict['l_0'] = np.array(df['Labor compensation (M€/y)'].values)/365
    pars_dict['c_0'] = np.array(df['Household demand (M€/y)'].values)/365
    pars_dict['f_0'] = np.array(df['Total other demand (M€/y)'].values)/365
    pars_dict['n'] = np.expand_dims(np.array(df['Desired stock (days)'].values), axis=1)
    pars_dict['on_site'] = np.array(df['On-site consumption (-)'].values)
    
    # shock to labor supply and househol demand
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df = pd.read_excel(os.path.join(par_interim_path,"model_parameters/shocks/shocks.xlsx"), sheet_name=shocks, header=[0],index_col=[0])
    pars_dict['l_s_1'] = -np.array(df['labor_supply_1'].values)/100
    pars_dict['l_s_2'] = -np.array(df['labor_supply_2'].values)/100
    pars_dict['l_s_1'] = np.where(pars_dict['l_s_1'] <= 0, 0, pars_dict['l_s_1'])
    pars_dict['l_s_2'] = np.where(pars_dict['l_s_2'] <= 0, 0, pars_dict['l_s_2'])
    pars_dict['c_s'] = -np.array(df['c_demand'].values)/100
    pars_dict['ratio_c_s'] = 0.75

    # shock to other demand
    # ~~~~~~~~~~~~~~~~~~~~~

    ## magnitude
    pars_dict['f_s'] = 0.15
    ## components of demand
    # retrieve data
    df = pd.read_csv(os.path.join(par_interim_path,"model_parameters/other_parameters.csv"), sep=',',header=[0],index_col=[0])
    pars_dict['f_gov'] = (np.array(df['Other consumption - government (M€/y)'].values) + np.array(df['Other consumption - IZW (M€/y)'].values))/365
    pars_dict['f_inv'] = np.array(df['Other consumption - investments (M€/y)'].values)/365
    f_exp=df['Other consumption - exports (M€/y)']/365
    # split exports of goods (A-F) and services (G-T) as these recover differently
    f_exp_goods = pd.Series(0, index=f_exp.index, name='exports_goods')
    f_exp_goods.loc[slice('A01', 'F41-43')] = f_exp.loc[slice('A01', 'F41-43')].values
    f_exp_services = pd.Series(0, index=f_exp.index, name='exports_goods')
    f_exp_services.loc[slice('G45', None)] = f_exp.loc[slice('G45', None)].values
    pars_dict['f_exp_goods'] = np.array(f_exp_goods.values)
    pars_dict['f_exp_services'] = np.array(f_exp_services.values)
    # compute relative fraction of total demand of each component
    pars_dict['f_gov'] = pars_dict['f_gov']/pars_dict['f_0']
    pars_dict['f_inv'] = pars_dict['f_inv']/pars_dict['f_0']
    pars_dict['f_exp_goods'] = pars_dict['f_exp_goods']/pars_dict['f_0']
    pars_dict['f_exp_services'] = pars_dict['f_exp_services']/pars_dict['f_0']

    # Critical inputs
    # ~~~~~~~~~~~~~~~

    df = pd.read_csv(os.path.join(par_interim_path,"model_parameters/IHS_critical_NACE64.csv"), sep=',',header=[0],index_col=[0])
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

    # Hardcoded model parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    pars_dict.update({'rho': 1-(1-0.60)/90,          
                      'delta_S': 0.75,                                                  
                      'L': 1,                                                        
                      'l_start_lockdown': sum((1-pars_dict['l_s_1'])*pars_dict['l_0']),                                                    
                      'tau': 14,                                                                                                 
                      'gamma_H': 56,
                      'gamma_F': 28 
                      })  

    # Time-dependent model parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pars_dict.update({'l1': 7,
                      'l2': 6*7,
                      't_start_lockdown_1': pd.Timestamp('2020-03-15'),
                      't_end_lockdown_1': pd.Timestamp('2020-05-01'),
                      't_start_lockdown_2': pd.Timestamp('2020-10-19'),
                      't_end_lockdown_2': pd.Timestamp('2020-11-19'),
                      't_start_final_relax': pd.Timestamp('2021-05-01'),
                    })

    pars_dict.update({'t_start_compensation': pars_dict['t_start_lockdown_1'],
                      't_end_compensation': pd.Timestamp('2021-12-01')})

    pars_dict.update({'epsilon_S': np.zeros([pars_dict['l_s_1'].shape[0]]),
                      'epsilon_D': np.zeros([pars_dict['l_s_1'].shape[0]]),
                      'epsilon_F': np.zeros([pars_dict['l_s_1'].shape[0]]),
                      'b': 0.7,
                      'b_s': 0.7,
                      'zeta': 1
                    })

    return pars_dict

def aggregate_shock(shock_in, demand, convmat):
    """
    Translates a shock on NACE64 level to a shock on NACE21 level, weighted with demand
    """
    return np.matmul(convmat*demand/np.expand_dims(np.sum(convmat*demand, axis=1),axis=1), np.expand_dims(shock_in, axis=1)).squeeze()