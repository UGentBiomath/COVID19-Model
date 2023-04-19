import random
import numpy as np

def draw_function(param_dict, samples_dict):
    """
    Samples calibrated magnitudes of household demand/exogeneous demand shocks during the summer of 2021
    Performs global sensitivity analysis on other parameters
    """

    # Option 1
    # Shocks in/out
    param_dict['l1'] = np.random.normal(loc=7, scale=2)
    param_dict['l2'] = np.random.normal(loc=2*28, scale=7)
    #Hiring and Firing speed
    param_dict['gamma_F'] = np.ones(len(param_dict['c_s']))*np.random.normal(loc=7, scale=2)
    param_dict['gamma_H'] = np.ones(len(param_dict['c_s']))*np.random.normal(loc=7, scale=2)
    #Household savings and prospects
    param_dict['delta_S'] = np.random.uniform(low=0.5, high=1)
    param_dict['L'] = np.random.uniform(low=0.5, high=1)
    param_dict['b_s'] = np.random.uniform(low=0.5, high=1)
    #restock rate
    param_dict['tau'] =  np.random.normal(loc=21, scale=7)
    #Shocks lockdowns
    param_dict['l_s_1'] =  np.random.normal(loc=1, scale=0.04)*param_dict['l_s_1']
    param_dict['l_s_2'] =  np.random.normal(loc=1, scale=0.04)*param_dict['l_s_2']
    param_dict['c_s'] =  np.random.normal(loc=1, scale=0.05)*param_dict['c_s']
    param_dict['f_s'] =  np.random.normal(loc=1, scale=0.05)*param_dict['f_s']
    param_dict['c_s'] = np.where(param_dict['c_s'] > 1, 1, param_dict['c_s'])
    param_dict['f_s'] = np.where(param_dict['c_s'] > 1, 1, param_dict['c_s'])
    #Shocks summer 2020 
    param_dict['ratio_c_s'] =  np.random.uniform(low=1/10, high=0.99)
    param_dict['ratio_f_s'] =  np.random.uniform(low=1/10, high=0.99)
    
    # Option 2
    # Calibrated c_s and f_s
    # idx = np.random.randint(low=0, high=len(samples_dict['c_s'][0]))
    # param_dict['c_s'] = np.array([slice[idx] for slice in samples_dict['c_s']], np.float64)
    # #param_dict['f_s'] = np.array([slice[idx] for slice in samples_dict['f_s']], np.float64)
    # param_dict['f_s'] =  np.random.normal(loc=1, scale=0.05)*param_dict['f_s']
    # # Perturbate l_s
    # param_dict['l_s_1'] =  np.random.normal(loc=1, scale=0.05)*param_dict['l_s_1']
    # param_dict['l_s_2'] =  np.random.normal(loc=1, scale=0.05)*param_dict['l_s_2']
    # # Perturbate ratio's during summer of 2020
    # param_dict['ratio_c_s'] =  np.random.uniform(low=1/10, high=0.99)
    # param_dict['ratio_f_s'] =  np.random.uniform(low=1/10, high=0.99)
    # # Hiring and Firing speed
    # param_dict['gamma_F'] = np.ones(len(param_dict['c_s']))*np.random.normal(loc=14, scale=3)
    # param_dict['gamma_H'] = np.ones(len(param_dict['c_s']))*np.random.normal(loc=28, scale=7)
    # # Household savings and prospects
    # param_dict['delta_S'] = np.random.uniform(low=0.50, high=1)
    # param_dict['L'] = np.random.uniform(low=0.50, high=1)
    # param_dict['b_s'] = np.random.uniform(low=0.50, high=1)
    # # restock rate
    # param_dict['tau'] =  np.random.normal(loc=10, scale=2)

    return param_dict