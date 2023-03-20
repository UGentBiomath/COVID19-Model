import numpy as np

def draw_function(param_dict, samples_dict):
    # Hiring and Firing speed
    param_dict['gamma_F'] = np.random.uniform(low=1, high=14)
    param_dict['gamma_H'] = np.random.uniform(low=7, high=28)
    # Household savings and prospects
    param_dict['delta_S'] = np.random.uniform(low=0, high=1)
    param_dict['L'] = np.random.uniform(low=0, high=1)
    # restock rate
    param_dict['tau'] =  np.random.uniform(low=7, high=14)
    # Shocks summer 2020
    param_dict['ratio_c_s'] =  np.random.uniform(low=1/6, high=0.99)
    param_dict['ratio_f_s'] =  np.random.uniform(low=1/6, high=0.99)      
    # Shocks lockdowns
    param_dict['l_s_1'] =  np.random.normal(loc=1, scale=0.05)*param_dict['l_s_1']
    param_dict['l_s_2'] =  np.random.normal(loc=1, scale=0.05)*param_dict['l_s_2']
    param_dict['c_s'] =  np.random.normal(loc=1, scale=0.05)*param_dict['c_s']
    param_dict['f_s'] =  np.random.normal(loc=1, scale=0.05)*param_dict['f_s']
    
    return param_dict