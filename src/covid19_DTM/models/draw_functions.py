import random
import numpy as np

def draw_fnc_COVID19_SEIQRD_hybrid_vacc(param_dict,samples_dict):
    """
    A function to draw samples from the estimated posterior distributions of the model parameters.
    Tailored for use with the national COVID-19 SEIQRD model with the hybrid vaccination implementation.

    Parameters
    ----------

    samples_dict : dict
        Dictionary containing the samples of the national COVID-19 SEIQRD model obtained through calibration of WAVE 1

    param_dict : dict
        Model parameters dictionary

    Returns
    -------
    param_dict : dict
        Modified model parameters dictionary

    """

    idx, param_dict['eff_work'] = random.choice(list(enumerate(samples_dict['eff_work'])))  
    #param_dict['eff_rest'] = samples_dict['eff_rest'][idx]
    param_dict['mentality'] = samples_dict['mentality'][idx]
    param_dict['k'] = samples_dict['k'][idx]
    param_dict['K_inf'] = np.array([slice[idx] for slice in samples_dict['K_inf']], np.float64)
    param_dict['amplitude'] = samples_dict['amplitude'][idx]
    param_dict['f_h'] = samples_dict['f_h'][idx]

    # Hospitalization
    # ---------------
    # Fractions
    names = ['c','m_C','m_ICU']
    for idx,name in enumerate(names):
        par=[]
        for jdx in range(len(param_dict['c'])):
            par.append(np.random.choice(samples_dict['samples_fractions'][idx,jdx,:]))
        param_dict[name] = np.array(par)
    # Residence times
    n=20
    distributions = [samples_dict['residence_times']['dC_R'],
                     samples_dict['residence_times']['dC_D'],
                     samples_dict['residence_times']['dICU_R'],
                     samples_dict['residence_times']['dICU_D'],
                     samples_dict['residence_times']['dICUrec']]

    names = ['dc_R', 'dc_D', 'dICU_R', 'dICU_D','dICUrec']
    for idx,dist in enumerate(distributions):
        param_val=[]
        for age_group in dist.index.get_level_values(0).unique().values[0:-1]:
            draw = np.random.gamma(dist['shape'].loc[age_group],scale=dist['scale'].loc[age_group],size=n)
            param_val.append(np.mean(draw))
        param_dict[names[idx]] = np.array(param_val)
    return param_dict

def draw_fnc_COVID19_SEIQRD_spatial_hybrid_vacc(param_dict,samples_dict):
    """
    A function to draw samples from the estimated posterior distributions of the model parameters.
    Tailored for use with the spatial COVID-19 SEIQRD model.

    Parameters
    ----------

    samples_dict : dict
        Dictionary containing the samples of the national COVID-19 SEIQRD model obtained through calibration of WAVE 1

    param_dict : dict
        Model parameters dictionary

    Returns
    -------
    param_dict : dict
        Modified model parameters dictionary

    """

    idx, param_dict['beta_R'] = random.choice(list(enumerate(samples_dict['beta_R'])))
    param_dict['beta_U'] = samples_dict['beta_U'][idx]  
    param_dict['beta_M'] = samples_dict['beta_M'][idx]
    param_dict['eff_work'] = samples_dict['eff_work'][idx]
    #param_dict['eff_rest'] = samples_dict['eff_rest'][idx]   
    param_dict['k'] = samples_dict['k'][idx]
    param_dict['mentality'] = samples_dict['mentality'][idx]
    param_dict['K_inf'] = np.array([slice[idx] for slice in samples_dict['K_inf']], np.float64)
    param_dict['amplitude'] = samples_dict['amplitude'][idx]
    param_dict['summer_rescaling_F'] = samples_dict['summer_rescaling_F'][idx]
    param_dict['summer_rescaling_W'] = samples_dict['summer_rescaling_W'][idx]

    # Hospitalization
    # ---------------
    # Fractions
    names = ['c','m_C','m_ICU']
    for idx,name in enumerate(names):
        par=[]
        for jdx in range(len(param_dict['c'])):
            par.append(np.random.choice(samples_dict['samples_fractions'][idx,jdx,:]))
        param_dict[name] = np.array(par)
    # Residence times
    n=20
    distributions = [samples_dict['residence_times']['dC_R'],
                     samples_dict['residence_times']['dC_D'],
                     samples_dict['residence_times']['dICU_R'],
                     samples_dict['residence_times']['dICU_D'],
                     samples_dict['residence_times']['dICUrec']]

    names = ['dc_R', 'dc_D', 'dICU_R', 'dICU_D','dICUrec']
    for idx,dist in enumerate(distributions):
        param_val=[]
        for age_group in dist.index.get_level_values(0).unique().values[0:-1]:
            draw = np.random.gamma(dist['shape'].loc[age_group],scale=dist['scale'].loc[age_group],size=n)
            param_val.append(np.mean(draw))
        param_dict[names[idx]] = np.array(param_val)
    return param_dict