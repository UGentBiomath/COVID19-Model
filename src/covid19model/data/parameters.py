import os
import datetime
import pandas as pd
import numpy as np
from covid19model.data import polymod

def get_COVID19_SEIRD_parameters(stratified=True):
    """
    Extracts and returns the parameters for the age-stratified deterministic model

    This function returns all parameters needed to run the age-stratified model.
    This function was created to group all parameters in one centralised location.

    Parameters
    ----------
    stratified : boolean
        If True: returns parameters stratified by age, for agestructured model
        If False: returns parameters for non-agestructured model

    Returns
    -----------
    A dictionary with following keys and values:

    Nc: np.array
        9x9 social interaction matrix; by default, the total interaction matrix Nc_total from the Polymod study is assigned to the parameters dictionary
    h : np.array
        fraction of the cases that require hospitalisation (-)
    icu: np.array
        fraction of the hospitalized in ICU
    c: np.array
        fraction of the hospitalized in Cohort, calculated as (c = 1 - icu) (-)
    m0: np.array
        fraction of the patients in ICU who die (-)
    a: np.array
        fraction of asymptomatic cases (-)
    m: np.array
        fraction of (mild) symptomatic cases (-)
    da: float64
        length of infectiousness in case of asymptomatic infection (days)
    dm: float64
        length of infectiousness in case of mild symptomatic infection (days)
    dc: float64
        length of stay in Cohort when symptoms don't worsen (days)
    dICU: float64
        length of stay in ICU (days)
    dICUrec: float64
        length of recovery stay in Cohort after stay in ICU (days)
    dhospital: float64
        time between first symptom and hospitalization (days)
    beta: float64
        chance of transmission when coming into contact with an infectious person (-)
    sigma: float64
        length of latent period (days)
    omega: float64
        length of pre-symptomatic infectious period (days)
    dq: float64
        length of quarantine when patient does not develop symptoms (days)

    Example use
    -----------
    parameters = get_COVID19_SEIRD_parameters()
    """

    abs_dir = os.path.dirname(__file__)
    par_path = os.path.join(abs_dir, "../../../data/raw/model_parameters/")

    # Initialize parameters dictionary
    pars_dict = {}

    # Assign Nc_total from the Polymod study to the parameters dictionary
    Nc_total = polymod.get_interaction_matrices()[-1]
    pars_dict['Nc'] = Nc_total

    if stratified == True:
        # Verity_etal
        df = pd.read_csv(os.path.join(par_path,"verity_etal.csv"), sep=',',header='infer')
        pars_dict['h'] =  np.array(df.loc[:,'symptomatic_hospitalized'].astype(float).tolist())/100
        pars_dict['icu'] = np.array(df.loc[:,'hospitalized_ICU'].astype(float).tolist())/100
        pars_dict['m0'] = np.array(df.loc[:,'CFR'].astype(float).tolist())/100/pars_dict['h']/pars_dict['icu']

        # Wu_etal
        df_asymp = pd.read_csv(os.path.join(par_path,"wu_etal.csv"), sep=',',header='infer')
        pars_dict['a'] =  np.array(df_asymp.loc[:,'fraction asymptomatic'].astype(float).tolist())

    else:
        non_strat = pd.read_csv(os.path.join(par_path,"non_stratified.csv"), sep=',',header='infer')
        pars_dict.update(non_strat.T.to_dict()[0])

    # deduced parameters
    pars_dict['c'] = 1-pars_dict['icu']
    pars_dict['m'] = 1-pars_dict['a']
        
    # Other parameters
    df_other_pars = pd.read_csv(os.path.join(par_path,"others.csv"), sep=',',header='infer')
    pars_dict.update(df_other_pars.T.to_dict()[0])

    # Fitted parameters
    pars_dict['beta'] = 0.03492

    return pars_dict
