import os
import datetime
import pandas as pd
import numpy as np
from covid19model.data import polymod

def get_COVID19_SEIRD_parameters(age_stratified=True, spatial=False):
    """
    Extracts and returns the parameters for the age-stratified deterministic model (spatial or non-spatial)

    This function returns all parameters needed to run the age-stratified model.
    This function was created to group all parameters in one centralised location.

    Parameters
    ----------

    stratified : boolean
        If True: returns parameters stratified by age, for agestructured model
        If False: returns parameters for non-agestructured model

    Returns
    -------

    pars_dict : dictionary
        containing the values of all parameters (both stratified and not)
        these can be obtained with the function parameters.get_COVID19_SEIRD_parameters()

        Non-stratified parameters
        -------------------------
        beta : probability of infection when encountering an infected person
        sigma : length of the latent period
        omega : length of the pre-symptomatic infectious period
        zeta : effect of re-susceptibility and seasonality
        m : probability of an initially mild infection (m=1-a)
        da : duration of the infection in case of asymptomatic
        dm : duration of the infection in case of mild
        der : duration of stay in emergency room/buffer ward
        dc : average length of a hospital stay when not in ICU
        dICU_R : average length of a hospital stay in ICU in case of recovery
        dICU_D: average length of a hospital stay in ICU in case of death
        dhospital : time before a patient reaches the hospital

        Age-stratified parameters
        --------------------
        s: relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU

    Example use
    -----------
    parameters = get_COVID19_SEIRD_parameters()
    """

    abs_dir = os.path.dirname(__file__)
    par_raw_path = os.path.join(abs_dir, "../../../data/raw/model_parameters/")
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/model_parameters/")

    # Initialize parameters dictionary
    pars_dict = {}

    if age_stratified == True:

        # Assign Nc_total from the Polymod study to the parameters dictionary
        Nc_total = polymod.get_interaction_matrices()[-1]
        pars_dict['Nc'] = Nc_total

        # Assign AZMM and UZG estimates to correct variables
        df = pd.read_csv(os.path.join(par_interim_path,"AZMM_UZG_hospital_parameters.csv"), sep=',',header='infer')
        pars_dict['c'] = np.array(df['c'].values[:-1])
        pars_dict['m_C'] = np.array(df['m0_{C}'].values[:-1])
        pars_dict['m_ICU'] = np.array(df['m0_{ICU}'].values[:-1])
        pars_dict['dc_R'] = np.array(df['dC_R'].values[:-1])
        pars_dict['dc_D'] = np.array(df['dC_D'].values[:-1])
        pars_dict['dICU_R'] = np.array(df['dICU_R'].values[:-1])
        pars_dict['dICU_D'] = np.array(df['dICU_D'].values[:-1])
        pars_dict['dICUrec'] = np.array(df['dICUrec'].values[:-1])

        # verity_etal
        df = pd.read_csv(os.path.join(par_raw_path,"verity_etal.csv"), sep=',',header='infer')
        pars_dict['h'] =  np.array(df.loc[:,'symptomatic_hospitalized'].astype(float).tolist())/100

        # davies_etal
        df_asymp = pd.read_csv(os.path.join(par_raw_path,"davies_etal.csv"), sep=',',header='infer')
        pars_dict['a'] =  np.array(df_asymp.loc[:,'fraction asymptomatic'].astype(float).tolist())
        pars_dict['s'] =  np.array(df_asymp.loc[:,'relative susceptibility'].astype(float).tolist())

    else:
        pars_dict['Nc'] = np.array([11.2])

        non_strat = pd.read_csv(os.path.join(par_raw_path,"non_stratified.csv"), sep=',',header='infer')
        pars_dict.update({key: np.array(value) for key, value in non_strat.to_dict(orient='list').items()})

    # Add recurrent mobility matrix to parameter dictionary
    if spatial == True:
        # Read recurrent mobility matrix, ordered in ascending NIS values
        mobility_df=pd.read_csv(os.path.join(abs_dir, '../../../data/interim/census_2011/recurrent_mobility.csv'), index_col=[0])
        # Make sure the regions are ordered well
        mobility_df=mobility_df.sort_index(axis=0).sort_index(axis=1)
        # Take only the values (matrix) and save in NIS
        NIS=mobility_df.values
        # Normalize recurrent mobility matrix
        for i in range(NIS.shape[0]):
            NIS[i,:]=NIS[i,:]/sum(NIS[i,:])
        pars_dict['place'] = NIS
        
        # Read areas per region, ordered in ascending NIS values
        area_df=pd.read_csv(os.path.join(abs_dir, '../../../data/interim/demographic/area_arrond.csv'), index_col='NIS')
        # Make sure the regions are ordered well
        area_df=area_df.sort_index(axis=0)
        area=area_df.values[:,0]
        pars_dict['area'] = area * 1e-6 # in square kilometer
        
        # Load mobility parameter, which is age-stratified and 1 by default
        pi = np.ones(pars_dict['Nc'].shape[0])
        pars_dict['pi'] = pi
        
        # Load average household size sigma_g per region. Set default to average 2.3 for now.
        sg = np.ones(pars_dict['place'].shape[0]) * 2.3
        pars_dict['sg'] = sg
        

    # Other parameters
    df_other_pars = pd.read_csv(os.path.join(par_raw_path,"others.csv"), sep=',',header='infer')
    pars_dict.update(df_other_pars.T.to_dict()[0])

    # Fitted parameters
    pars_dict['beta'] = 0.03492

    return pars_dict
