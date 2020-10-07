import os
import datetime
import pandas as pd
import numpy as np

def get_interaction_matrices(intensity='all'):
    """Extracts and returns interaction matrices for given contact intensity (Willem 2012) and extracts and returns demographic data for Belgium (2020)

	Parameters
	-----------
	intensity : string
		the extracted interaction matrix can be altered based on the nature or duration of the social contacts
		this is necessary because a contact is defined as any conversation longer than 3 sentences
		however, an infectious disease may only spread upon more 'intense' contact, hence the need to exclude the so-called 'non-physical contacts'
		valid options include 'all' (default), 'physical_only', 'less_5_min', 'more_5_min', less_15_min', 'more_15_min', 'more_one_hour', 'more_four_hours'

    Returns
    -----------
    initN : np.array
        number of Belgian individuals, regardless of sex, in ten year age bins
    Nc_home :  np.array (9x9)
        number of daily contacts at home of individuals in age group X with individuals in age group Y
    Nc_work :  np.array (9x9)
        number of daily contacts in the workplace of individuals in age group X with individuals in age group Y
    Nc_schools :  np.array (9x9)
        number of daily contacts in schools of individuals in age group X with individuals in age group Y
    Nc_transport :  np.array (9x9)
        number of daily contacts on public transport of individuals in age group X with individuals in age group Y
    Nc_leisure :  np.array (9x9)
        number of daily contacts during leisure activities of individuals in age group X with individuals in age group Y
    Nc_others :  np.array (9x9)
        number of daily contacts in other places of individuals in age group X with individuals in age group Y
    Nc_total :  np.array (9x9)
        total number of daily contacts of individuals in age group X with individuals in age group Y, calculated as the sum of all the above interaction

    Notes
    ----------
    The interaction matrices are extracted using the SOCRATES data tool made by Lander Willem: https://lwillem.shinyapps.io/socrates_rshiny/.
    During the data extraction, reciprocity is assumed, weighing by age and weighing by week/weekend were enabled.
    The demographic data was retreived from https://statbel.fgov.be/en/themes/population/structure-population

    Example use
    -----------
    initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = get_interaction_matrices()
    """

    # Define data path
    abs_dir = os.path.dirname(__file__)
    matrix_path = os.path.join(abs_dir, "../../../data/interim/interaction_matrices/willem_2012")

    # Input check on user-defined intensity
    if intensity not in pd.ExcelFile(os.path.join(matrix_path, "total.xlsx")).sheet_names:
        raise ValueError(
            "The specified intensity '{0}' is not a valid option, check the sheet names of the raw data spreadsheets".format(intensity))

    # Extract interaction matrices
    Nc_home = pd.read_excel(os.path.join(matrix_path, "home.xlsx"), index_col=0, header=0, sheet_name=intensity).values
    Nc_work = pd.read_excel(os.path.join(matrix_path, "work.xlsx"), index_col=0, header=0, sheet_name=intensity).values
    Nc_schools = pd.read_excel(os.path.join(matrix_path, "school.xlsx"), index_col=0, header=0, sheet_name=intensity).values
    Nc_transport = pd.read_excel(os.path.join(matrix_path, "transport.xlsx"), index_col=0, header=0, sheet_name=intensity).values
    Nc_leisure = pd.read_excel(os.path.join(matrix_path, "leisure.xlsx"), index_col=0, header=0, sheet_name=intensity).values
    Nc_others = pd.read_excel(os.path.join(matrix_path, "otherplace.xlsx"), index_col=0, header=0, sheet_name=intensity).values
    Nc_total = pd.read_excel(os.path.join(matrix_path, "total.xlsx"), index_col=0, header=0, sheet_name=intensity).values

    # Extract demographic data
    initN = np.loadtxt(os.path.join(matrix_path, "../demographic/BELagedist_10year.txt"), dtype='f', delimiter='\t')

    return initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total

def get_COVID19_SEIRD_parameters(age_stratified=True, spatial=False, intensity='all'):
    """
    Extracts and returns the parameters for the age-stratified deterministic model (spatial or non-spatial)

    This function returns all parameters needed to run the age-stratified model.
    This function was created to group all parameters in one centralised location.

    Parameters
    ----------

    age_stratified : boolean
        If True: returns parameters stratified by age, for agestructured model
        If False: returns parameters for non-agestructured model
    
    spatial : boolean
        If True: returns parameters for the age-stratified spatially explicit model
        If False: returns parameters for the age-stratified national-level model

    intensity : string
        the extracted interaction matrix can be altered based on the nature or duration of the social contacts
		this is necessary because a contact is defined as any conversation longer than 3 sentences
		however, an infectious disease may only spread upon more 'intense' contact, hence the need to exclude the so-called 'non-physical contacts'
		valid options include 'all' (default), 'physical_only', 'less_5_min', less_15_min', 'more_one_hour', 'more_four_hours'

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

        # Assign total Flemish interaction matrix from Lander Willem study to the parameters dictionary
        Nc_total = get_interaction_matrices(intensity=intensity)[-1]
        pars_dict['Nc'] = Nc_total

        # Assign AZMM and UZG estimates to correct variables
        df = pd.read_csv(os.path.join(par_interim_path,"AZMM_UZG_hospital_parameters.csv"), sep=',',header='infer')
        pars_dict['c'] = np.array(df['c'].values[:-1])
        pars_dict['m_C'] = np.array(df['m0_{C}'].values[:-1])
        pars_dict['m_ICU'] = np.array(df['m0_{ICU}'].values[:-1])
        pars_dict['dc_R'] = np.array(df['dC_R'].values[-1])
        pars_dict['dc_D'] = np.array(df['dC_D'].values[-1])
        pars_dict['dICU_R'] = np.array(df['dICU_R'].values[-1])
        pars_dict['dICU_D'] = np.array(df['dICU_D'].values[-1])
        pars_dict['dICUrec'] = np.array(df['dICUrec'].values[-1])

        # verity_etal
        df = pd.read_csv(os.path.join(par_raw_path,"verity_etal.csv"), sep=',',header='infer')
        pars_dict['h'] =  np.array(df.loc[:,'symptomatic_hospitalized'].astype(float).tolist())/100

        # davies_etal
        df_asymp = pd.read_csv(os.path.join(par_raw_path,"davies_etal.csv"), sep=',',header='infer')
        pars_dict['a'] =  np.array(df_asymp.loc[:,'fraction asymptomatic'].astype(float).tolist())
        pars_dict['s'] =  np.array(df_asymp.loc[:,'relative susceptibility'].astype(float).tolist())

    else:
        pars_dict['Nc'] = np.array([17.65]) # Average interactions assuming weighing by age, by week/weekend and the inclusion of supplemental professional contacts (SPC)

        non_strat = pd.read_csv(os.path.join(par_raw_path,"non_stratified.csv"), sep=',',header='infer')
        pars_dict.update({key: np.array(value) for key, value in non_strat.to_dict(orient='list').items()})

    # Add recurrent mobility matrix to parameter dictionary
    if spatial == True:
        mobility_df=pd.read_csv(os.path.join(abs_dir, '../../../data/interim/census_2011/recurrent_mobility.csv'), index_col=[0])
        NIS=mobility_df.values
        # Normalize recurrent mobility matrix
        for i in range(NIS.shape[0]):
            NIS[i,:]=NIS[i,:]/sum(NIS[i,:])
        pars_dict.update({'place': NIS})

    # Other parameters
    df_other_pars = pd.read_csv(os.path.join(par_raw_path,"others.csv"), sep=',',header='infer')
    pars_dict.update(df_other_pars.T.to_dict()[0])

    # Fitted parameters
    pars_dict['beta'] = 0.03492

    return pars_dict
