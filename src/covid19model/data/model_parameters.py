import os
import datetime
import pandas as pd
import numpy as np

def get_interaction_matrices(dataset='willem_2012', wave = 1, intensity='all', spatial=None):
    """Extracts and returns interaction matrices of the CoMiX or Willem 2012 dataset for a given contact intensity.
    Extracts and returns demographic data for Belgium (2020).

	Parameters
	-----------
    dataset : string
        The desired interaction matrices to be extracted. These can either be the pre-pandemic matrices for Belgium ('willem_2012') or pandemic matrices for Belgium ('comix').
        The pandemic data are 'time-dependent', i.e. associated with a date at which the survey was conducted.
        Default dataset: pre-pandemic Willem 2012.
    wave : int
        The wave number of the comix data.
        Defaults to the first wave.
	intensity : string

		The extracted interaction matrix can be altered based on the nature or duration of the social contacts.
		This is necessary because a contact is defined as any conversation longer than 3 sentences however, an infectious disease may only spread upon more 'intense' contact.
		Valid options for Willem 2012 include 'all' (default), 'physical_only', 'less_5_min', 'more_5_min', less_15_min', 'more_15_min', 'more_one_hour', 'more_four_hours'.
        Valid options for CoMiX include 'all' (default) or 'physical_only'.

    spatial : string
        Takes either None (default), 'mun', 'arr', 'prov' or 'test', and influences the geographical stratification of the Belgian population in the first return value (initN).
        When 'test' is chosen, it only returns the population of the arrondissements for the test scenario (Antwerpen, Brussel-Hoofdstad, Gent, in that order).

    Returns
    ------------

    Willem 2012:
    ------------
    initN : np.array
        number of Belgian individuals, regardless of sex, in ten year age bins. If spatial is not None, this value is also geographically stratified
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

    CoMiX:
    ----------
    initN : np.array
        number of Belgian individuals, regardless of sex, in ten year age bins
    Nc : np.array (9x9)
        total number of daily contacts of individuals in age group X with individuals in age group Y
    dates : str
        date associated with the comix survey wave

    Notes
    ----------
    The interaction matrices are extracted using the SOCRATES data tool made by Lander Willem: https://lwillem.shinyapps.io/socrates_rshiny/.
    During the data extraction, reciprocity is assumed, weighing by age and weighing by week/weekend were enabled.
    The demographic data was retreived from https://statbel.fgov.be/en/themes/population/structure-population

    Example use
    -----------
    initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = get_interaction_matrices()
    initN, Nc, dates = get_interaction_matrices(dataset='comix', wave = 3)
    """

    # Extract demographic data
    abs_dir = os.path.dirname(__file__)
    if spatial:
        if spatial not in ['mun', 'arr', 'prov', 'test']:
            raise ValueError(
                        "spatial stratification '{0}' is not legitimate. Possible spatial "
                        "stratifications are 'mun', 'arr', 'prov' or 'test'".format(spatial)
                    )
        initN_data = "../../../data/interim/demographic/initN_" + spatial + ".csv"
        initN_df = pd.read_csv(os.path.join(abs_dir, initN_data), index_col='NIS')
        initN = initN_df.values[:,:-1]
    if not spatial:
        initN_data = "../../../data/interim/demographic/initN_arr.csv"
        initN_df = pd.read_csv(os.path.join(abs_dir, initN_data), index_col='NIS')
        initN = initN_df.values[:,:-1].sum(axis=0)    
    
    if dataset == 'willem_2012':
        # Define data path
        matrix_path = os.path.join(abs_dir, "../../../data/interim/interaction_matrices/willem_2012")

        # Input check on user-defined intensity
        if intensity not in pd.ExcelFile(os.path.join(matrix_path, "total.xlsx")).sheet_names:
            raise ValueError(
                "The specified intensity '{0}' is not a valid option, check the sheet names of the data spreadsheets".format(intensity))

        # Extract interaction matrices
        Nc_home = pd.read_excel(os.path.join(matrix_path, "home.xlsx"), index_col=0, header=0, sheet_name=intensity).values
        Nc_work = pd.read_excel(os.path.join(matrix_path, "work.xlsx"), index_col=0, header=0, sheet_name=intensity).values
        Nc_schools = pd.read_excel(os.path.join(matrix_path, "school.xlsx"), index_col=0, header=0, sheet_name=intensity).values
        Nc_transport = pd.read_excel(os.path.join(matrix_path, "transport.xlsx"), index_col=0, header=0, sheet_name=intensity).values
        Nc_leisure = pd.read_excel(os.path.join(matrix_path, "leisure.xlsx"), index_col=0, header=0, sheet_name=intensity).values
        Nc_others = pd.read_excel(os.path.join(matrix_path, "otherplace.xlsx"), index_col=0, header=0, sheet_name=intensity).values
        Nc_total = pd.read_excel(os.path.join(matrix_path, "total.xlsx"), index_col=0, header=0, sheet_name=intensity).values

        return initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total

    elif dataset == 'comix':
        # Define data path
        matrix_path = os.path.join(abs_dir, "../../../data/raw/interaction_matrices/comix")
        # Input check on user-defined intensity
        if intensity not in pd.ExcelFile(os.path.join(matrix_path, "wave1.xlsx")).sheet_names:
            raise ValueError(
                "The specified intensity '{0}' is not a valid option, check the sheet names of the data spreadsheets".format(intensity))
        # Allow for both string or digit input for the wave number
        if type(wave) is not int:
            raise ValueError(
                "The specified comix survey wave number '{0}' must be an integer number".format(wave))
        # Extract interaction matrices
        Nc = pd.read_excel(os.path.join(matrix_path, "wave"+str(wave)+".xlsx"), index_col=0, header=0, sheet_name=intensity).values
        # Convert interaction matrices
        Nc[0,:] = Nc[:,0] # Assume reciprocity
        Nc[0,0] = Nc[1,1] # Assume interactions of 0-10 yo are equal to interactions 10-20 yo

        # Date list of comix waves
        dates = ['24-04-2020','08-05-2020','21-05-2020','04-06-2020','18-06-2020','02-07-2020','20-07-2020','03-08-2020']

        return initN, Nc, dates[wave-1]

    else:
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

def get_COVID19_SEIRD_parameters(age_stratified=True, spatial=None, intensity='all'):
    """
    Extracts and returns the parameters for the age-stratified deterministic model (spatial or non-spatial)

    This function returns all parameters needed to run the age-stratified and/or spatially stratified model.
    This function was created to group all parameters in one centralised location.

    Parameters
    ----------

    age_stratified : boolean
        If True: returns parameters stratified by age, for agestructured model
        If False: returns parameters for non-agestructured model

    spatial : string
        Can be either None (default), 'mun', 'arr', 'prov' or 'test' for various levels of geographical stratification. Note that
        'prov' contains the arrondissement Brussels-Capital. When 'test' is chosen, the mobility matrix for the test scenario is provided:
        mobility between Antwerp, Brussels-Capital and Ghent only (all other outgoing traffic is kept inside the home arrondissement).

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
        xi : factor controlling the contact dependence on density f (spatial only)

        Age-stratified parameters
        -------------------------
        s : relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU
        pi : mobility parameter per age class. Only loads when spatial is not None
        v : daily vaccination rate (percentage of population to be vaccinated)
        e : vaccine effectivity

        Spatially stratified parameters
        -------------------------------
        place : normalised mobility data. place[g][h] denotes the fraction of the population in patch g that goes to patch h
        area : area[g] is the area of patch g in square kilometers. Used for the density dependence factor f.
        sg : average size of a household per patch. Not used as of yet.

    Example use
    -----------
    parameters = get_COVID19_SEIRD_parameters()
    """

    abs_dir = os.path.dirname(__file__)
    par_raw_path = os.path.join(abs_dir, "../../../data/raw/model_parameters/")
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/model_parameters/COVID19_SEIRD")

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

        pars_dict['dc_R'] = np.array(df['dC_R'].values[-1]) # Better is .values[:-1], but there is no sufficient data
        pars_dict['dc_D'] = np.array(df['dC_D'].values[-1]) # Better is .values[:-1], but there is no sufficient data
        pars_dict['dICU_R'] = np.array(df['dICU_R'].values[-1]) # Better is .values[:-1], but there is no sufficient data
        pars_dict['dICU_D'] = np.array(df['dICU_D'].values[-1]) # Better is .values[:-1], but there is no sufficient data
        pars_dict['dICUrec'] = np.array(df['dICUrec'].values[-1]) # Better is .values[:-1], but there is no sufficient data

        # verity_etal
        df = pd.read_csv(os.path.join(par_raw_path,"verity_etal.csv"), sep=',',header='infer')
        pars_dict['h'] =  np.array(df.loc[:,'symptomatic_hospitalized'].astype(float).tolist())/100

        # davies_etal
        df_asymp = pd.read_csv(os.path.join(par_raw_path,"davies_etal.csv"), sep=',',header='infer')
        pars_dict['a'] =  np.array(df_asymp.loc[:,'fraction asymptomatic'].astype(float).tolist())
        pars_dict['s'] =  np.array(df_asymp.loc[:,'relative susceptibility'].astype(float).tolist())

        # vaccination
        df_vacc = pd.read_csv(os.path.join(par_raw_path,"vaccination.csv"), sep=',',header='infer')
        pars_dict['v'] =  np.array(df_vacc.loc[:,'fraction_vaccinated'].astype(float).tolist())
        pars_dict['e'] =  np.array(df_vacc.loc[:,'effectivity'].astype(float).tolist())

    else:
        pars_dict['Nc'] = np.array([17.65]) # Average interactions assuming weighing by age, by week/weekend and the inclusion of supplemental professional contacts (SPC)

        # Assign AZMM and UZG estimates to correct variables
        df = pd.read_csv(os.path.join(par_interim_path,"AZMM_UZG_hospital_parameters.csv"), sep=',',header='infer')
        pars_dict['c'] = np.array([df['c'].values[-1]])
        pars_dict['m_C'] = np.array([df['m0_{C}'].values[-1]])
        pars_dict['m_ICU'] = np.array([df['m0_{ICU}'].values[-1]])

        pars_dict['dc_R'] = np.array(df['dC_R'].values[-1])
        pars_dict['dc_D'] = np.array(df['dC_D'].values[-1])
        pars_dict['dICU_R'] = np.array(df['dICU_R'].values[-1])
        pars_dict['dICU_D'] = np.array(df['dICU_D'].values[-1])
        pars_dict['dICUrec'] = np.array(df['dICUrec'].values[-1])

        # verity_etal
        df = pd.read_csv(os.path.join(par_raw_path,"verity_etal.csv"), sep=',',header='infer')
        pars_dict['h'] =  np.array([0.0812]) # age-weiged average

        # davies_etal
        df_asymp = pd.read_csv(os.path.join(par_raw_path,"davies_etal.csv"), sep=',',header='infer')
        pars_dict['a'] =  np.array([0.579]) # age-weighed average
        pars_dict['s'] =  np.array([0.719]) # age-weighed average. Ideally this is equal to one, I would think

#         non_strat = pd.read_csv(os.path.join(par_raw_path,"non_stratified.csv"), sep=',',header='infer')
#         pars_dict.update({key: np.array(value) for key, value in non_strat.to_dict(orient='list').items()})

    # Add spatial parameters to dictionary
    if spatial:
        if spatial not in ['mun', 'arr', 'prov', 'test']:
            raise ValueError(
                        "spatial stratification '{0}' is not legitimate. Possible spatial "
                        "stratifications are 'mun', 'arr', 'prov', or 'test'".format(spatial)
                    )

        # Read recurrent mobility matrix per region
        mobility_data = '../../../data/interim/census_2011/census-2011-updated_row-commutes-to-column_' + spatial + '.csv'
        mobility_df=pd.read_csv(os.path.join(abs_dir, mobility_data), index_col='NIS')
        # Make sure the regions are ordered according to ascending NIS values
        mobility_df=mobility_df.sort_index(axis=0).sort_index(axis=1)
        # Take only the values (matrix) and save in NIS as floating points
        NIS=mobility_df.values.astype(float)
        # Normalize recurrent mobility matrix
        for i in range(NIS.shape[0]):
            NIS[i,:]=NIS[i,:]/sum(NIS[i,:])
        pars_dict['place'] = NIS

        # Read areas per region, ordered in ascending NIS values
        area_data = '../../../data/interim/demographic/area_' + spatial + '.csv'
        area_df=pd.read_csv(os.path.join(abs_dir, area_data), index_col='NIS')
        # Make sure the regions are ordered well
        area_df=area_df.sort_index(axis=0)
        area=area_df.values[:,0]
        pars_dict['area'] = area * 1e-6 # in square kilometer

        # Load mobility parameter, which is doubly stratified and 1 by default (no measures)
        pi = np.ones([pars_dict['place'].shape[0], pars_dict['Nc'].shape[0]])
        pars_dict['pi'] = pi

        # Load average household size sigma_g (sg) per region. Set default to average 2.3 for now. Currently not used
        sg = np.ones(pars_dict['place'].shape[0]) * 2.3
        pars_dict['sg'] = sg
        
        # Define factor controlling the contact dependence on density f (hard-coded)
        xi = 0.01 # km^-2
        pars_dict['xi'] = xi


    # Other parameters
    df_other_pars = pd.read_csv(os.path.join(par_raw_path,"others.csv"), sep=',',header='infer')
    pars_dict.update(df_other_pars.T.to_dict()[0])

    # Fitted parameters
    pars_dict['beta'] = 0.03492

    return pars_dict
