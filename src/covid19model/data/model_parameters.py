import os
import json
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from covid19model.data.utils import construct_initN, convert_age_stratified_property


def get_interaction_matrices(dataset='willem_2012', wave=1, intensity='all', age_path='0_12_18_25_35_45_55_65_75_85/'):
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
                Valid options for Willem 2012 include 'all' (default), 'physical_only', 'less_5_min', less_15_min', 'more_one_hour', 'more_four_hours'.
        Valid options for CoMiX include 'all' (default) or 'physical_only'.

    age_stratification_size : int
        The the desired number of age groups in the model. Three options are programmed for use with the COVID19-SEIQRD model:
        3.  [0,20(, [20,60(, [60,120(
        9.  [0,10(,[10,20(,[20,30(,[30,40(,[40,50(,[50,60(,[60,70(,[70,80(,[80,120(,
        10. [0,12(,[12,18(,[18,25(,[25,35(,[35,45(,[45,55(,[55,65(,[65,75(,[75,85(,[85,120(


    Returns
    -------

    Willem 2012:
    ------------

    Nc_dict : dictionary
        dictionary containing the desired interaction matrices
        the dictionary has the following keys: ['home', 'work', 'schools', 'transport', 'leisure', 'others', 'total'] and contains the following interaction matrices:

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
    Nc : np.array (9x9)
        total number of daily contacts of individuals in age group X with individuals in age group Y
    dates : str
        date associated with the comix survey wave

    Notes
    ----------
    The interaction matrices are extracted using the SOCRATES data tool made by Lander Willem: https://lwillem.shinyapps.io/socrates_rshiny/.
    During the data extraction, reciprocity is assumed, weighing by age and weighing by week/weekend were enabled. Contacts with friends are moved from the home to the leisure interaction matrix.

    Example use
    -----------
    Nc_dict = get_interaction_matrices()
    Nc, dates = get_interaction_matrices(dataset='comix', wave = 3)
    """

    abs_dir = os.path.dirname(__file__)

    ##########################################
    ## Extract the Willem or Comix matrices ##
    ##########################################

    if dataset == 'willem_2012':
        # Define data path
        matrix_path = "../../../data/raw/interaction_matrices/willem_2012/"
        path = os.path.join(abs_dir, matrix_path+age_path)

        # Input check on user-defined intensity
        if intensity not in pd.ExcelFile(os.path.join(path, "total.xlsx"), engine='openpyxl').sheet_names:
            raise ValueError(
                "The specified intensity '{0}' is not a valid option, check the sheet names of the data spreadsheets".format(intensity))

        # Extract interaction matrices
        Nc_home = np.ascontiguousarray(pd.read_excel(os.path.join(
            path, "home.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values)
        Nc_work = np.ascontiguousarray(pd.read_excel(os.path.join(
            path, "work.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values)
        Nc_schools = np.ascontiguousarray(pd.read_excel(os.path.join(
            path, "school.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values)
        Nc_transport = np.ascontiguousarray(pd.read_excel(os.path.join(
            path, "transport.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values)
        Nc_leisure = np.ascontiguousarray(pd.read_excel(os.path.join(
            path, "leisure.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values)
        Nc_others = np.ascontiguousarray(pd.read_excel(os.path.join(
            path, "otherplace.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values)
        Nc_total = np.ascontiguousarray(pd.read_excel(os.path.join(
            path, "total.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values)
        Nc_dict = {'total': Nc_total, 'home': Nc_home, 'work': Nc_work, 'schools': Nc_schools,
                   'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}
        return Nc_dict

    elif dataset == 'comix':
        # Define data path
        matrix_path = os.path.join(
            abs_dir, "../../../data/raw/interaction_matrices/comix")
        # Input check on user-defined intensity
        if intensity not in pd.ExcelFile(os.path.join(matrix_path, "wave1.xlsx")).sheet_names:
            raise ValueError(
                "The specified intensity '{0}' is not a valid option, check the sheet names of the data spreadsheets".format(intensity))
        # Allow for both string or digit input for the wave number
        if type(wave) is not int:
            raise ValueError(
                "The specified comix survey wave number '{0}' must be an integer number".format(wave))
        # Extract interaction matrices
        Nc = pd.read_excel(os.path.join(matrix_path, "wave"+str(wave)+".xlsx"),
                           index_col=0, header=0, sheet_name=intensity).values
        # Convert interaction matrices
        Nc[0, :] = Nc[:, 0]  # Assume reciprocity
        # Assume interactions of 0-10 yo are equal to interactions 10-20 yo
        Nc[0, 0] = Nc[1, 1]

        # Date list of comix waves
        dates = ['24-04-2020', '08-05-2020', '21-05-2020', '04-06-2020',
                 '18-06-2020', '02-07-2020', '20-07-2020', '03-08-2020']

        return Nc, dates[wave-1]

    else:
        raise ValueError(
            "The specified intensity '{0}' is not a valid option, check the sheet names of the raw data spreadsheets".format(intensity))


def get_integrated_willem2012_interaction_matrices(age_path='0_12_18_25_35_45_55_65_75_85/'):
    """
    Extracts and returns interaction matrices of the Willem 2012 dataset, integrated with the duration of the contact.
    The relative share of contacts changes as follows by integrating with the duration of the contact (absolute number vs. time integrated):
        home: 12% --> 22%
        work: 35% --> 30%
        schools: 11% --> 13%
        leisure: 20% --> 18%
        transport: 4% --> 2%
        others: 19% --> 15%

    Parameters
    ----------
    age_stratification_size : int
        The the desired number of age groups in the model. Three options are programmed for use with the COVID19-SEIQRD model:
        3.  [0,20(, [20,60(, [60,120(
        9.  [0,10(,[10,20(,[20,30(,[30,40(,[40,50(,[50,60(,[60,70(,[70,80(,[80,120(,
        10. [0,12(,[12,18(,[18,25(,[25,35(,[35,45(,[45,55(,[55,65(,[65,75(,[75,85(,[85,120(

        Returns
        -------
    Nc_dict: dict
        Dictionary containing the integrated interaction matrices per place.
        Dictionary keys: ['home', 'work', 'schools', 'transport', 'leisure', 'others', 'total']

    """

    ################################################
    ## Extract and integrate Willem 2012 matrices ##
    ################################################

    intensities = ['all', 'less_5_min', 'less_15_min',
                   'more_one_hour', 'more_four_hours']
    # Define places
    places = ['home', 'work', 'schools',
              'transport', 'leisure', 'others', 'total']
    # Get matrices at defined intensities
    matrices_raw = {}
    for idx, intensity in enumerate(intensities):
        Nc_dict = get_interaction_matrices(
            dataset='willem_2012', intensity=intensity, age_path=age_path)
        matrices_raw.update({intensities[idx]: Nc_dict})
    # Integrate matrices at defined intensities
    Nc_dict = {}
    for idx, place in enumerate(places):
        integration = matrices_raw['less_5_min'][place]*2.5/60 + (matrices_raw['less_15_min'][place] - matrices_raw['less_5_min'][place])*10/60 + ((matrices_raw['all'][place] - matrices_raw['less_15_min']
                                                                                                                                                    [place]) - matrices_raw['more_one_hour'][place])*37.5/60 + (matrices_raw['more_one_hour'][place] - matrices_raw['more_four_hours'][place])*150/60 + matrices_raw['more_four_hours'][place]*240/60
        Nc_dict.update({place: integration})

    return Nc_dict


def get_COVID19_SEIQRD_parameters(age_classes=pd.IntervalIndex.from_tuples([(0, 12), (12, 18), (18, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75), (75, 85), (85, 120)], closed='left'),
                                  spatial=None):
    """
    Extracts and returns the parameters for the age-stratified deterministic COVID-19 model (spatial or non-spatial)

    This function returns all parameters needed to run the age-stratified and/or spatially stratified model.
    This function was created to group all parameters in a centralised location.

    Parameters
    ----------

    age_stratification_size : int
        The the desired number of age groups in the model. Three options are programmed for use with the COVID19-SEIQRD model:
        3.  [0,20(, [20,60(, [60,120(
        9.  [0,10(,[10,20(,[20,30(,[30,40(,[40,50(,[50,60(,[60,70(,[70,80(,[80,120(,
        10. [0,12(,[12,18(,[18,25(,[25,35(,[35,45(,[45,55(,[55,65(,[65,75(,[75,85(,[85,120(

    spatial : string
        Can be either None (default), 'mun', 'arr' or 'prov' for various levels of geographical stratification. Note that
        'prov' contains the arrondissement Brussels-Capital. When 'test' is chosen, the mobility matrix for the test scenario is provided:
        mobility between Antwerp, Brussels-Capital and Ghent only (all other outgoing traffic is kept inside the home arrondissement).

    Returns
    -------

    initN : np.array (size: n_spatial_patches * n_age_groups)
        Number of individuals per age group and per geographic location. Used to initialize the number of susceptibles in the model.

    pars_dict : dictionary

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
        -------------------------
        s : relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU
        p : mobility parameter per region. Only loads when spatial is not None

        Spatially stratified parameters
        -------------------------------
        place : normalised mobility data. place[g][h] denotes the fraction of the population in patch g that goes to patch h
        area : area[g] is the area of patch g in square kilometers. Used for the density dependence factor f.

        Other stratified parameters
        ---------------------------

    Example use
    -----------
    initN, Nc_dict, parameters = get_COVID19_SEIRD_parameters()
    """

    abs_dir = os.path.dirname(__file__)
    par_interim_path = os.path.join(
        abs_dir, "../../../data/interim/model_parameters/COVID19_SEIQRD")

    # Initialize parameters dictionary
    pars_dict = {}

    ########################
    ## Initial population ##
    ########################

    if spatial:
        if spatial not in ['mun', 'arr', 'prov', 'test']:
            raise ValueError(
                "spatial stratification '{0}' is not legitimate. Possible spatial "
                "stratifications are 'mun', 'arr', 'prov' or 'test'".format(
                    spatial)
            )

    initN = construct_initN(age_classes, spatial)
    age_stratification_size = len(age_classes)
    if age_stratification_size == 3:
        age_path = '0_20_60/'
    elif age_stratification_size == 9:
        age_path = '0_10_20_30_40_50_60_70_80/'
    elif age_stratification_size == 10:
        age_path = '0_12_18_25_35_45_55_65_75_85/'
    else:
        raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(
                age_stratification_size)
        )
    par_interim_path = os.path.join(par_interim_path, 'hospitals/'+age_path)

    ##########################
    ## Interaction matrices ##
    ##########################

    # Assign total Flemish interaction matrix from Lander Willem study to the parameters dictionary (integrated version)
    Nc_dict = get_integrated_willem2012_interaction_matrices(age_path)
    pars_dict['Nc'] = Nc_dict['total']

    ##########################################################################
    ## Susceptibility, hospitalization propensity and asymptomatic fraction ##
    ##########################################################################

    # Susceptibility (Davies et al.)
    pars_dict['s'] = np.ones(age_stratification_size, np.float64)

    # Hospitalization propensity (manually fitted)
    hosp_prop = pd.Series(index=pd.IntervalIndex.from_tuples([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 120)], closed='left'),
                          data=[0.015, 0.020, 0.03, 0.03, 0.03, 0.06, 0.15, 0.35, 0.80])

    # Relative symptoms dataframe (Wu et al., 2020)
    rel_symptoms = pd.Series(index=pd.IntervalIndex.from_tuples([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 120)], closed='left'),
                             data=[0.053, 0.072, 0.408, 1.000, 1.349, 1.993, 2.849, 3.046, 3.240])

    def rescale_relative_to_absolute(relative_data, desired_pop_avg_fraction):
        """ A function to rescale age-structured relative information into absolute population information.
            F.i. The relative fraction of symptomatic individuals per age group is given but must be converted to a fraction between [0,1] for every age group.
            This can only be accomplished if an overall population average fraction is provided.
        """
        n = sum(relative_data *
                construct_initN(age_classes=relative_data.index).values)
        n_desired = desired_pop_avg_fraction * sum(construct_initN(None, None))

        def errorfcn(multiplier, n, n_desired):
            return (multiplier*n - n_desired)**2
        return minimize(errorfcn, 0, args=(n, n_desired))['x'] * relative_data

    pars_dict['h'] = np.array(convert_age_stratified_property(
        hosp_prop, age_classes).values, np.float64)
    rel_symptoms = rescale_relative_to_absolute(rel_symptoms, 0.43)
    pars_dict['a'] = np.array(
        1 - convert_age_stratified_property(rel_symptoms, age_classes).values, np.float64)

    #########################
    ## Hospital parameters ##
    #########################

    fractions = pd.read_excel(os.path.join(par_interim_path, 'sciensano_hospital_parameters.xlsx'),
                              sheet_name='fractions', index_col=0, header=[0, 1], engine='openpyxl')
    pars_dict['c'] = np.array(
        fractions['c', 'point estimate'].values[:-1], np.float64)
    pars_dict['m_C'] = np.array(
        fractions['m0_{C}', 'point estimate'].values[:-1], np.float64)
    pars_dict['m_ICU'] = np.array(
        fractions['m0_{ICU}', 'point estimate'].values[:-1], np.float64)

    residence_times = pd.read_excel(os.path.join(par_interim_path, 'sciensano_hospital_parameters.xlsx'),
                                    sheet_name='residence_times', index_col=0, header=[0, 1], engine='openpyxl')
    pars_dict['dc_R'] = np.array(
        residence_times['dC_R', 'median'].values[:-1], np.float64)
    pars_dict['dc_D'] = np.array(
        residence_times['dC_D', 'median'].values[:-1], np.float64)
    pars_dict['dICU_R'] = np.array(
        residence_times['dICU_R', 'median'].values[:-1], np.float64)
    pars_dict['dICU_D'] = np.array(
        residence_times['dICU_D', 'median'].values[:-1], np.float64)
    pars_dict['dICUrec'] = np.array(
        residence_times['dICUrec', 'median'].values[:-1], np.float64)

    ###################################
    ## Non-age-stratified parameters ##
    ###################################

    pars_dict['l1'] = 21
    pars_dict['l2'] = 7
    pars_dict['da'] = 7
    pars_dict['dm'] = 7
    pars_dict['sigma'] = 4.54
    pars_dict['omega'] = 0.66
    pars_dict['dhospital'] = 6.4
    pars_dict['zeta'] = 1/365

    #######################
    ## Dummy seasonality ##
    #######################

    # Value of one equals no seasonality --> value is modified in time-dependant parameter function
    pars_dict['seasonality'] = 1
    pars_dict['peak_shift'] = 0
    pars_dict['amplitude'] = 0

    ###############
    ## Dummy VOC ##
    ###############

    pars_dict['f_VOC'] = [[1, 0],]
    pars_dict['K_inf'] = []
    pars_dict['K_hosp'] = []

    ########################
    ## Spatial parameters ##
    ########################

    if spatial:

        # Read recurrent mobility matrix per region
        # Note: this is still 2011 census data, loaded by default. A time-dependant function should update mobility_data
        mobility_data = '../../../data/interim/census_2011/census-2011-updated_row-commutes-to-column_' + spatial + '.csv'
        mobility_df = pd.read_csv(os.path.join(
            abs_dir, mobility_data), index_col='NIS')
        # Make sure the regions are ordered according to ascending NIS values
        mobility_df = mobility_df.sort_index(axis=0).sort_index(axis=1)
        # Infer spatial stratification size
        G = len(mobility_df.index.get_level_values('NIS').unique())
        # Take only the values (matrix) and save in NIS as floating points
        NIS = mobility_df.values.astype(float)
        # Normalize recurrent mobility matrix
        for i in range(NIS.shape[0]):
            NIS[i, :] = NIS[i, :]/sum(NIS[i, :])
        pars_dict['place'] = NIS
        # Read areas per region, ordered in ascending NIS values
        area_data = '../../../data/interim/demographic/area_' + spatial + '.csv'
        area_df = pd.read_csv(os.path.join(
            abs_dir, area_data), index_col='NIS')
        # Make sure the regions are ordered well
        area_df = area_df.sort_index(axis=0)
        area = area_df.values[:, 0]
        pars_dict['area'] = area * 1e-6  # in square kilometer
        # Load mobility parameter, which is regionally stratified and 1 by default (no user-defined mobility changes)
        p = np.ones(pars_dict['place'].shape[0])
        pars_dict['p'] = p
        # Add Nc_work and Nc to parameters
        # np.expand_dims(Nc_dict['total'],axis=0) # dims (1, N, N) # suggestion errors in validate
        pars_dict['Nc'] = Nc_dict['total']
        # np.expand_dims(Nc_dict['work'],axis=0) # dims (1, N, N)
        pars_dict['Nc_work'] = Nc_dict['work']

    #################################
    ## Dummy rescaling vaccination ##
    #################################

    if spatial:
        # Value of one equals no vaccination --> value is modified in time-dependant parameter function
        pars_dict['E_susc'] = pars_dict['E_inf'] = pars_dict['E_hosp'] = np.ones([G, age_stratification_size])
    else:
        pars_dict['E_susc'] = pars_dict['E_inf'] = pars_dict['E_hosp'] = np.ones(age_stratification_size)

    ############################
    ## BASE fitted parameters ##
    ############################

    if not spatial:
        # Set the average values for beta, seasonality, contact effectivities and mentality according to 'BASE' calibration dictionary
        samples_path = '../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'
        base_dict_name = 'BE_BASE_SAMPLES_2022-04-19.json'
        base_samples_dict = json.load(
            open(os.path.join(samples_path, base_dict_name)))
        pars_dict.update({
            'beta': np.mean(base_samples_dict['beta']),
            'zeta': np.mean(base_samples_dict['zeta']),
            'eff_schools': np.mean(base_samples_dict['eff_schools']),
            'eff_work': np.mean(base_samples_dict['eff_work']),
            'eff_rest': np.mean(base_samples_dict['eff_rest']),
            'eff_home': np.mean(base_samples_dict['eff_home']),
            'mentality': np.mean(base_samples_dict['mentality']),
            'amplitude': np.mean(base_samples_dict['amplitude']),
        })
    else:
        # Set the average values for beta, seasonality, contact effectivities and mentality according to 'BASE' calibration dictionary
        samples_path = '../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/prov/'
        base_dict_name = 'prov_BASE_SAMPLES_2022-04-21.json'
        base_samples_dict = json.load(
            open(os.path.join(samples_path, base_dict_name)))
        pars_dict.update({
            'beta_R': np.mean(base_samples_dict['beta_R']),
            'beta_U': np.mean(base_samples_dict['beta_U']),
            'beta_M': np.mean(base_samples_dict['beta_M']),
            'zeta': np.mean(base_samples_dict['zeta']),
            'eff_schools': np.mean(base_samples_dict['eff_schools']),
            'eff_work': np.mean(base_samples_dict['eff_work']),
            'eff_rest': np.mean(base_samples_dict['eff_rest']),
            'eff_home': np.mean(base_samples_dict['eff_home']),
            'mentality': np.mean(base_samples_dict['mentality']),
            'amplitude': np.mean(base_samples_dict['amplitude']),
        })

    return initN, Nc_dict, pars_dict, base_samples_dict


def get_COVID19_SEIQRD_VOC_parameters(VOCs=['WT', 'abc', 'delta', 'omicron'], pars_dict=None):
    """
    A function to load all parameters that in some way depend on what VOCs you consider in the model.
    If pars_dict is provided, relevant values of VOC parameters are set and pars_dict is returned.
    """

    abs_dir = os.path.dirname(__file__)
    save_path = os.path.join(
        abs_dir, "../../../data/interim/model_parameters/COVID19_SEIQRD/VOCs")

    ###########################################################
    ## Build a dataframe with all properties of the variants ##
    ###########################################################

    columns = [('logistic_growth', 't_introduction'), ('logistic_growth', 't_sigmoid'), ('logistic_growth', 'k'),
               ('variant_properties', 'sigma'), ('variant_properties',
                                                 'f_VOC'), ('variant_properties', 'f_immune_escape'),
               ('variant_properties', 'K_hosp'), ('variant_properties', 'K_inf')]
    VOC_parameters = pd.DataFrame(
        index=['WT', 'abc', 'delta', 'omicron'], columns=pd.MultiIndex.from_tuples(columns))

    # Define logistic growth properties
    VOC_parameters.loc['WT']['logistic_growth'] = [
        '2019-01-01', '2019-02-01', 0.20]
    VOC_parameters.loc['abc']['logistic_growth'] = [
        '2020-12-01', '2021-02-14', 0.07]
    VOC_parameters.loc['delta']['logistic_growth'] = [
        '2021-05-01', '2021-06-25', 0.11]
    VOC_parameters.loc['omicron']['logistic_growth'] = [
        '2021-11-26', '2021-12-24', 0.19]

    # Define variant properties
    VOC_parameters['variant_properties', 'sigma'] = [4.54, 4.54, 4.54, 2.34]
    VOC_parameters['variant_properties', 'f_VOC'] = [[1, 0], [0, 0], [0, 0], [0, 0]]
    VOC_parameters['variant_properties', 'f_immune_escape'] = [0, 0, 0, 1.5]
    VOC_parameters.loc[('abc', 'delta', 'omicron'), ('variant_properties', 'K_hosp')] = [1.62, 1.62, 1.62*0.30]
    VOC_parameters.loc[('abc', 'delta', 'omicron'),('variant_properties', 'K_inf')] = [1.50, 1.50, 3.00]

    ###############################################
    ## Build a dataframe with vaccine properties ##
    ###############################################

    iterables = [['WT', 'abc', 'delta', 'omicron'], [
        'none', 'partial', 'full', 'waned', 'boosted']]
    index = pd.MultiIndex.from_product(iterables, names=['VOC', 'dose'])
    vaccine_parameters = pd.DataFrame(
        index=index, columns=['e_s', 'e_i', 'e_h', 'waning', 'onset_immunity'], dtype=np.float64)

    # e_s, e_i for (WT, Alpha), Delta, Omicron: see email with VET estimates update from Toon Braeye 
    # Other references for e_s of Omicron: https://www.nejm.org/doi/full/10.1056/NEJMoa2119451, https://www.nature.com/articles/d41586-022-00775-3
    # e_h for Delta, Omicron: "COVID-19 VACCINE EFFECTIVENESS AGAINST SYMPTOMATIC INFECTION AND HOSPITALIZATION IN BELGIUM, JULY 2021-APRIL 2022", see email with preprint by Toon Braeye
    # Other references for e_h of omicron: https://www.nature.com/articles/s41591-022-01753-y/tables/2, https://pubmed.ncbi.nlm.nih.gov/35468336/

    vaccine_parameters.loc[('WT', slice(None)), 'e_s'] = [
        0, 0.87/2, 0.87, 0.64, 0.87]
    vaccine_parameters.loc[('abc', slice(None)), 'e_s'] = [
        0, 0.87/2, 0.87, 0.64, 0.87]
    vaccine_parameters.loc[('delta', slice(None)), 'e_s'] = [
        0, 0.79/2, 0.79, 0.54, 0.80]
    vaccine_parameters.loc[('omicron', slice(None)), 'e_s'] = [
        0, 0.05/2, 0.05, 0.02, 0.45] # https://www.nature.com/articles/s41591-022-01753-y

    # e_h = protection against symptomatic disease (=e_s) * protection against severe disease (=e_h_star)
    vaccine_parameters.loc[('WT', slice(None)), 'e_h'] = [
        0, 0.94/2, 0.94, 0.81, 0.94]
    vaccine_parameters.loc[('abc', slice(None)), 'e_h'] = [
        0, 0.94/2, 0.94, 0.81, 0.94]
    vaccine_parameters.loc[('delta', slice(None)), 'e_h'] = [
        0, 0.94/2, 0.94, 0.81, 0.94]
    vaccine_parameters.loc[('omicron', slice(None)), 'e_h'] = [
        0, 0.66/2, 0.66, 0.45, 0.87]

    # e_h_star
    for VOC in vaccine_parameters.index.get_level_values('VOC').unique():
        # e_h cannot be smaller than e_s
        if any( vaccine_parameters.loc[(VOC, ['partial', 'full', 'waned', 'boosted']), 'e_h'].values <= vaccine_parameters.loc[(VOC, ['partial', 'full', 'waned', 'boosted']), 'e_s'].values):
            raise ValueError(f"The reduction in hospitalization propensity cannot be lower than the reduction in susceptibility to a symptomatic infection for VOC '{VOC}'")
        # Compute reduction in hospitalization propensity "atop" of the reduction in developping symptoms
        vaccine_parameters.loc[(VOC, slice(None)), 'e_h'] = 1 - (1-vaccine_parameters.loc[(
            VOC, slice(None)), 'e_h'].values)/(1-vaccine_parameters.loc[(VOC, slice(None)), 'e_s'].values)

    # e_i
    vaccine_parameters.loc[('WT', slice(None)), 'e_i'] = [
        0, 0.62/2, 0.62, 0.43, 0.62]
    vaccine_parameters.loc[('abc', slice(None)), 'e_i'] = [
        0, 0.62/2, 0.62, 0.43, 0.62]
    vaccine_parameters.loc[('delta', slice(None)), 'e_i'] = [
        0, 0.38/2, 0.38, 0.25, 0.34]
    vaccine_parameters.loc[('omicron', slice(None)), 'e_i'] = [
        0, 0.22/2, 0.22, 0.14, 0.34]

    # onset: 14 days for every vaccine dose
    vaccine_parameters.loc[(
        slice(None), ['partial', 'full', 'boosted']), 'onset_immunity'] = 14

    # waning:
    vaccine_parameters.loc[(
        slice(None), ['partial', 'full', 'boosted']), 'waning'] = 200

    # Cut everything not needed from the VOC dataframe
    VOC_parameters = VOC_parameters.loc[VOCs]

    # Save a copy in a pickle
    VOC_parameters.to_pickle(os.path.join(save_path, 'VOC_parameters.pkl'), protocol=4)
    vaccine_parameters.to_pickle(os.path.join(save_path, 'vaccine_parameters.pkl'), protocol=4)


    #############################
    ## Set relevant VOC values ##
    #############################

    if pars_dict:
        # Update the relevant model parameters
        pars_dict.update({'sigma': np.array(VOC_parameters['variant_properties', 'sigma'].tolist(), np.float64),
                    'f_VOC': np.transpose(np.array(VOC_parameters['variant_properties', 'f_VOC'].tolist(), np.float64)),
                    'K_inf': np.array(VOC_parameters['variant_properties', 'K_inf'].tolist()[1:], np.float64),
                    'K_hosp': np.array(VOC_parameters['variant_properties', 'K_hosp'].tolist()[1:], np.float64)})
        if not pd.isnull(list(VOC_parameters['variant_properties', 'K_hosp'].values)[0]):
            pars_dict.update(
                {'h': pars_dict['h']*list(VOC_parameters['variant_properties', 'K_hosp'].values)[0]})
        return VOC_parameters, vaccine_parameters, pars_dict
    else:
        return VOC_parameters, vaccine_parameters
