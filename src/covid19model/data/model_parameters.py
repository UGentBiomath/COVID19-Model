import os
import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from covid19model.data.utils import construct_initN, convert_age_stratified_property

def get_interaction_matrices(dataset='willem_2012', wave = 1, intensity='all', age_stratification_size=10):
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
        # Define age path
        if age_stratification_size == 3:
            age_path = "0_20_60"
        elif age_stratification_size == 9:
            age_path = "0_10_20_30_40_50_60_70_80"
        elif age_stratification_size == 10:
            age_path = "0_12_18_25_35_45_55_65_75_85"
        else:
            raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
        )

        path = os.path.join(abs_dir, matrix_path+age_path)

        # Input check on user-defined intensity
        if intensity not in pd.ExcelFile(os.path.join(path, "total.xlsx"), engine='openpyxl').sheet_names:
            raise ValueError(
                "The specified intensity '{0}' is not a valid option, check the sheet names of the data spreadsheets".format(intensity))

        # Extract interaction matrices
        Nc_home = pd.read_excel(os.path.join(path, "home.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values
        Nc_work = pd.read_excel(os.path.join(path, "work.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values
        Nc_schools = pd.read_excel(os.path.join(path, "school.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values
        Nc_transport = pd.read_excel(os.path.join(path, "transport.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values
        Nc_leisure = pd.read_excel(os.path.join(path, "leisure.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values
        Nc_others = pd.read_excel(os.path.join(path, "otherplace.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values
        Nc_total = pd.read_excel(os.path.join(path, "total.xlsx"), index_col=0, header=0, sheet_name=intensity, engine='openpyxl').values
        Nc_dict = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}

        return Nc_dict


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

        return Nc, dates[wave-1]

    else:
        raise ValueError(
            "The specified intensity '{0}' is not a valid option, check the sheet names of the raw data spreadsheets".format(intensity))

def get_integrated_willem2012_interaction_matrices(age_stratification_size=10):
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

    intensities = ['all', 'less_5_min', 'less_15_min', 'more_one_hour', 'more_four_hours']
    # Define places
    places = ['home', 'work', 'schools', 'transport', 'leisure', 'others', 'total']
    # Get matrices at defined intensities
    matrices_raw = {}
    for idx, intensity in enumerate(intensities):
        Nc_dict = get_interaction_matrices(dataset='willem_2012', intensity = intensity, age_stratification_size=age_stratification_size)
        matrices_raw.update({intensities[idx]: Nc_dict})
    # Integrate matrices at defined intensities
    Nc_dict = {}
    for idx, place in enumerate(places):
        integration = matrices_raw['less_5_min'][place]*2.5/60 + (matrices_raw['less_15_min'][place] - matrices_raw['less_5_min'][place])*10/60 + ((matrices_raw['all'][place] - matrices_raw['less_15_min'][place]) - matrices_raw['more_one_hour'][place])*37.5/60 + (matrices_raw['more_one_hour'][place] - matrices_raw['more_four_hours'][place])*150/60 + matrices_raw['more_four_hours'][place]*240/60
        Nc_dict.update({place: integration})

    return Nc_dict

def get_COVID19_SEIQRD_parameters(age_stratification_size=10, spatial=None, vaccination=False, VOC=True):
    """
    Extracts and returns the parameters for the age-stratified deterministic COVID-19 model (spatial or non-spatial)

    This function returns all parameters needed to run the age-stratified and/or spatially stratified model.
    This function was created to group all parameters in one centralised location.

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

    vaccination : boolean
        Attach the vaccination parameters to the model parameters dictionary.
    
    VOC : boolean
        Attach the variant-of-concern parameters to the model parameters dictionary.

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
        xi : factor controlling the contact dependence on density f (spatial only)

        Age-stratified parameters
        -------------------------
        s : relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU
        p : mobility parameter per region. Only loads when spatial is not None
        v : daily vaccination rate (percentage of population to be vaccinated)
        e : vaccine effectivity
        leakiness : leakiness of the vaccine (proportion of vaccinated people that contribute to infections)

        Spatially stratified parameters
        -------------------------------
        place : normalised mobility data. place[g][h] denotes the fraction of the population in patch g that goes to patch h
        area : area[g] is the area of patch g in square kilometers. Used for the density dependence factor f.
        sg : average size of a household per patch. Not used as of yet.

        Other stratified parameters
        ---------------------------


    Example use
    -----------
    initN, Nc_dict, parameters = get_COVID19_SEIRD_parameters()
    """

    abs_dir = os.path.dirname(__file__)
    par_raw_path = os.path.join(abs_dir, "../../../data/raw/model_parameters/")
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/model_parameters/COVID19_SEIQRD")
    
    # Initialize parameters dictionary
    pars_dict = {}

    ########################
    ## Initial population ##
    ########################

    if spatial:
        if spatial not in ['mun', 'arr', 'prov', 'test']:
            raise ValueError(
                        "spatial stratification '{0}' is not legitimate. Possible spatial "
                        "stratifications are 'mun', 'arr', 'prov' or 'test'".format(spatial)
                    )

    if age_stratification_size == 3:
        initN = construct_initN(pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left'), spatial)
        age_path = '0_20_60/'
    elif age_stratification_size == 9:
        initN = construct_initN(pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left'), spatial)
        age_path = '0_10_20_30_40_50_60_70_80/'
    elif age_stratification_size == 10:
        initN = construct_initN(pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'), spatial)
        age_path = '0_12_18_25_35_45_55_65_75_85/'
    else:
        raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
        )
    par_interim_path = os.path.join(par_interim_path, 'hospitals/'+age_path)

    ##########################
    ## Interaction matrices ##
    ##########################

    # Assign total Flemish interaction matrix from Lander Willem study to the parameters dictionary (integrated version)
    Nc_dict = get_integrated_willem2012_interaction_matrices(age_stratification_size)
    pars_dict['Nc'] = Nc_dict['total']

    ##########################################################################
    ## Susceptibility, hospitalization propensity and asymptomatic fraction ##
    ##########################################################################

    # Susceptibility (Davies et al.)
    pars_dict['s'] =  np.ones(age_stratification_size)

    # Hospitalization propensity (manually fitted)
    hosp_prop = pd.Series(index = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left'),
                             data = [0.015, 0.020, 0.03, 0.03, 0.03, 0.06, 0.15, 0.35, 0.80])

    # Relative symptoms dataframe (Wu et al., 2020)
    rel_symptoms = pd.Series(index = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left'),
                             data = [0.053, 0.072, 0.408, 1.000, 1.349, 1.993, 2.849, 3.046, 3.240])

    def rescale_relative_to_absolute(relative_data, desired_pop_avg_fraction):
        """ A function to rescale age-structured relative information into absolute population information.
            F.i. The relative fraction of symptomatic individuals per age group is given but must be converted to a fraction between [0,1] for every age group.
            This can only be accomplished if an overall population average fraction is provided.
        """
        n = sum(relative_data * construct_initN(age_classes=relative_data.index).values)
        n_desired = desired_pop_avg_fraction * sum(construct_initN(None,None))
        def errorfcn(multiplier, n, n_desired):
            return (multiplier*n - n_desired)**2
        return minimize(errorfcn, 0, args=(n, n_desired))['x'] * relative_data
    
    rel_symptoms = rescale_relative_to_absolute(rel_symptoms, 0.43)

    if age_stratification_size == 3:
        pars_dict['h'] = convert_age_stratified_property(hosp_prop, pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')).values
        pars_dict['a'] = 1 - convert_age_stratified_property(rel_symptoms, pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')).values
    elif age_stratification_size == 9:
        pars_dict['h'] = convert_age_stratified_property(hosp_prop, pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')).values
        pars_dict['a'] = 1 - convert_age_stratified_property(rel_symptoms, pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')).values
    elif age_stratification_size == 10:
        pars_dict['h'] = convert_age_stratified_property(hosp_prop, pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')).values
        pars_dict['a'] = 1 - convert_age_stratified_property(rel_symptoms, pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')).values

    #########################
    ## Hospital parameters ##
    #########################

    fractions = pd.read_excel(os.path.join(par_interim_path,'sciensano_hospital_parameters.xlsx'), sheet_name='fractions', index_col=0, header=[0,1], engine='openpyxl')
    pars_dict['c'] = np.array(fractions['c','point estimate'].values[:-1])
    pars_dict['m_C'] = np.array(fractions['m0_{C}','point estimate'].values[:-1])
    pars_dict['m_ICU'] = np.array(fractions['m0_{ICU}', 'point estimate'].values[:-1])

    residence_times = pd.read_excel(os.path.join(par_interim_path,'sciensano_hospital_parameters.xlsx'), sheet_name='residence_times', index_col=0, header=[0,1], engine='openpyxl')
    pars_dict['dc_R'] = np.array(residence_times['dC_R','median'].values[:-1])
    pars_dict['dc_D'] = np.array(residence_times['dC_D','median'].values[:-1])
    pars_dict['dICU_R'] = np.array(residence_times['dICU_R','median'].values[:-1])
    pars_dict['dICU_D'] = np.array(residence_times['dICU_D','median'].values[:-1])
    pars_dict['dICUrec'] = np.array(residence_times['dICUrec', 'median'].values[:-1])

    ###################################
    ## Non-age-stratified parameters ##
    ###################################

    # Other parameters
    pars_dict['da'] = 7
    pars_dict['dm'] = 7
    pars_dict['sigma'] = 4.54
    pars_dict['omega'] = 0.66
    pars_dict['zeta'] = 0.003206
    pars_dict['dhospital'] = 7.543

    # Fitted parameters
    if not spatial:
        pars_dict['beta'] = 0.03492
    else:
        pars_dict['beta_R'] = 0.03492 # rural
        pars_dict['beta_U'] = 0.03492 # urban
        pars_dict['beta_M'] = 0.03492 # metropolitan


    #######################################################
    ## Google community mobility social contact function ##
    #######################################################

    pars_dict.update({'l1' : 23.0,
                'l2' : 5.72,
                'prev_schools' : .333,
                'prev_work' : .0771,
                'prev_rest_lockdown' : .014,
                'prev_rest_relaxation' : .444,
                'prev_home' : .206})

    #################
    ## Vaccination ##
    #################

    if vaccination == True:
        # Model parameters
        pars_dict['N_vacc'] = np.zeros(age_stratification_size) # Default: no vaccination at simulation start
        pars_dict['e_s'] = np.array([0.80, 0.80, 0.75]) # Default: 95% lower susceptibility to SARS-CoV-2 on a per contact basis
        pars_dict['e_h'] = np.array([0.95, 0.95, 0.95]) # Default: 100% protection against severe COVID-19
        pars_dict['e_a'] = 1.00*np.ones(3) # Default: vaccination works in 100% of people
        pars_dict['e_i'] = 0.5*np.ones(3)# Default: vaccinated infectious individual is equally infectious as non-vaccinated individual
        pars_dict['d_vacc'] = 10*12*30 # Default: 36 months coverage of vaccine
        # TDPF parameters
        pars_dict.update({'initN' : initN.values,
                          'daily_doses' : 60000, # copy default values from vaccination_function, which are curently not used I think
                          'delay_immunity' : 14,
                          'vacc_order' : [8, 7, 6, 5, 4, 3, 2, 1, 0],
                          'stop_idx' : 9,
                          'refusal' : [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]})

    ##########
    ## VOCs ##
    ##########

    if VOC:
        pars_dict['alpha'] = [1, 0, 0] # Must be a list so we can check if "sum(alpha) == 1"
        pars_dict['K_inf1'] = 1.45 # British variant infectivity gain
        pars_dict['K_inf2'] = 1.45*1.5 # Indian variant infectivity gain
        pars_dict['K_hosp'] = np.ones(3)

    #################
    ## Seasonality ##
    #################

    pars_dict.update({'amplitude' : 0.104,
                    'peak_shift' :22.2})

    ########################
    ## Spatial parameters ##
    ########################

    if spatial:
        # Read recurrent mobility matrix per region
        # Note: this is still 2011 census data, loaded by default. A time-dependant function should update mobility_data
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

        # Load mobility parameter, which is regionally stratified and 1 by default (no user-defined mobility changes)
        p = np.ones(pars_dict['place'].shape[0])
        pars_dict['p'] = p

        # Load average household size sigma_g (sg) per region. Set default to average 2.3 for now.
        # NB Currently not used
        #sg = np.ones(pars_dict['place'].shape[0]) * 2.3
        #pars_dict['sg'] = sg

        # Define factor controlling the contact dependence on density f (hard-coded)
        #xi = 0.01 # km^-2
        #pars_dict['xi'] = xi

        # TDPF parameters
        pars_dict['default_mobility'] = None


    return initN, Nc_dict, pars_dict
