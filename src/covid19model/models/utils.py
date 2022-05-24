import sys
import os
import warnings
import random
from numba import jit
import numpy as np
import pandas as pd
import xarray as xr
import json
import pickle

abs_dir = os.path.dirname(__file__)
data_path = os.path.join(abs_dir, "../../../data/")

def initialize_COVID19_SEIQRD_hybrid_vacc(age_stratification_size=10, VOCs=['WT', 'abc', 'delta'], start_date='2020-03-15', update_data=False):

    ###########################################################
    ## Convert age_stratification_size to desired age groups ##
    ###########################################################

    if age_stratification_size == 3:
        age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
    elif age_stratification_size == 9:
        age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
    elif age_stratification_size == 10:
        age_classes = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')
    else:
        raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
        )

    #####################################
    ## Import necessary pieces of code ##
    #####################################

    # Import the SEIQRD model with VOCs, vaccinations, seasonality
    from covid19model.models import models
    # Import time-dependent parameter functions for resp. P, Nc, alpha, N_vacc, season_factor
    from covid19model.models.time_dependant_parameter_fncs import   make_contact_matrix_function, \
                                                                    make_VOC_function, \
                                                                    make_N_vacc_function, \
                                                                    make_vaccination_efficacy_function, \
                                                                    make_seasonality_function
    # Import packages containing functions to load in data used in the model and the time-dependent parameter functions
    from covid19model.data import mobility, sciensano, model_parameters, VOC
    from covid19model.data.utils import convert_age_stratified_quantity

    #########################
    ## Load necessary data ##
    #########################

    # Population size, interaction matrices and the model parameters
    initN, Nc_dict, params, BASE_samples_dict = model_parameters.get_COVID19_SEIQRD_parameters(age_classes=age_classes)
    # Load previous vaccine parameters and currently saved VOC/vaccine parameters
    vaccine_params_previous = pd.read_pickle(os.path.join(abs_dir, '../../../data/interim/model_parameters/COVID19_SEIQRD/VOCs/vaccine_parameters.pkl'))
    VOC_params, vaccine_params, params = model_parameters.get_COVID19_SEIQRD_VOC_parameters(VOCs=VOCs, pars_dict=params)
    # Sciensano hospital and vaccination data
    df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=update_data)
    df_hosp = df_hosp.groupby(by=['date']).sum()
    # Using the weekly vaccination data
    df_vacc = sciensano.get_public_spatial_vaccination_data(update=update_data)
    # Google Mobility data
    df_google = mobility.get_google_mobility_data(update=update_data)

    ##################################################
    ## Construct time-dependent parameter functions ##
    ##################################################

    # Time-dependent VOC function, updating alpha
    VOC_function = make_VOC_function(VOC_params['logistic_growth'])
    # Time-dependent social contact matrix over all policies, updating Nc
    policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_all
    # Time-dependent seasonality function, updating season_factor
    seasonality_function = make_seasonality_function()
    # Time-dependent (first) vaccination function, updating N_vacc. Hypothetical functions administers no boosters but extends the dataframe of incidences with half a year.
    df_incidences_previous = pd.read_pickle(os.path.join(abs_dir, '../../../data/interim/sciensano/vacc_incidence_national.pkl'))
    N_vacc_function = make_N_vacc_function(df_vacc['INCIDENCE'], age_classes=age_classes, hypothetical_function=True)
    # Extract the smoothed dataframe
    df_incidences = N_vacc_function.df
    # Check if the vaccination efficacies must be updated
    rescaling_update=False
    if ((not vaccine_params_previous.equals(vaccine_params)) or (not df_incidences_previous.equals(df_incidences))):
        rescaling_update = True
    elif update_data == True:
        rescaling_update = True
    # Construct the efficacy function subject to waning
    efficacy_function = make_vaccination_efficacy_function(update=rescaling_update, df_incidences=df_incidences, vaccine_params=vaccine_params,
                                              VOCs=VOCs, age_classes=age_classes)

    ####################
    ## Initial states ##
    ####################

    samples_path = os.path.join(abs_dir, data_path + '/interim/model_parameters/COVID19_SEIQRD/initial_conditions/national/')

    if start_date == '2020-03-15':
        with open(samples_path+'initial_states-COVID19_SEIQRD_hybrid_vacc.pickle', 'rb') as handle:
            load = pickle.load(handle)
            initial_states = load[start_date]

    elif ((start_date == '2021-08-01') | (start_date == '2021-09-01')):
        with open(samples_path+'summer_2021-COVID19_SEIQRD_hybrid_vacc.pickle', 'rb') as handle:
            load = pickle.load(handle)
            initial_states = load[start_date]
    else:
        raise ValueError("Chosen startdate '{0}' is not valid. Choose: 2020-03-15, 2021-08-01 or 2021-09-01".format(start_date))

    for key,value in initial_states.items():
        converted_value = np.zeros([len(age_classes),value.shape[1]])
        for i in range(value.shape[1]):
            column = value[:,i]
            data = pd.Series(index=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'), data=column)
            converted_value[:,i] = convert_age_stratified_quantity(data, age_classes).values
        initial_states.update({key: converted_value})

    ##############################################################################
    ## Explicit vaccination module requires quite some parameters to be defined ##
    ##############################################################################
    
    # TODO: Make this default after making the spatial version

    # Drop rescaling parameters
    for key in ['E_susc', 'E_inf', 'E_hosp']:
        params.pop(key)

    # Define dummy vaccine efficacies
    e_i = np.ones([len(age_classes), len(efficacy_function.df_efficacies.index.get_level_values('dose').unique()), len(vaccine_params.index.get_level_values('VOC').unique())])
    e_h = np.ones([len(age_classes), len(efficacy_function.df_efficacies.index.get_level_values('dose').unique()), len(vaccine_params.index.get_level_values('VOC').unique())])
    e_s = np.ones([len(age_classes), len(efficacy_function.df_efficacies.index.get_level_values('dose').unique()), len(vaccine_params.index.get_level_values('VOC').unique())])

    # Vaccination parameters when using the stratified vaccination model
    params.update({'N_vacc': np.zeros([age_stratification_size, len(df_vacc.index.get_level_values('dose').unique())]),
                   'doses': np.zeros(len(df_vacc.index.get_level_values('dose').unique())),
                   'e_i': e_i,
                   'e_s': e_s,
                   'e_h': e_h,
                   })

    ##########################
    ## Initialize the model ##
    ##########################

    # Initialize model
    model = models.COVID19_SEIQRD_hybrid_vacc(initial_states, params,
                        time_dependent_parameters={'Nc': policy_function,
                                                   'seasonality': seasonality_function, 
                                                   'N_vacc': N_vacc_function,
                                                   'e_i' : efficacy_function.e_i,
                                                   'e_s' : efficacy_function.e_s,
                                                   'e_h' : efficacy_function.e_h,
                                                   'f_VOC':VOC_function})

    return model, BASE_samples_dict, initN

def initialize_COVID19_SEIQRD_spatial_rescaling(age_stratification_size=10, agg='prov', VOCs=['WT', 'abc', 'delta'], start_date='2020-03-17', virgin=False, update_data=False):
    
    abs_dir = os.path.dirname(__file__)

    ###########################################################
    ## Convert age_stratification_size to desired age groups ##
    ###########################################################

    if age_stratification_size == 3:
        age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
    elif age_stratification_size == 9:
        age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
    elif age_stratification_size == 10:
        age_classes = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')
    else:
        raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
        )

    #####################################
    ## Import necessary pieces of code ##
    #####################################

    # Import the file with the model definitions
    from covid19model.models.models import COVID19_SEIQRD_spatial_rescaling
    # Import time-dependent parameter functions for resp. P, Nc, alpha, N_vacc, season_factor
    from covid19model.models.time_dependant_parameter_fncs import make_mobility_update_function, \
                                                          make_contact_matrix_function, \
                                                          make_VOC_function, \
                                                          make_vaccination_rescaling_function, \
                                                          make_seasonality_function
    # Import packages containing functions to load in data used in the model and the time-dependent parameter functions
    from covid19model.data import mobility, sciensano, model_parameters
    from covid19model.data.utils import convert_age_stratified_quantity

    #########################
    ## Load necessary data ##
    #########################

    # Population size, interaction matrices and the model parameters
    initN, Nc_dict, params, BASE_samples_dict = model_parameters.get_COVID19_SEIQRD_parameters(age_classes=age_classes,spatial='prov')
    # Google Mobility data (for social contact Nc)
    df_google = mobility.get_google_mobility_data(update=update_data, provincial=True)
    # Load and format mobility dataframe (for mobility place)
    proximus_mobility_data = mobility.get_proximus_mobility_data('prov')

    ##############################################
    ## Construct vaccination rescaling function ##
    ##############################################

    # Variants of concern
    # Load previously saved VOC parameters and 
    VOC_params_previous = pd.read_pickle(os.path.join(abs_dir, '../../../data/interim/model_parameters/COVID19_SEIQRD/VOCs/VOC_parameters.pkl'))
    vaccine_params_previous = pd.read_pickle(os.path.join(abs_dir, '../../../data/interim/model_parameters/COVID19_SEIQRD/VOCs/vaccine_parameters.pkl'))
    # Load currently saved VOC parameters
    VOC_params, vaccine_params, params = model_parameters.get_COVID19_SEIQRD_VOC_parameters(VOCs=VOCs, pars_dict=params)
    # Time-dependent VOC function, updating alpha
    VOC_function = make_VOC_function(VOC_params['logistic_growth'])
    # Assert if there are differences in the VOC parameters, if yes, the rescaling dataframe must be updated
    # If (vaccine) data is updated, so must the rescaling dataframe
    rescaling_update=False
    if ((not VOC_params_previous.equals(VOC_params)) or (not vaccine_params_previous.equals(vaccine_params))):
        rescaling_update = True
    elif update_data == True:
        rescaling_update = True

    if rescaling_update == True:
        # Load and format local vaccination-induced rescaling data
        df_vacc = sciensano.get_public_spatial_vaccination_data(update=update_data, agg=agg)
        # Compute the rescaling parameters
        vaccination_rescaling_function = make_vaccination_rescaling_function(update=rescaling_update, agg=agg, age_classes=age_classes, df_incidences=df_vacc, VOC_function=VOC_function, vaccine_params=vaccine_params)
    else:
        vaccination_rescaling_function = make_vaccination_rescaling_function(update=rescaling_update, agg=agg, age_classes=age_classes)

    ##################################################
    ## Construct time-dependent parameter functions ##
    ##################################################

    # Time-dependent social contact matrix over all policies, updating Nc
    policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_all_spatial
    policy_function_work = make_contact_matrix_function(df_google, Nc_dict).policies_all_work_only
    # Time-dependent mobility function, updating P (place)
    mobility_function = \
        make_mobility_update_function(proximus_mobility_data).mobility_wrapper_func
    # Time-dependent vaccination-induced rescaling parameter functions, updating E_susc^g, E_inf^g, E_hosp^g
    E_susc_function = vaccination_rescaling_function.E_susc
    E_inf_function = vaccination_rescaling_function.E_inf
    E_hosp_function = vaccination_rescaling_function.E_hosp
    # Time-dependent seasonality function, updating season_factor
    seasonality_function = make_seasonality_function()

    ####################
    ## Initial states ##
    ####################

    # Get correct initial condition
    samples_path = os.path.join(abs_dir, data_path + 'interim/model_parameters/COVID19_SEIQRD/initial_conditions/'+agg+'/')
    if ((start_date == '2020-03-23') | (start_date == '2020-03-22') | (start_date == '2020-03-21') | (start_date == '2020-03-20') | (start_date == '2020-03-19') |
        (start_date == '2020-03-18') | (start_date == '2020-03-17') | (start_date == '2020-03-16') | (start_date == '2020-03-15') ) :
        with open(samples_path+'initial_states-COVID19_SEIQRD_spatial_rescaling.pickle', 'rb') as handle:
            load = pickle.load(handle)
            initial_states = load[start_date]

    elif ((start_date == '2021-08-01') | (start_date == '2021-09-01') | (start_date == '2021-10-01')):
        with open(samples_path+'summer_2021-COVID19_SEIQRD_spatial.pickle', 'rb') as handle:
            load = pickle.load(handle)
            initial_states = load[start_date]
    else:
        raise ValueError("Chosen startdate '{0}' is not valid. Choose: 2020-03-17, 2021-08-01 or 2021-09-01".format(start_date))
    # Convert to right age groups using demographic wheiging
    for key,value in initial_states.items():
        converted_value = np.zeros([value.shape[0], len(age_classes)])
        for i in range(value.shape[0]):
            column = value[i,:]
            data = pd.Series(index=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'), data=column)
            converted_value[i,:] = convert_age_stratified_quantity(data, age_classes).values
        initial_states.update({key: converted_value})
        
    ##########################
    ## Initialize the model ##
    ##########################

    time_dependent_parameters={'Nc' : policy_function,
                               'Nc_work' : policy_function_work,
                               'place' : mobility_function,
                               'f_VOC' : VOC_function,
                               'E_susc' : E_susc_function,
                               'E_inf' : E_inf_function,
                               'E_hosp' : E_hosp_function,
                               'seasonality' : seasonality_function}
    if virgin:
        # Drop VOCs and vaccination
        for key in ['f_VOC', 'E_susc', 'E_inf', 'E_hosp']:
            time_dependent_parameters.pop(key)

    model = COVID19_SEIQRD_spatial_rescaling(initial_states, params, spatial='prov', time_dependent_parameters=time_dependent_parameters)

    return model, BASE_samples_dict, initN

def load_samples_dict(filepath, age_stratification_size=10):
    """
    A function to load the samples dictionary from the model calibration, and append the hospitalization parameters bootstrapped samples and resusceptibility to them.

    Parameters
    ----------

    filepath : str
        Path to samples dictionary
    
    Returns
    -------

    samples_dict: dict
        Original samples dict plus bootstrapped samples of hospitalization mortalities ('samples_fractions') and parameters of distributions of residence times in hospital ('residence_times')
        The natural re-susceptibility samples (parameter 'zeta'), calibrated to the first 2020 COVID-19 wave serodata are also appended to the dictionary
    """
    
    # Set correct age_paths to find the hospital data
    if age_stratification_size == 3:
        age_path = '0_20_60/'
    elif age_stratification_size == 9:
        age_path = '0_10_20_30_40_50_60_70_80/'
    elif age_stratification_size == 10:
        age_path = '0_12_18_25_35_45_55_65_75_85/'
    else:
        raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
            )
    # Load raw samples dict
    samples_dict = json.load(open(filepath))
    # Append data on hospitalizations
    residence_time_distributions = pd.read_excel('../../data/interim/model_parameters/COVID19_SEIQRD/hospitals/'+age_path+'sciensano_hospital_parameters.xlsx', sheet_name='residence_times', index_col=0, header=[0,1])
    samples_dict.update({'residence_times': residence_time_distributions})
    bootstrap_fractions = np.load('../../data/interim/model_parameters/COVID19_SEIQRD/hospitals/'+age_path+'sciensano_bootstrap_fractions.npy')
    samples_dict.update({'samples_fractions': bootstrap_fractions})
    return samples_dict

def draw_fnc_COVID19_SEIQRD_national(param_dict,samples_dict):
    """
    A function to draw samples from the estimated posterior distributions of the model parameters.
    Tailored for use with the national COVID-19 SEIQRD model.

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

    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['eff_schools'] = samples_dict['eff_schools'][idx]    
    param_dict['eff_home'] = samples_dict['eff_home'][idx]      
    param_dict['eff_work'] = samples_dict['eff_work'][idx]       
    param_dict['eff_rest'] = samples_dict['eff_rest'][idx]
    param_dict['mentality'] = samples_dict['mentality'][idx]
    param_dict['K_inf'] = np.array([samples_dict['K_inf_abc'][idx], samples_dict['K_inf_delta'][idx]], np.float64)
    param_dict['amplitude'] = samples_dict['amplitude'][idx]
    param_dict['zeta'] = samples_dict['zeta'][idx]  

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

def draw_fnc_COVID19_SEIQRD_spatial(param_dict,samples_dict):
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
    param_dict['eff_schools'] = samples_dict['eff_schools'][idx]    
    param_dict['eff_home'] = samples_dict['eff_home'][idx]      
    param_dict['eff_work'] = samples_dict['eff_work'][idx]       
    param_dict['eff_rest'] = samples_dict['eff_rest'][idx]
    param_dict['mentality'] = samples_dict['mentality'][idx]
    param_dict['K_inf'] = np.array([samples_dict['K_inf_abc'][idx], samples_dict['K_inf_delta'][idx]], np.float64)
    param_dict['amplitude'] = samples_dict['amplitude'][idx]
    param_dict['zeta'] = samples_dict['zeta'][idx]

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

def output_to_visuals(output, states, alpha=1e-6, n_draws_per_sample=1, UL=1-0.05*0.5, LL=0.05*0.5):
    """
    A function to add the a-posteriori poisson uncertainty on the relationship between the model output and data
    and format the model output in a pandas dataframe for easy acces


    Parameters
    ----------

    output : xarray
        Simulation output xarray

    states : xarray
        Model states on which to add the a-posteriori poisson uncertainty

    alpha: float
        Overdispersion factor of the negative binomial distribution. For alpha --> 0, the negative binomial converges to the poisson distribution.

    n_draws_per_sample : int
        Number of poisson experiments to be added to each simulated trajectory (default: 1)

    UL : float
        Upper quantile of simulation result (default: 97.5%)

    LL : float
        Lower quantile of simulation result (default: 2.5%)

    Returns
    -------

    df : pd.DataFrame
        contains for every model state the mean, median, lower- and upper quantiles
        index is equal to simtime

    Example use
    -----------

    simtime, df_2plot = output_to_visuals(output, 100, 1, LL = 0.05/2, UL = 1 - 0.05/2)
    # x-values do not need to be supplied when using `plt.plot`
    plt.plot(df_2plot['H_in', 'mean'])
    # x-values must be supplied when using `plt.fill_between`
    plt.fill_between(simtime, df_2plot['H_in', 'LL'], df_2plot['H_in', 'UL'])

    """
    # Check if dimension draws is present
    if not 'draws' in list(output.dims):
        raise ValueError(
            "dimension 'draws' is not present in model output xarray"
        )
    # Check if the states are present
    for state_name in states:
        if not state_name in list(output.data_vars):
            raise ValueError(
                "variable state_name '{0}' is not a model state".format(state_name)
            )
    # Initialize a pandas dataframe for results
    columns = [[],[]]
    tuples = list(zip(*columns))
    columns = pd.MultiIndex.from_tuples(tuples, names=["model state", "quantity"])
    df = pd.DataFrame(index=pd.to_datetime(output['time'].values), columns=columns)
    df.index.name = 'simtime'
    # Deepcopy xarray output (it is mutable like a dictionary!)
    copy = output.copy(deep=True)
    # Loop over output states
    for state_name in states:
        # Automatically sum all dimensions except time and draws
        for dimension in output.dims:
            if ((dimension != 'time') & (dimension != 'draws')):
                copy[state_name] = copy[state_name].sum(dim=dimension)
        mean, median, lower, upper = add_negative_binomial(copy[state_name].values, alpha, n_draws_per_sample, UL, LL, add_to_mean=False)
        # Add to dataframe
        df[state_name,'mean'] = mean
        df[state_name,'median'] = median
        df[state_name,'lower'] = lower
        df[state_name,'upper'] = upper
    return df

def add_negative_binomial(output_array, alpha, n_draws_per_sample=100, LL=0.05*0.5, UL=1-0.05*0.5, add_to_mean=True):
    """ A function to add a-posteriori negative binomial uncertainty on the relationship between the model output and data
    
    Parameters
    ----------

    output_array: np.array
        2D numpy array containing the simulation result. First axis: draws, second axis: time.

    alpha: float
        Negative binomial overdispersion coefficient

    n_draws_per_sample: int
        Number of draws to take from the negative binomial distribution at each timestep and then average out.
    
    LL: float
        Lower quantile limit.

    UL: float
        Upper quantile limit.

    add_to_mean: boolean
        If True, `n_draws_per_sample` negative binomial draws are added to the mean model prediction. If False, `n_draws_per_sample` negative binomial draws are added to each of the `n_samples` model predictions.
        Both options converge for large `n_draws_per_sample`.

    Returns
    -------

    mean: np.array
        1D numpy array containing the mean model prediction at every timestep
    
    median: np.array
        1D numpy array containing the mean model prediction at every timestep
    
    lower: np.array
        1D numpy array containing the lower quantile of the model prediction at every timestep

    upper: np.array
        1D numpy array containing the upper quantile of the model prediction at every timestep
    """

    # Determine number of samples and number of timesteps
    simtime = output_array.shape[1]
    if add_to_mean:
        output_array= np.mean(output_array, axis=0)
        output_array=output_array[np.newaxis, :]
        n_samples=1
    else:
        n_samples = output_array.shape[0]
    # Initialize a column vector to append to
    vector = np.zeros((simtime,1))
    # Loop over dimension draws
    for n in range(n_samples):
        try:
            for draw in range(n_draws_per_sample):
                vector = np.append(vector, np.expand_dims(np.random.negative_binomial(1/alpha, (1/alpha)/(output_array[n,:] + (1/alpha)), size = output_array.shape[1]), axis=1), axis=1)
        except:
            warnings.warn("I had to remove a simulation result from the output because there was a negative value in it..")
    # Remove first column
    vector = np.delete(vector, 0, axis=1)
    #  Compute mean and median
    mean = np.mean(vector,axis=1)
    median = np.median(vector,axis=1)    
    # Compute quantiles
    lower = np.quantile(vector, q = LL, axis = 1)
    upper = np.quantile(vector, q = UL, axis = 1)

    return mean, median, lower, upper

def add_poisson(output_array, n_draws_per_sample=1, UL=1-0.05*0.5, LL=0.05*0.5):
    """ A function to add the a-posteriori poisson uncertainty on the relationship between the model output and data

    Parameters
    ----------

    output : np.array
        Simulation output vector to add a-posteriori poisson uncertainty to
        Must be of shape: [n_samples, simtime]

    n_draws_per_sample : int
        Number of poisson experiments to be added to each simulated trajectory (default: 1)

    UL : float
        Upper quantile of simulation result (default: 97.5%)

    LL : float
        Lower quantile of simulation result (default: 2.5%)

    Returns
    -------

    mean: np.array
        contains mean model prediction for 'state_name' at simulation times 'simtime'

    median : np.array
        contains median model prediction for 'state_name' at simulation times 'simtime'

    lower : np.array
        contains lower quantile of model prediction for 'state_name' at simulation times 'simtime'

    upper : np.array
        contains upper quantile of model prediction for 'state_name' at simulation times 'simtime'

    Example use
    -----------
    simtime, mean, median, lower, upper = add_poisson('H_in', output, 100, 1)

    """

    # Determine number of samples and number of timesteps
    simtime = output_array.shape[1]
    n_samples = output_array.shape[0]
    # Initialize a column vector to append to
    vector = np.zeros((simtime,1))
    # Loop over dimension draws
    for n in range(n_samples):
        try:
            result = np.random.poisson( np.expand_dims(output_array[n,:],axis=1),size = (output_array.shape[1],n_draws_per_sample))
            vector = np.append(vector, result, axis=1)
        except:
            warnings.warn("I had to remove a simulation result from the output because there was a negative value in it..")
    # Remove first column
    vector = np.delete(vector, 0, axis=1)
    #  Compute mean and median
    mean = np.mean(vector,axis=1)
    median = np.median(vector,axis=1)    
    # Compute quantiles
    lower = np.quantile(vector, q = LL, axis = 1)
    upper = np.quantile(vector, q = UL, axis = 1)

    return mean, median, lower, upper

def name2nis(name):
    """
    A function to convert the name of a Belgian municipality/arrondissement/province/etc. to its NIS code

    Parameters
    ----------
    name : str
        the name of the municipality/arrondissement/province/etc.

    Returns
    -------
    NIS : float
        the NIS code corresponding with the given name

    """
    # Load the list of name-NIS couples
    name_df=pd.read_csv(os.path.join(data_path, 'raw/GIS/NIS_name.csv'))
    pos_name = name_df['name'].values
    # Convert list of possible names to lowercase only
    pos_name_lower = [string.lower() for string in pos_name]
    name_df['name'] = pos_name_lower
    # Check if input is a string
    if not isinstance(name,str):
        raise TypeError(
            "name2nis input must be a string"
            )
    # Convert input to lowercase
    name = name.lower()
    # Search for a match and return NIS code
    if not name in pos_name_lower:
        raise ValueError(
                "No match for '{0}' found".format(name)
            )
    else:
        return name_df[name_df['name'] == name]['NIS'].values[0]

@jit(nopython=True)
def stratify_beta(beta_R, beta_U, beta_M, areas, pops, RU_threshold=400, UM_threshold=4000):
    """
    Function that returns a spatially stratified infectivity parameter. IMPORTANT: this assumes that throughout the model, all NIS values are in order (e.g. 11000 to 93000). Currently hard-coded on threshold densities of 400/km2 and 4000/km2. Indices indicated in order of density.
    
    Input
    -----
    beta_R : float
        Infecitivity in rural areas
    beta_U : float
        Infectivity in urban areas
    beta_M : float
        Infectivity in metropolitan areas
    areas : np.array
        G-fold numpy.array with areas of all regions in order of increasing NIS code
    pops : np.array
        G-fold numpy.array with populations in all regions in order of increasing NIS code
    RU_threshold : float
        Threshold population density to distinguish between rural and urbanised regions. Default: 400/km2
    UM_threshold : float
        Threshold population density to distinguish between urbanised and metropolitan regions. Default: 4000/km2

    Returns
    -------
    beta : np.array of floats
        Array with length fitting to aggregation level agg, and three degrees of freedom depending on beta_R, beta_U, beta_M
    """

    # Exceptions
    if (RU_threshold >= UM_threshold) or (RU_threshold < 0) or (UM_threshold < 0):
        raise Exception("RU_threshold must be smaller than UM_threshold and both values must be positive (units of people/km2).")
        
    # Define densities
    dens = pops/areas

    # Initialise and fill beta array
    beta = np.full(len(dens), beta_U, np.float64) # np.ones(len(dens))*beta_U # inbetween values
    beta[dens < RU_threshold] = beta_R # lower-than-threshold values
    beta[dens >= UM_threshold] = beta_M # higher-than-threshold values

    return beta


def initial_state(dist='bxl', agg='arr', number=1, age=-1, age_stratification_size=9):
    """
    Function determining the initial state of a model compartment.
    
    Input
    -----
    dist: str or int
        Spatial distribution of the initial state. Choose between 'bxl' (Brussels-only, default), 'hom' (homogeneous), 'data' (data-inspired), or 'frac' (fraction of hospitalisations at March 20th 2020), or choose a NIS code (5-digit int) corresponding to the aggregation level.
    agg: str
        Level of spatial aggregation. Choose between 'mun' (581 municipalities), 'arr' (43 arrondissements, default), or 'prov' (10+1 provinces)
    number: int
        Total number of people initialised in the compartment. 1 by default. Note that this generally needs to be changed if dist != 'bxl'
    age: int
        Integer larger than -1. If -1 (default), random ages are chosen (following demography and age stratification). If a non-negative integer is chosen, this corresponds to the index of the stratified class. Exception is raised when an age is chosen beyond the number of age classes.
    age_stratification_size: int
        The stratification size of the ages considered in the model. Choose between 3, 9 (default) or 10.

    Returns
    -------
    init: np.array containing integers
        The initial state with 11, 43 or 581 rows and 9 columns, representing the initial age and spatial distribution of people in a particular SEIR compartment.
    """
    
    from covid19model.data.model_parameters import construct_initN
    
    # Raise exceptions if input is wrong
    if not isinstance(dist, int) and (dist not in ['bxl', 'hom', 'data', 'frac']):
        raise Exception(f"Input dist={dist} is not recognised. Choose between 'bxl', 'hom', 'data' or 'frac', or pick a NIS code (integer).")
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception(f"Aggregation level {agg} not recognised. Choose between 'prov', 'arr' or 'mun'.")
    if not ((number > 0) and float(number).is_integer()):
        raise Exception(f"Input number={number} is not acceptable. Choose a natural number.")
    if not ((-1 <= age) and float(number).is_integer()):
        raise Exception(f"Input age={age} is not acceptable. Choose an integer -1 (random) or positive (reduces to age decade).")
    if age >= age_stratification_size:
        raise Exception(f"Age index {age} falls outside the number of stratified age classes ({age_stratification_size}).")

    if age_stratification_size == 3:
        initN = construct_initN(pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left'), agg).values
    elif age_stratification_size == 9:
        initN = construct_initN(pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left'), agg).values
    elif age_stratification_size == 10:
        initN = construct_initN(pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'), agg).values
    else:
        raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
        )      
            
    # Initialise matrix with proper dimensions.
    G, N = initN.shape
    initE = np.zeros((G,N))
    
    # Case for chosen NIS code
    if isinstance(dist,int):
        # Find coordinate of chosen NIS code
        gg=np.where(read_coordinates_nis(spatial=agg)==dist)[0][0]
        initE[gg] = _initial_age_dist(number, age, initN[gg], age_stratification_size=age_stratification_size)
    
    # Case for Brussels
    elif dist=='bxl':
        # Find coordinate of bxl NIS code
        if agg in ['arr', 'prov']:
            gg=np.where(read_coordinates_nis(spatial=agg)==21000)[0][0]
        else:
            gg=np.where(read_coordinates_nis(spatial=agg)==21004)[0][0] # Choice is made for historical Brussels
        initE[gg]= _initial_age_dist(number, age, initN[gg], age_stratification_size=age_stratification_size)
    
    # Case for data-inspired initial conditions, based on highest concentration in first peak
    # Note: the cases are spread almost equally (local population is only a secondary attention point)
    elif dist=='data':
        # Find coordinates of NIS codes for Alken, Sint-Truiden, Quévy
        coordinates_nis=read_coordinates_nis(spatial=agg)
        gg_array=[]
        if agg == 'arr':
            gg_array.append(np.where(coordinates_nis==73000)[0][0]) # arrondissement Tongeren (Alken)
            gg_array.append(np.where(coordinates_nis==71000)[0][0]) # arrondissement Hasselt (Sint-Truiden)
            gg_array.append(np.where(coordinates_nis==53000)[0][0]) # arrondissement Mons (Quévy)
        elif agg == 'prov':
            gg_array.append(np.where(coordinates_nis==70000)[0][0]) # province Limburg (Alken)
            gg_array.append(np.where(coordinates_nis==70000)[0][0]) # province Limburg (Sint-Truiden), count double
            gg_array.append(np.where(coordinates_nis==50000)[0][0]) # province Hainaut (Quévy)
        else:
            gg_array.append(np.where(coordinates_nis==73001)[0][0]) # municipality Alken
            gg_array.append(np.where(coordinates_nis==71053)[0][0]) # municipality Sint-Truiden
            gg_array.append(np.where(coordinates_nis==53084)[0][0]) # municipality Quévy
        initN_tot=np.array([initN[gg_array[i]].sum(axis=0) for i in range(3)])
        initE_all_ages = np.array([number//3 for i in range(3)])
        for i in range(number%3):
            jj = np.where(initN_tot==np.sort(initN_tot)[-i-1])[0][0] # find index of highest populations
            initE_all_ages[jj] += 1 # add remaining initial states to region with highest population first
        for i in range(3):
            initE[gg_array[i]] += _initial_age_dist(initE_all_ages[i], age, initN[gg_array[i]], age_stratification_size=age_stratification_size)
            
    # Case for homogeneous initial conditions: equal country-wide distribution
    # Note: the cases are spread almost equally (local population is only a secondary attention point)
    elif dist=='hom':
        initN_tot = initN.sum(axis=1)
        initE_all_ages = np.array([number//G for i in range(G)])
        initN_tot_sorted = np.sort(initN_tot)
        for i in range(number%G):
            jj = np.where(initN_tot==initN_tot_sorted[i-1])[0][0] # find index of highest populations
            initE_all_ages[jj] += 1 # add remaining initial states to region with highest population first
        for i in range(G):
            initE[i] += _initial_age_dist(initE_all_ages[i], age, initN[i], age_stratification_size=age_stratification_size)
          
    # Case for initial conditions based on fraction of hospitalisations on 20 March
    # If age < 0, the number of people is distributed over the age classes fractionally
    elif dist=='frac':
        from covid19model.data.sciensano import get_sciensano_COVID19_data_spatial
        # Note that this gives non-natural numbers as output
        max_date = '2020-03-20' # Hard-coded and based on Arenas's convention
        values = 'hospitalised_IN' # Hard-coded and 
        df = get_sciensano_COVID19_data_spatial(agg=agg, values=values, moving_avg=True)
        max_value = df.loc[max_date].sum()
        df_frac = (df.loc[max_date] / max_value * number)
        for nis in df_frac.index:
            gg = np.where(read_coordinates_nis(spatial=agg)==nis)[0][0]
            initE[gg] = _initial_age_dist(df_frac.loc[nis], age, initN[gg], fractional=True, age_stratification_size=age_stratification_size)
            
    return initE

def _initial_age_dist(number, age,  pop, fractional=False, age_stratification_size=9):
    """
    Help function for initial_state, for the distribution of the initial state over the required age classes.
    
    Input
    -----
    number: int
        Total number of people initialised in the compartment.
    age: int
        Integer ranging from -1 to age_stratification_size-1. If -1, random ages are chosen (following demography). If 0 to age_stratification_size-1 is chosen, the number corresponds to the index of the stratified age class (e.g. 1 = ages 10-19)
    pop: np.array
        Contains population in the various age classes
    fractional: Boolean
        If True, the number is distributed over the age classes fractionally (such that we are no longer dealing with a whole number of people)
    age_stratification_size: int
        The stratification size of the ages considered in the model. Choose between 3, 9 (default) or 10.
        
    Returns
    -------
    init_per_age: np.array with integers of dimension 9
        The distribution of the people in a particular state in one particular region per age    
    """
    # Initialise age vector
    init_per_age = np.zeros(age_stratification_size)
    
    # Return vector with people in one particular age class
    if age > -1:
        init_per_age[int(age)] = number
    
    elif not fractional:
        indices = list(range(0,age_stratification_size))
        probs = pop/pop.sum(axis=0)
        index_choices = np.random.choice(indices, p=probs, size=number)
        unique, counts = np.unique(index_choices, return_counts=True)
        index_dict = dict(zip(unique, counts))
        for key in index_dict:
            init_per_age[key] = index_dict[key]
            
    elif fractional:
        indices = list(range(0,age_stratification_size))
        probs = pop/pop.sum(axis=0)
        init_per_age = number * probs
    
    return init_per_age

def read_coordinates_nis(spatial='arr'):
    """
    A function to extract from /data/interim/demographic/initN_arrond.csv the list of arrondissement NIS codes

    Parameters
    ----------
    spatial : str
        choose geographical aggregation. Pick between 'arr', 'mun', 'prov', 'test'. Default is 'arr'.

    Returns
    -------
     NIS: list
        a list containing the NIS codes of the 43 Belgian arrondissements, 581 municipalities, 10 provinces (+ Brussels-Capital), or 3
        arrondissements in the 'test' case (Antwerp, Brussels, Gent)

    """

    initN_df=pd.read_csv(os.path.join(data_path, 'interim/demographic/initN_' + spatial + '.csv'), index_col=[0])
    NIS = initN_df.index.values

    return NIS

def dens_dep(rho, xi=0.01):
    """
    A function used by Arenas et al. (2020) and justified by Hu et al. (2013) (https://pubmed.ncbi.nlm.nih.gov/23665296/)

    Parameters
    ----------
    rho : population density
    xi : scale parameter. Default value is 0.01 square kilometer

    Returns
    -------
    f : density dependence value, ranges between 1 and 2
    """

    f = 1 + ( 1 - np.exp(-xi * rho) )

    return f

def read_areas(spatial='arr'):
    """
    Reads full CSV with area per NIS code

    Parameters
    ----------
    spatial : str
        Choose between municipalities ('mun'), arrondissements ('arr'), provinces ('prov') or Antwerp-Brussel-Gent ('test').
        Defaults is 'arr'

    Returns
    -------
    areas : dictionary
        NIS codes are keys, values are population in square meters
    """

    areas_df = pd.read_csv(os.path.join(data_path, 'interim/demographic/area_' + spatial + '.csv'), index_col='NIS')
    areas = areas_df['area'].to_dict()

    return areas

def read_pops(spatial='arr',age_stratification_size=9,return_matrix=False,drop_total=False):
    """
    Reads initial population per age and per area

    Parameters
    ----------
    spatial : str
        choose geographical aggregation. Pick between 'arr', 'mun', 'prov', or 'test'. Default is 'arr'.
    return_matrix : boolean
        if True, return np.array instead of dictionary
    drop_total : boolean
        if True, drop the 10th column containing the sums of the rows (total population)

    Returns
    -------
    pops : dictionary
        NIS codes are keys, values are dictionaries. The inner dictionary has population classes as keys and
        population per age class and per NIS code as values.
        Age classes are [0,10), [10,20), ... , [80, 110)
    """

    if age_stratification_size not in [3, 9, 10]:
        raise Exception(f"Age stratification {age_stratification_size} is not allowed. Choose between either 3, 9 (default), or 10.")
    
    pops_df = pd.read_csv(os.path.join(data_path, 'interim/demographic/initN_' + spatial + '.csv'), index_col='NIS')
    if drop_total:
        pops_df.drop(columns='total', inplace=True)
    if return_matrix:
        pops = pops_df.values
    else:
        pops = pops_df.T.to_dict()

    return pops

def save_sim(out, name, group, new=False, descr=None, verbose=True):
    """Save the output xarray of a simulation as a zarr file for future reference. Especially pertinent when simulation takes a long time to process, such that simulations may be saved in a database. Note that data variable values are saved as Disk arrays (but open_sim converts them back to Numpy arrays)
    
    Parameters
    ----------
    out : xarray object
        output of the base sim function: xarray with dimensions 'time' and 'draws', and possibly 'Nc' (age stratification) and 'place' (geographical stratification), and attributes "parameters" (which holds a dictionary)
    name : string
        Name under which to save the zarr directory. If no directory path is provided, the directory is saved in the current location.
    group : string
        Name of the group in the zarr file to save the particular model scenario under
    new : Boolean
        If True, creates a new zarr file with name provided in the function argument. False by default.
    descr : string
        Description of the particular simulation inside the group. This is saved as an attribute.
    verbose : Boolean
        Print under which file and group name the xarray data has been saved. True by default.
    """
    
    ### EXCEPTIONS
    
    # Check the zarr name format
    if (type(name) != str) or (name[-5:] != '.zarr'):
        raise Exception(f"The 'name' parameter value '{name}' is invalid. 'name' should be a string with a .zarr extension.")
    
    # Check whether the zarr directory exists
    if (new == False) and not os.path.isdir(name):
        raise Exception(f"The 'name' parameter value '{name}' does not exist. If you want to create a new zarr file with this name, include 'new=True' in the function arguments.")
    
    if (new == True) and os.path.isdir(name):
        raise Exception(f"Cannot create new zarr file with name '{name}' because it already exists. Either add a group to this zarr file, or create a new zarr directory with a different name.")
        
    if os.path.isdir(name + '/' + group):
        raise Exception(f"The group '{group}' already exists in the zarr directory '{name}'. Create a new group or a new zarr directory.")
    
    # Check description properties
    if not descr:
        descr = 'No description provided'
    if type(descr) != str:
        raise Exception(f"The description '{descr}' is invalid. The 'descr' parameter value should be a string.")
    
    ### CORE
    
    # Save parameters dictionary one level up (otherwise it cannot be saved in zarr format)
    out.attrs = out.attrs['parameters']
    
    # Add description attribute
    out.attrs['description'] = descr
    
    # Save to zarr under the right name/group
    out.to_zarr(name, group=group)
    
    # Include message if verbose (default)
    if verbose:
        if new:
            message = f"Saved simulation output in newly created zarr directory '{name}' under the group '{group}'"
        else:
            message = f"Added simulation output to zarr directory '{name} under the group '{group}'"
        print(message)
        print("Description:")
        print("------------")
        print(f"'{descr}'")
    
    
def open_sim(name, group, verbose=True):
    """Open the saved simulation output xarray and (optionally) display the main characteristics. Note: can only handle parameter attributes + description (additional attributes will be wrongly categorised under 'parameters'). Note: changes the order of the parameters in the dictionary
    
    Parameters
    ----------
    name : str
        directory and name of the main zarr file
    group : str
        name of the group within the zarr directory in which the xarray is saved
    verbose : Boolean
        Print description and dimensional information of the simulation. Default is True.
    """
    
    ### EXCEPTIONS
    
    # Check the zarr name format
    if (type(name) != str) or (name[-5:] != '.zarr'):
        raise Exception(f"The 'name' parameter value '{name}' is invalid. 'name' should be a string with a .zarr extension.")
    
    # Verify whether the file and group exists
    if not os.path.isdir(name + '/' + group):
        raise Exception(f"The group '{name}/{group}' does not exist. Check the 'name' and 'group' parameter values.")
    
    ### CORE
    
    # Open xarray from zarr
    out = xr.open_zarr(name, group=group)
    if 'description' in out.attrs.keys():
        # Copy and delete description from attributes
        descr = out.attrs.pop('description')

    # Repack parameters into a dictionary
    param_dict = dict({})
    for key in out.attrs.copy():
        param_dict[key] = out.attrs[key]
        out.attrs.pop(key)
    out.attrs['parameters'] = param_dict
    
    # Convert Disk arrays values of data variables back to Numpy arrays
    for var in out:
        out[var].values = np.array(out[var].values)
        
    if verbose:
        print(f"Opened simulation output that is saved in {name}/{group}")
        print('')
        print('Dimensions:')
        print('-----------')
        for key, value in out.dims.items(): print(f"{key}: {value}")
        print('')
        print('Description')
        print('-----------')
        if descr:
            print(f"'{descr}'")
        else:
            print("WARNING: No 'description' attribute in opened model output. Has it been saved properly, using the save_sim() function?")
            
    return out

def double_heaviside(t, t0, delta_t=None):
    """
    Function used to inject one additional exposed subjects per day at time t0 for delta_t days
    
    Parameters
    ----------
    t : float
        Time to be evaluated. Returns 1 only if t is within [t0, t0+delta_t] and 0 otherwise.
    t0 : array of floats
        Times at which new exposed individuals are injected in units of days since a particular start day. This is a G-dimensional array (one time per region)
    delta_t: array of floats
        Length of injection window in units of days. This is a G-dimensional array (one duration per region). Defaults to one day.
        
    Returns
    -------
    dh : array of floats
        G-dimensional array with values 0 or 1.
    
    """
    if delta_t is None:
        delta_t = np.ones(len(t0))
    if len(t0) != len(delta_t):
        raise Exception("Dimension of t0 and delta_t must be identical.")
    dh=[]
    for h in range(len(t0)):
        dh_h = np.heaviside(t-t0[h],1) * np.heaviside(t0[h]+delta_t[h]-t,1)
        dh.append(dh_h)
    return np.array(dh)



