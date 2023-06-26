import os
import json
import warnings
from numba import jit
import numpy as np
import pandas as pd
import xarray as xr

abs_dir = os.path.dirname(__file__)
data_path = os.path.join(abs_dir, "../../../data/")

def initialize_COVID19_SEIQRD_hybrid_vacc(age_stratification_size=10, VOCs=['WT', 'abc', 'delta'], start_date=None, update_data=False,
                                            vaccination=True,  stochastic=False, distinguish_day_type=True):

    ###########################################################
    ## Convert age_stratification_size to desired age groups ##
    ###########################################################

    if age_stratification_size == 3:
        age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
    elif age_stratification_size == 9:
        age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
    elif age_stratification_size == 10:
        age_classes = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')
    elif age_stratification_size == 18:
        age_classes = pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50),
                                                        (50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,85),(85,120)], closed='left')
    else:
        raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
        )

    #####################################
    ## Import necessary pieces of code ##
    #####################################

    # Import the SEIQRD model with VOCs, vaccinations, seasonality
    from covid19_DTM.models import ODE_models, SDE_models
    # Import time-dependent parameter functions for resp. P, Nc, alpha, N_vacc, season_factor
    from covid19_DTM.models.TDPF import   make_contact_matrix_function, \
                                                                    make_VOC_function, \
                                                                    make_N_vacc_function, \
                                                                    make_vaccination_efficacy_function, \
                                                                    make_seasonality_function, \
                                                                    h_func
    # Import packages containing functions to load in data used in the model and the time-dependent parameter functions
    from covid19_DTM.data import mobility, sciensano, model_parameters

    #########################
    ## Load necessary data ##
    #########################

    # Interaction matricesm model parameters, samples dictionary
    Nc_dict, params, samples_dict, initN = model_parameters.get_model_parameters(age_classes=age_classes, distinguish_day_type=distinguish_day_type)
    # Load previous vaccine parameters and currently saved VOC/vaccine parameters
    vaccine_params_previous = pd.read_pickle(os.path.join(abs_dir, '../../../data/covid19_DTM/interim/model_parameters/VOCs/vaccine_parameters.pkl'))
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
    df_incidences_previous = pd.read_pickle(os.path.join(abs_dir, '../../../data/covid19_DTM/interim/sciensano/vacc_incidence_national.pkl'))
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

    # age- and dose stratification size
    N = len(age_classes)
    D = len(efficacy_function.df_efficacies.index.get_level_values('dose').unique())

    if not start_date:
        # Start with one exposed and one presymptomatic individual in every age group
        E0 = I0 = np.zeros([age_stratification_size,D])
        E0[:,0] = I0[:,0] = 1/(10/N) # Model calibrated with one sick individual in 10 age groups by default --> needs to be rescaled to remain accurate
        # Construct initial states dictionary (other states get filled with zeros automatically)
        initial_states={"S": np.concatenate( (np.expand_dims(initN, axis=1), np.ones([age_stratification_size,D-1])), axis=1),
                        "E": E0,
                        "I": E0
                        }
    else:
        reference_sim_path = os.path.join(abs_dir, '../../../data/covid19_DTM/interim/model_parameters/initial_conditions/national/')
        reference_sim_name = 'national_REF_SIMULATION_2022-09-13.nc'
        out = xr.open_dataset(reference_sim_path+reference_sim_name)
        initial_states={}
        for data_var in out.keys():
            initial_states.update({data_var: out.sel(time=start_date)[data_var].values})     

    ##########################################################################
    ## Vaccination module requires some additional parameters to be defined ##
    ##########################################################################
    
    # Define dummy vaccine efficacies
    e_i=e_s=e_h=np.ones([len(age_classes), len(efficacy_function.df_efficacies.index.get_level_values('dose').unique()), len(vaccine_params.index.get_level_values('VOC').unique())])

    # Vaccination parameters when using the stratified vaccination model
    params.update({'N_vacc': np.zeros([age_stratification_size, len(df_vacc.index.get_level_values('dose').unique())]),
                   'e_i': e_i,
                   'e_s': e_s,
                   'e_h': e_h,
                   })

    ##########################
    ## Initialize the model ##
    ##########################

    # Define coordinates
    coordinates = {'age_groups':construct_coordinates_Nc(age_stratification_size=age_stratification_size),
                    'doses': ['none', 'partial', 'full', 'boosted']}

    # Construct dictionary of time dependent parameters
    time_dependent_parameters={'Nc' : policy_function,
                               'f_VOC' : VOC_function,
                               'seasonality' : seasonality_function,}
    if vaccination:
        time_dependent_parameters.update({'N_vacc' : N_vacc_function,
                               'e_s' : efficacy_function.e_s,
                               'e_i' : efficacy_function.e_i,
                               'e_h' : efficacy_function.e_h,
                               'h': h_func})                      
    
    # Initialize model
    if stochastic == True:
        for key,state in initial_states.items():
            initial_states.update({key: np.rint(state)})
        model = SDE_models.COVID19_SEIQRD_hybrid_vacc_sto(initial_states, params, coordinates=coordinates, time_dependent_parameters=time_dependent_parameters)
    else:
        model = ODE_models.COVID19_SEIQRD_hybrid_vacc(initial_states, params, coordinates=coordinates, time_dependent_parameters=time_dependent_parameters)

    return model, samples_dict, initN

def initialize_COVID19_SEIQRD_spatial_hybrid_vacc(age_stratification_size=10, agg='prov', VOCs=['WT', 'abc', 'delta'], start_date=None,
                                                    vaccination=True, update_data=False, stochastic=False, distinguish_day_type=True):
    
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
    elif age_stratification_size == 18:
        age_classes = pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50),
                                                        (50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,85),(85,120)], closed='left')
    else:
        raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
        )

    #####################################
    ## Import necessary pieces of code ##
    #####################################

    # Import the SEIQRD model with VOCs, vaccinations, seasonality
    from covid19_DTM.models import ODE_models, SDE_models
    # Import time-dependent parameter functions for resp. P, Nc, alpha, N_vacc, season_factor
    from covid19_DTM.models.TDPF import make_mobility_update_function, \
                                        make_contact_matrix_function, \
                                        make_VOC_function, \
                                        make_N_vacc_function, \
                                        make_vaccination_efficacy_function, \
                                        make_seasonality_function, \
                                        h_func
    # Import packages containing functions to load in data used in the model and the time-dependent parameter functions
    from covid19_DTM.data import mobility, sciensano, model_parameters
    from covid19_DTM.data.utils import convert_age_stratified_quantity

    #########################
    ## Load necessary data ##
    #########################

    # Model parameters and population sizes
    Nc_dict, params, samples_dict, initN = model_parameters.get_model_parameters(age_classes=age_classes, agg=agg, distinguish_day_type=distinguish_day_type)
    # Load previous vaccine parameters and currently saved VOC/vaccine parameters
    vaccine_params_previous = pd.read_pickle(os.path.join(abs_dir, '../../../data/covid19_DTM/interim/model_parameters/VOCs/vaccine_parameters.pkl'))
    VOC_params, vaccine_params, params = model_parameters.get_COVID19_SEIQRD_VOC_parameters(VOCs=VOCs, pars_dict=params)
    # Using the weekly vaccination data
    df_vacc = sciensano.get_public_spatial_vaccination_data(update=update_data, agg=agg)
    # Proximus mobility data
    proximus_mobility_data = mobility.get_proximus_mobility_data(agg)
    # Google Mobility data
    if agg == 'prov':
        df_google = mobility.get_google_mobility_data(update=update_data, provincial=False)
    elif agg == 'arr':
        df_google = mobility.get_google_mobility_data(update=update_data, provincial=False)

    #####################################################################
    ## Construct time-dependent parameter functions except vaccination ##
    #####################################################################

    # Time-dependent VOC function, updating alpha
    VOC_function = make_VOC_function(VOC_params['logistic_growth'])
    # Time-dependent social contact matrix over all policies, updating Nc
    policy_function = make_contact_matrix_function(df_google, Nc_dict, G=len(df_vacc.index.get_level_values('NIS').unique())).policies_all_spatial
    policy_function_home = make_contact_matrix_function(df_google, Nc_dict, G=len(df_vacc.index.get_level_values('NIS').unique())).policies_all_home_only
    # Time-dependent mobility function, updating P (place)
    mobility_function = make_mobility_update_function(proximus_mobility_data).mobility_wrapper_func
    # Time-dependent seasonality function, updating season_factor
    seasonality_function = make_seasonality_function()

    ##############################################################
    ## Construct vaccination time-dependent parameter functions ##
    ##############################################################

    try:
        # Check if dataframe with incidences is available
        df_incidences_previous = pd.read_pickle(os.path.join(abs_dir, '../../../data/covid19_DTM/interim/sciensano/vacc_incidence_'+agg+'.pkl'))
        # Time-dependent (first) vaccination function, updating N_vacc.
        N_vacc_function = make_N_vacc_function(df_vacc['INCIDENCE'], age_classes=age_classes, agg=agg, hypothetical_function=False)
        # Extract the smoothed dataframe
        df_incidences = N_vacc_function.df
        # Check if the vaccination efficacies must be updated
        rescaling_update=False
        if ((not vaccine_params_previous.equals(vaccine_params)) or (not df_incidences_previous.equals(df_incidences))):
            rescaling_update = True
        elif update_data == True:
            rescaling_update = True
        # Construct the efficacy function subject to waning
        efficacy_function = make_vaccination_efficacy_function(update=rescaling_update, agg=agg, df_incidences=df_incidences, vaccine_params=vaccine_params,
                                                VOCs=VOCs, age_classes=age_classes)
    except:
        N_vacc_function = make_N_vacc_function(df_vacc['INCIDENCE'], age_classes=age_classes, agg=agg, hypothetical_function=False)
        # Extract the smoothed dataframe
        df_incidences = N_vacc_function.df
        # Construct the efficacy function subject to waning
        efficacy_function = make_vaccination_efficacy_function(update=True, agg=agg, df_incidences=df_incidences, vaccine_params=vaccine_params,
                                                VOCs=VOCs, age_classes=age_classes)

    ####################
    ## Initial states ##
    ####################

    # Space, age, dose sizes
    G = len(df_incidences.index.get_level_values('NIS').unique())
    N = len(age_classes)
    D = len(efficacy_function.df_efficacies.index.get_level_values('dose').unique())

    if not start_date:
        # Number of susceptibles
        S0 = np.concatenate( (np.expand_dims(initN,axis=2), 0.01*np.ones([G,N,3])), axis=2)
        # Start with one exposed and one presymptomatic individual in every age group
        E0 = I0 = np.zeros([G,N,D])
        E0[:,:,0] = I0[:,:,0] = 1 # Model calibrated with one sick individual in 10 age groups by default --> needs to be rescaled to remain accurate
        # Construct initial states dictionary (other states get filled with zeros automatically)
        initial_states={"S": S0,
                        "E": E0,
                        "I": E0
                        }
    else:
        reference_sim_path = os.path.join(abs_dir, f'../../../data/covid19_DTM/interim/model_parameters/initial_conditions/{agg}/')
        reference_sim_name = f'{agg}_INITIAL-CONDITION.nc'
        out = xr.open_dataset(reference_sim_path+reference_sim_name)
        initial_states={}
        for data_var in out.keys():
            try:
                initial_states.update({data_var: out.sel(date=start_date)[data_var].values})
            except:
                raise ValueError("Chosen startdate '{0}' not found.".format(start_date))
            
    # Convert to the right age groups
    for key,value in initial_states.items():
        converted_value = np.zeros([G, N, D])
        for i in range(value.shape[0]):
            for j in range(value.shape[2]):
                column = value[i,:,j]
                data = pd.Series(index=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'), data=column)
                converted_value[i,:,j] = convert_age_stratified_quantity(data, age_classes).values
        initial_states.update({key: converted_value})

    if stochastic == True:
        initial_states.update({'S_work': np.zeros([G,N,D])})

    ##########################################################################
    ## Vaccination module requires some additional parameters to be defined ##
    ##########################################################################
    
    # Define dummy vaccine efficacies
    e_i=e_h=e_s=np.ones([G, N, D, len(vaccine_params.index.get_level_values('VOC').unique())])
    # Add vaccination parameters to parameter dictionary
    params.update({'N_vacc': np.zeros([G, N, D]),
                   'e_i': e_i,
                   'e_s': e_s,
                   'e_h': e_h,
                    })  

    ##########################
    ## Initialize the model ##
    ##########################

    params.update({'summer_rescaling_F': 0, 'summer_rescaling_W': 0, 'summer_rescaling_B': 0})

    # Define coordinates
    coordinates = {'NIS': read_coordinates_place(agg=agg),
                   'age_groups': construct_coordinates_Nc(age_stratification_size=age_stratification_size),
                   'doses': ['none', 'partial', 'full', 'boosted']}

    # Define time-dependent-parameters
    time_dependent_parameters={'Nc' : policy_function,
                               'Nc_home' : policy_function_home,
                               'NIS' : mobility_function,
                               'f_VOC' : VOC_function,
                               'seasonality' : seasonality_function,
                               'h': h_func}
    if vaccination:
        time_dependent_parameters.update({'N_vacc' : N_vacc_function,
                                          'e_s' : efficacy_function.e_s,
                                          'e_i' : efficacy_function.e_i,
                                          'e_h' : efficacy_function.e_h})       
                                                        
    # Setup model
    if stochastic == True:
        for key,state in initial_states.items():
            initial_states.update({key: np.rint(state)})
        model = SDE_models.COVID19_SEIQRD_spatial_hybrid_vacc_sto(initial_states, params, coordinates=coordinates, time_dependent_parameters=time_dependent_parameters)
    else:
        model = ODE_models.COVID19_SEIQRD_spatial_hybrid_vacc(initial_states, params, coordinates=coordinates, time_dependent_parameters=time_dependent_parameters)

    return model, samples_dict, initN

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
    
    abs_dir = os.path.dirname(__file__)

    # Set correct age_paths to find the hospital data
    if age_stratification_size == 3:
        age_path = '0_20_60/'
    elif age_stratification_size == 9:
        age_path = '0_10_20_30_40_50_60_70_80/'
    elif age_stratification_size == 10:
        age_path = '0_12_18_25_35_45_55_65_75_85/'
    elif age_stratification_size == 18:
        age_path = '0_5_10_15_20_25_30_35_40_45_50_55_60_65_70_75_80_85/'
    else:
        raise ValueError(
            "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
            )
    # Load raw samples dict
    samples_dict = json.load(open(filepath))
    # Append data on hospitalizations
    residence_time_distributions = pd.read_excel(os.path.join(abs_dir,'../../../data/covid19_DTM/interim/model_parameters/hospitals/'+age_path+'sciensano_hospital_parameters.xlsx'), sheet_name='residence_times', index_col=0, header=[0,1])
    samples_dict.update({'residence_times': residence_time_distributions})
    bootstrap_fractions = np.load(os.path.join(abs_dir,'../../../data/covid19_DTM/interim/model_parameters/hospitals/'+age_path+'sciensano_bootstrap_fractions.npy'))
    samples_dict.update({'samples_fractions': bootstrap_fractions})
    return samples_dict

def aggregate_Brussels_Brabant_data(initN, df_hosp):
    """
    An ugly function to aggregate the provincial hospitalisation data and demographics for Brussels (NIS: 21000), Brabant-Wallon (NIS: 20002) and Vlaams-Brabant (NIS: 20001)
    """

    # Aggregate Bxl and Brabant hospitalisation data
    df_hosp.loc[slice(None), 21000] = (df_hosp.loc[slice(None), 20001] + df_hosp.loc[slice(None), 20002] + df_hosp.loc[slice(None), 21000]).values
    df_hosp = df_hosp.reset_index()
    df_hosp = df_hosp[((df_hosp['NIS'] != 20001) & (df_hosp['NIS'] != 20002))]
    df_hosp = df_hosp.groupby(by=['date','NIS']).sum().squeeze()

    # Aggregate initN
    initN.loc[21000] = (initN.loc[20001] + initN.loc[20002] + initN.loc[21000]).values
    initN = initN.reset_index()
    initN = initN[((initN['NIS'] != 20001) & (initN['NIS'] != 20002))]
    initN = initN.groupby(by=['NIS']).sum().squeeze()

    return initN, df_hosp

import xarray as xr
def aggregate_Brussels_Brabant_Dataset(simulation_in):
    """
    A wrapper for `aggregate_Brussels_Brabant()`, converting all model states into the aggregated format

    Input
    =====
    
    simulation_in: xarray.Dataset
        Simulation result (arrondissement or provincial level)
    
    Output
    ======
    
    simulation_out: xarray.Dataset
        Simulation result. Provincial spatial aggregation with Bruxelles and Brabant aggregated into NIS 21000
    """
    output = []
    for state in simulation_in.keys():
        o = aggregate_Brussels_Brabant_DataArray(simulation_in[state])
        o.name = state
        output.append(o)
    return xr.merge(output)

def aggregate_Brussels_Brabant_DataArray(simulation_in):
    """
    A function to aggregate an arrondissement simulation to the provincial level.
    A function to aggregate the provinces of Bruxelles, Brabant Wallon and Vlaams Brabant into one province.
    
    Input
    =====
    
    simulation_in: xarray.DataArray
        Simulation result (arrondissement or provincial level)
    
    Output
    ======
    
    simulation_out: xarray.DataArray
        Simulation result. Provincial spatial aggregation with Bruxelles and Brabant aggregated into NIS 21000
    """

    # Conversion keys
    prov = [10000, 21000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
    arr2prov = [
            [11000, 12000, 13000],
            [21000, 23000, 24000, 25000],
            [31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000],
            [41000, 42000, 43000, 44000, 45000, 46000],
            [51000, 52000, 53000, 55000, 56000, 57000, 58000],
            [61000, 62000, 63000, 64000],
            [71000, 72000, 73000],
            [81000, 82000, 83000, 84000, 85000],
            [91000, 92000, 93000]
        ] 
    # Preallocate tensor
    if 'draws' in simulation_in.dims:
        data = np.zeros([len(prov),
                        len(simulation_in.coords['draws']),
                        len(simulation_in.coords['date']),
                        len(simulation_in.coords['age_groups']),
                        len(simulation_in.coords['doses'])
                        ])
    else:
        data = np.zeros([len(prov),
                        len(simulation_in.coords['date']),
                        len(simulation_in.coords['age_groups']),
                        len(simulation_in.coords['doses'])
                        ])
    # Aggregate data
    if 41000 in simulation_in.coords['NIS']:
        for i,arr_lst in enumerate(arr2prov):
            som=0
            for arr_NIS in arr_lst:
                som+=simulation_in.sel(NIS=arr_NIS).values
            data[i,...] = som
    else:
        for i,NIS in enumerate(prov):
            if NIS != 21000:
                data[i,...] = simulation_in.sel(NIS=NIS).values
            else:
                data[i,...] = simulation_in.sel(NIS=20001).values + simulation_in.sel(NIS=20002).values + simulation_in.sel(NIS=21000).values            
    # Send to simulation out
    if 'draws' in simulation_in.dims:
        data = np.swapaxes(np.swapaxes(data,0,1), 1,2)
        coords=dict(NIS = (['NIS'], prov),
                    draws = simulation_in.coords['draws'],
                    date = simulation_in.coords['date'],
                    age_groups = simulation_in.coords['age_groups'],
                    doses = simulation_in.coords['doses'],
                    )
    else:
        data = np.swapaxes(data,0,1)
        coords=dict(NIS = (['NIS'], prov),
            date = simulation_in.coords['date'],
            age_groups = simulation_in.coords['age_groups'],
            doses = simulation_in.coords['doses'],
            )
    return xr.DataArray(data, dims=simulation_in.dims, coords=coords)

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
    df = pd.DataFrame(index=pd.to_datetime(output['date'].values), columns=columns)
    df.index.name = 'simtime'
    # Deepcopy xarray output (it is mutable like a dictionary!)
    copy = output.copy(deep=True)
    # Loop over output states
    for state_name in states:
        # Automatically sum all dimensions except time and draws
        for dimension in output.dims:
            if ((dimension != 'date') & (dimension != 'draws')):
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
    name_df=pd.read_csv(os.path.join(data_path, 'covid19_DTM/raw/GIS/NIS_name.csv'))
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
def stratify_beta_density(beta_R, beta_U, beta_M, areas, pops, RU_threshold=400, UM_threshold=4000):
    """
    Function that returns a spatially stratified infectivity parameter.
    Assumes that the NIS values are in order (e.g. 11000 to 93000).
    
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
    beta : np.array
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

@jit(nopython=True)
def stratify_beta_regional(beta_W, beta_FL, beta_Bxl, G):
    """
    Function that returns a spatially stratified infectivity parameter. Stratified per Belgian region (Flanders, Wallonia, Brussels).

    Input
    -----

    beta_FL : float
        Infecitivity in Flanders
    beta_W : float
        Infectivity in Wallonia
    beta_Bxl : float
        Infectivity in Brussels
    G : int
        Number of spatial patches in the model

     Returns
    -------
    beta : np.array
        Array of length G containing the infectivity parameter per spatial patch.
    """

    # Initialize array containing Flanders infectivity
    beta = beta_FL*np.ones(G, np.float64)
    # Set Brussels infectivity
    beta[3] = beta_Bxl
    # Hardcode indices for prov and arr
    if G == 11:
        idx_W = [2, 6, 7, 9, 10]
    elif G == 43:                                    
        idx_W = [6,                                # Waals-Brabant
                21, 22, 23, 24, 25, 26, 27,        # Henegouwen
                28, 29, 30, 31,                    # Luik
                35, 36, 37, 38, 39,                # Luxemburg
                40, 41, 42]                        # Namen

    # Set correct infectivities for Wallonia
    for i in idx_W:
        beta[i] = beta_W

    return beta

def read_coordinates_place(agg='arr'):
    """
    A function to extract from /data/covid19_DTM/interim/demographic/initN_arrond.csv the list of arrondissement NIS codes

    Parameter
    ---------
    spatial : str
        choose geographical aggregation. Pick between 'arr', 'mun', 'prov', 'test'. Default is 'arr'.

    Returns
    -------
     NIS: list
        a list containing the NIS codes of the 43 Belgian arrondissements, 581 municipalities, 10 provinces (+ Brussels-Capital), or 3
        arrondissements in the 'test' case (Antwerp, Brussels, Gent)

    """

    initN_df=pd.read_csv(os.path.join(data_path, 'covid19_DTM/interim/demographic/initN_' + agg + '.csv'), index_col=[0])
    return list(initN_df.index.values)

def construct_coordinates_Nc(age_stratification_size=10):
    """
    A function to return a list with the labels of the age groups used in the model

    Parameter
    ---------
    N: int
        Number of age groups (3, 9, 10 or 18)
    
    Returns
    -------
    coordinates: list
        List containing labels of age groups
    

    """
    if age_stratification_size not in [3, 9, 10, 18]:
        raise Exception(f"Age stratification {age_stratification_size} is not allowed. Choose between either 3, 9, 10 (default) or 18.")

    if age_stratification_size == 3:
        return pd.IntervalIndex.from_tuples([(0, 20), (20, 60), (60, 120)], closed='left')
    elif age_stratification_size == 9:
        return pd.IntervalIndex.from_tuples([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 120)], closed='left')
    elif age_stratification_size == 10:
        return pd.IntervalIndex.from_tuples([(0, 12), (12, 18), (18, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75), (75, 85), (85,120)], closed='left')
    elif age_stratification_size == 18:
        return pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,85),(85,120)], closed='left')

def read_areas(agg='arr'):
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

    areas_df = pd.read_csv(os.path.join(data_path, 'covid19_DTM/interim/demographic/area_' + agg + '.csv'), index_col='NIS')
    areas = areas_df['area'].to_dict()

    return areas

def read_pops(agg='arr',age_stratification_size=10,return_matrix=False,drop_total=False):
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
    
    pops_df = pd.read_csv(os.path.join(data_path, 'covid19_DTM/interim/demographic/initN_' + agg + '.csv'), index_col='NIS')
    if drop_total:
        pops_df.drop(columns='total', inplace=True)
    if return_matrix:
        pops = pops_df.values
    else:
        pops = pops_df.T.to_dict()

    return pops



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
    
    from covid19_DTM.data.model_parameters import construct_initN
    
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
        from covid19_DTM.data.sciensano import get_sciensano_COVID19_data_spatial
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
    A function to extract from /data/covid19_DTM/interim/demographic/initN_arrond.csv the list of arrondissement NIS codes

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

    initN_df=pd.read_csv(os.path.join(data_path, 'covid19_DTM/interim/demographic/initN_' + spatial + '.csv'), index_col=[0])
    NIS = initN_df.index.values

    return NIS

import datetime
from dateutil.easter import easter

def is_Belgian_primary_secundary_school_holiday(d):
    """
    A function returning 'True' if a given date is a school holiday or primary and secundary schools in Belgium
    
    Input
    -----
    
    d: datetime.datetime
        Current simulation date
    
    Returns
    -------
    
    is_Belgian_primary_secundary_school_holiday: bool
        True: date `d` is a school holiday for primary and secundary schools
    """
    
    # Pre-allocate a vector containing the year's holiday weeks
    holiday_weeks = []
    
    # Herfstvakantie
    holiday_weeks.append(44)
    
    # Extract date of easter
    d_easter = easter(d.year)
    # Convert from datetime.date to datetime.datetime
    d_easter = datetime(d_easter.year, d_easter.month, d_easter.day)
    # Get week of easter
    w_easter = d_easter.isocalendar().week

    # Default logic: Easter holiday starts first monday of April
    # Unless: Easter falls after 04-15: Easter holiday ends with Easter
    # Unless: Easter falls in March: Easter holiday starts with Easter
    if d_easter >= datetime(year=d.year,month=4,day=15):
        w_easter_holiday = w_easter - 1
    elif d_easter.month == 3:
        w_easter_holiday = w_easter + 1
    else:
        w_easter_holiday = datetime.date(d.year, 4, (8 - datetime.date(d.year, 4, 1).weekday()) % 7).isocalendar().week
    holiday_weeks.append(w_easter_holiday)
    holiday_weeks.append(w_easter_holiday+1)

    # Krokusvakantie
    holiday_weeks.append(w_easter-6)

    # Extract week of Christmas
    # If Christmas falls on Saturday or Sunday, Christams holiday starts week after
    w_christmas_current = datetime(year=d.year,month=12,day=25).isocalendar().week
    if datetime(year=d.year,month=12,day=25).isoweekday() in [6,7]:
        w_christmas_current += 1
    w_christmas_previous = datetime(year=d.year-1,month=12,day=25).isocalendar().week
    if datetime(year=d.year-1,month=12,day=25).isoweekday() in [6,7]:
        w_christmas_previous += 1
    # Christmas logic
    if w_christmas_previous == 52:
        if datetime(year=d.year-1, month=12, day=31).isocalendar().week != 53:
            holiday_weeks.append(1)   
    if w_christmas_current == 51:
        holiday_weeks.append(w_christmas_current)
        holiday_weeks.append(w_christmas_current+1)
    if w_christmas_current == 52:
        holiday_weeks.append(w_christmas_current)
        if datetime(year=d.year, month=12, day=31).isocalendar().week == 53:
            holiday_weeks.append(w_christmas_current+1)

    # Define Belgian Public holidays
    public_holidays = [
        datetime(year=d.year, month=1, day=1),       # New Year
        d_easter + timedelta(days=1),                # Easter monday
        datetime(year=d.year, month=5, day=1),       # Labor day
        d_easter + timedelta(days=40),               # Acension Day
        datetime(year=d.year, month=7, day=21),      # National holiday
        datetime(year=d.year, month=8, day=15),      # Assumption Mary
        datetime(year=d.year, month=11, day=1),      # All Saints
        datetime(year=d.year, month=11, day=11),     # Armistice
        datetime(year=d.year, month=12, day=25),     # Christmas
    ]
    
    # Logic
    if ((d.isocalendar().week in holiday_weeks) | \
            (d in public_holidays)) | \
                ((datetime(year=d.year, month=7, day=1) <= d < datetime(year=d.year, month=9, day=1))):
        return True
    else:
        return False