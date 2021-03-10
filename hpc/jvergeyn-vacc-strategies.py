 
# *Authored by J. Vergeynst*
# 
# Copyright (c) 2021 by J. Vergeynst, BIOMATH, Ghent University. All Rights Reserved.


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime
import scipy
import json
import random
from math import floor, ceil

from covid19model.optimization import objective_fcns
from covid19model.models import models
from covid19model.models.utils import draw_sample_COVID19_SEIRD
from covid19model.models.time_dependant_parameter_fncs import ramp_fun, google_lockdown
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.visualization.output import population_status, infected, _apply_tick_locator
from covid19model.visualization.optimization import plot_fit, traceplot


# # Data, policy and model initialization

# ## Load contact data

# Load the interaction matrices (size: 9x9)
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012',intensity='all')
# Define the number of age categories
levels = initN.size

initN, Nc_home_15, Nc_work_15, Nc_schools_15, Nc_transport_15, Nc_leisure_15, Nc_others_15, Nc_total_15 = model_parameters.get_interaction_matrices(dataset='willem_2012',intensity='more_15_min')
initN, Nc_home_1hr, Nc_work_1hr, Nc_schools_1hr, Nc_transport_1hr, Nc_leisure_1hr, Nc_others_1hr, Nc_total_1hr = model_parameters.get_interaction_matrices(dataset='willem_2012',intensity='more_one_hour')

Nc_all = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}
Nc_15min = {'total': Nc_total_15, 'home': Nc_home_15, 'work': Nc_work_15, 'schools': Nc_schools_15, 'transport': Nc_transport_15, 'leisure': Nc_leisure_15, 'others': Nc_others_15}
Nc_1hr = {'total': Nc_total_1hr, 'home': Nc_home_1hr, 'work': Nc_work_1hr, 'schools': Nc_schools_1hr, 'transport': Nc_transport_1hr, 'leisure': Nc_leisure_1hr, 'others': Nc_others_1hr}


# ## Load publically available data from Sciensano

df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
df_sciensano['D_cum'] = df_sciensano.D_tot.cumsum()
df_google = mobility.get_google_mobility_data(update=False, plot=False)


# ## Load posterior parameter distributions of calibrated parameters

# Load samples dictionaries
with open('../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_4_prev_full_2021-01-30_WAVE2_GOOGLE.json', 'r') as fp:
    samples_dict = json.load(fp)

with open('../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states_sept = json.load(fp) 
    
# Set initial states of VE (vaccine elegible)
initial_states_sept['VE'] = (np.array(initial_states_sept['S'])+
                             np.array(initial_states_sept['R'])+
                             np.array(initial_states_sept['E'])+
                             np.array(initial_states_sept['I'])+
                             np.array(initial_states_sept['A'])).squeeze()

## Scenarios

start_sim = '2020-09-01'
end_sim = '2021-09-01'
start_fig = '2021-01-01'
warmup = 0
n_samples = 100
n_draws_per_sample = 1000

Re_1feb = 0.958*1.4
# current estimation of UK prevalence between 20 and 40% => take 30%
current_UK = 0.3
# 5-6 days incubation
incubation_period = 5
n_periods = 30/incubation_period
# How much on January 1? (30 days ago)
portion_new_strain_introduced = current_UK/(Re_1feb**n_periods) #0.001
injection_day = (pd.Timestamp('2021-01-01') - pd.Timestamp(start_sim))/pd.Timedelta('1D')

UL = 0.975
LL = 0.025

# doses per day (Regeringscommissariaat Corona, published by De Tijd)
d = {}
d['jan'] = 31765
d['feb'] = 45897
d['mar-apr'] = 128499
d['may-aug'] = 78358

# proportion of age group living in nursing homes
# for simplicity we will use the data for respectively age groups 60+, 70+ and 80+
NH = {}
NH['65+'] = 0.007
NH['75+'] = (0.05+0.029)/2
NH['85+'] = (0.137+0.267)/2

## Helpfunction vaccination

def smoothclamp(x, mi, mx): 
    return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )

def smooth_dose(dose, sense, start_dose=0, end_dose=0, N=1000):
    if sense == 'up':
        x = np.linspace(start_dose, dose, num=N)
        return smoothclamp(x, start_dose, dose)
    elif sense == 'down':
        x = np.linspace(dose, end_dose, num=N)
        return smoothclamp(x, dose, end_dose)
    else:
        raise ValueError(
                "sense must be 'up' or 'down' "
            )

def smooth_dose_ts(daily_dose, start_t, end_t, spread, start_dose=0, end_dose=0, N=1000):
    """
    max_to_vacc : max number to vaccinate
    daily_dose : daily number of vaccinations
    spread : number of days at start and end to spread out step function
    N : resolution of the smoothed step function
    start_dose : level to start smoothing from (0 or previous daily dose)
    
    """
    constant_period = np.linspace(start_t+spread, end_t-spread, num=ceil(end_t))
    
    ts = np.concatenate((np.linspace(start_t, start_t+spread, num=N), 
                         constant_period,
                         np.linspace(end_t-spread, end_t, num=N)  
                         )) 
    
    doses = np.concatenate((smooth_dose(daily_dose, 'up', start_dose=start_dose, N=N),
                    np.repeat(np.array(daily_dose), len(constant_period)),
                    smooth_dose(daily_dose, 'down', end_dose=end_dose, N=N)))
    return ts, doses

def smooth_dose_ts_open_end(daily_dose, start_t, spread, start_dose=0, N_days=100, N=1000):
    """
    max_to_vacc : max number to vaccinate
    daily_dose : daily number of vaccinations
    N_days : number of days to spread out open end
    spread : number of days at start/end to spread out step function
    N : resolution of the smoothed step function
    
    """
    constant_period = np.linspace(start_t+spread, 100, num=N_days)
    
    ts = np.concatenate((np.linspace(start_t, start_t+spread, num=N), 
                         constant_period
                         )) 
    
    doses = np.concatenate((smooth_dose(daily_dose, 'up', start_dose=start_dose, N=N),
                            np.repeat(np.array(daily_dose), len(constant_period))
                            ))
    return ts, doses

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def calc_N_vacc(N_vacc, age, prev_age, VE_states, dose, leftover, flag):
    if (VE_states[:,[age]] > leftover) and flag:
        dose = dose-N_vacc[:,[prev_age]]
        N_vacc[:,[age]] = dose 
        flag=False
        if (VE_states[:,[age]]-leftover) <= dose:
            N_vacc[:,[age]] = VE_states[:,[age]]-leftover 
            flag=True
    return N_vacc[:,[age]], flag

## Vaccination functions

NH_8_to_vacc = NH['85+']*initial_states_sept['VE'][8]
NH_7_to_vacc = NH['75+']*initial_states_sept['VE'][7]
NH_6_to_vacc = NH['65+']*initial_states_sept['VE'][6]

def vacc_strategy(t, states, param, d, NH, order, elder):
    """
    time-dependent function for vaccination strategy
    
    states : dictionary
        model states, VE = vaccine eligible states = S+R+E+I+A
    d : dictionary
        daily number of doses for that month
        daily vaccinated persons on immunity date = daily dose on vaccination date / 2
    NH : dictionary
        proportion of residents in nursing homes per age group
    
    """
    
    leftover = 100
    
    VE_states = states['VE'].copy()
    
    if VE_states.ndim == 1:
        VE_states = VE_states[np.newaxis]

    N_vacc = np.zeros(VE_states.shape)

    delay = pd.Timedelta('30D')
    
    t1 = pd.Timestamp('2021-01-01') + delay
    t2 = pd.Timestamp('2021-02-01') + delay
    t3 = pd.Timestamp('2021-03-01') + delay
    t4 = pd.Timestamp('2021-05-01') + delay
    daily_dose_jan = d['jan']/2
    daily_dose_feb = d['feb']/2
    daily_dose_ma_ap = d['mar-apr']/2
    daily_dose_may_aug = d['may-aug']/2
    
    if t < t1:
        N_vacc = N_vacc
    
    elif t1 <= t < t2: # January : nursing homes + part of care personnel
        t_num = (t-t1)/pd.Timedelta('1D')
        end_phase1 = NH_8_to_vacc/daily_dose_jan
        end_phase2 = end_phase1 + NH_7_to_vacc/daily_dose_jan
        end_phase3 = end_phase2 + NH_6_to_vacc/daily_dose_jan
        end_t2 = (t2-t1)/pd.Timedelta('1D')
        if t_num < end_phase1:
            ts, doses = smooth_dose_ts(daily_dose_jan, 0, end_phase1, spread=2)
            N_vacc[:,[8]] = doses[find_nearest_idx(ts, t_num)]
        elif t_num < end_phase2:
            ts, doses = smooth_dose_ts(daily_dose_jan, end_phase1, end_phase2, spread=1)
            N_vacc[:,[7]] = doses[find_nearest_idx(ts, t_num)]
        elif t_num < end_phase3:
            ts, doses = smooth_dose_ts(daily_dose_jan, end_phase2, end_phase3, spread=0.3)
            N_vacc[:,[6]] = doses[find_nearest_idx(ts, t_num)]
        else: 
            ts, doses = smooth_dose_ts(daily_dose_jan, end_phase3, end_t2, spread=4, end_dose=daily_dose_feb)
            N_vacc[:,[5,4,3,2]] = doses[find_nearest_idx(ts, t_num)]/4

    elif t2 <= t < t3: # February : care personnel
        t_num = (t-t2)/pd.Timedelta('1D')
        end_t3 = (t3-t2)/pd.Timedelta('1D')
        ts, doses = smooth_dose_ts(daily_dose_feb, 0, end_t3, spread=4, start_dose=daily_dose_jan)
        N_vacc[:,[5,4,3,2]] = doses[find_nearest_idx(ts, t_num)]/4
        
    elif t3 <= t < t4: # March-April : 65+ and risk patients
        t_num = (t-t3)/pd.Timedelta('1D')
        if VE_states[:,[elder[0]]] > leftover:
            ts, doses = smooth_dose_ts_open_end(daily_dose_ma_ap, 0, spread=2)
            N_vacc[:,[elder[0]]] = doses[find_nearest_idx(ts, t_num)]
            ## One daily dose before all 80+ are vaccinated, start building down
            if (VE_states[:,[elder[0]]]-leftover) <= daily_dose_ma_ap:
                N_vacc[:,[elder[0]]] = VE_states[:,[elder[0]]]-leftover
                
                flag = True
                N_vacc[:,[elder[1]]],flag = calc_N_vacc(N_vacc, elder[1], elder[0], VE_states, daily_dose_ma_ap, leftover,flag)
                N_vacc[:,[elder[2]]],flag = calc_N_vacc(N_vacc, elder[2], elder[1], VE_states, daily_dose_ma_ap, leftover,flag)
                if flag==True and len(np.unique(elder))==3:
                    N_vacc[:,[5,4,3,2]] = (daily_dose_ma_ap - N_vacc[:,[elder[2]]])/4
                if flag==True and len(np.unique(elder))==2:
                    N_vacc[:,[6,5,4,3,2]] = (daily_dose_ma_ap - N_vacc[:,[elder[2]]])/5
                                

    else: # May-August : all 18+ 
        t_num = (t-t4)/pd.Timedelta('1D')
        ts, doses = smooth_dose_ts_open_end(daily_dose_may_aug, 0, spread=3)#, start_dose=daily_dose_ma_ap/4
        if len(np.unique(elder))==3:
            N_vacc[:,[5,4,3,2]] = doses[find_nearest_idx(ts, t_num)]/4
        elif len(np.unique(elder))==2:
            N_vacc[:,[6,5,4,3,2]] = doses[find_nearest_idx(ts, t_num)]/5
    return N_vacc.squeeze()


def vacc_strategy_priors(t, states, param, d, NH, order, elder):
    """
    time-dependent function for vaccination strategy
    
    states : dictionary
        model states, VE = vaccine eligible states = S+R+E+I+A
    d : dictionary
        daily number of doses for that month
        daily vaccinated persons on immunity date = daily dose on vaccination date / 2
    NH : dictionary
        proportion of residents in nursing homes per age group
    order : list
        order of age categories to be vaccinated after the elderly
    elder : list
        order of elderly age categories ([8,7,6] or if 60+ not prioritised [8,7,7])
    
    """
    if len(order)!=7:
        raise ValueError(
                "order must have length 7"
            )
    if len(elder)!=3:
        raise ValueError(
                "elder must have length 3"
            )        
    
    leftover = 100
    
    VE_states = states['VE'].copy()
    if VE_states.ndim == 1:
        VE_states = VE_states[np.newaxis]

    N_vacc = np.zeros(VE_states.shape)

    delay = pd.Timedelta('30D')
    
    t1 = pd.Timestamp('2021-01-01') + delay
    t2 = pd.Timestamp('2021-02-01') + delay
    t3 = pd.Timestamp('2021-03-01') + delay
    t4 = pd.Timestamp('2021-05-01') + delay
    daily_dose_jan = d['jan']/2
    daily_dose_feb = d['feb']/2
    daily_dose_ma_ap = d['mar-apr']/2
    daily_dose_may_aug = d['may-aug']/2
    
    if t < t1:
        N_vacc = N_vacc
    
    elif t1 <= t < t2: # January : nursing homes + part of care personnel
        t_num = (t-t1)/pd.Timedelta('1D')
        end_phase1 = NH_8_to_vacc/daily_dose_jan
        end_phase2 = end_phase1 + NH_7_to_vacc/daily_dose_jan
        end_phase3 = end_phase2 + NH_6_to_vacc/daily_dose_jan
        end_t2 = (t2-t1)/pd.Timedelta('1D')
        if t_num < end_phase1:
            ts, doses = smooth_dose_ts(daily_dose_jan, 0, end_phase1, spread=2)
            N_vacc[:,[8]] = doses[find_nearest_idx(ts, t_num)]
        elif t_num < end_phase2:
            ts, doses = smooth_dose_ts(daily_dose_jan, end_phase1, end_phase2, spread=1)
            N_vacc[:,[7]] = doses[find_nearest_idx(ts, t_num)]
        elif t_num < end_phase3:
            ts, doses = smooth_dose_ts(daily_dose_jan, end_phase2, end_phase3, spread=0.3)
            N_vacc[:,[6]] = doses[find_nearest_idx(ts, t_num)]
        else: 
            ts, doses = smooth_dose_ts(daily_dose_jan, end_phase3, end_t2, spread=4, end_dose=daily_dose_feb)
            N_vacc[:,[5,4,3,2]] = doses[find_nearest_idx(ts, t_num)]/4

    elif t2 <= t < t3: # February : care personnel
        t_num = (t-t2)/pd.Timedelta('1D')
        end_t3 = (t3-t2)/pd.Timedelta('1D')
        ts, doses = smooth_dose_ts(daily_dose_feb, 0, end_t3, spread=4, start_dose=daily_dose_jan)
        N_vacc[:,[5,4,3,2]] = doses[find_nearest_idx(ts, t_num)]/4
         
    elif t3 <= t < t4: # March-April : 65+ and risk patients
        t_num = (t-t3)/pd.Timedelta('1D')
        if VE_states[:,[elder[0]]] > leftover:
            ts, doses = smooth_dose_ts_open_end(daily_dose_ma_ap, 0, spread=2)
            N_vacc[:,[elder[0]]] = doses[find_nearest_idx(ts, t_num)]
            ## One daily dose before all 80+ are vaccinated, start building down
            if (VE_states[:,[elder[0]]]-leftover) <= daily_dose_ma_ap:
                N_vacc[:,[elder[0]]] = VE_states[:,[elder[0]]]-leftover
                
                flag = True
                N_vacc[:,[elder[1]]],flag = calc_N_vacc(N_vacc, elder[1], elder[0], VE_states, daily_dose_ma_ap, leftover,flag)
                N_vacc[:,[elder[2]]],flag = calc_N_vacc(N_vacc, elder[2], elder[1], VE_states, daily_dose_ma_ap, leftover,flag)
                N_vacc[:,[order[0]]],flag = calc_N_vacc(N_vacc, order[0], elder[2], VE_states, daily_dose_ma_ap, leftover,flag)
                N_vacc[:,[order[1]]],flag = calc_N_vacc(N_vacc, order[1], order[0], VE_states, daily_dose_ma_ap, leftover,flag)
                N_vacc[:,[order[2]]],flag = calc_N_vacc(N_vacc, order[2], order[1], VE_states, daily_dose_ma_ap, leftover,flag)          
                
    else: # May-August : all 18+
        
        t_num = (t-t4)/pd.Timedelta('1D')
        if VE_states[:,[order[0]]] > leftover:
            ts, doses = smooth_dose_ts_open_end(daily_dose_may_aug, 0, spread=2)
            N_vacc[:,[order[0]]] = doses[find_nearest_idx(ts, t_num)]
            ## One daily dose before all 80+ are vaccinated, start building down
            if (VE_states[:,[order[0]]]-leftover) <= daily_dose_may_aug:
                N_vacc[:,[order[0]]] = VE_states[:,[order[0]]]-leftover  
                
                flag = True
                N_vacc[:,[order[1]]],flag = calc_N_vacc(N_vacc, order[1], order[0], VE_states, daily_dose_may_aug, leftover,flag)
                N_vacc[:,[order[2]]],flag = calc_N_vacc(N_vacc, order[2], order[1], VE_states, daily_dose_may_aug, leftover,flag)
                N_vacc[:,[order[3]]],flag = calc_N_vacc(N_vacc, order[3], order[2], VE_states, daily_dose_may_aug, leftover,flag)
                N_vacc[:,[order[4]]],flag = calc_N_vacc(N_vacc, order[4], order[3], VE_states, daily_dose_may_aug, leftover,flag)
                N_vacc[:,[order[5]]],flag = calc_N_vacc(N_vacc, order[5], order[4], VE_states, daily_dose_may_aug, leftover,flag)
                N_vacc[:,[order[6]]],flag = calc_N_vacc(N_vacc, order[6], order[5], VE_states, daily_dose_may_aug, leftover,flag)

                
    return N_vacc.squeeze()

## Functions

def sample_from_binomial(sim_result, variable, n_draws_per_sample, n_samples,
                         Y0_new=None):
                         #Y0_mean=[], Y0_median=[], Y0_LL=[], Y0_UL=[]):
    """
    Function to sample from binomial, and add the result to an existing list (if given)
    """
    
    
    Y = sim_result[variable].sum(dim="Nc").values
    # Initialize vectors
    Y_new = np.zeros((Y.shape[1],n_draws_per_sample*n_samples))
    # Loop over dimension draws
    for n in range(Y.shape[0]):
        binomial_draw = np.random.poisson( np.expand_dims(Y[n,:],axis=1),size = (Y.shape[1],n_draws_per_sample))
        Y_new[:,n*n_draws_per_sample:(n+1)*n_draws_per_sample] = binomial_draw
    # Compute mean and median
    if Y0_new is None:
        Y_new = Y_new
    else:
        Y_new = np.append(Y0_new, Y_new, axis=0)

    return Y_new
    
    
def draw_fcn(param_dict,samples_dict):
    # Sample
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['l'] = samples_dict['l'][idx] 
    param_dict['tau'] = samples_dict['tau'][idx]    
    param_dict['prev_home'] = samples_dict['prev_home'][idx] 
    param_dict['prev_schools'] = samples_dict['prev_schools'][idx]    
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx] 
    #param_dict['K'] = np.random.uniform(infectivity_gain_lower,infectivity_gain_upper)
    return param_dict

# # Time-dep functions

from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

def report7_policy_function(t, states, param, l , tau, prev_home, prev_schools, prev_work, prev_rest,scenario='1a'):
    """
    Scenario's:
        '1' : current behaviour
        '2' : opening March 1
        '3' : opening April 1
        '4' : opening May 1
    """
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Define key policy dates
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer: COVID-urgency very low
    t4 = pd.Timestamp('2020-08-01')
    t5 = pd.Timestamp('2020-09-01') # september: lockdown relaxation narrative in newspapers reduces sense of urgency
    t6 = pd.Timestamp('2020-10-19') # lockdown
    t7 = pd.Timestamp('2020-11-16') # schools re-open
    t8 = pd.Timestamp('2020-12-18') # start Christmas holidays (schools close)
    t13 = pd.Timestamp('2020-01-04') # end Christmas holidays (schools open)
    #t14 = pd.Timestamp('2021-01-18') # start of alternative policies
    t15 = pd.Timestamp('2021-02-15') # Start of Krokus holidays (schools close)
    t16 = pd.Timestamp('2021-02-21') # End of Krokus holidays (schools open)
    t17 = pd.Timestamp('2021-03-01') # release to SB March 1
    t18 = pd.Timestamp('2021-04-01') # release to SB April 1
    t19 = pd.Timestamp('2021-04-05') # start Eastern Holidays (schools close)
    t20 = pd.Timestamp('2021-04-19') # end Eastern Holidays (schools open)
    t21 = pd.Timestamp('2021-05-01') # release to SB May 1

    # Average out september mobility

    if t5 < t <= t6 + tau_days:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, school=1)
    elif t6 + tau_days < t <= t6 + tau_days + l_days:
        t = pd.Timestamp(t.date())
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t6)
    elif t6 + tau_days + l_days < t <= t7:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t7 < t <= t8:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t8 < t <= t13:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t13 < t <= t15:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                        school=0.6)
    elif t15 < t <= t16:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                        school=0)    
    else:
        # Scenario 1: Current contact behaviour
        if scenario == '1':
            if t16 < t <= t19:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0.6)
            elif t19 < t <= t20:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)
            elif t20 < t:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0.6)
            else:
                raise Exception ('scenario '+scenario+' t:'+str(t))
                    
        # Scenario 2: increases in work or leisure mobility on March 1
        elif scenario == '2':
            if t16 < t <= t17:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0.6)  
            elif t17 < t <= t19:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, 1, 1, 1, 1, 
                                school=0.6,SB='2a')
            elif t19 < t <= t20:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, 1, 1, 1, 1, 
                                school=0,SB='2a')
            elif t20 < t:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, 1, 1, 1, 1, 
                                school=0.6,SB='2a')
            else:
                raise Exception ('scenario '+scenario+' t:'+str(t))
        # Scenario 3: increases in work or leisure mobility on April 1
        elif scenario == '3':
            if t16 < t <= t18:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0.6)  
            elif t18 < t <= t19:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, 1, 1, 1, 1, 
                                school=0.6,SB='2a')
            elif t19 < t <= t20:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, 1, 1, 1, 1, 
                                school=0,SB='2a')
            elif t20 < t:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, 1, 1, 1, 1, 
                                school=0.6,SB='2a')
            
            else:
                raise Exception ('scenario '+scenario+' t:'+str(t))
        # Scenario 4: increases in work or leisure mobility on May 1
        elif scenario == '4':
            
            if t16 < t <= t19:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0.6)  
            elif t19 < t <= t20:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)  
            elif t20 < t <= t21:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0.6) 
            elif t > t21:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, 1, 1, 1, 1, 
                                school=0.6,SB='2a')
            else:
                raise Exception ('scenario '+scenario+' t:'+str(t))
        else:
            raise Exception ('scenario '+scenario+' non-existing')


def vaccin_model(initial_states, scenario, order=None, elder=None, effectivity=None, injection_day=0, injection_ratio=0,
                 Nc_fun=None, N_vacc_fun=vacc_strategy, levels=levels):
    """
    Function to initialize the model given a certain vaccination strategy
    """
    params = model_parameters.get_COVID19_SEIRD_parameters()
    params.update({
            'l' : 5,
            'tau' : 5,
            'prev_home': 0.5,
            'prev_schools': 0.5,
            'prev_work': 0.5,
            'prev_rest': 0.5,
            'scenario': scenario,
            'injection_day' : injection_day,
            'injection_ratio' : injection_ratio
                  })
    
    tdp = {}
    
    if Nc_fun is not None:
        tdp.update({'Nc': Nc_fun})
    if N_vacc_fun is not None:
        tdp.update({'N_vacc':N_vacc_fun})
        params.update({
            'd' : d,
            'NH' : NH,
            'e' : np.array([effectivity]*levels),
            'order': order,
            'elder': elder
        })
    return models.COVID19_SEIRD(initial_states, params, time_dependent_parameters=tdp)

# ## Run and save all scenarios

scenario_settings = pd.DataFrame({
    'Scenario_name':['S1a','S1b','S1c','S1d','S1e',
                     'S2a','S2b','S2c','S2d','S2e',
                     'S3a','S3b','S3c','S3d','S3e',
                     'S4a','S4b','S4c','S4d','S4e'],
    'opening':(5*['3']+5*['4']+5*['3']+5*['4']),
    'effectivity':20*[0.7],
    'K':20*[1.5],
    'order':2*[None,[5,4,3,2,2,2,2],[2,3,4,5,5,5,5],[3,4,2,5,5,5,5],[0,1,2,3,4,5,5]]+2*[None,[6,5,4,3,2,2,2],[2,3,4,5,6,6,6],[3,4,2,5,6,6,6],[0,1,2,3,4,5,6]],
    'elder': 10*[[8,7,6]]+10*[[8,7,7]],
    'vacc_fun':4*([vacc_strategy]+4*[vacc_strategy_priors])
})

scenario_settings = scenario_settings.set_index('Scenario_name')

#import pickle

results = pd.DataFrame()
for scen in scenario_settings.index:
    print(scen)
    scenario = scenario_settings.loc[scen,'opening']
    effectivity = scenario_settings.loc[scen,'effectivity']
    K = scenario_settings.loc[scen,'K']
    vacc_fun = scenario_settings.loc[scen,'vacc_fun']
    order = scenario_settings.loc[scen,'order']
    elder = scenario_settings.loc[scen,'elder']

    scenario_model = vaccin_model(initial_states_sept, scenario=scenario, order=order, elder=elder, effectivity=effectivity, 
                             injection_day=injection_day, injection_ratio=portion_new_strain_introduced,
                             Nc_fun=report7_policy_function, N_vacc_fun=vacc_fun)
    scenario_model.parameters.update({'K':K})
    out = scenario_model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict,verbose=True)
    #with open('../results/temp/'+scen+'_out.pkl', 'wb') as handle: pickle.dump(out, handle)

    H_in_binom = sample_from_binomial(out, 'H_in', n_draws_per_sample, n_samples)
    H_tot_binom = sample_from_binomial(out, 'H_tot', n_draws_per_sample, n_samples)
    time = out["time"].values
    
    results['Date'] =  time
    results[scen+'_incidences_mean'] = np.mean(H_in_binom, axis=1)
    results[scen+'_incidences_median'] = np.median(H_in_binom, axis=1)
    results[scen+'_incidences_LL'] = np.quantile(H_in_binom, q = LL, axis = 1)
    results[scen+'_incidences_UL'] = np.quantile(H_in_binom, q = UL, axis = 1)
    results[scen+'_load_mean'] = np.mean(H_tot_binom, axis=1)
    results[scen+'_load_median'] = np.median(H_tot_binom, axis=1)
    results[scen+'_load_LL'] = np.quantile(H_tot_binom, q = LL, axis = 1)
    results[scen+'_load_UL'] = np.quantile(H_tot_binom, q = UL, axis = 1)

    # Figures to check vaccination strategy

    out_single = out.sel(draws=1)
    dates = pd.date_range(start_fig, end_sim, periods=5)

    plt.ioff()

    fig,ax = plt.subplots(figsize=(9,15), nrows=9, sharex=True)
    for i in range(9):
        ax[i].plot(out_single['time'], out_single['V_new'].sel(Nc=i), c='green', linestyle='--', label='model')
        ax[i].set_ylabel('Age '+str(i))
    ax[0].set_title('Daily vaccination number')
    ax[0].set_xticks(dates)
    ax[0].set_xlim(start_fig, end_sim)
    fig_path =  '../results/predictions/national/vaccination_study_2/'
    fig.savefig(fig_path+scen+'_daily_vacc'+'.jpg', dpi=400, bbox_inches='tight')

    fig,ax = plt.subplots(figsize=(12,15), nrows=9, sharex=True)
    for i in range(9):
        ax[i].plot(out_single['time'], out_single['VE'].sel(Nc=i), c='green', linestyle='--', label='model')
        ax[i].set_ylabel('Age '+str(i))
    ax[0].set_title('VE state')
    ax[0].set_xticks(dates)
    ax[0].set_xlim(start_fig, end_sim)
    fig_path =  '../results/predictions/national/vaccination_study_2/'
    fig.savefig(fig_path+scen+'_VE_state'+'.jpg', dpi=400, bbox_inches='tight')

results.to_csv('../results/predictions/national/vaccination_study_2/simulations.csv')
