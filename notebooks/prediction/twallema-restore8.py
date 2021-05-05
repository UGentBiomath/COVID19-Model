"""
This script contains code to simulate the scenarios for RESTORE report 8.
Deterministic, national-level BIOMATH COVID-19 SEIRD

Example use:
------------
python twallema-restore8.py -f BE_WAVE2_R0_COMP_EFF_2021-04-28.json -s 0 1 2 3 4 5 6 -n 100

    Runs all 7 social scenarios with 100 simulations per scenario.

"""

__author__      = "Tijs Alleman and Jenna Vergeynst"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import gc
import sys, getopt
import ujson as json
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from covid19model.models import models
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-s", "--scenarios", help="Scenarios to be simulated", nargs='+')
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
args = parser.parse_args()

###########################
### Simulation control  ###
###########################

import sys, getopt
scenarios = args.scenarios
report_version = '8.0'
start_sim = '2020-09-01'
end_sim = '2021-09-01'
start_calibration = start_sim
end_calibration = '2021-04-21'
model = 'BIOMATH COVID-19 SEIRD national'
n_samples = args.n_samples
n_draws = args.n_draws_per_sample
warmup = 0
conf_int = 0.05

# Scenario settings
descriptions_scenarios = ['Current contact behaviour', 'Relaxation of work-at-home - schools open', 'Relaxation of work-at-home - schools closed',
                    'Relaxation of leisure - schools open', 'Relaxation of leisure - schools closed',
                    'Relaxation of work-at-home and leisure - schools open', 'Relaxation of work-at-home and leisure - schools closed']
relaxdates = ['2021-05-08','2021-06-01']
doses = [60000,80000]
orders = [np.array(range(9))[::-1]]#[np.array(range(9)), np.array(range(9))[::-1]]
description_order = ['old --> young']#['young (0 yo.) --> old', 'old --> young'] # Add contact order, and/or add young to old, starting at 20 yo.

# Upper- and lower confidence level
UL = 1-conf_int/2
LL = conf_int/2

print('\n##################################')
print('### RESTORE SIMULATION SUMMARY ###')
print('##################################\n')

# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False, plot=False)

print('report: v' + report_version)
print('scenarios: '+ ', '.join(map(str, scenarios)))
print('model: ' + model)
print('number of samples: ' + str(n_samples))
print('confidence level: ' + str(conf_int*100) +' %')
print('start of simulation: ' + start_sim)
print('end of simulation: ' + end_sim)
print('last hospitalization datapoint: '+str(df_sciensano.index[-1]))
print('last vaccination datapoint: '+str(df_sciensano.index[-1]))
print('last mobility datapoint: '+str(df_google.index[-1]))
print('simulation date: '+ str(datetime.date.today())+'\n')

print('###############')
print('### WORKING ###')
print('###############\n')

print('1) Loading data\n')

# -------------------
# Load remaining data
# -------------------

# Time-integrated contact matrices
initN, Nc_all = model_parameters.get_integrated_willem2012_interaction_matrices()
levels = initN.size
# Model initial condition on September 1st
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)  

# -----------------------
# Load samples dictionary
# -----------------------

# Path where MCMC samples are saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'

from covid19model.models.utils import load_samples_dict
samples_dict = load_samples_dict(samples_path+str(args.filename), wave=2)
warmup = int(samples_dict['warmup'])

print('2) Initializing model\n')

# ---------------------------
# Time-dependant VOC function
# ---------------------------

from covid19model.models.time_dependant_parameter_fncs import make_VOC_function
VOC_function = make_VOC_function()

def VOC_wrapper_func(t,states,param):
    t = pd.Timestamp(t.date())
    return VOC_function(t)

# -----------------------------------
# Time-dependant vaccination function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import  make_vaccination_function
sciensano_first_dose, df_sciensano_start, df_sciensano_end = make_vaccination_function(df_sciensano)

def vacc_strategy(t, states, param, df_sciensano_start, df_sciensano_end,
                    daily_dose=30000, delay = 21, vacc_order = [8,7,6,5,4,3,2,1,0], refusal = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]):
    """
    time-dependent function for the Belgian vaccination strategy
    First, all available data from Sciensano are used. Then, the user can specify a custom vaccination strategy of "daily_dose" doses per day,
    given in the order specified by the vector "vacc_order" with a refusal propensity of "refusal" in every age group.
  
    Parameters
    ----------
    t : int
        Simulation time
    states: dict
        Dictionary containing values of model states
    param : dict
        Model parameter dictionary
    sciensano_first_dose : function
        Function returning the number of (first dose) vaccinated individuals at simulation time t, according to the data made public by Sciensano.
    df_sciensano_start : date
        Start date of Sciensano vaccination data frame
    df_sciensano_end : date
        End date of Sciensano vaccination data frame
    daily_dose : int
        Number of doses administered per day. Default is 30000 doses/day.
    delay : int
        Time delay between first dose vaccination and start of immunity. Default is 21 days.
    vacc_order : array
        Vector containing vaccination prioritization preference. Default is old to young. Must be equal in length to the number of age bins in the model.
    refusal: array
        Vector containing the fraction of individuals refusing a vaccine per age group. Default is 30% in every age group. Must be equal in length to the number of age bins in the model.

    Return
    ------
    N_vacc : array
        Number of individuals to be vaccinated at simulation time "t"
        
    """

    # Convert time to suitable format
    t = pd.Timestamp(t.date())
    # Convert delay to a timedelta
    delay = pd.Timedelta(str(int(delay))+'D')
    # Compute the number of vaccine eligible individuals
    VE = states['S'] + states['R']
    
    if t <= df_sciensano_start + delay:
        return np.zeros(9)
    elif df_sciensano_start + delay < t <= df_sciensano_end + delay:
        return sciensano_first_dose(t-delay)
    else:
        N_vacc = np.zeros(9)
        # Vaccines distributed according to vector 'order'
        # With residue 'refusal' remaining in each age group
        idx = 0
        while daily_dose > 0:
            if VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]]) > daily_dose:
                N_vacc[vacc_order[idx]] = daily_dose
                daily_dose = 0
            else:
                N_vacc[vacc_order[idx]] = VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                daily_dose = daily_dose - VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                idx = idx + 1
        return N_vacc

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, delayed_ramp_fun, ramp_fun
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_RESTORE8(t, states, param, l , l_relax, prev_schools, prev_work, prev_rest, prev_home, relaxdate, scenario=0):
    
    t = pd.Timestamp(t.date())

    # Convert compliance tau and l to dates
    l_days = pd.Timedelta(l, unit='D')

    # Convert relaxation l to dates
    l_relax_days = pd.Timedelta(l_relax, unit='D')

    # Define key dates of first wave
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-09-01') # end of summer holidays

    # Define key dates of second wave
    t5 = pd.Timestamp('2020-10-19') # lockdown (1)
    t6 = pd.Timestamp('2020-11-02') # lockdown (2)
    t7 = pd.Timestamp('2020-11-16') # schools re-open
    t8 = pd.Timestamp('2020-12-18') # Christmas holiday starts
    t9 = pd.Timestamp('2021-01-04') # Christmas holiday ends
    t10 = pd.Timestamp('2021-02-15') # Spring break starts
    t11 = pd.Timestamp('2021-02-21') # Spring break ends
    t12 = pd.Timestamp('2021-02-28') # Contact increase in children
    t13 = pd.Timestamp('2021-03-26') # Start of Easter holiday
    t14 = pd.Timestamp('2021-04-18') # End of Easter holiday
    t15 = pd.Timestamp(relaxdate) # Relaxation date
    t16 = pd.Timestamp('2021-07-01') # Start of Summer holiday
    t17 = pd.Timestamp('2021-09-01') 

    if t <= t1:
        return all_contact(t)
    elif t1 < t < t1:
        return all_contact(t)
    elif t1  < t <= t1 + l_days:
        policy_old = all_contact(t)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, t1, l)
    elif t1 + l_days < t <= t2:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t3 < t <= t4:
        return contact_matrix_4prev(t, school=0)

    # Second wave
    elif t4 < t <= t5:
        return contact_matrix_4prev(t, school=1)
    elif t5  < t <= t5 + l_days:
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                    school=1)
        return ramp_fun(policy_old, policy_new, t, t5, l)
    elif t5 + l_days < t <= t6:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t6 < t <= t7:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t7 < t <= t8:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1) 
    elif t8 < t <= t9:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t9 < t <= t10:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t10 < t <= t11:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)    
    elif t11 < t <= t12:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t12 < t <= t13:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest,
                            school=1)
    elif t13 < t <= t14:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)
                                   
    if scenario == 0:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)    
        elif t15 < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)             
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
    elif scenario == 1:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)    
        elif t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=1)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=1)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=0.8, transport=0.90, school=0) # 20% less work mobility during summer, 10% less public transport during summer                                      
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=1)
    elif scenario == 2:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)    
        elif t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=0)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax,)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=0)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=0.8, transport=0.90, school=0)                                  
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=1)    
    elif scenario == 3:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)   
        elif t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=1)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=1)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=0.90, school=0)                                       
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=1)  
    elif scenario == 4:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=0)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1,  school=0)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=0.90, school=0)                                           
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=1)                                     
    elif scenario == 5:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)   
        elif t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=1)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=0.8, leisure=1, transport=0.90, others=1, school=0)                                      
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=1)

    elif scenario == 6:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)   
        if t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=0)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=0)                           
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=0.8, leisure=1, transport=0.90, others=1, school=0)                
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_work, 
                                work=1, leisure=1, transport=1, others=1, school=1)
    
# ----------------
# Helper functions
# ----------------

from covid19model.models.utils import add_poisson

def draw_fcn(param_dict,samples_dict):
    """ 
    This draw function differes from the one located in the `~/src/models/utils.py` because daily_dose is not included
    """

    # Calibration of WAVE 1
    # ---------------------
    idx, param_dict['zeta'] = random.choice(list(enumerate(samples_dict['zeta'])))

    # Calibration of WAVE 2
    # ---------------------
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    model.parameters['da'] = samples_dict['da'][idx]
    model.parameters['l'] = samples_dict['l'][idx]  
    model.parameters['prev_schools'] = samples_dict['prev_schools'][idx]    
    model.parameters['prev_home'] = samples_dict['prev_home'][idx]      
    model.parameters['prev_work'] = samples_dict['prev_work'][idx]       
    model.parameters['prev_rest'] = samples_dict['prev_rest'][idx]
    model.parameters['K_inf'] = samples_dict['K_inf'][idx]
    model.parameters['K_hosp'] = samples_dict['K_hosp'][idx]

    # Vaccination
    # -----------
    param_dict['e_i'] = np.random.uniform(low=0.8,high=1) # Vaccinated individual is 80-100% less infectious than non-vaccinated indidivudal
    param_dict['e_s'] = np.random.uniform(low=0.90,high=0.99) # Vaccine results in a 85-95% lower susceptibility
    param_dict['e_h'] = np.random.uniform(low=0.8,high=1.0) # Vaccine blocks hospital admission between 50-100%
    param_dict['delay'] = np.mean(np.random.triangular(1, 40, 40, size=30))

    # Hospitalization
    # ---------------
    # Fractions
    names = ['c','m_C','m_ICU']
    for idx,name in enumerate(names):
        par=[]
        for jdx in range(9):
            par.append(np.random.choice(samples_dict['samples_fractions'][idx,jdx,:]))
        param_dict[name] = np.array(par)
    # Residence times
    n=100
    distributions = [samples_dict['residence_times']['dC_R'],
                     samples_dict['residence_times']['dC_D'],
                     samples_dict['residence_times']['dICU_R'],
                     samples_dict['residence_times']['dICU_D']]
    names = ['dc_R', 'dc_D', 'dICU_R', 'dICU_D']
    for idx,dist in enumerate(distributions):
        param_val=[]
        for age_group in dist.index.get_level_values(0).unique().values[0:-1]:
            draw = np.random.gamma(dist['shape'].loc[age_group],scale=dist['scale'].loc[age_group],size=n)
            param_val.append(np.mean(draw))
        param_dict[names[idx]] = np.array(param_val)

    return param_dict

# -------------------------------------
# Initialize the model with vaccination
# -------------------------------------


# Model initial condition on September 1st
warmup = 0
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters(vaccination=True)
# Add the time-dependant parameter function arguments
# Social policies
params.update({'l': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5, 'scenario': 0, 'relaxdate': '2021-05-08', 'l_relax': 28})
# Vaccination
params.update(
    {'vacc_order': np.array(range(9))[::-1], 'daily_dose': 55000,
     'refusal': 0.2*np.ones(9), 'delay': 21, 'df_sciensano_start': df_sciensano_start,
     'df_sciensano_end': df_sciensano_end}
)
# Initialize model
model = models.COVID19_SEIRD_vacc(initial_states, params,
                        time_dependent_parameters={'Nc': policies_RESTORE8, 'N_vacc': vacc_strategy, 'alpha': VOC_wrapper_func})

# ----------------------------
# Initialize results dataframe
# ----------------------------
index = pd.date_range(start=start_sim, end=end_sim)
columns = [[],[],[],[],[]]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["social scenario", "relaxation date", "daily vaccinations", "vaccination order", "results"])
df_sim = pd.DataFrame(index=index, columns=columns)

# ---------------------
# Start simulation loop
# ---------------------

colors = ['blue', 'green', 'red', 'orange', 'black', 'brown', 'purple']

print('3) Starting scenario loop\n')
for idx,scenario in enumerate(scenarios):
    print('\n\t# scenario '+scenario)
    model.parameters.update({'scenario': int(scenario)})
    fig,axes = plt.subplots(ncols=1,nrows=len(relaxdates),figsize=(10,len(relaxdates)*4),sharex=True)

    for jdx,relaxdate in enumerate(relaxdates):
        model.parameters.update({'relaxdate': relaxdate})
        print('\t## relaxdate '+relaxdate)
        k=0
        legend_text=[]

        for kdx, daily_dose in enumerate(doses):
            print('\t### '+str(daily_dose)+' doses per day')
            model.parameters.update({'daily_dose': daily_dose})

            for ldx, order in enumerate(orders):
                print('\t#### Vaccination order: '+description_order[ldx])
                model.parameters.update({'vacc_order': order})
                # --------------
                # Simulate model
                # --------------
                out_vacc = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)
                mean_Hin, median_Hin, LL_Hin, UL_Hin = add_poisson('H_in', out_vacc, n_samples, n_draws)
                mean_Htot, median_Htot, LL_Htot, UL_Htot = add_poisson('H_tot', out_vacc, n_samples, n_draws)
                # Append to dataframe
                columnnames = ['incidences_mean', 'incidences_median', 'incidences_LL', 'incidences_UL',
                                'load_mean', 'load_median', 'load_LL', 'load_UL']
                data = [mean_Hin, median_Hin, LL_Hin, UL_Hin, mean_Htot, median_Htot, LL_Htot, UL_Htot]
                for i in range(len(columnnames)):
                    df_sim[scenario, relaxdate, daily_dose, description_order[ldx], columnnames[i]] = data[i]
                # Append to figure
                axes[jdx].plot(df_sim.index, df_sim[scenario, relaxdate, daily_dose, description_order[ldx], 'incidences_mean'],'--', linewidth=1, color=colors[k])
                axes[jdx].fill_between(df_sim.index, df_sim[scenario, relaxdate, daily_dose, description_order[ldx], 'incidences_LL'], df_sim[scenario, relaxdate, daily_dose, description_order[ldx], 'incidences_UL'], color=colors[k], alpha=0.20)
                legend_text.append(str(daily_dose)+' doses/day - '+description_order[ldx])
                k=k+1

        axes[jdx].scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        axes[jdx].scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        axes[jdx] = _apply_tick_locator(axes[jdx])
        axes[jdx].set_xlim('2020-09-01',end_sim)
        axes[jdx].set_ylim(0,1200)
        axes[jdx].set_ylabel('$H_{in}$ (-)')
        axes[jdx].set_title('Relaxation on '+relaxdate, fontsize=13)
        if jdx == 0:
            axes[jdx].legend(legend_text, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
    fig.suptitle('Scenario '+scenario+': '+descriptions_scenarios[int(scenario)]+'\n', x=0.92, y=0.99, ha='right')
    

    fig.savefig('../../results/predictions/national/restore_v8.0/scenario_'+scenario+'.pdf', dpi=400, bbox_inches='tight')
    fig.savefig('../../results/predictions/national/restore_v8.0/scenario_'+scenario+'.png', dpi=400, bbox_inches='tight')

df_sim.to_csv('../../results/predictions/national/restore_v8.0/RESTORE8_UGent_simulations.csv')
