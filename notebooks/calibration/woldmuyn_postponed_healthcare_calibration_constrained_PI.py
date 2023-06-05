"""
This script contains a calibration of an constrained PI postponed healthcare model using data from 2020-2021.
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import json
import argparse
import sys,os
import random
import datetime
import pandas as pd
import numpy as np
from math import factorial
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import font_manager
import csv
# pySODM packages
from pySODM.models.base import ODEModel
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_gaussian, ll_poisson, validate_calibrated_parameters, expand_bounds
# COVID-19 package
from covid19_DTM.data.sciensano import get_sciensano_COVID19_data
from functools import lru_cache

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--high_performance_computing", help="Disable visualizations of fit for hpc runs", action="store_true")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-s", "--start_calibration", help="Calibration startdate. Format 'YYYY-MM-DD'.", default='2020-08-01')
parser.add_argument("-e", "--end_calibration", help="Calibration enddate. Format 'YYYY-MM-DD'.",default='2021-03-01')
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=1)
parser.add_argument("-n_nm", "--n_nm", help="Maximum number of Nelder Mead iterations.", default=1)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default =1)
parser.add_argument("-ID", "--identifier", help="Name in output files.", default = 'constrained_PI_model_test')
parser.add_argument("-MDC_sim", "--MDC_sim", help="MDC classes to plot, setting to 'all' equals all MDC keys. \ndefault=['03','04','05','06']",default=['03','04','05','06'])
parser.add_argument("-MDC_plot", "--MDC_plot", help="MDC classes to plot, setting to 'all' equals all MDC keys. \ndefault=['03','04','05','06']",default=['03','04','05','06'])
parser.add_argument("-filter", "--filter_args", help="Arguments for Savitzky Golay Filtering, [window_size, order]",default=None)
parser.add_argument("-pars", "--pars", help="parameters to calibrate",default=['epsilon'])
parser.add_argument("-covid_data", "--covid_data", help="Use COVID-19 data to estimate covid patients in UZ Gent",action="store_true")
parser.add_argument("-samples","--samples",help="file_name of uncomplete samples",default='constrained_PI_model_500it_second_wave_strict_bounds_SAMPLES_2023-05-16.json')
args = parser.parse_args()

# Backend
if args.backend == False:
    backend = None
else:
    backend = True
# HPC
if args.high_performance_computing:
    high_performance_computing = True
else:
    high_performance_computing = False
# Identifier (name)
if args.identifier:
    identifier = str(args.identifier)
else:
    raise Exception("The script must have a descriptive name for its output.")
# Maximum number of NM iterations
n_pso = int(args.n_pso)
# Maximum number of NM iterations
n_nm = int(args.n_nm)
# Maximum number of MCMC iterations
n_mcmc = int(args.n_mcmc)
# Date at which script is started
run_date = str(datetime.date.today())

# Keep track of runtime
initial_time = datetime.datetime.now()
# Start and end of calibration
start_date = pd.to_datetime(args.start_calibration)
if args.end_calibration:
    end_date = pd.to_datetime(args.end_calibration)

abs_dir = os.path.dirname(__file__)
rel_dir = '../../data/QALY_model/interim/postponed_healthcare/UZG/'
# MDC_dict to translate keys to disease group
MDC_dict={}
file_name = 'MDC_dict.csv'
with open(os.path.join(abs_dir,rel_dir,file_name), mode='r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        MDC_dict.update({row[0]:row[1]})

MDC_dict.pop('.')
MDC_keys = sorted(list(MDC_dict.keys()))

# MDC classes to plot
if args.MDC_sim == 'all':
    MDC_sim  = np.array(MDC_keys)
else:
    if isinstance(args.MDC_sim , str):
        MDC_sim  = np.array(args.MDC_sim [1:-1].split(','))
    else:
        MDC_sim = np.array(args.MDC_sim )

# MDC classes to plot
if args.MDC_plot == 'all':
    MDC_plot  = np.array(MDC_keys)
else:
    if isinstance(args.MDC_plot , str):
        MDC_plot  = np.array(args.MDC_plot [1:-1].split(','))
    else:
        MDC_plot  = np.array(args.MDC_plot )

MDC_plot = np.array(sorted(MDC_plot))
MDC_sim = np.array(sorted(MDC_sim))

# smoother
if args.filter_args:
    if isinstance(args.filter_args, str):
        filter_args  = np.array(args.filter_args[1:-1].split(','))
    else:
        filter_args  = np.array(args.filter_args)
else:
    filter_args = None

#pars to calibrate
if isinstance(args.pars, str):
    pars = np.array(args.pars[1:-1].split(','))
else:
    pars = np.array(args.pars)

# use covid data to estimate UZ Gent covid patients
use_covid_data = args.covid_data
# use_covid_data = True
# filter_args = [61,4]
# MDC_sim = np.array(MDC_keys)
if args.samples == None:
    samples_dict_all_MDCs = None
else:
    rel_dir = '../../data/QALY_model/interim/postponed_healthcare/model_parameters/constrained_PI_model/calibrations'
    f = open(os.path.join(abs_dir,rel_dir,args.samples))
    samples_dict_all_MDCs = json.load(f)
    print(len(samples_dict_all_MDCs['alpha'])) 

##############################
## Define results locations ##
##############################

abs_dir = os.path.dirname(__file__)
# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = os.path.join(abs_dir,'../../results/QALY_model/postponed_healthcare/calibrations/constrained_PI_model/')
# Path where MCMC samples should be saved
samples_path = os.path.join(abs_dir,'../../data/QALY_model/interim/postponed_healthcare/model_parameters/constrained_PI_model/')
# Path where samples backend should be stored
backend_folder = os.path.join(abs_dir,'../../results/QALY_model/postponed_healthcare/calibrations/constrained_PI_model/')
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path, backend_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
    for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/", fig_path+"fits/", fig_path+"variance/"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

###############
## Load data ##
###############

# Postponed healthcare data
abs_dir = os.path.dirname(__file__)
rel_dir = '../../data/QALY_model/interim/postponed_healthcare/UZG'
file_name = '2020_2021_normalized.csv'
types_dict = {'APR_MDC_key': str}
# mean data
hospitalizations_normalized = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True)['mean']
hospitalizations_normalized = hospitalizations_normalized.reorder_levels(['date','APR_MDC_key'])
hospitalizations_normalized=hospitalizations_normalized.sort_index()
hospitalizations_normalized=hospitalizations_normalized.reindex(hospitalizations_normalized.index.rename('MDC',level=1))

# MDC_keys = hospitalizations_normalized.index.get_level_values('MDC').unique().values
dates = hospitalizations_normalized.index.get_level_values('date').unique().values

# lower and upper quantiles
hospitalizations_normalized_quantiles = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True).loc[(slice(None), slice(None)), ('q0.025','q0.975')]
hospitalizations_normalized_quantiles = hospitalizations_normalized_quantiles.reorder_levels(['date','APR_MDC_key'])
hospitalizations_normalized_quantiles=hospitalizations_normalized_quantiles.sort_index()
hospitalizations_normalized_quantiles=hospitalizations_normalized_quantiles.reindex(hospitalizations_normalized_quantiles.index.rename('MDC',level=1))

file_name = 'MZG_2016_2021.csv'
types_dict = {'APR_MDC_key': str}
hospitalizations = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2,3],dtype=types_dict).squeeze()
hospitalizations = hospitalizations.groupby(['APR_MDC_key','date']).sum()
hospitalizations.index = hospitalizations.index.set_levels([hospitalizations.index.levels[0], pd.to_datetime(hospitalizations.index.levels[1])])
hospitalizations = hospitalizations.reorder_levels(['date','APR_MDC_key'])
hospitalizations=hospitalizations.sort_index()
hospitalizations=hospitalizations.reindex(hospitalizations.index.rename('MDC',level=1))

# COVID-19 data
covid_data, _ , _ , _ = get_sciensano_COVID19_data(update=False)
new_index = pd.MultiIndex.from_product([pd.to_datetime(hospitalizations_normalized.index.get_level_values('date').unique()),covid_data.index.get_level_values('NIS').unique()])
covid_data = covid_data.reindex(new_index,fill_value=0)
df_covid_H_in = covid_data['H_in'].loc[:,40000]
df_covid_H_tot = covid_data['H_tot'].loc[:,40000]
df_covid_dH = df_covid_H_tot.diff().fillna(0)

# hospitalisation baseline
file_name = 'MZG_baseline.csv'
types_dict = {'APR_MDC_key': str, 'week_number': int, 'day_number':int}
hospitalization_baseline = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2,3],dtype=types_dict).squeeze()
hospitalization_baseline = hospitalization_baseline.groupby(['APR_MDC_key','week_number','day_number']).mean()

# mean hospitalisation length
file_name = 'MZG_residence_times.csv'
types_dict = {'APR_MDC_key': str, 'age_group': str, 'stay_type':str}
residence_times = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2],dtype=types_dict).squeeze()
mean_residence_times = residence_times.groupby(by=['APR_MDC_key']).mean()

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

multi_index = pd.MultiIndex.from_product([MDC_keys,dates])
baseline_in_date_form = pd.Series(index=multi_index,dtype='float')
for idx,(disease,date) in enumerate(multi_index):
    date = pd.to_datetime(date)
    baseline_in_date_form[idx] = hospitalization_baseline.loc[(disease,date.isocalendar().week,date.isocalendar().weekday)]

if filter_args is not None:
    window = filter_args[0]
    order = filter_args[1]

    hospitalizations_normalized_smooth = hospitalizations_normalized.copy()
    for MDC_key in MDC_keys:
        hospitalizations_normalized_smooth.loc[slice(None),MDC_key] = savitzky_golay(hospitalizations_normalized.loc[slice(None),MDC_key],window,order)

    hospitalizations_smooth = hospitalizations.copy()
    for MDC_key in MDC_keys:
        hospitalizations_smooth.loc[slice(None),MDC_key] = savitzky_golay(hospitalizations.loc[slice(None),MDC_key],window,order)

    hospitalizations_baseline_smooth = baseline_in_date_form.copy()
    for MDC_key in MDC_keys:
        hospitalizations_baseline_smooth[MDC_key] = savitzky_golay(baseline_in_date_form[MDC_key],window,order)

rel_dir = '../../data/QALY_model/interim/postponed_healthcare/model_parameters/constrained_PI_model/'

# Load samples
file_names = ['constrained_PI_model_first_wave_THETA_init.json',
              'constrained_PI_model_second_wave_THETA_init.json']

thetas_init = {'first':{},'second':{}}
for period,file_name in zip(thetas_init.keys(),file_names):
    f = open(os.path.join(abs_dir,rel_dir,file_name))
    thetas_init[period] = json.load(f)

##################
## Define model ##
##################

class Constrained_PI_Model(ODEModel):    
    state_names = ['H_norm','E']
    parameter_names = ['covid_H','covid_dH']
    parameter_stratified_names = ['Kp', 'Ki', 'alpha','epsilon','covid_capacity']
    dimension_names = ['MDC']
    
    @staticmethod
    def integrate(t, H_norm, E, covid_H, covid_dH, Kp, Ki, alpha, epsilon, covid_capacity):

        error = 1-H_norm
        dE = error - epsilon*E
        u = np.where(covid_H <= covid_capacity, Kp * np.where(E<=0,error,0) + Ki * np.where(E>0,E,0), 0)
        dH_norm = -alpha*covid_dH + u

        return dH_norm, dE 

#################
## Define TDPF ##
#################

class get_covid_H():

    def __init__(self, use_covid_data, covid_data=None, baseline=None, hospitalizations=None):
        self.use_covid_data = use_covid_data
        if use_covid_data:
            self.covid_data = covid_data
        else:
            self.baseline_04 = baseline['04']
            self.hospitalizations_04 = hospitalizations.loc[(slice(None),'04')]

    def H_wrapper_func(self, t, states, param):
        return self.__call__(t)

    def __call__(self, t):
        t = pd.to_datetime(t).round(freq='D')
        if self.use_covid_data:
            try:
                covid_H = self.covid_data.loc[t]
            except:
                covid_H = 0

        else:
            covid_H = max(self.hospitalizations_04.loc[t] - self.baseline_04.loc[t],0)
        
        return covid_H
    
class get_covid_dH():

    def __init__(self, data):
        self.data = data

    def dH_wrapper_func(self, t, states, param):
        return self.__call__(t)

    @lru_cache
    def __call__(self, t):
        t = pd.to_datetime(t).round(freq='D')
        try:
            covid_dH = self.data.loc[t]
        except:
            covid_dH = 0

        return covid_dH
    
#################
## Setup model ##
#################

def init_constrained_PI_model(start_date,end_date,MDC_sim,wave='first'):
    # Define model parameters, initial states and coordinates
    n = len(MDC_sim)

    #theta = thetas_init[wave]
    #alpha, epsilon, Kp, Ki, covid_capacity = [],[],[],[],[]
    #for MDC_key in MDC_sim:
    #    MDC_idx = MDC_keys.index(MDC_key)
    #    alpha.append(theta['alpha'][MDC_idx])
    #    epsilon.append(theta['epsilon'][MDC_idx])
    #    Kp.append(theta['Kp'][MDC_idx])
    #    Ki.append(theta['Ki'][MDC_idx])
    #    covid_capacity.append(theta['covid_capacity'][MDC_idx])
    #alpha = np.array(alpha)
    #epsilon = np.array(epsilon)
    #Kp = np.array(Kp)
    #Ki = np.array(Ki)
    #covid_capacity = np.array(covid_capacity)
    
    alpha = 0.0005*np.ones(n)
    epsilon = 0.06*np.ones(n)
    Kp = 0.1*np.ones(n)
    Ki = 0.005*np.ones(n)
    covid_capacity = 500*np.ones(n)

    start_date_string = start_date.strftime('%Y-%m-%d')

    if filter_args is not None:
        baseline = hospitalizations_baseline_smooth
    else:
        baseline = baseline_in_date_form

    covid_H = df_covid_H_tot.loc[start_date_string]
    covid_dH = df_covid_dH.loc[start_date_string]

    params={'covid_H':covid_H,'covid_dH':covid_dH,'epsilon':epsilon, 'alpha':alpha, 'Kp':Kp,'Ki':Ki,'covid_capacity':covid_capacity}

    init_states = {'H_norm':np.ones(n)}
    coordinates={'MDC':MDC_sim}

    covid_H_func = get_covid_H(use_covid_data,df_covid_H_tot,baseline,hospitalizations).H_wrapper_func
    dH_function = get_covid_dH(df_covid_dH).dH_wrapper_func

    # Initialize model
    model = Constrained_PI_Model(init_states,params,coordinates,
                          time_dependent_parameters={'covid_H':covid_H_func,'covid_dH':dH_function})

    return model

##########################################################
## Compute the overdispersion parameters for our H data ##
##########################################################

method = 'gaussian'

if filter_args is not None:
    data = hospitalizations_normalized_smooth.loc[start_date:end_date,MDC_sim]
else:
    data = hospitalizations_normalized.loc[start_date:end_date,MDC_sim]

sigmas_ll = []
results, ax = variance_analysis(data, resample_frequency='W')
for MDC_key in MDC_sim:
    sigmas_ll.append(results['theta'].loc[MDC_key,method])
plt.close()
    
if __name__ == '__main__':

    #############################
    ## plot initial conditions ##
    #############################

    # plot 
    fig,axs = plt.subplots(len(MDC_sim),1,sharex=True,figsize=(9,2*len(MDC_sim)))
    plot_time = pd.date_range(start_date,end_date)
    
    model = init_constrained_PI_model(start_date,end_date,MDC_sim)
 
    calibrated_param_dict = model.parameters.copy()
    for param in model.parameters:
        if isinstance(calibrated_param_dict[param], np.ndarray):
            calibrated_param_dict[param] = list(calibrated_param_dict[param])
        else:
            calibrated_param_dict[param] = float(calibrated_param_dict[param])

    out = model.sim([start_date, end_date],method='LSODA')
    
    for i,MDC_key in enumerate(MDC_sim):
        axs[i].plot(plot_time,hospitalizations_normalized.loc[plot_time,MDC_key].values,color='blue')
        if filter_args is not None:
            axs[i].plot(plot_time,hospitalizations_normalized_smooth.loc[plot_time,MDC_key].values,color='red')
        
        axs[i].plot(plot_time,out['H_norm'].sel(MDC=MDC_key),color='black')

        axs[i].set_ylabel(MDC_key)
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        axs[i].grid(False)
        axs[i].tick_params(axis='both', which='major', labelsize=8)
        axs[i].tick_params(axis='both', which='minor', labelsize=8)
        axs[i].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5) 

    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,str(identifier)+'_init_condition.pdf'))
    # plt.show()
    plt.close()

    #############################
    ## set up calibration plot ##
    #############################

    fig,axs = plt.subplots(len(MDC_plot),3,sharex=True,figsize=(9,2*len(MDC_plot)))

    axs[0,0].set_title('PSO')
    axs[0,1].set_title('Nelder-Mead')
    axs[0,2].set_title('MCMC')
    for i,MDC_key in enumerate(MDC_plot):
        axs[i,0].set_ylabel(MDC_key)
        for j in range(3):
            axs[i,j].xaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i,j].grid(False)
            axs[i,j].tick_params(axis='both', which='major', labelsize=8)
            axs[i,j].tick_params(axis='both', which='minor', labelsize=8)
            axs[i,j].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5) 

    ######################
    ## Calibrate models ##
    ######################
    print(f'start calibration: {initial_time}')
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    processes = 1
    
    # parameters and bounds
    labels_dict = {'epsilon':'$\\epsilon$', 'alpha':'$\\alpha$', 'Kp':'$K_p$','Ki':'$K_I$','covid_capacity':'$COVID_{capacity}$'}
    labels = [labels_dict[par] for par in pars]
    
    
    for i,MDC_key in enumerate(MDC_sim):
        string = f'Calibrating MDC:{MDC_key}'
        print("*"*len(string)+"\n"+string+"\n"+"*"*len(string))

        if MDC_key == '04':
            bounds_dict = {'epsilon':(1*10**-10,1),'alpha':(-1,-1*10**-10), 'Kp':(1*10**-10,1),'Ki':(1*10**-10,1),'covid_capacity':(1*10**-10,1000)}
            bounds = [bounds_dict[par] for par in pars]
        elif MDC_key == '07':
            bounds_dict = {'epsilon':(1*10**-10,1),'alpha':(-1,1), 'Kp':(1*10**-10,1),'Ki':(1*10**-10,1),'covid_capacity':(1*10**-10,1000)}
            bounds = [bounds_dict[par] for par in pars]
        else:
            bounds_dict = {'epsilon':(1*10**-10,1),'alpha':(10**-10,1), 'Kp':(1*10**-10,1),'Ki':(1*10**-10,1),'covid_capacity':(1*10**-10,1000)}
            bounds = [bounds_dict[par] for par in pars]

        model = init_constrained_PI_model(start_date,end_date,[MDC_key])

        parameter_sizes, _ = validate_calibrated_parameters(pars, model.parameters)
        expanded_bounds = expand_bounds(parameter_sizes, bounds)

        # Define dataset
        if filter_args is not None:
            data = hospitalizations_normalized_smooth.loc[start_date:end_date,MDC_key]
        else:
            data = hospitalizations_normalized.loc[start_date:end_date,MDC_key]
        calibration_data=[data.to_frame(),]
        
        states = ['H_norm',]
        # Setup likelihood functions and arguments
        log_likelihood_fnc = [ll_gaussian,]
        log_likelihood_fnc_args = [np.array([sigmas_ll[i]]),]

        # Setup objective function (no priors --> uniform priors based on bounds) 
        objective_function = log_posterior_probability(model,pars,bounds,calibration_data,states,
                                            log_likelihood_fnc,log_likelihood_fnc_args,labels=labels)
        # --- #
        # PSO #
        # --- #
        if n_pso > 0:
            theta = pso.optimize(objective_function, swarmsize=30*len(expanded_bounds), max_iter=n_pso, processes=processes,kwargs={'simulation_kwargs':{'method': 'LSODA'}})[0]

            param_dict = assign_theta(model.parameters,pars,theta)
            for param in pars:
                calibrated_param_dict[param][i] = float(param_dict[param])

            with open(os.path.join(samples_path, str(identifier)+'_THETA_PSO_'+run_date+'.json'), 'w') as fp:
                json.dump(calibrated_param_dict, fp)       
        else:
            theta = np.array([])
            for param in pars:
                theta = np.append(theta,model.parameters[param][0])

            param_dict = assign_theta(model.parameters,pars,theta)
        
        # simulate model
        model.parameters.update(param_dict)
        out_pso = model.sim([start_date, end_date],method = 'LSODA')

        # visualization
        axs[i,0].plot(plot_time,hospitalizations_normalized.loc[plot_time,MDC_key], label='ruw data', alpha=0.7, color='blue')
        if filter_args is not None:
            axs[i,0].plot(plot_time,hospitalizations_normalized_smooth.loc[plot_time,MDC_key], label='filtered data', alpha=0.7, color='red')
        axs[i,0].plot(plot_time,out_pso.sel(date=plot_time,MDC=MDC_key)['H_norm'],label='model',color='black')
        
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path,'fits/',str(identifier)+ '_' +run_date+'.pdf'))
        plt.close()

        # ----------- #
        # Nelder-mead #
        # ----------- #
        if n_nm > 0:
            step = len(theta)*[0.3,]
            theta,_ = nelder_mead.optimize(objective_function, theta, step, processes=processes, max_iter=n_nm,kwargs={'simulation_kwargs':{'method': 'LSODA'}})

            param_dict = assign_theta(model.parameters,pars,theta)
            for param in pars:
                calibrated_param_dict[param][i] = float(param_dict[param])
            with open(os.path.join(samples_path, str(identifier)+'_THETA_NM_'+run_date+'.json'), 'w') as fp:
                json.dump(calibrated_param_dict, fp)

            # simulate model
            model.parameters.update(param_dict)
            out_nm = model.sim([start_date, end_date], method = 'LSODA')

            # visualization
            axs[i,1].plot(plot_time,hospitalizations_normalized.loc[plot_time,MDC_key], label='ruw data', alpha=0.7, color='blue')
            if filter_args is not None:
                axs[i,1].plot(plot_time,hospitalizations_normalized_smooth.loc[plot_time,MDC_key], label='filtered data', alpha=0.7, color='red')
            axs[i,1].plot(plot_time,out_nm.sel(date=plot_time,MDC=MDC_key)['H_norm'],label='model',color='black')

            fig.tight_layout()
            fig.savefig(os.path.join(fig_path,'fits/',str(identifier) + '_' +run_date+'.pdf'))
            plt.close()

        # ---- #
        # MCMC #
        # ---- #
        if n_mcmc > 0:
            multiplier_mcmc = 10
            print_n = 10

            theta = np.where(theta==0,0.00001,theta)

            # Perturbate previously obtained estimate
            ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.3,]*len(theta), bounds=expanded_bounds, multiplier=multiplier_mcmc, verbose=True)

            # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
            settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"), 'n_chains': nwalkers,
                        'labels': labels, 'parameters': list(pars), 'starting_estimate': list(theta),"MDC":list(MDC_sim)}
            # Start calibration
            sys.stdout.flush()
            # Sample n_mcmc iterations
            sampler = run_EnsembleSampler(pos, n_mcmc, identifier+'_'+MDC_key, objective_function, objective_function_kwargs={'simulation_kwargs':{'method': 'LSODA'}},
                                        fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                        settings_dict=settings) 
            # Generate a sample dictionary
            thin = 1
            try:
                autocorr = sampler.get_autocorr_time(flat=True)
                thin = max(1,int(0.5 * np.min(autocorr)))
            except:
                pass
            if sampler.get_chain(discard=int(n_mcmc*3/4), thin=thin, flat=True).shape[0] > 200:
                discard = int(n_mcmc*3/4)
            elif sampler.get_chain(discard=int(n_mcmc*1/2), thin=thin, flat=True).shape[0] > 200:
                discard = int(n_mcmc*1/2)
            else:
                discard = 0
            samples_dict = emcee_sampler_to_dictionary(discard=discard, thin=thin, samples_path=samples_path,identifier=identifier+'_'+MDC_key)

            if samples_dict_all_MDCs is None:
                samples_dict_all_MDCs = samples_dict.copy()
                for param in pars:
                    samples_dict_all_MDCs[param] = [samples_dict_all_MDCs[param]]
            else:
                for param in pars:
                    samples_dict_all_MDCs[param].append(samples_dict[param])

            with open(os.path.join(samples_path, str(identifier)+'_SAMPLES_'+run_date+'.json'), 'w') as fp:
                json.dump(samples_dict_all_MDCs, fp)

            # Define draw function
            def draw_fcn(param_dict, samples_dict):
                pars = samples_dict['parameters']

                idx = random.randint(0,len(samples_dict[pars[0]])-1)
                
                for param in pars:
                        param_dict.update({param:samples_dict[param][idx]})
                return param_dict

            # Simulate model
            N=5
            out_mcmc = model.sim([start_date, end_date], N=N, samples=samples_dict, draw_function=draw_fcn, processes=processes,method = 'LSODA')

            # visualization
            axs[i,2].plot(plot_time,hospitalizations_normalized.loc[plot_time,MDC_key], label='ruw data', alpha=0.7, color='blue')
            if filter_args is not None:
                axs[i,2].plot(plot_time,hospitalizations_normalized_smooth.loc[plot_time,MDC_key], label='filtered data', alpha=0.7, color='red')
            axs[i,2].plot(plot_time,out_mcmc['H_norm'].sel(date=plot_time,MDC=MDC_key).mean(dim='draws'), color='black', label='simulated_H')
            axs[i,2].fill_between(plot_time,out_mcmc['H_norm'].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.025),
                                out_mcmc['H_norm'].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.975), color='black', alpha=0.2, label='Simulation 95% CI')

            fig.tight_layout()
            fig.savefig(os.path.join(fig_path,'fits/',str(identifier) + '_' +run_date+'.pdf'))
            plt.close()
    print(f'start calibration: {initial_time}')
    print(f'end calibration: {datetime.datetime.now()}')