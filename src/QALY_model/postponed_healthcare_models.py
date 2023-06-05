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
from pySODM.models.base import ODEModel
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_gaussian
from covid19_DTM.data.sciensano import get_sciensano_COVID19_data
from functools import lru_cache
from decimal import Decimal
import xarray as xar
import math
from math import factorial

class Queuing_Model(ODEModel):

    state_names = ['W','H','H_adjusted','H_norm','R','NR','X']
    parameter_names = ['X_tot','f_UZG','covid_H','alpha','post_processed_H']
    parameter_stratified_names = ['A','gamma','epsilon','sigma']
    dimension_names = ['MDC']
    
    @staticmethod
    def integrate(t, W, H, H_adjusted, H_norm, R, NR, X, X_tot, f_UZG, covid_H, alpha, post_processed_H, A, gamma, epsilon, sigma):
        X_new = (X_tot-alpha*covid_H-sum(H - (1/gamma*H))) * (sigma*(A+W))/sum(sigma*(A+W))

        W_to_H = np.where(W>X_new,X_new,W)
        W_to_NR = epsilon*(W-W_to_H)
        A_to_H = np.where(A>(X_new-W_to_H),(X_new-W_to_H),A)
        A_to_W = A-A_to_H
        
        dX = -X + X_new - W_to_H - A_to_H
        dW = A_to_W - W_to_H - W_to_NR
        dH = A_to_H + W_to_H - (1/gamma*H) 
        dR = (1/gamma*H)
        dNR = W_to_NR

        dH_adjusted = -H_adjusted + post_processed_H[0]
        dH_norm = -H_norm + post_processed_H[1]
        
        return dW, dH, dH_adjusted, dH_norm, dR, dNR, dX
    
class Constrained_PI_Model(ODEModel):

    state_names = ['H_norm','E']
    parameter_names = ['covid_H','covid_dH','covid_capacity']
    parameter_stratified_names = ['Kp', 'Ki', 'alpha','epsilon']
    dimension_names = ['MDC']
    
    @staticmethod
    def integrate(t, H_norm, E, covid_H, covid_dH, Kp, Ki, alpha, epsilon, covid_capacity):

        error = 1-H_norm
        dE = error - epsilon*E
        u = np.where(covid_H <= covid_capacity, Kp * np.where(E<=0,error,0) + Ki * np.where(E>0,E,0), 0)
        dH_norm = -alpha*covid_dH + u

        return dH_norm, dE
    
class PI_Model(ODEModel):
    
    state_names = ['H_norm','E']
    parameter_names = ['covid_H']
    parameter_stratified_names = ['Kp', 'Ki', 'alpha', 'epsilon']
    dimension_names = ['MDC']
    
    @staticmethod
    def integrate(t, H_norm, E, covid_H, Kp, Ki, alpha, epsilon):

        error = 1-H_norm
        dE = error - epsilon*E
        u = Kp*error + Ki*E
        dH_norm = -alpha*covid_H + u

        return dH_norm, dE 
      
class get_A():
    def __init__(self, baseline, mean_residence_times):
        self.baseline = baseline
        self.mean_residence_times = mean_residence_times
        
    def A_wrapper_func(self, t, states, param):
        return self.__call__(t)
    
    @lru_cache()
    def __call__(self, t):

        A = (self.baseline.loc[(t,slice(None))]/self.mean_residence_times).values

        return A 
    
class H_post_processing():

    def __init__(self, baseline, MDC_sim):
        self.MDC = np.array(MDC_sim)
        self.baseline = baseline

    def H_post_processing_wrapper_func(self, t, states, param, covid_H):
        H = states['H']
        return self.__call__(t, H, covid_H)

    def __call__(self, t, H, covid_H):
        H_adjusted = np.where(self.MDC=='04',H+covid_H,H)
        H_norm = H_adjusted/self.baseline.loc[(t,slice(None))]
        return (H_adjusted,H_norm)
    
class get_covid_H():

    def __init__(self, use_covid_data, covid_data=None, baseline=None, hospitalizations=None):
        self.use_covid_data = use_covid_data
        if use_covid_data:
            self.covid_data = covid_data
        else:
            self.baseline_04 = baseline.loc[(slice(None),'04')]
            self.hospitalizations_04 = hospitalizations.loc[(slice(None),'04')]

    def H_wrapper_func(self, t, states, param, f_UZG):
        return self.__call__(t, f_UZG)

    @lru_cache
    def __call__(self, t, f_UZG):
        if self.use_covid_data:
            t = pd.to_datetime(t).round(freq='D')
            try:
                covid_H = self.covid_data.loc[t]*f_UZG
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

def draw_fcn(param_dict, samples_dict):
        pars = samples_dict['parameters']

        if hasattr(samples_dict[pars[0]][0],'__iter__'):
            idx = random.randint(0,len(samples_dict[pars[0]][0])-1)
        else:
            idx = random.randint(0,len(samples_dict[pars[0]])-1)
        
        for param in pars:
            if hasattr(samples_dict[param][0],'__iter__'):
                par_array = np.array([])
                for samples in samples_dict[param]:
                    par_array = np.append(par_array,samples[idx])
                param_dict.update({param:par_array})
            else:
                param_dict.update({param:samples_dict[param][idx]})

        return param_dict

class Postponed_healthcare_models_and_data:
    def __init__(self):
        
        def get_lower(x):
            return np.quantile(x,0.025)
        def get_upper(x):
            return np.quantile(x,0.975)

        dates = pd.date_range('2020-01-01','2021-12-31')

        # hospitalization data
        abs_dir = os.path.dirname(__file__)
        rel_dir = '../../data/QALY_model/interim/postponed_healthcare/UZG'

        file_name = '2020_2021_normalized.csv'
        types_dict = {'APR_MDC_key': str}
        hospitalizations_normalized_data = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True)

        hospitalizations_normalized = hospitalizations_normalized_data['mean']
        hospitalizations_normalized=hospitalizations_normalized.sort_index()
        hospitalizations_normalized=hospitalizations_normalized.reindex(hospitalizations_normalized.index.rename('MDC',level=0))
        hospitalizations_normalized = hospitalizations_normalized.reorder_levels(['date','MDC'])
        self.hospitalizations_normalized = hospitalizations_normalized

        hospitalizations_normalized_quantiles = hospitalizations_normalized_data.loc[(slice(None), slice(None)), ('q0.025','q0.975')]
        hospitalizations_normalized_quantiles=hospitalizations_normalized_quantiles.sort_index()
        hospitalizations_normalized_quantiles=hospitalizations_normalized_quantiles.reindex(hospitalizations_normalized_quantiles.index.rename('MDC',level=0))
        hospitalizations_normalized_quantiles = hospitalizations_normalized_quantiles.reorder_levels(['date','MDC'])
        self.hospitalizations_normalized_quantiles = hospitalizations_normalized_quantiles

        file_name = 'MZG_2016_2021.csv'
        types_dict = {'APR_MDC_key': str}
        hospitalizations = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2,3],dtype=types_dict).squeeze()
        hospitalizations = hospitalizations.groupby(['APR_MDC_key','date']).sum()
        hospitalizations.index = hospitalizations.index.set_levels([hospitalizations.index.levels[0], pd.to_datetime(hospitalizations.index.levels[1])])
        hospitalizations=hospitalizations.sort_index()
        hospitalizations=hospitalizations.reindex(hospitalizations.index.rename('MDC',level=0))
        hospitalizations = hospitalizations.reorder_levels(['date','MDC'])
        self.hospitalizations = hospitalizations

        file_name = 'MZG_baseline.csv'
        types_dict = {'APR_MDC_key': str, 'week_number': int, 'day_number':int}
        hospitalizations_baseline = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2,3],dtype=types_dict).squeeze()
        self.hospitalizations_baseline_lower = hospitalizations_baseline.groupby(['APR_MDC_key','week_number','day_number']).apply(get_lower)
        self.hospitalizations_baseline_upper = hospitalizations_baseline.groupby(['APR_MDC_key','week_number','day_number']).apply(get_upper)
        self.hospitalizations_baseline_mean = hospitalizations_baseline.groupby(['APR_MDC_key','week_number','day_number']).mean()
        self.hospitalizations_baseline_mean=self.hospitalizations_baseline_mean.reindex(self.hospitalizations_baseline_mean.index.rename('MDC',level=0))

        # mean hospitalisation length
        file_name = 'MZG_residence_times.csv'
        types_dict = {'APR_MDC_key': str, 'age_group': str, 'stay_type':str}
        residence_times = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2],dtype=types_dict).squeeze()
        self.mean_residence_times = residence_times.groupby(by=['APR_MDC_key']).mean()

        self.all_MDC_keys = sorted(hospitalizations_normalized.index.get_level_values('MDC').unique().values)

        # COVID-19 data
        covid_data, _ , _ , _ = get_sciensano_COVID19_data(update=False)
        new_index = pd.MultiIndex.from_product([pd.to_datetime(hospitalizations.index.get_level_values('date').unique()),covid_data.index.get_level_values('NIS').unique()])
        covid_data = covid_data.reindex(new_index,fill_value=0)
        self.df_covid_H_in = covid_data['H_in'].loc[:,40000]
        self.df_covid_H_tot = covid_data['H_tot'].loc[:,40000]
        self.df_covid_dH = self.df_covid_H_tot.diff().fillna(0)

        # smoothing data
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

        multi_index = pd.MultiIndex.from_product([dates,self.all_MDC_keys])
        mean_baseline_in_date_form = pd.Series(index=multi_index,dtype='float')
        lower_baseline_in_date_form = pd.Series(index=multi_index,dtype='float')
        upper_baseline_in_date_form = pd.Series(index=multi_index,dtype='float')
        for idx,(date,disease) in enumerate(multi_index):
            date = pd.to_datetime(date)
            mean_baseline_in_date_form[idx] = self.hospitalizations_baseline_mean.loc[(disease,date.isocalendar().week,date.isocalendar().weekday)]
            lower_baseline_in_date_form[idx] = self.hospitalizations_baseline_lower.loc[(disease,date.isocalendar().week,date.isocalendar().weekday)]
            upper_baseline_in_date_form[idx] = self.hospitalizations_baseline_upper.loc[(disease,date.isocalendar().week,date.isocalendar().weekday)]

        window = 61
        order = 4

        self.hospitalizations_smooth = hospitalizations.copy()
        self.hospitalizations_baseline_mean_smooth = mean_baseline_in_date_form.copy()
        self.hospitalizations_baseline_lower_smooth = mean_baseline_in_date_form.copy()
        self.hospitalizations_baseline_upper_smooth = mean_baseline_in_date_form.copy()
        self.hospitalizations_normalized_smooth = hospitalizations_normalized.copy()
        self.hospitalizations_normalized_lower_smooth = hospitalizations_normalized_quantiles['q0.025'].copy()
        self.hospitalizations_normalized_upper_smooth = hospitalizations_normalized_quantiles['q0.975'].copy()
        for MDC_key in self.all_MDC_keys:
            self.hospitalizations_normalized_smooth.loc[slice(None),MDC_key] = savitzky_golay(hospitalizations_normalized.loc[slice(None),MDC_key],window,order)
            self.hospitalizations_normalized_lower_smooth.loc[slice(None),MDC_key] = savitzky_golay(hospitalizations_normalized_quantiles['q0.025'].loc[slice(None),MDC_key],window,order)
            self.hospitalizations_normalized_upper_smooth.loc[slice(None),MDC_key] = savitzky_golay(hospitalizations_normalized_quantiles['q0.975'].loc[slice(None),MDC_key],window,order)
            self.hospitalizations_smooth.loc[slice(None),MDC_key] = savitzky_golay(hospitalizations.loc[slice(None),MDC_key],window,order)
            self.hospitalizations_baseline_mean_smooth.loc[slice(None),MDC_key] = savitzky_golay(mean_baseline_in_date_form.loc[slice(None),MDC_key],window,order)
            self.hospitalizations_baseline_lower_smooth.loc[slice(None),MDC_key] = savitzky_golay(lower_baseline_in_date_form.loc[slice(None),MDC_key],window,order)
            self.hospitalizations_baseline_upper_smooth.loc[slice(None),MDC_key] = savitzky_golay(upper_baseline_in_date_form.loc[slice(None),MDC_key],window,order)

    def init_models(self,start_date,MDC_keys='all'):
        if MDC_keys == 'all':
            MDC_keys = self.all_MDC_keys
        n = len(MDC_keys)

        # init queuing model
        gamma =  self.mean_residence_times.loc[MDC_keys].values
        epsilon = np.ones(n)*0.1
        sigma = np.ones(n)
        alpha = 5

        f_UZG = 0.13
        X_tot = 1049

        start_date_string = start_date.strftime('%Y-%m-%d')
        covid_H = self.df_covid_H_tot.loc[start_date_string]*f_UZG
        H_init = self.hospitalizations_baseline_mean_smooth.loc[(start_date,MDC_keys)].values
        H_init_normalized = (H_init/self.hospitalizations_baseline_mean_smooth.loc[((start_date,MDC_keys))]).values
        A = H_init/self.mean_residence_times.loc[MDC_keys]

        params={'A':A,'covid_H':0,'alpha':alpha,'post_processed_H':(H_init,H_init_normalized),'X_tot':X_tot, 'gamma':gamma, 'epsilon':epsilon, 'sigma':sigma,'f_UZG':f_UZG}

        init_states = {'H':H_init,'H_norm':np.ones(len(MDC_keys)), 'H_adjusted':H_init}
        coordinates={'MDC':MDC_keys}

        daily_hospitalizations_func = get_A(self.hospitalizations_baseline_mean_smooth,self.mean_residence_times.loc[MDC_keys]).A_wrapper_func
        covid_H_func = get_covid_H(True,self.df_covid_H_tot,self.hospitalizations_baseline_mean_smooth,self.hospitalizations).H_wrapper_func
        post_processing_H_func = H_post_processing(self.hospitalizations_baseline_mean_smooth,MDC_keys).H_post_processing_wrapper_func

        # Initialize model
        queuing_model = Queuing_Model(init_states,params,coordinates,
                                time_dependent_parameters={'A': daily_hospitalizations_func,'covid_H':covid_H_func,'post_processed_H':post_processing_H_func})
        
        n = len(MDC_keys) 
        # init constrained PI model
        alpha = 0.0005*np.ones(n)
        epsilon = 0.05*np.ones(n)

        Kp = 0.05*np.ones(n)
        Ki = 0.003*np.ones(n)

        covid_capacity = 20

        start_date_string = start_date.strftime('%Y-%m-%d')
        covid_H = self.df_covid_H_tot.loc[start_date_string]
        covid_dH = self.df_covid_dH.loc[start_date_string]
        params={'covid_H':covid_H,'covid_dH':covid_dH,'epsilon':epsilon, 'alpha':alpha, 'Kp':Kp,'Ki':Ki,'covid_capacity':covid_capacity,'f_UZG':1}

        init_states = {'H_norm':np.ones(n)}
        coordinates={'MDC':MDC_keys}

        covid_H_func = get_covid_H(True,self.df_covid_H_tot,self.hospitalizations_baseline_mean_smooth,self.hospitalizations).H_wrapper_func
        dH_function = get_covid_dH(self.df_covid_dH).dH_wrapper_func

        constrained_PI_model = Constrained_PI_Model(init_states,params,coordinates,
                                time_dependent_parameters={'covid_H':covid_H_func,'covid_dH':dH_function})
        
        # init PI model
        alpha = 0.00003*np.ones(n)
        epsilon = 0.03*np.ones(n)

        Kp = 0.01*np.ones(n)
        Ki = 0.005*np.ones(n)

        start_date_string = start_date.strftime('%Y-%m-%d')
        covid_H = self.df_covid_H_tot.loc[start_date_string]

        params={'covid_H':covid_H,'epsilon':epsilon, 'alpha':alpha, 'Kp':Kp,'Ki':Ki,'f_UZG':1}

        init_states = {'H_norm':np.ones(n)}
        coordinates={'MDC':MDC_keys}

        covid_H_func = get_covid_H(True,self.df_covid_H_tot,self.hospitalizations_baseline_mean_smooth,self.hospitalizations).H_wrapper_func

        PI_model = PI_Model(init_states,params,coordinates,
                                time_dependent_parameters={'covid_H':covid_H_func})
        
        return queuing_model,constrained_PI_model,PI_model