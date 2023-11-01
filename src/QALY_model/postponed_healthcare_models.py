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
from pySODM.models.base import ODE
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

class Queuing_Model(ODE):

    states = ['W','H','H_adjusted','H_norm','R','NR','X']
    parameters = ['X_tot','f_UZG','covid_H','alpha','post_processed_H']
    stratified_parameters = ['A','gamma','epsilon','sigma']
    dimensions = ['MDC']
    
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
    
class Constrained_PI_Model(ODE):

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
    
class PI_Model(ODE):
    
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