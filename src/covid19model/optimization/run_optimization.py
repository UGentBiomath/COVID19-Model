import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import scipy
from scipy.integrate import odeint
import matplotlib.dates as mdates
import matplotlib
import scipy.stats as st

import math
import xarray as xr
import emcee
import json
import corner

from covid19model.optimization import objective_fcns
from covid19model.optimization import MCMC
from covid19model.models import models
from covid19model.data import google
from covid19model.data import sciensano
from covid19model.data import model_parameters
from covid19model.visualization.optimization import traceplot

def full_calibration_wave1(model, timeseries, spatial_unit, start_date, end_beta, end_ramp,
                     fig_path, samples_path, initN, Nc_total,
                     maxiter=50, popsize=50, steps_mcmc=10000):

    """
    Function to calibrate the first wave in different steps with pso and mcmc
    Step 1: calibration of beta and lag_time
    Step 2: calibation of compliance parameters

    model : object
        initialized model
    timeseries : Series
        data to fit with date in index
    spatial_unit : string
        name of the spatial_unit, e.g. Gent, Antwerp, Belgium or NIS-code (for writing out files)
    start_date, end_beta, end_ramp : string, format YYYY-MM-DD
        date of first data point, last date for fitting beta and last date
        for fitting the compliance ramp
    fig_path : string
        path to folder where to save figures
    samples_path : string
        path to folder where to save samples
    maxiter: int (default 50)
        maximum number of pso iterations
    popsize: int (default 50)
        population size of particle swarm
        increasing this variable lowers the chance of finding local minima but
        slows down calculations
    steps_mcmc : int (default 10000)
        number of steps in MCMC calibration


    """
    plt.ioff()
    # define dataset
    data=[timeseries[start_date:end_beta]]
    states = [["H_in"]]

    #############################################
    ####### CALIBRATING BETA AND LAG_TIME #######
    #############################################
    # set optimisation settings
    parNames_pso = ['sigma_data','extraTime','beta'] # must be a list!
    bounds_pso=((1,100),(30,60),(0.02,0.06)) # must be a list!
    # run pso optimisation
    theta = MCMC.fit_pso(model,data,parNames_pso,states,bounds_pso,maxiter=maxiter,popsize=popsize)

    sigma_data = theta[0]
    lag_time = int(round(theta[1]))
    beta = theta[2]
    # Assign 'extraTime' or lag_time as a model attribute --> is needed to perform the optimalization
    model.extraTime = lag_time
    model.parameters.update({'beta': beta})
    

    # run MCMC calibration

    parNames_mcmc = ['sigma_data','beta'] # must be a list!
    bounds_mcmc=((1,200),(0.01,0.10))

    pos = [sigma_data,beta] + [1, 1e-2 ]* np.random.randn(4, 2)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                                    args=(model, bounds_mcmc, data, states, parNames_mcmc))
    sampler.run_mcmc(pos, steps_mcmc, progress=True);

    samples_beta = sampler.get_chain(discard=500,flat=False)
    flat_samples_beta = sampler.get_chain(discard=500,flat=True)

    try:
        sampler.get_autocorr_time()
    except:
        print('Calibrating beta. Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s). Use this estimate with caution and run a longer chain!')


    traceplot(samples_beta,labels=['$\sigma_{data}$','$\\beta$'],plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
    plt.savefig(fig_path+'traceplots/beta_'+str(spatial_unit)+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')

    fig = corner.corner(flat_samples_beta,labels=['$\sigma_{data}$','$\\beta$'])
    fig.set_size_inches(8, 8)
    plt.savefig(fig_path+'cornerplots/beta_'+str(spatial_unit)+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')
    
    samples_beta = {'beta': flat_samples_beta[:,1].tolist()}

    model.parameters.update({'policy_time': lag_time})

    #############################################
    ####### CALIBRATING COMPLIANCE PARAMS #######
    #############################################
    l = None
    tau = None
    prevention = None

    # define dataset
    data=[timeseries[start_date:end_ramp]]
    # set optimisation settings
    parNames_pso2 = ['sigma_data','l','tau','prevention'] # must be a list!
    bounds_pso2=((1,100),(0.1,20),(0,20),(0,1)) # must be a list!
    # run optimisation
    theta_comp = MCMC.fit_pso(model, data, parNames_pso2, states, bounds_pso2,
                            samples=samples_beta, maxiter=maxiter,popsize=popsize)

    model.parameters.update({'l': theta_comp[1], 
                            'tau': theta_comp[2],
                            'prevention': theta_comp[3]})

    bounds_mcmc2=((1,100),(0.001,20),(0,20),(0,1)) # must be a list!
    pos = theta_comp + [1, 0.1, 0.1, 0.1 ]* np.random.randn(8, 4)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                                    args=(model,bounds_mcmc2,data,states,parNames_pso2,samples_beta))

    sampler.run_mcmc(pos, steps_mcmc, progress=True);

    try:
        sampler.get_autocorr_time()
    except:
        print('Calibrating compliance ramp. Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s). Use this estimate with caution and run a longer chain!')


    samples_ramp = sampler.get_chain(discard=200,flat=False)
    flat_samples_ramp = sampler.get_chain(discard=200,flat=True)

    traceplot(samples_ramp, labels=["$\sigma_{data}$","l","$\\tau$","prevention"],
                plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
    plt.savefig(fig_path+'traceplots/ramp_'+str(spatial_unit)+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')

    fig = corner.corner(flat_samples_ramp, labels=["$\sigma_{data}$","l","$\\tau$","$\Omega$"])
    fig.set_size_inches(9, 9)
    plt.savefig(fig_path+'cornerplots/ramp_'+str(spatial_unit)+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')
    
    sigma_data = flat_samples_ramp[:,0].tolist()
    l = flat_samples_ramp[:,1].tolist()
    tau = flat_samples_ramp[:,2].tolist()
    prevention = flat_samples_ramp[:,3].tolist()

    #############################################
    ####### CALCULATING R0 ######################
    #############################################

    R0 =[]
    for i in range(len(samples_beta['beta'])):
        R0.append(sum((model.parameters['a']*model.parameters['da']+model.parameters['omega'])*samples_beta['beta'][i]*model.parameters['s']*np.sum(Nc_total,axis=1)*(initN/sum(initN))))
    R0_stratified = np.zeros([initN.size,len(samples_beta['beta'])])
    for i in range(len(samples_beta['beta'])):
        R0_stratified[:,i]= (model.parameters['a']*model.parameters['da']+model.parameters['omega'])*samples_beta['beta'][i]*model.parameters['s']*np.sum(Nc_total,axis=1)
    R0_stratified_dict = pd.DataFrame(R0_stratified).T.to_dict(orient='list')

    samples_dict={'calibration_data':states[0][0], 'start_date':start_date,
                  'end_beta':end_beta, 'end_ramp':end_ramp,
                  'maxiter': maxiter, 'popsize':popsize, 'steps_mcmc':steps_mcmc,
                  'R0':R0, 'R0_stratified_dict':R0_stratified_dict,
                  'lag_time': lag_time, 'beta': samples_beta['beta'],
                  'sigma_data':sigma_data, 'l': l,'tau': tau,
                  'prevention':prevention}

    with open(samples_path+str(spatial_unit)+'_'+str(datetime.date.today())+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    plt.ion()
    return samples_dict


def full_calibration_wave2(model, timeseries, spatial_unit, start_date, end_beta, 
                           beta_init, sigma_data_init, beta_norm_params, sigma_data_norm_params, 
                           fig_path, samples_path,initN, Nc_total,
                           steps_mcmc=10000):

    """

    Function to calibrate the second wave: only mcmc, 
    based on initial values for beta and sigma_data from the first waves
    Only beta is calibrated in this function.

    model : object
        initialized model
    timeseries : Series
        data to fit with date in index
    spatial_unit : string
        name of the spatial_unit, e.g. Gent, Antwerp, Belgium
    start_date, end_beta : string, format YYYY-MM-DD
        date of first data point, last date for fitting beta and last date
        for fitting the compliance ramp
    fig_path : string
        path to folder where to save figures
    samples_path : string
        path to folder where to save samples
    steps_mcmc : int (default 10000)
        number of steps in MCMC calibration


    """
    plt.ioff()
    # define dataset
    data=[timeseries[start_date:end_beta]]
    states = [["H_in"]]

    #############################################
    ############# CALIBRATING BETA ##############
    #############################################
    lag_time = 0
    model.extraTime = lag_time
    model.parameters.update({'beta': beta_init})
    
    # run MCMC calibration

    parNames_mcmc = ['sigma_data','beta'] # must be a list!
    norm_params = (sigma_data_norm_params, beta_norm_params)
    bounds_mcmc = ((1,200),(0.01,0.10))

    pos = [sigma_data_init, beta_init] + [1, 1e-2 ]* np.random.randn(4, 2)
    nwalkers, ndim = pos.shape
    if beta_norm_params is not None: # use normal prior
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability_normal,
                                    args=(model, norm_params, data, states, parNames_mcmc))
    else: # use uniform prior
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                                    args=(model, bounds_mcmc, data, states, parNames_mcmc))
    sampler.run_mcmc(pos, steps_mcmc, progress=True);

    samples_beta = sampler.get_chain(discard=500,flat=False)
    flat_samples_beta = sampler.get_chain(discard=500,flat=True)

    try:
        sampler.get_autocorr_time()
    except:
        print('Calibrating beta. Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s). Use this estimate with caution and run a longer chain!')


    traceplot(samples_beta,labels=['$\sigma_{data}$','$\\beta$'],plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
    plt.savefig(fig_path+'traceplots/beta_'+str(spatial_unit)+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')

    fig = corner.corner(flat_samples_beta,labels=['$\sigma_{data}$','$\\beta$'])
    fig.set_size_inches(8, 8)
    plt.savefig(fig_path+'cornerplots/beta_'+str(spatial_unit)+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')
    
    samples_beta = {'beta': flat_samples_beta[:,1].tolist()}

       #############################################
    ####### CALCULATING R0 ######################
    #############################################

    R0 =[]
    for i in range(len(samples_beta['beta'])):
        R0.append(sum((model.parameters['a']*model.parameters['da']+model.parameters['omega'])*samples_beta['beta'][i]*model.parameters['s']*np.sum(Nc_total,axis=1)*(initN/sum(initN))))
    R0_stratified = np.zeros([initN.size,len(samples_beta['beta'])])
    for i in range(len(samples_beta['beta'])):
        R0_stratified[:,i]= (model.parameters['a']*model.parameters['da']+model.parameters['omega'])*samples_beta['beta'][i]*model.parameters['s']*np.sum(Nc_total,axis=1)
    R0_stratified_dict = pd.DataFrame(R0_stratified).T.to_dict(orient='list')

    samples_dict={'calibration_data':states[0][0], 'start_date':start_date,
                  'end_beta':end_beta,
                  'steps_mcmc':steps_mcmc,
                  'R0':R0, 'R0_stratified_dict':R0_stratified_dict,
                  'lag_time': lag_time, 'beta': samples_beta['beta'],
                  'sigma_data':sigma_data}

    with open(samples_path+str(spatial_unit)+'_'+str(datetime.date.today())+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    plt.ion()
    return samples_dict
