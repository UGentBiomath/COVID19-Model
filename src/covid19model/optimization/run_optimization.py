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
from covid19model.data import polymod
from covid19model.data import model_parameters
from covid19model.visualization.optimization import traceplot

initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = polymod.get_interaction_matrices()

def full_calibration(model, timeseries, spatial_unit, start_date, end_beta, end_ramp,
                     fig_path, samples_path,
                     maxiter=50, popsize=50, steps_mcmc=10000):

    """
    model : object
        initialized model
    timeseries : Series
        data to fit with date in index
    spatial_unit : string
        name of the spatial_unit, e.g. Gent, Antwerp, Belgium
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

    lag_time = int(round(theta[1]))
    # Assign 'extraTime' or lag_time as a model attribute --> is needed to perform the optimalization
    model.extraTime = int(round(theta[1]))

    model.parameters.update({'beta': theta[2]})

    parNames_mcmc = ['sigma_data','beta'] # must be a list!
    bounds_mcmc=((1,200),(0.01,0.10))

    # run MCMC calibration
    pos = [theta[0],theta[2]] + [1, 1e-2 ]* np.random.randn(4, 2)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                                    args=(model, bounds_mcmc, data, states, parNames_mcmc))
    sampler.run_mcmc(pos, steps_mcmc, progress=True);

    samples_beta = sampler.get_chain(discard=100,flat=False)
    flat_samples_beta = sampler.get_chain(discard=100,flat=True)

    try:
        sampler.get_autocorr_time()
    except:
        print('Calibrating beta. Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s). Use this estimate with caution and run a longer chain!')


    traceplot(samples_beta,labels=['$\sigma_{data}$','$\\beta$'],plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
    plt.savefig(fig_path+'traceplots/beta_'+spatial_unit+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')

    fig = corner.corner(flat_samples_beta,labels=['$\sigma_{data}$','$\\beta$'])
    fig.set_size_inches(8, 8)
    plt.savefig(fig_path+'cornerplots/beta_'+spatial_unit+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')

    #############################################
    ####### CALIBRATING COMPLIANCE PARAMS #######
    #############################################

    samples_beta = {'beta': flat_samples_beta[:,1].tolist()}

    # Create checkpoints dictionary
    chk_beta_pso = {
        'time':  [lag_time],
        'Nc':    [0.2*Nc_home + 0.3*Nc_work + 0.2*Nc_transport],
    }
    # define dataset
    data=[timeseries[start_date:end_ramp]]
    # set optimisation settings
    parNames_pso2 = ['sigma_data','l','tau','prevention'] # must be a list!
    bounds_pso2=((1,100),(0.1,20),(0,20),(0,1)) # must be a list!
    # run optimisation
    theta = MCMC.fit_pso(model, data, parNames_pso2, states, bounds_pso2,
                         checkpoints=chk_beta_pso, samples=samples_beta, maxiter=maxiter,popsize=popsize)

    model.parameters.update({'l': theta[1], 'tau': theta[2]})
    prevention = theta[2]

    # Create checkpoints dictionary
    chk_beta_MCMC = {
        'time':  [lag_time],
        'Nc':    [prevention*(1.0*Nc_home + 0.4*Nc_work + 0.3*Nc_transport + 0.7*Nc_others + 0.2*Nc_leisure)]}


    bounds_mcmc2=((1,100),(0.001,20),(0,20),(0,1)) # must be a list!
    pos = theta + [1, 0.1, 0.1, 0.1 ]* np.random.randn(8, 4)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                                    args=(model,bounds_mcmc2,data,states,parNames_pso2,chk_beta_MCMC,samples_beta))

    sampler.run_mcmc(pos, steps_mcmc, progress=True);

    try:
        sampler.get_autocorr_time()
    except:
        print('Calibrating compliance ramp. Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s). Use this estimate with caution and run a longer chain!')


    samples_ramp = sampler.get_chain(discard=200,flat=False)
    flat_samples_ramp = sampler.get_chain(discard=200,flat=True)

    traceplot(samples_ramp, labels=["$\sigma_{data}$","l","$\\tau$","prevention"],
              plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
    plt.savefig(fig_path+'traceplots/ramp_'+spatial_unit+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')

    fig = corner.corner(flat_samples_ramp, labels=["$\sigma_{data}$","l","$\\tau$","$\Omega$"])
    fig.set_size_inches(9, 9)
    plt.savefig(fig_path+'cornerplots/ramp_'+spatial_unit+'_'+str(datetime.date.today())+'.pdf',
                dpi=600, bbox_inches='tight')

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
                  'l': flat_samples_ramp[:,1].tolist(),'tau':flat_samples_ramp[:,2].tolist(),
                  'prevention':flat_samples_ramp[:,3].tolist()}

    with open(samples_path+spatial_unit+'_'+str(datetime.date.today())+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    plt.ion()
    return samples_dict
