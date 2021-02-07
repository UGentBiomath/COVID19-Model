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
from covid19model.optimization import pso
from covid19model.models import models
from covid19model.models.time_dependant_parameter_fncs import google_lockdown
from covid19model.data import sciensano
from covid19model.data import model_parameters
from covid19model.visualization.optimization import traceplot, autocorrelation_plot
from covid19model.models.utils import draw_sample_COVID19_SEIRD_google

def checkplots(sampler, discard, thin, fig_path, spatial_unit, figname, labels):
    
    samples = sampler.get_chain(discard=discard,thin=thin,flat=False)
    flatsamples = sampler.get_chain(discard=discard,thin=thin,flat=True)
    
    # Traceplots of samples
    traceplot(samples,labels=labels,plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
    plt.savefig(fig_path+'traceplots/'+str(spatial_unit)+'_TRACE_'+figname+'_'+str(datetime.date.today())+'.pdf',
                dpi=400, bbox_inches='tight')

    # Autocorrelation plots of chains
    autocorrelation_plot(samples)
    plt.savefig(fig_path+'autocorrelation/'+str(spatial_unit)+'_AUTOCORR_'+figname+'_'+str(datetime.date.today())+'.pdf',
                dpi=400, bbox_inches='tight')

    # Cornerplots of samples
    fig = corner.corner(flatsamples,labels=labels)
    plt.savefig(fig_path+'cornerplots/'+str(spatial_unit)+'_CORNER_'+figname+'_'+str(datetime.date.today())+'.pdf',
                dpi=400, bbox_inches='tight')

    return

def samples_dict_to_emcee_chain(samples_dict,keys,n_chains,discard=0,thin=1):
    """
    A function to convert a samples dictionary into a 2D and 3D np.array, similar to using the emcee method `sampler.get_chain()`

    Parameters
    ----------
    samples_dict : dict
        Dictionary containing MCMC samples
    
    keys : lst
        List containing the names of the sampled parameters

    n_chains: int
        Number of parallel Markov Chains run during the inference

    discard: int
        Number of samples to be discarded from the start of each Markov chain (=burn-in).

    thin: int
        Thinning factor of the Markov Chain. F.e. thin = 5 extracts every fifth sample from each chain.

    Returns
    -------
    samples : np.array
        A 3D np.array with dimensions:
            x: number of samples per Markov chain
            y: number of parallel Markov chains
            z: number of parameters
    flat_samples : np.array
        A 2D np.array with dimensions:
            x: total number of samples per Markov chain (= user defined number of samples per Markov Chain * number of parallel chains)
            y: number of parameters

    Example use
    -----------
    samples, flat_samples = samples_dict_to_emcee_chain(samples_dict, ['l', 'tau'], 4, discard=1000, thin=20)
    """

    # Convert to raw flat samples
    flat_samples_raw = np.zeros([len(samples_dict[keys[0]]),len(keys)])
    for idx,key in enumerate(keys):
        flat_samples_raw[:,idx] = samples_dict[key]
    # Convert to raw samples
    samples_raw = np.zeros([int(flat_samples_raw.shape[0]/n_chains),n_chains,flat_samples_raw.shape[1]])
    for i in range(samples_raw.shape[0]): # length of chain
        for j in range(samples_raw.shape[1]): # chain number
            samples_raw[i,:,:] = flat_samples_raw[i*n_chains:(i+1)*n_chains,:]
    # Do discard
    samples_discard = np.zeros([(samples_raw.shape[0]-discard),n_chains,flat_samples_raw.shape[1]])
    for i in range(samples_raw.shape[1]):
        for j in range(flat_samples_raw.shape[1]):
            samples_discard[:,i,j] = samples_raw[discard:,i,j]  

    # Do thin
    samples = samples_discard[::thin,:,:]
    # Convert to flat samples
    flat_samples = samples[:,0,:]
    for i in range(1,samples.shape[1]):
        flat_samples=np.append(flat_samples,samples[:,i,:],axis=0)

    return samples,flat_samples

def calculate_R0(samples_beta, model, initN, Nc_total):
    spatial=False
    N = initN.size
    sample_size = len(samples_beta['beta'])
    if 'place' in model.parameters.keys():
        spatial=True
        G = initN.shape[0]
        N = initN.shape[1]
        # Define values for 'normalisation' of contact matrices
        T_eff = np.zeros([G,N])
        for ii in range(N):
            for gg in range(G):
                som = 0
                for hh in range(G):
                    som += model.parameters['place'][hh][gg] * initN[hh][ii] # pi = 1 for calculation of R0
                T_eff[gg][ii] = som
        density = np.sum(T_eff,axis=1) / model.parameters['area']
        f = 1 + ( 1 - np.exp(-model.parameters['xi'] * density) )
        zi_denom = np.zeros(N)
        for ii in range(N):
            som = 0
            for hh in range(G):
                som += f[hh] * T_eff[hh][ii]
            zi_denom[ii] = som
        zi = np.sum(initN, axis=0) / zi_denom
        Nc_total_spatial = np.zeros([G,N,N])
        for ii in range(N):
            for jj in range(N):
                for hh in range(G):
                    Nc_total_spatial[hh][ii][jj] = zi[ii] * f[hh] * Nc_total[ii][jj]
        
    R0 =[]
    # Weighted average R0 value over all ages (and all places). This needs to be modified if beta is further stratified
    for j in range(sample_size):
        som = 0
        if spatial:
            for gg in range(G):
                for i in range(N):
                    som += (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * samples_beta['beta'][j] * \
                            model.parameters['s'][i] * np.sum(Nc_total_spatial, axis=2)[gg][i] * initN[gg][i]
            R0_temp = som / np.sum(initN)
        else:
            for i in range(N):
                som += (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * samples_beta['beta'][j] * \
                        model.parameters['s'][i] * np.sum(Nc_total, axis=1)[i] * initN[i]
            R0_temp = som / np.sum(initN)
        R0.append(R0_temp)
        
    # Stratified R0 value: R0_stratified[place][age][chain] or R0_stratified[age][chain]
    # This needs to be modified if 'beta' is further stratified
    R0_stratified_dict = dict({})
    if spatial:
        for gg in range(G):
            R0_stratified_dict[gg] = dict({})
            for i in range(N):
                R0_list = []
                for j in range(sample_size):
                    R0_temp = (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * \
                            samples_beta['beta'][j] * model.parameters['s'][i] * np.sum(Nc_total_spatial,axis=2)[gg][i]
                    R0_list.append(R0_temp)
                R0_stratified_dict[gg][i] = R0_list
    else:
        for i in range(N):
            R0_list = []
            for j in range(sample_size):
                R0_temp = (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * \
                        samples_beta['beta'][j] * model.parameters['s'][i] * np.sum(Nc_total,axis=1)[i]
                R0_list.append(R0_temp)
            R0_stratified_dict[i] = R0_list

    return R0, R0_stratified_dict


def google_calibration_wave1(model, timeseries, spatial_unit, start_data, end_beta_ramp, start_recalibrate_beta, end_recalibrate_beta, fig_path, samples_path, initN, Nc_total,warmup=0,
                     maxiter=50, popsize=50, n=30, steps_mcmc=10000, discard=500, omega=0.8, phip=0.8, phig=0.8, processes=-1):

    plt.ioff()

    ####################################################
    ####### CALIBRATING BETA AND COMPLIANCE RAMP #######
    ####################################################

    print('------------------------------------')
    print('CALIBRATING BETA AND COMPLIANCE RAMP')
    print('------------------------------------\n')
    print('Using data from '+start_data+' until '+end_beta_ramp+'\n')
    print('1) Particle swarm optimization\n')

    # define dataset
#     ts = moving_avg(timeseries, days=avg_window, win_type=None, params=None).T.squeeze()
    data=[timeseries[start_data:end_beta_ramp].squeeze().values]
    states = [["H_in"]]

    # set PSO optimisation settings
    parNames = ['sigma_data','beta','l','tau']
    bounds=((30,200),(0.030,0.040),(0.01,20),(0.01,20))
    # run PSO optimisation
    if processes == -1: # use all but one processor
        theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,start_date=start_data,warmup=warmup)
    else: # use indicated number of processors
        theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,start_date=start_data,warmup=warmup, processes=processes)
        
    # run MCMC sampler
    print('\n2) Markov-Chain Monte-Carlo sampling\n')
    parNames_mcmc = parNames
    bounds_mcmc=((1,200),(0.020,0.060),(0.01,20),(0.01,20))

    pos = theta + [1, 1e-3, 1e-3, 1e-3 ]* np.random.randn(8, 4)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                     args=(model, bounds_mcmc, data, states, parNames_mcmc, None, start_data, warmup))
    sampler.run_mcmc(pos, steps_mcmc, progress=True)
    # Check chain length
    thin = 0
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s).\nUse this estimate with caution and run a longer chain!')
        
    # Make and save diagnostic visualizations
    checkplots(sampler, discard, thin, fig_path, spatial_unit, 
                figname='BETA_RAMP_GOOGLE_WAVE1_ ', labels=['$\sigma_{data}$','$\\beta$','l','$\\tau$'])

    # Save output in parameter dictionary
    print('\n3) Saving chains\n')

    samples_dict = {
        'warmup': warmup,
        'beta': sampler.get_chain(discard=discard,flat=True)[:,1].tolist(),
        'l': sampler.get_chain(discard=discard,flat=True)[:,2].tolist(),
        'tau': sampler.get_chain(discard=discard,flat=True)[:,3].tolist(),
        'sigma_data': sampler.get_chain(discard=discard,flat=True)[:,0].tolist(),
        'calibration_data': states[0][0],
        'start_data': start_data,
        'end_beta_ramp': end_beta_ramp,
        'maxiter': maxiter,
        'popsize': popsize,
        'steps_mcmc': steps_mcmc,
        'discard' : discard
    }

    #############################################
    ####### CALCULATING R0 ######################
    #############################################

    print('\n4) Computing R0\n')

    R0, R0_stratified_dict = calculate_R0(samples_dict, model, initN, Nc_total)

    samples_dict.update({
        'R0': R0,
        'R0_stratified_dict': R0_stratified_dict,
    })

    ###################################
    ####### RECALIBRATING BETA  #######
    ###################################

    print('------------------')
    print('RECALIBRATING BETA')
    print('------------------\n')
    print('Using data from '+start_recalibrate_beta+' until '+end_recalibrate_beta+'\n')
    print('1) Computing model states on ' + start_recalibrate_beta)

    out = model.sim('2020-08-01',start_date=start_data,excess_time=samples_dict['warmup'],N=n,draw_fcn=draw_sample_COVID19_SEIRD_google,samples=samples_dict)

    print('2) Re-initialize model on ' + start_recalibrate_beta)

    model.initial_states = {'S': out['S'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'E': out['E'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'I': out['I'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'A': out['A'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'M': out['M'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'ER': out['ER'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'C': out['C'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'C_icurec': out['C_icurec'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'ICU': out['ICU'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'R': out['R'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'D': out['D'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'H_in': out['H_in'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'H_out': out['H_out'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta)),
                  'H_tot': out['H_tot'].mean(dim="draws").sel(time=pd.to_datetime(start_recalibrate_beta))}

    print('3) Particle swarm optimization\n')

    # define dataset
    data=[timeseries[start_recalibrate_beta:end_recalibrate_beta]]
    states = [["H_in"]]

    # set PSO optimisation settings
    parNames = ['sigma_data','beta']
    bounds=((1,100),(0.010,0.060))
    # run PSO optimisation
    if processes == -1:
        theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,start_date=start_recalibrate_beta,warmup=0)
    else:
        theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,start_date=start_recalibrate_beta,warmup=0, processes=processes)
        
    # run MCMC sampler
    print('\n4) Markov-Chain Monte-Carlo sampling\n')
    parNames_mcmc = parNames
    bounds_mcmc=((1,200),(0.010,0.060))

    pos = theta + [1, 1e-4]* np.random.randn(4, 2)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                        args=(model, bounds_mcmc, data, states, parNames_mcmc, None, start_recalibrate_beta, 0))
    sampler.run_mcmc(pos, steps_mcmc, progress=True)
    # Check chain length
    thin = 0
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s).\nUse this estimate with caution and run a longer chain!')
    # Make and save diagnostic visualizations
    checkplots(sampler, discard, thin, fig_path, spatial_unit, 
                figname='BETA_RECALIBRATE_GOOGLE_', labels=['$\sigma_{data}$','$\\beta$'])

    print('\n5) Saving chains\n')
    samples_dict.update({
        'beta_summer': sampler.get_chain(discard=discard,flat=True)[:,1].tolist(),
        'start_recalibrate_beta': start_recalibrate_beta,
        'end_recalibrate_beta': end_recalibrate_beta})

    with open(samples_path+str(spatial_unit)+'_'+str(datetime.date.today())+'_google.json', 'w') as fp:
        json.dump(samples_dict, fp)

    #####################################
    ####### SAVING END OF WAVE 1  #######
    #####################################

    print('-------------------------------------')
    print('COMPUTING MODEL STATES ON ' + end_recalibrate_beta)
    print('-------------------------------------\n')
    print('1) Initializing COVID-19 SEIRD\n')

    def switch_beta(t,param,samples_dict):
        if t < pd.to_datetime('2020-05-04'):
            return np.random.choice(samples_dict['beta'],1,replace=False)
        elif pd.to_datetime('2020-05-04') < t <= pd.to_datetime('2020-09-01'):
            return np.random.choice(samples_dict['beta_summer'],1,replace=False)
        else:
            return np.random.choice(samples_dict['beta'],1,replace=False)

    params = model.parameters
    params.update({'samples_dict': samples_dict})

    # Define the initial condition: one exposed inidividual in every age category
    initial_states = {'S': initN, 'E': np.ones(9)}
    # Initialize the model
    model = models.COVID19_SEIRD(initial_states, params, time_dependent_parameters={'Nc': google_lockdown, 'beta': switch_beta})

    print('2) Simulating COVID-19 SEIRD ' + str(n) + ' times\n')
    print('from ' + start_data + ' until ' + end_recalibrate_beta+'\n')

    # Simulate the model until start_calibration
    out = model.sim(end_recalibrate_beta,start_date=start_data,excess_time=samples_dict['warmup'],N=n,draw_fcn=draw_sample_COVID19_SEIRD_google,samples=samples_dict)
    # Define new initial states
    initial_states = {'S': out['S'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'E': out['E'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'I': out['I'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'A': out['A'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'M': out['M'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'ER': out['ER'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'C': out['C'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'C_icurec': out['C_icurec'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'ICU': out['ICU'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'R': out['R'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'D': out['D'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'H_in': out['H_in'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'H_out': out['H_out'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist(),
                  'H_tot': out['H_tot'].mean(dim="draws").sel(time=pd.to_datetime(end_recalibrate_beta)).values.tolist()
                  }

    print('3) Dumping results\n')
    # Dump initial condition in a .json
    with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_'+end_recalibrate_beta+'.json', 'w') as fp:
        json.dump(initial_states,fp)

    plt.ion()

    print('DONE\n')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_'+str(datetime.date.today())+'_WAVE1_GOOGLE.json'+'"\n')

    return samples_dict

    print('---------------------------------------------------------------------------------------------------------\n')


def full_calibration_wave1(model, timeseries, spatial_unit, start_date, end_beta, end_ramp,
                     fig_path, samples_path, initN, Nc_total,
                     maxiter=50, popsize=50, steps_mcmc=10000, discard=500, omega=0.8, phip=0.8, phig=0.8, processes=-1):

    """
    Function to calibrate the first wave in different steps with pso and mcmc
    Step 1: calibration of beta and warmup
    Step 2: calibation of compliance parameters

    Parameters
    ----------
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
    initN : int
        total population in spatial unit
    Nc_total : array
        general contact matrix
    maxiter: int (default 100)
        maximum number of pso iterations
    popsize: int (default 50)
        population size of particle swarm
        increasing this variable increases the chance of finding local minima but
        slows down calculations
    steps_mcmc : int (default 10000)
        number of steps in MCMC calibration
    processes : int
        number of processors used in the PSO. -1 means "use all but one" (default).

    
    Returns
    -------
    samples_dict: dictionary
        dictionary with keys 'warmup', 'beta', 'l', 'tau' and 'prevention', as well as some other lists
        and values that are of interest for inspecting the working of the optimization.
    
    """
    plt.ioff()
    # define dataset
    data=[timeseries[start_date:end_beta]]
    states = [["H_in"]]

    #############################################
    ####### CALIBRATING BETA AND warmup #######
    #############################################

    if processes == -1:
        nr_processes = 'all but one'
    else:
        nr_processes = str(processes)

    print(f'Using {nr_processes} logical processors\n')
    print('---------------------------')
    print('CALIBRATING BETA AND WARMUP')
    print('---------------------------\n')
    print('1) Particle swarm optimization\n')

    # set optimisation settings
    parNames_pso = ['sigma_data','warmup','beta'] # must be a list!
    bounds_pso=((1,100),(40,70),(0.025,0.04)) # must be a list!
    # run pso optimisation
    if processes == -1:
        theta = pso.fit_pso(model,data,parNames_pso,states,bounds_pso,maxiter=maxiter,popsize=popsize,
                        start_date=start_date, omega=omega, phip=phip, phig=phig)
    else:
        theta = pso.fit_pso(model,data,parNames_pso,states,bounds_pso,maxiter=maxiter,popsize=popsize,
                        start_date=start_date, omega=omega, phip=phip, phig=phig, processes=processes)
    sigma_data = theta[0]
    warmup = int(round(theta[1]))
    beta = theta[2]
    model.parameters.update({'beta': beta})


    # run MCMC calibration
    print('\n2) Markov-Chain Monte-Carlo sampling\n')
    parNames_mcmc = ['sigma_data','beta'] # must be a list!
    bounds_mcmc=((1,200),(0.01,0.10))

    pos = [sigma_data,beta] + [1, 1e-2 ]* np.random.randn(4, 2)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                     args=(model, bounds_mcmc, data, states, parNames_mcmc, None, start_date, warmup))
    sampler.run_mcmc(pos, steps_mcmc, progress=True);

    thin = 0
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s).\nUse this estimate with caution and run a longer chain!')

    checkplots(sampler, discard, thin, fig_path, spatial_unit, 
                figname='beta_', labels=['$\sigma_{data}$','$\\beta$'])

    samples_dict = {'warmup': warmup,
                    'beta': sampler.get_chain(discard=discard,flat=True)[:,1].tolist()}

    print('---------------------------------------------------------------------------------------------------------\n')

    #############################################
    ####### CALIBRATING COMPLIANCE PARAMS #######
    #############################################

    print('CALIBRATING COMPLIANCE RAMP')
    print('---------------------------\n')
    print('1) Particle swarm optimization\n')

    # define dataset
    data=[timeseries[start_date:end_ramp]]
    # set optimisation settings
    parNames_pso2 = ['sigma_data','l','tau','prevention'] # must be a list!
    bounds_pso2=((1,100),(0.1,20),(0,20),(0,1)) # must be a list!

    # Import a function to draw values of beta and assign them to the model parameter dictionary
    from covid19model.models.utils import draw_sample_beta_COVID19_SEIRD
    global draw_sample_beta_COVID19_SEIRD

    # run optimisation
    print('\n2) Markov-Chain Monte-Carlo sampling\n')
    if processes == -1:
        theta_comp = pso.fit_pso(model, data, parNames_pso2, states, bounds_pso2,
                            draw_fcn=draw_sample_beta_COVID19_SEIRD, samples=samples_dict, maxiter=maxiter,popsize=popsize, start_date=start_date, warmup=warmup)
    else:
        theta_comp = pso.fit_pso(model, data, parNames_pso2, states, bounds_pso2,
                            draw_fcn=draw_sample_beta_COVID19_SEIRD, samples=samples_dict, maxiter=maxiter,popsize=popsize, start_date=start_date, warmup=warmup, processes=processes)

    model.parameters.update({'l': theta_comp[1],
                            'tau': theta_comp[2],
                            'prevention': theta_comp[3]})

    bounds_mcmc2=((1,100),(0.001,20),(0,20),(0,1)) # must be a list!
    pos = theta_comp + [1, 0.1, 0.1, 0.1 ]* np.random.randn(8, 4)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                                    args=(model,bounds_mcmc2,data,states,parNames_pso2,samples_dict, start_date, warmup))
    sampler.run_mcmc(pos, steps_mcmc, progress=True)

    # Check autocorrelation time as a measure of the adequacy of the sample size
    thin = 0
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s). Use this estimate with caution and run a longer chain!')
        
    checkplots(sampler, discard, thin, fig_path, spatial_unit, 
                figname='ramp_', labels=["$\sigma_{data}$","l","$\\tau$","prevention"])
    print('---------------------------------------------------------------------------------------------------------\n')

    #############################################
    ####### CALCULATING R0 ######################
    #############################################

    R0, R0_stratified_dict = calculate_R0(samples_dict, model, initN, Nc_total)

    #############################################
    ####### Output to dictionary ################
    #############################################

    samples_dict.update({'l': sampler.get_chain(discard=discard,flat=True)[:,1].tolist(),
                        'tau': sampler.get_chain(discard=discard,flat=True)[:,2].tolist(),
                        'prevention': sampler.get_chain(discard=discard,flat=True)[:,3].tolist(),
                        'sigma_data': sampler.get_chain(discard=discard,flat=True)[:,0].tolist(),
                        'calibration_data':states[0][0],
                        'start_date':start_date,
                        'end_beta':end_beta,
                        'end_ramp':end_ramp,
                        'maxiter': maxiter,
                        'popsize': popsize,
                        'steps_mcmc': steps_mcmc,
                        'discard' : discard,
                        'R0': R0,
                        'R0_stratified_dict': R0_stratified_dict,
    })

    with open(samples_path+str(spatial_unit)+'_'+str(datetime.date.today())+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    plt.ion()
    print('DONE\n')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_'+str(datetime.date.today())+'.json'+'"')
    return samples_dict


def full_calibration_wave2(model, timeseries, spatial_unit, start_date, end_beta,
                           beta_init, sigma_data_init, beta_norm_params, sigma_data_norm_params,
                           fig_path, samples_path,initN, Nc_total,
                           steps_mcmc=10000, discard=500):

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

    warmup = 0
    model.parameters.update({'beta': beta_init})

    # run MCMC calibration
    parNames_mcmc = ['sigma_data','beta'] # must be a list!
    norm_params = (sigma_data_norm_params, beta_norm_params)
    bounds_mcmc = ((1,200),(0.0001,0.10))

    pos = [sigma_data_init, beta_init] + [1, 1e-2 ]* np.random.randn(4, 2)
    nwalkers, ndim = pos.shape

    if beta_norm_params is not None: # use normal prior
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability_normal,
                                    args=(model, norm_params, data, states, parNames_mcmc, None, start_date, warmup))
    else: # use uniform prior
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
                                    args=(model, bounds_mcmc, data, states, parNames_mcmc, None, start_date, warmup))
    sampler.run_mcmc(pos, steps_mcmc, progress=True);

    # Check autocorrelation time as a measure of the adequacy of the sample size
    thin = 0
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Calibrating beta. Warning: The chain is shorter than 50 times the integrated autocorrelation time for 4 parameter(s). Use this estimate with caution and run a longer chain!')

    checkplots(sampler, discard, thin, fig_path, spatial_unit, 
                figname='beta_', labels=['$\sigma_{data}$','$\\beta$'])

    samples_dict = {'warmup': warmup,
                    'beta': sampler.get_chain(discard=discard,flat=True)[:,1].tolist()}

    #############################################
    ####### CALCULATING R0 ######################
    #############################################

    R0, R0_stratified_dict = calculate_R0(samples_dict, model, initN, Nc_total)

    samples_dict.update({
        'calibration_data': states[0][0],
        'start_date': start_date,
        'end_beta': end_beta,
        'steps_mcmc': steps_mcmc,
        'discard': discard,
        'R0': R0,
        'R0_stratified_dict': R0_stratified_dict,
    })

    #############################################
    ####### Output to dictionary ################
    #############################################

    with open(samples_path+str(spatial_unit)+'_'+str(datetime.date.today())+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    plt.ion()
    return samples_dict
