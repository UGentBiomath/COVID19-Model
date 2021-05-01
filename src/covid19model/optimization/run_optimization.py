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
from covid19model.data import sciensano
from covid19model.data import model_parameters
from covid19model.visualization.optimization import traceplot, autocorrelation_plot
from covid19model.models.utils import draw_sample_COVID19_SEIRD_google
from covid19model.models.utils import stratify_beta # used to determine whether the local region

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

def calculate_R0(samples_beta, model, initN, Nc_total, agg=None):
    """
    Function to calculate the initial R value, based on prepandemic social contact and a dictionary of infectivity values.
    TO DO: the syntax of this function is very unpythonic.
    
    Input
    -----
    samples_beta: dict
        Dictionary with i.a. infectivity samples from MCMC-based calibration
    model: covid19model.models.models
        Model that contains the parameters as properties
    initN: np.array
        Initial population per age (and per region if agg==True)
    Nc_total: np.array
        Intergenerational contact matrices
    agg: str
        If not None (default), choose between 'arr', 'prov' or 'mun', depending on spatial aggregation
    
    Return
    ------
    R0 : float
        Resulting R0 value
    R0_stratified_dict: dict of float
        Resulting R0 value per age (and per region if agg==True)    
    """
    
    if agg:
        
        # Read areas per region, ordered in ascending NIS values
        abs_dir = os.path.dirname(__file__)
        area_data = f'../../../data/interim/demographic/area_{agg}.csv'
        area_df=pd.read_csv(os.path.join(abs_dir, area_data), index_col='NIS')
        # Make sure the regions are ordered well
        area_df=area_df.sort_index(axis=0)
        area=area_df.values[:,0]
        areas = area * 1e-6 # in square kilometer
        
        # Take total population per region
        pops = initN.sum(axis=1)
        
        beta = stratify_beta('beta_R','beta_U', 'beta_M', agg, areas, pops) # name at correct spatial index
        sample_size = len(samples_beta['beta_M']) # or beta_U or beta_R, it's the same.
        G = initN.shape[0]
        N = initN.shape[1]
    else:
        sample_size = len(samples_beta['beta'])
        N = initN.size
        
    if agg:
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
        if agg:
            for gg in range(G):
                for i in range(N):
                    som += (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * samples_beta[beta[gg]][j] * \
                            model.parameters['s'][i] * np.sum(Nc_total_spatial, axis=2)[gg][i] * initN[gg][i]
            R0_temp = som / np.sum(initN)
        else:
            for i in range(N):
                som += (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * samples_beta[beta[gg]][j] * \
                        model.parameters['s'][i] * np.sum(Nc_total, axis=1)[i] * initN[i]
            R0_temp = som / np.sum(initN)
        R0.append(R0_temp)
        
    # Stratified R0 value: R0_stratified[place][age][chain] or R0_stratified[age][chain]
    # This needs to be modified if 'beta' is further stratified
    R0_stratified_dict = dict({})
    if agg:
        for gg in range(G):
            R0_stratified_dict[gg] = dict({})
            for i in range(N):
                R0_list = []
                for j in range(sample_size):
                    R0_temp = (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * \
                            samples_beta[beta[gg]][j] * model.parameters['s'][i] * np.sum(Nc_total_spatial,axis=2)[gg][i]
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
