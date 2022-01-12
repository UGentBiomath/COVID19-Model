"""
This script contains code to estimate an appropriate initial condition of the spatial COVID-19 SEIQRD model on March 15th-17th, 2020.
Two initial conditions are saved: 1) for the "virgin" model (COVID19_SEIQRD_spatial) and 2) for the stratified vaccination model (COVID19_SEIQRD_spatial_stratified_vacc)

The model is initialized 31 days prior to March 15th, 2020 with one exposed individual in every of the 10 (!) age groups.
The infectivity that results in the best fit to the hospitalization data is determined using PSO.
Next, the fit is visualized to allow for further manual tweaking of the PSO result.
Finally, the model states on March 15,16 and 17 are pickled.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import sys
import click
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.data import sciensano
from covid19model.optimization.pso import *
from covid19model.optimization.utils import assign_PSO, plot_PSO
from covid19model.visualization.output import _apply_tick_locator 

############################
## Parse script arguments ##
############################

parser = argparse.ArgumentParser()
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr (default) or prov.")
args = parser.parse_args()

# Maximum number of PSO iterations
n_pso = int(args.n_pso)
# Number of age groups used in the model: default is maximum
age_stratification_size=10
# Use public data by default
public=False
# Update data
update = False

#############################
## Define results location ##
#############################

# Path where the pickle with initial conditions should be stored
results_path = f'../../data/interim/model_parameters/COVID19_SEIQRD/initial_conditions/{args.agg}/'
# Generate path
if not os.path.exists(results_path):
    os.makedirs(results_path)

#############################################################
## Define a custom version of the spatially-explicit model ##
#############################################################

from covid19model.models.base import BaseModel

class custom_COVID19_SEIQRD_spatial_stratified_vacc(BaseModel):
    """
    A version of the spatially-, age-, and dose-stratified COVID-19 SEIQRD model that uses only one 'beta', which is intended to be a vector with dimension equal to the size of the spatial stratification
    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'C_icurec', 'ICU', 'R', 'D', 'H_in', 'H_out', 'H_tot']
    parameter_names = ['beta', 'alpha', 'K_inf1', 'K_inf2', 'K_hosp', 'sigma', 'omega', 'zeta', 'da', 'dm', 'dc_R', 'dc_D', 'dICU_R', 'dICU_D', 'dICUrec', 'dhospital', 'N_vacc', 'e_i', 'e_s', 'e_h', 'd_vacc', 'Nc_work']
    parameters_stratified_names = [['area', 'p'], ['s','a','h', 'c', 'm_C','m_ICU'],[]]
    stratification = ['place','Nc','doses'] # mobility and social interaction: name of the dimension (better names: ['nis', 'age'])
    coordinates = ['place', None, None] # 'place' is interpreted as a list of NIS-codes appropriate to the geography

    # ..transitions/equations
    @staticmethod

    def integrate(t, S, E, I, A, M, C, C_icurec, ICU, R, D, H_in, H_out, H_tot, # time + SEIRD classes
                  beta, alpha, K_inf1, K_inf2, K_hosp, sigma, omega, zeta, da, dm, dc_R, dc_D, dICU_R, dICU_D, dICUrec, dhospital, N_vacc, e_i, e_s, e_h, d_vacc, Nc_work,# SEIRD parameters
                  area, p,  # spatially stratified parameters. 
                  s, a, h, c, m_C, m_ICU, # age-stratified parameters
                  place, Nc, doses): # stratified parameters that determine stratification dimensions

        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        K_inf = np.array([1, K_inf1, K_inf2])
        if sum(alpha) != 1:
            raise ValueError(
                "The sum of the fractions of the VOCs is not equal to one, please check your time dependant VOC function"
            )
        h = np.sum(np.outer(h, alpha*K_hosp),axis=1)
        e_i = np.matmul(alpha, e_i)
        e_s = np.matmul(alpha, e_s)
        e_h = np.matmul(alpha, e_h)

        ####################################################
        ## Expand dims on first stratification axis (age) ##
        ####################################################

        a = np.expand_dims(a, axis=1)
        h = np.expand_dims(h, axis=1)
        c = np.expand_dims(c, axis=1)
        m_C = np.expand_dims(m_C, axis=1)
        m_ICU = np.expand_dims(m_ICU, axis=1)
        dc_R = np.expand_dims(dc_R, axis=1)
        dc_D = np.expand_dims(dc_D, axis=1)
        dICU_R = np.expand_dims(dICU_R, axis=1)
        dICU_D = np.expand_dims(dICU_D, axis=1)
        dICUrec = np.expand_dims(dICUrec, axis=1)

        ############################################
        ## Compute the vaccination transitionings ##
        ############################################

        dS = np.zeros(S.shape)
        dR = np.zeros(R.shape)

        # 0 --> 1 and  0 --> 2
        # ~~~~~~~~~~~~~~~~~~~~
        # Compute vaccine eligible population
        VE = S[:,:,0] + R[:,:,0]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,0]/VE
        f_R = R[:,:,0]/VE
        # Compute transisitoning in zero syringes
        dS[:,:,0] = - (N_vacc[:,:,0] + N_vacc[:,:,2])*f_S 
        dR[:,:,0] = - (N_vacc[:,:,0]+ N_vacc[:,:,2])*f_R
        # Compute transitioning in one short circuit
        dS[:,:,1] =  N_vacc[:,:,0]*f_S # 0 --> 1 dose
        dR[:,:,1] =  N_vacc[:,:,0]*f_R # 0 --> 1 dose
        # Compute transitioning in two shot circuit
        dS[:,:,2] =  N_vacc[:,:,2]*f_S # 0 --> 2 doses
        dR[:,:,2] =  N_vacc[:,:,2]*f_R # 0 --> 2 doses

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,1] + E[:,:,1] + I[:,:,1] + A[:,:,1] + R[:,:,1]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,1]/VE
        f_R = R[:,:,1]/VE
        # Compute transitioning in one short circuit
        dS[:,:,1] = dS[:,:,1] - N_vacc[:,:,1]*f_S
        dR[:,:,1] = dR[:,:,1] - N_vacc[:,:,1]*f_R
        # Compute transitioning in two shot circuit
        dS[:,:,2] = dS[:,:,2] + N_vacc[:,:,1]*f_S
        dR[:,:,2] = dR[:,:,2] + N_vacc[:,:,1]*f_R

        # waned vaccine, 2 --> B
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,2]+ R[:,:,2] + S[:,:,3] + R[:,:,3]
        # 2 dose circuit
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,2]/VE
        f_R = R[:,:,2]/VE
        # Compute transitioning in two shot circuit
        dS[:,:,2] = dS[:,:,2] - N_vacc[:,:,3]*f_S
        dR[:,:,2] = dR[:,:,2] - N_vacc[:,:,3]*f_R
        # Compute transitioning in booster circuit
        dS[:,:,4] = dS[:,:,4] + N_vacc[:,:,3]*f_S
        dR[:,:,4] = dR[:,:,4] + N_vacc[:,:,3]*f_R
        # waned vaccine circuit
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,3]/VE
        f_R = R[:,:,3]/VE
        # Compute transitioning in two shot circuit
        dS[:,:,3] = dS[:,:,3] - N_vacc[:,:,3]*f_S
        dR[:,:,3] = dR[:,:,3] - N_vacc[:,:,3]*f_R
        # Compute transitioning in booster circuit
        dS[:,:,4] = dS[:,:,4] + N_vacc[:,:,3]*f_S
        dR[:,:,4] = dR[:,:,4] + N_vacc[:,:,3]*f_R

        # Update the S and R state
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        S_post_vacc = S + dS
        R_post_vacc = R + dR

        # Compute dS that makes S and R equal to zero
        dS[np.where(S_post_vacc < 0)] = 0 - S[np.where(S_post_vacc < 0)]
        dR[np.where(R_post_vacc < 0)] = 0 - R[np.where(R_post_vacc < 0)]
        # Set S and R equal to zero
        S_post_vacc[np.where(S_post_vacc < 0)] = 0
        R_post_vacc[np.where(R_post_vacc < 0)] = 0

        ################################
        ## calculate total population ##
        ################################

        T = np.sum(S + E + I + A + M + C + C_icurec + ICU + R, axis=2) # Sum over doses

        ################################
        ## Compute infection pressure ##
        ################################

        # For total population and for the relevant compartments I and A
        G = place.shape[0] # spatial stratification
        N = Nc.shape[1] # age stratification
        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*place + np.identity(G)*np.matmul(place, (1-np.outer(p,p)))
        # infer aggregation (prov, arr or mun)
        agg = None
        if G == 11:
            agg = 'prov'
        elif G == 43:
            agg = 'arr'
        elif G == 581:
            agg = 'mun'
        else:
            raise Exception(f"Space is {G}-fold stratified. This is not recognized as being stratification at Belgian province, arrondissement, or municipality level.")
        # Expand beta to size G
        beta = beta*sum(alpha*K_inf)
        # Compute populations after application of 'place' to obtain the S, I and A populations
        T_work = np.matmul(np.transpose(place_eff), T)
        T_work = np.expand_dims(T_work, axis=2)
        # I have verified on a dummy example that the following line of code:
        S_work = np.transpose(np.matmul(np.transpose(S_post_vacc), place_eff))
        # Is equivalent to the following for loop:
        # S_work = np.zeros(S.shape)
        #for idx in range(S.shape[2]):
        #    S_work[:,:,idx] = np.matmul(np.transpose(place_eff), S[:,:,idx]) 
        I_work = np.transpose(np.matmul(np.transpose(I), place_eff))
        A_work = np.transpose(np.matmul(np.transpose(A), place_eff))
        infpop = np.sum( (I_work + A_work)/T_work*(1-e_i), axis=2)
        # (11, 10, 10) x (11, 10, 5)
        multip_work = np.matmul(Nc_work, infpop[:,:,np.newaxis])
        multip_work *= beta[:,np.newaxis, np.newaxis]
        # Apply all other contacts to non-place modified populations
        infpop = np.sum( (I + A)/np.expand_dims(T, axis=2)*(1-e_i), axis=2)
        multip_rest = np.matmul(Nc-Nc_work, infpop[:,:,np.newaxis])
        multip_rest *= beta[:,np.newaxis,np.newaxis]
        # Compute rates of change
        dS_inf = (S_work * multip_work + S_post_vacc * multip_rest)*(1-e_s)

        ############################
        ## Compute system of ODEs ##
        ############################

        h_acc = (1-e_h)*h

        dS  = dS - dS_inf
        dE  = dS_inf - E/sigma 
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - A/da
        dM = ((1-a)/omega)*I - M*((1-h_acc)/dm) - M*h_acc/dhospital
        dC = M*(h_acc/dhospital)*c - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D))
        dICUstar = M*(h_acc/dhospital)*(1-c) - (1-m_ICU)*ICU/(dICU_R) - m_ICU*ICU/(dICU_D)

        dC_icurec = (1-m_ICU)*ICU/(dICU_R) - C_icurec*(1/dICUrec)
        dR  = dR + A/da + ((1-h_acc)/dm)*M + (1-m_C)*C*(1/(dc_R)) + C_icurec*(1/dICUrec)
        dD  = (m_ICU/(dICU_D))*ICU + (m_C/(dc_D))*C 
        dH_in = M*(h_acc/dhospital) - H_in
        dH_out =  (1-m_C)*C*(1/(dc_R)) +  m_C*C*(1/(dc_D)) + m_ICU/(dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = M*(h_acc/dhospital) - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D)) - m_ICU*ICU/(dICU_D)- C_icurec*(1/dICUrec) 

        ########################
        ## Waning of immunity ##
        ########################

        # Waning of second dose
        r_waning_vacc = 1/((5/12)*365)
        dS[:,:,2] = dS[:,:,2] - r_waning_vacc*S_post_vacc[:,:,2]
        dR[:,:,2] = dR[:,:,2] - r_waning_vacc*R_post_vacc[:,:,2]
        dS[:,:,3] = dS[:,:,3] + r_waning_vacc*S_post_vacc[:,:,2]
        dR[:,:,3] = dR[:,:,3] + r_waning_vacc*R_post_vacc[:,:,2]
        
        # Waning of booster dose
        # No waning of booster dose

        # Waning of natural immunity
        dS[:,:,0] = dS[:,:,0] + zeta*R_post_vacc[:,:,0] 
        dR[:,:,0] = dR[:,:,0] - zeta*R_post_vacc[:,:,0]       

        return (dS, dE, dI, dA, dM, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot)

##########################
## Initialize the model ##
##########################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Convert age_stratification_size to desired age groups
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if age_stratification_size == 3:
    age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
elif age_stratification_size == 9:
    age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
elif age_stratification_size == 10:
    age_classes = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')
else:
    raise ValueError(
        "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import necessary pieces of code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import the spatially explicit SEIQRD model with VOCs, vaccinations, seasonality
from covid19model.models import models
# Import time-dependent parameter functions for resp. P, Nc, alpha, N_vacc, season_factor
from covid19model.models.time_dependant_parameter_fncs import make_mobility_update_function, \
                                                            make_contact_matrix_function, \
                                                            make_VOC_function, \
                                                            make_vaccination_function, \
                                                            make_seasonality_function
# Import packages containing functions to load in data used in the model and the time-dependent parameter functions
from covid19model.data import mobility, sciensano, model_parameters, VOC

# ~~~~~~~~~~~~~~~~~~~
# Load necessary data
# ~~~~~~~~~~~~~~~~~~~

# Population size, interaction matrices and the model parameters
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_classes=age_classes, spatial=args.agg, vaccination=True, VOC=True)
# Raw local hospitalisation data used in the calibration. Moving average disabled for calibration.
df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=args.agg, values='hospitalised_IN', moving_avg=False, public=public)
# Google Mobility data (for social contact Nc)
provincial=False
if args.agg == 'prov':
    provincial=True
df_google = mobility.get_google_mobility_data(update=False, provincial=provincial)
# Load and format mobility dataframe (for mobility place)
proximus_mobility_data, proximus_mobility_data_avg = mobility.get_proximus_mobility_data(args.agg, dtype='fractional', beyond_borders=False)
# Load and format national VOC data (for time-dependent VOC fraction)
df_VOC_abc = VOC.get_abc_data()
# Load and format local vaccination data, which is also under the sciensano object
public_spatial_vaccination_data = sciensano.get_public_spatial_vaccination_data(update=update,agg=args.agg)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct time-dependent parameter functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Time-dependent social contact matrix over all policies, updating Nc
policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_all_spatial
policy_function_work = make_contact_matrix_function(df_google, Nc_dict).policies_all_work_only
# Time-dependent mobility function, updating P (place)
mobility_function = make_mobility_update_function(proximus_mobility_data, proximus_mobility_data_avg).mobility_wrapper_func
# Time-dependent VOC function, updating alpha
VOC_function = make_VOC_function(df_VOC_abc)
# Time-dependent (first) vaccination function, updating N_vacc
vaccination_function = make_vaccination_function(public_spatial_vaccination_data['INCIDENCE'], age_classes=age_classes)
# Time-dependent seasonality function, updating season_factor
seasonality_function = make_seasonality_function()

# ~~~~~~~~~~~~~~~~~~~~
# Initialize the model
# ~~~~~~~~~~~~~~~~~~~~

# Warmup of one month by default
warmup = 31
# Determine size of space and dose stratification size
spatial_stratification_size = params['place'].shape[0]
dose_stratification_size = len(public_spatial_vaccination_data.index.get_level_values('dose').unique()) + 2 # Added +2 for waning of 2nd dose vaccination + boosters
# 0.1 Exposed inidividual per age group and per NIS code
S = np.zeros([spatial_stratification_size, age_stratification_size, dose_stratification_size])
S[:,:,0] = initN.values
S[:,:,1:3] = 1e-1
E = np.zeros([spatial_stratification_size, age_stratification_size, dose_stratification_size])
E[:,:,0] = 1e-1
initial_states = {"S": S, "E": E}
# Update size of N_vacc
params.update({'N_vacc': np.zeros([params['place'].shape[0], age_stratification_size, len(public_spatial_vaccination_data.index.get_level_values('dose').unique())+1])}) # Added +1 because vaccination dataframe does not include boosters yet
# Swap trifold beta for one beta
params.pop('beta_R')
params.pop('beta_U')
params.pop('beta_M')
params.update({'beta': np.ones(spatial_stratification_size)*0.024})
# Set l1, prevention parameters and seasonality to guestimates obtained form model calibration
params.update({'l1':16.0, 'prev_schools':0.166,'prev_work':0.56, 'prev_home':0.501,'prev_rest_lockdown':0.0195, 'prev_rest_relaxation':0.88, 'amplitude': 0.227, 'peak_shift': -6.77})
# Initiate model with initial states, defined parameters, and proper time dependent functions
model = custom_COVID19_SEIQRD_spatial_stratified_vacc(initial_states, params, spatial=args.agg,
                        time_dependent_parameters={'Nc' : policy_function,
                                                    'Nc_work' : policy_function_work,
                                                    'place' : mobility_function,
                                                    'N_vacc' : vaccination_function, 
                                                    'alpha' : VOC_function,
                                                    'beta' : seasonality_function})

#######################################
## Write a custom objective function ##
#######################################

# Initial guess: you can replace this to make this faster
theta = [0.024*np.ones(spatial_stratification_size),]
theta = [np.array([0.024,0.023,0.020,0.02475,0.02225,0.023,0.02375,0.023,0.023,0.02,0.021]),] # Manual fit to provinces
pars = ['beta',]

# Start- and enddates of visualizations
start_calibration=df_sciensano.index.min()
end_calibration='2020-04-07'
end_visualization=end_calibration
data=[df_sciensano[start_calibration:end_calibration]]

###########################################
## Visualize the result per spatial unit ##
###########################################

for idx,NIS in enumerate(data[0].columns):
    # Assign estimate
    pars_PSO = assign_PSO(model.parameters, pars, theta)
    model.parameters = pars_PSO
    # Perform simulation
    out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
    # Visualize
    fig,ax = plt.subplots(figsize=(12,4))
    ax.plot(out['time'],out['H_in'].sel(place=NIS).sum(dim='Nc').sum(dim='doses'),'--', color='blue')
    ax.scatter(data[0].index,data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.axvline(x=pd.Timestamp('2020-03-14'),linestyle='--',linewidth=1,color='black')
    ax.axvline(x=pd.Timestamp('2020-03-18'),linestyle='--',linewidth=1,color='black')
    # Add a box with the NIS code
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.02, 0.88, 'NIS: '+str(NIS), transform=ax.transAxes, fontsize=13, verticalalignment='center', bbox=props)
    # Format axis
    ax = _apply_tick_locator(ax)
    # Display figure
    plt.show()
    plt.close()

    satisfied = not click.confirm('Do you want to make manual tweaks to beta?', default=False)
    while not satisfied:
        # Prompt for input
        theta[0][idx] = float(input("What should the value of beta be? "))
        # Assign estimate
        pars_PSO = assign_PSO(model.parameters, pars, theta)
        model.parameters = pars_PSO
        # Perform simulation
        out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
        # Visualize new fit
        fig,ax = plt.subplots(figsize=(12,4))
        ax.plot(out['time'],out['H_in'].sel(place=NIS).sum(dim='Nc').sum(dim='doses'),'--', color='blue')
        ax.scatter(data[0].index,data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.axvline(x=pd.Timestamp('2020-03-14'),linestyle='--',linewidth=1,color='black')
        ax.axvline(x=pd.Timestamp('2020-03-18'),linestyle='--',linewidth=1,color='black')
        # Add a box with the NIS code
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.02, 0.88, 'NIS: '+str(NIS), transform=ax.transAxes, fontsize=13, verticalalignment='center', bbox=props)
        # Format axis
        ax = _apply_tick_locator(ax)
        # Display figure
        plt.show()
        plt.close()
        # Satisfied?
        satisfied = not click.confirm('Would you like to make further changes?', default=False)

print('Resulting vector of infectivities: ' + str(theta[0]))

##########################################################
## Save initial states for the vaccine-stratified model ##
##########################################################

dates = ['2020-03-15', '2020-03-16', '2020-03-17']
initial_states={}
for date in dates:
    initial_states_per_date = {}
    for state in out.data_vars:
        initial_states_per_date.update({state: out[state].sel(time=pd.to_datetime(date)).values})
    initial_states.update({date: initial_states_per_date})
with open(results_path+'initial_states-COVID19_SEIQRD_spatial_stratified_vacc.pickle', 'wb') as fp:
    pickle.dump(initial_states, fp)

##############################################
## Save initial states for the virgin model ##
##############################################

initial_states={}
for date in dates:
    initial_states_per_date = {}
    for state in out.data_vars:
        # Select first column only for non-dose stratified model
        initial_states_per_date.update({state: out[state].sel(time=pd.to_datetime(date)).values[:,:,0]})
    initial_states.update({date: initial_states_per_date})
with open(results_path+'initial_states-COVID19_SEIQRD_spatial.pickle', 'wb') as fp:
    pickle.dump(initial_states, fp)

# Work is done
sys.stdout.flush()
sys.exit()