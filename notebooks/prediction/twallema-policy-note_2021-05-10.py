"""
This script contains code to simulate a set of scenarios/infographics for policy makers.
Deterministic, national-level BIOMATH COVID-19 SEIRD

Example use:
------------
python twallema-restore8.py -f BE_WAVE2_R0_COMP_EFF_2021-05-07.json -s 0 1 2 3 4 5 6 -n 100

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
from covid19model.data import mobility, sciensano, model_parameters, VOC
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save figures",action='store_true')
args = parser.parse_args()

# Path where figures and results should be stored
fig_path = '../../results/predictions/national/twallema-policy-note-2021-05-10/'

# -----------------------
# Load samples dictionary
# -----------------------

# Path where MCMC samples are saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'

from covid19model.models.utils import load_samples_dict
samples_dict = load_samples_dict(samples_path+str(args.filename), wave=2)
warmup = int(samples_dict['warmup'])

###########################
### Simulation control  ###
###########################

# Scenario settings
relaxdates = ['2021-05-01', '2021-05-15', '2021-06-01']

import sys, getopt
report_version = 'policy note 1.0'
start_sim = '2020-09-01'
end_sim = '2021-10-01'
start_calibration = samples_dict['start_calibration']
end_calibration = samples_dict['end_calibration']
model = 'BIOMATH COVID-19 SEIRD national'
n_samples = args.n_samples
n_draws = args.n_draws_per_sample
conf_int = 0.05

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

print('report: ' + report_version)
print('model: ' + model)
print('number of samples: ' + str(n_samples))
print('confidence level: ' + str(conf_int*100) +' %')
print('start of simulation: ' + start_sim)
print('end of simulation: ' + end_sim)
print('start of calibration: ' + start_calibration)
print('end of calibration: ' + end_calibration)
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
# VOC data
df_VOC_501Y = VOC.get_501Y_data()
# Model initial condition on September 1st
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)  

print('2) Initializing model\n')

# ---------------------------
# Time-dependant VOC function
# ---------------------------

from covid19model.models.time_dependant_parameter_fncs import make_VOC_function
VOC_function = make_VOC_function(df_VOC_501Y)

# -----------------------------------
# Time-dependant vaccination function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import  make_vaccination_function
vacc_strategy = make_vaccination_function(df_sciensano)

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, delayed_ramp_fun, ramp_fun
contact_matrix_4prev = make_contact_matrix_function(df_google, Nc_all)
policies_WAVE2_full_relaxation = make_contact_matrix_function(df_google, Nc_all).policies_WAVE2_full_relaxation
    
# ----------------
# Helper functions
# ----------------

from covid19model.models.utils import output_to_visuals, draw_fcn_WAVE2

def draw_fcn_no_vacc(param_dict,samples_dict):
    """ 
    This draw function differes from the one located in the `~/src/models/utils.py` because the vaccination is excluded
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
    model.parameters['K_inf1'] = samples_dict['K_inf'][idx]
    model.parameters['K_inf2'] = samples_dict['K_inf'][idx]*np.random.uniform(low=1.3,high=1.5)
    model.parameters['K_hosp'] = np.array([1, np.random.uniform(low=1.3,high=1.5), np.random.uniform(low=1.3,high=1.5)])

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

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters(vaccination=True)
# Add the time-dependant parameter function arguments
# Social policies
params.update({'l': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5, 'relaxdate': '2021-05-08', 'l_relax': 31})
# VOC
params.update({'t_sig': '2021-07-01'})
# Vaccination
params.update(
    {'vacc_order': np.array(range(9))[::-1], 'daily_dose': 55000,
     'refusal': 0.2*np.ones(9), 'delay': 20}
)
# Initialize model
model = models.COVID19_SEIRD_vacc(initial_states, params,
                        time_dependent_parameters={'Nc': policies_WAVE2_full_relaxation, 'N_vacc': vacc_strategy, 'alpha': VOC_function})
                        
# ----------------------------
# Initialize results dataframe
# ----------------------------
index = pd.date_range(start=start_sim, end=end_sim)
columns = [[],[],[]]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["relaxation date", "state", "results"])
df_sim_virgin = pd.DataFrame(index=index, columns=columns)

# --------------
# Run simulation
# --------------

print('3) Simulating model\n')

states = ['H_in','H_tot']
quantities = ['mean','LL','UL']

df_sim = df_sim_virgin.copy(deep=True)
for idx,relaxdate in enumerate(relaxdates):
    model.parameters.update({'relaxdate': relaxdate})
    print('\t# relaxdate '+relaxdate)
    out_vacc = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn_WAVE2,samples=samples_dict)
    simtime, df_2plot = output_to_visuals(out_vacc, states, n_samples, args.n_draws_per_sample, LL = conf_int/2, UL = 1 - conf_int/2)
    for idx,state in enumerate(states):
        for jdx, quantity in enumerate(quantities):
            df_sim[relaxdate, state, quantity] = df_2plot[state,quantity]

# -------------------
# Make visualizations
# -------------------

print('4) Making visualizations\n')
colors = ['red','orange','green']

# Hospital load
fig,ax = plt.subplots(figsize=(15,4))
for idx,relaxdate in enumerate(relaxdates):
    ax.plot(df_sim[relaxdate,'H_tot','mean'],'--', color=colors[idx], linewidth=1.5)
    ax.fill_between(simtime, df_sim[relaxdate,'H_tot','LL'], df_sim[relaxdate,'H_tot','UL'],alpha=0.15, color = colors[idx])
ax.scatter(df_sciensano[start_calibration:].index,df_sciensano['H_tot'][start_calibration:], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
# Annotations
ax.text(x='2020-09-01',y=11800+1000,s='The Hammer', fontsize=16)
ax.text(x='2020-09-01',y=9600+1000,s='- Avoid total collapse of\n  public health care system\n- Take lockdown measures', fontsize=12)
ax.text(x='2020-12-14',y=11800+1000,s='The Dance', fontsize=16)
ax.text(x='2020-12-14',y=9600+1000,s='- Keep reproduction number below one\n- Retain most cost-effective measures only\n- Start vaccination campaign', fontsize=12)
ax.text(x='2021-06-07',y=11800+1000,s='The Tipping Point', fontsize=16)
ax.text(x='2021-06-07',y=8900+1000,s='- Vaccination creates ever more\n  margin for relaxations\n- Collapse of health care system\n  gradually becomes impossible', fontsize=12)
# ICU lines
ax.text(x='2020-09-01',y=4200,s='ICU treshold', fontsize=16)
plt.axhline(y=4000, linestyle = '--', color='black', linewidth=1.5)
# Arrows
ax.annotate('Relax May 1st', fontsize=14, color='red', xy=('2021-05-15',3500),xytext=('2021-02-15',6000), xycoords='data',arrowprops=dict(arrowstyle="->", facecolor='red', edgecolor='red', lw=1.5) )
ax.annotate('Relax May 15th', fontsize=14, color='orange', xy=('2021-06-05',4000),xytext=('2021-04-15',7000), xycoords='data',arrowprops=dict(arrowstyle="->", facecolor='orange', edgecolor='orange', lw=1.5) )
ax.annotate('Relax June 1st', fontsize=14, color='green', xy=('2021-07-01',3500),xytext=('2021-07-01',6500), xycoords='data',arrowprops=dict(arrowstyle="->", facecolor='green', edgecolor='green', lw=1.5) )
# Credits
plt.gcf().text(0.125, 0.01, 'Alleman et. al., (2021) Ghent University', fontsize=6)
# Stylize figure
ax = _apply_tick_locator(ax)
ax.grid(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xlim('2020-09-01','2021-09-01')
ax.set_ylim(0,12000)
plt.tight_layout()
plt.show()
plt.close()
if args.save:
    fig.savefig(fig_path+args.filename[:-5]+'_HAMMER_DANCE_TIPPING_POINT_RELAXDATE.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_path+args.filename[:-5]+'_HAMMER_DANCE_TIPPING_POINT_RELAXDATE.png', dpi=300, bbox_inches='tight')


# Hospital load
fig,ax = plt.subplots(figsize=(15,4))
relaxdate = '2021-06-01'
ax.plot(df_sim[relaxdate,'H_tot','mean'],'--', color='green', linewidth=1.5)
ax.fill_between(simtime, df_sim[relaxdate,'H_tot','LL'], df_sim[relaxdate,'H_tot','UL'],alpha=0.15, color = 'green')
ax.scatter(df_sciensano[start_calibration:].index,df_sciensano['H_tot'][start_calibration:], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
# Without vaccine
print('Simulating model without vaccination\n')
rm = ['df_sciensano_end','df_sciensano_start','refusal','delay','daily_dose','vacc_order']
for key in rm:
    del params[key]
model = models.COVID19_SEIRD_vacc(initial_states, params,
                        time_dependent_parameters={'Nc': policies_full_relaxation, 'alpha': stratified_VOC_func})
model.parameters['relaxdate'] = '2021-06-01'
model.parameters['N_vacc'] = np.zeros(9)
out_no_vacc = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn_no_vacc,samples=samples_dict)
simtime, df_2plot = output_to_visuals(out_no_vacc, states, n_samples, args.n_draws_per_sample, LL = conf_int/2, UL = 1 - conf_int/2)
ax.plot(df_2plot['H_tot','mean'],'--', color='red', linewidth=1.5)
ax.fill_between(simtime, df_2plot['H_tot','LL'], df_2plot['H_tot','UL'],alpha=0.15, color = 'red')
# Annotations
ax.text(x='2020-09-01',y=11800+1000,s='The Hammer', fontsize=16)
ax.text(x='2020-09-01',y=9600+1000,s='- Avoid total collapse of\n  public health care system\n- Take lockdown measures', fontsize=12)
ax.text(x='2020-12-14',y=11800+1000,s='The Dance', fontsize=16)
ax.text(x='2020-12-14',y=9600+1000,s='- Keep reproduction number below one\n- Retain most cost-effective measures only\n- Start vaccination campaign', fontsize=12)
ax.text(x='2021-06-07',y=11800+1000,s='The Tipping Point', fontsize=16)
ax.text(x='2021-06-07',y=8900+1000,s='- Vaccination creates ever more\n  margin for relaxations\n- Collapse of health care system\n  gradually becomes impossible', fontsize=12)
# ICU lines
ax.text(x='2020-09-01',y=4200,s='ICU treshold', fontsize=16)
plt.axhline(y=4000, linestyle = '--', color='black', linewidth=1.5)
# Arrows
ax.annotate('Relax June 1st with vaccination', fontsize=14, color='green', xy=('2021-06-07',3200),xytext=('2021-02-15',6000), xycoords='data',arrowprops=dict(arrowstyle="->", facecolor='green', edgecolor='green', lw=1.5) )
ax.annotate('Relax June 1st w/o vaccination', fontsize=14, color='red', xy=('2021-06-10',7000), xytext=('2021-02-15',8000), xycoords='data',arrowprops=dict(arrowstyle="->", facecolor='red', edgecolor='red', lw=1.5) )
# Credits
plt.gcf().text(0.015, 0.01, 'Alleman et. al., (2021) Ghent University', fontsize=6)
# Stylize figure
ax = _apply_tick_locator(ax)
ax.grid(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xlim('2020-09-01','2021-09-01')
ax.set_ylim(0,12000)
plt.tight_layout()
plt.show()
plt.close()
if args.save:
    fig.savefig(fig_path+args.filename[:-5]+'_HAMMER_DANCE_TIPPING_POINT_VACCINES.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_path+args.filename[:-5]+'_HAMMER_DANCE_TIPPING_POINT_VACCINES.png', dpi=300, bbox_inches='tight')
