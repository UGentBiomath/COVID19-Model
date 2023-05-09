##############
## Packages ##
##############

import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# EPNM functions
from covid19_DTM.visualization.output import _apply_tick_locator 
from EPNM.models.utils import initialize_model
from EPNM.data.parameters import get_model_parameters
from EPNM.data.utils import get_sector_labels, get_sector_names, aggregate_simulation, get_sectoral_conversion_matrix
from EPNM.models.TDPF import household_demand_shock, compute_income_expectations
from EPNM.models.draw_functions import draw_function as draw_function
from EPNM.data.calibration_data import get_NAI_value_added, get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand

##############
## Settings ##
##############

# Start- and enddate simulation
start_sim = '2020-03-01'
end_sim = '2021-10-01'
state = 'x'

###########################
## Define draw functions ##
###########################

def draw_function_b(param_dict, samples_dict):
    param_dict['b_s'] = np.random.uniform(low=0.5, high=1)
    return param_dict

def draw_function_L(param_dict, samples_dict):
    param_dict['L'] = np.random.uniform(low=0.5, high=1)
    return param_dict

def draw_function_delta_S(param_dict, samples_dict):
    param_dict['delta_S'] = np.random.uniform(low=0.5, high=1)
    return param_dict

def draw_function_rho(param_dict, samples_dict):
    param_dict['rho'] = np.random.uniform(low=0.99, high=0.999) # 0.1 to 1 quarter
    return param_dict

def draw_function_l_s(param_dict, samples_dict):
    r = np.random.uniform(low=0.70, high=1.30)
    param_dict['l_s_1'] =  r*param_dict['l_s_1']
    param_dict['l_s_2'] =  r*param_dict['l_s_2']
    return param_dict

draw_functions = [draw_function_l_s, draw_function_b, draw_function_rho, draw_function_L, draw_function_delta_S]
parameter_names = ['$\\epsilon_{i,t}^S$', '$b_t$', '$\\rho$', '$L$', '$\Delta S$']

######################
## Initialize model ##
######################

params, model = initialize_model(shocks='alleman', prodfunc='half_critical')

#####################################
## Run simulation loop & Visualize ##
#####################################

# Initialize figure
fig,ax=plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(8.27,11.7))

for i,draw_function in enumerate(draw_functions):
    # Simulate model
    out = model.sim([start_sim, end_sim], method='RK45', rtol=1e-4, N=5*18, processes=18, samples={}, draw_function=draw_function)
    simtime = out['date'].values
    # Skip row 0, column 1

    if i < 1:
        # Visualize prediction
        ax.flatten()[i].fill_between(simtime, out[state].sum(dim='NACE64').quantile(dim='draws', q=0.025)/out[state].sum(dim='NACE64').mean(dim='draws').isel(date=0)*100-100,
                                    out[state].sum(dim='NACE64').quantile(dim='draws', q=0.975)/out[state].sum(dim='NACE64').mean(dim='draws').isel(date=0)*100-100,
                                    color='black', alpha=0.90, label='95% CI')
        # Text box with parameter name
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.flatten()[i].text(0.05, 0.95, parameter_names[i], transform=ax.flatten()[i].transAxes, fontsize=11,
                            verticalalignment='top', bbox=props)    
        # Attributes
        ax.flatten()[i].grid(False)
        # Ylim
        ax.flatten()[i].set_ylim([-50, 8])
        ax.flatten()[i].tick_params(axis='both', which='major', labelsize=13)
    else:
        # Visualize prediction
        ax.flatten()[i+1].fill_between(simtime, out[state].sum(dim='NACE64').quantile(dim='draws', q=0.025)/out[state].sum(dim='NACE64').mean(dim='draws').isel(date=0)*100-100,
                                    out[state].sum(dim='NACE64').quantile(dim='draws', q=0.975)/out[state].sum(dim='NACE64').mean(dim='draws').isel(date=0)*100-100,
                                    color='black', alpha=0.90, label='95% CI')
        # Text box with parameter name
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.flatten()[i+1].text(0.05, 0.95, parameter_names[i], transform=ax.flatten()[i+1].transAxes, fontsize=11,
                            verticalalignment='top', bbox=props)    
        # Attributes
        ax.flatten()[i+1].grid(False)
        # Ylim
        ax.flatten()[i+1].set_ylim([-50, 9.5])
        ax.flatten()[i+1].tick_params(axis='both', which='major', labelsize=13)

# Manually set xticks and rotate
ax[2,0].set_xticks([pd.to_datetime('2020-03-31'), pd.to_datetime('2020-09-30'),pd.to_datetime('2021-03-31'),pd.to_datetime('2021-09-30')])
ax[2,0].set_xticklabels(['2020-03-31', '2020-09-30', '2021-03-31', '2021-09-30'], rotation=30, fontsize=13)
ax[2,1].set_xticks([pd.to_datetime('2020-03-31'), pd.to_datetime('2020-09-30'),pd.to_datetime('2021-03-31'),pd.to_datetime('2021-09-30')])
ax[2,1].set_xticklabels(['2020-03-31', '2020-09-30', '2021-03-31', '2021-09-30'], rotation=30, fontsize=13)
# Y labels
ax[0,0].set_ylabel('Gross output reduction (%)', fontsize=13)
ax[1,0].set_ylabel('Gross output reduction (%)', fontsize=13)
ax[2,0].set_ylabel('Gross output reduction (%)', fontsize=13)
# Legend
#ax[2,1].legend(loc=4, framealpha=1, fontsize=11)
# Remove second subplot
fig.delaxes(ax[0,1])
# Print figure
plt.tight_layout()
#plt.show()
fig.savefig('sensitivity-b-L-deltaS-ls.pdf')
plt.close()