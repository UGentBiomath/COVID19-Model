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
# Colorscale
colors = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
          "green" : "#009E73", "yellow" : "#F0E442",
          "blue" : "#0072B2", "red" : "#D55E00",
          "pink" : "#CC79A7", "black" : "#000000"}

##############
## Settings ##
##############

# Start- and enddate simulation
start_sim = '2020-03-01'
end_sim = '2020-10-01'
state = 'x'

#######################
## Initialize models ##
#######################

# Load models
params, model_1 = initialize_model(shocks='alleman', prodfunc='strongly_critical')
params, model_2 = initialize_model(shocks='alleman', prodfunc='weakly_critical')
# Alter enddate of first lockdown: no end of lockdown
model_1.parameters['t_end_lockdown_1'] = pd.Timestamp(end_sim)
model_2.parameters['t_end_lockdown_1'] = pd.Timestamp(end_sim)
# Group models
models = [model_1, model_2]
# Figure attributes
colors_models = ['black', colors['red']]
taus = [1, 30]
linestyle_taus = ['solid','dotted']
labels_tau = ['$\\tau=1$', '$\\tau=30$']
prodfuncs = ['strongly critical', 'weakly critical']

#####################################
## Run simulation loop & Visualize ##
#####################################

# Initialize figure
fig,ax=plt.subplots(figsize=(8.27,4))

for i,model in enumerate(models):
    for j,tau in enumerate(taus):
        # Set tau
        model.parameters['tau'] = tau
        # Simulate model
        out = model.sim([start_sim, end_sim], method='RK45', rtol=1e-4)
        simtime = out['date'].values
        ax.plot(simtime, out[state].sum(dim='NACE64')/sum(params['x_0'])*100-100,linestyle=linestyle_taus[j], color=colors_models[i], label=prodfuncs[i]+f", $\\tau = {tau}$")
ax.legend(fontsize=13)
# Size of ticks
ax.tick_params(axis='both', which='major', labelsize=13)
# Ylabel
ax.set_ylabel('Gross output reduction (%)', fontsize=13)
ax.grid(False)
# Y limit
ax.set_ylim([-55,5])
# Limit number of axis ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
plt.xticks(rotation=30)
# Print figure
plt.tight_layout()
#plt.show()
fig.savefig('sensitivity-tau.pdf')
plt.close()