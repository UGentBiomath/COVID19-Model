##############
## Packages ##
##############

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
# Nicer colors
colors = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
          "green" : "#009E73", "yellow" : "#F0E442",
          "blue" : "#0072B2", "red" : "#D55E00",
          "pink" : "#CC79A7", "black" : "#000000"}

##############
## Settings ##
##############

# Start- and enddate simulation
start_sim = '2020-03-01'
end_sim = '2021-03-31'

####################
## Simulate model ##
####################

# Initialize model
params, model = initialize_model(shocks='alleman', prodfunc='half_critical')

# Load (relative) data
data_employment = get_employment_survey().loc[slice(start_sim,end_sim), slice(None)].loc[slice(None), 'BE']
data_revenue = get_revenue_survey().loc[slice(start_sim,end_sim), slice(None)].loc[slice(None), 'BE']
data_GDP = get_synthetic_GDP().loc[slice(start_sim,end_sim), slice(None)].loc[slice(None), 'BE']
data_B2B = get_B2B_demand().loc[slice(start_sim,end_sim), slice(None)].drop('U', level='NACE21', axis=0, inplace=False)
# Weighted aggregation of B2B demand data
B2B_demand = np.matmul(params['O_j'], np.transpose(get_sectoral_conversion_matrix('NACE64_NACE21')))[:-1]
B2B_demand = B2B_demand/sum(B2B_demand)
data_B2B = data_B2B.to_frame()
data_B2B['weighted']=0
for date in data_B2B.index.get_level_values('date').unique():
    v = data_B2B.loc[date, slice(None)]['B2B demand']*B2B_demand
    data_B2B.loc[(date,slice(None)), 'weighted'] = v.values
data_B2B = data_B2B['weighted'].groupby(by='date').sum().ewm(span=1).mean()

# Draw function
from EPNM.models.draw_functions import draw_function

# Simulate model
out = model.sim([start_sim, end_sim], tau=1, N=18, processes=18, samples={}, draw_function=draw_function)
simtime = out['date'].values

###############
## Visualize ##
###############

# Aggregate data
datasets = [data_B2B, data_GDP, data_revenue, data_employment]
# States
states = ['O', 'x', 'x', 'l']
# Ylabels
ylabels = ['B2B transactions\nreduction (%)', 'Synthetic GDP\nreduction (%)', 'Revenue\nreduction (%)', 'Employment\nreduction (%)']
# Sectoral dimension name
dims = ['NACE21', 'NACE64', 'NACE64', 'NACE64']
# Initialize figure
fig,ax=plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8.27,8.27))

for i, (data, state, ylabel, dim) in enumerate(zip(datasets, states, ylabels, dims)):
    # Data
    ax[i].scatter(data.index.get_level_values('date').unique(), -100+data*100, color='black', alpha=0.9, linestyle='None', facecolors='black', s=20, linewidth=2)
    # Model prediction
    ax[i].plot(simtime, out[state].sum(dim='NACE64').mean(dim='draws')/out[state].sum(dim='NACE64').mean(dim='draws').isel(date=0)*100-100, color='blue', linestyle='--', linewidth=1)
    ax[i].fill_between(simtime, out[state].sum(dim='NACE64').quantile(dim='draws', q=0.025)/out[state].sum(dim='NACE64').mean(dim='draws').isel(date=0)*100-100,
                                out[state].sum(dim='NACE64').quantile(dim='draws', q=0.975)/out[state].sum(dim='NACE64').mean(dim='draws').isel(date=0)*100-100,
                                color=colors['blue'], alpha=0.2)
    # Formatting
    ax[i].set_ylabel(ylabel)
    ax[i].set_ylim([-50,5])
    ax[i].grid(False)
    #ax[i].set_xlim([start_sim, end_sim])
    #ax[i].set_xticks([pd.to_datetime('2020-03-31'), pd.to_datetime('2020-06-30'), pd.to_datetime('2020-09-30'),pd.to_datetime('2020-12-31'),pd.to_datetime('2021-03-31')])

plt.xticks(rotation = 30) 
# Show figure
plt.tight_layout()
plt.show()
fig.savefig('plot-fit-national.pdf')
plt.close()