# General packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# EPNM functions
from EPNM.models.utils import initialize_model
from EPNM.data.parameters import get_model_parameters
from EPNM.data.utils import get_sector_labels, get_sector_names, aggregate_simulation, get_sectoral_conversion_matrix
from EPNM.models.TDPF import household_demand_shock, compute_income_expectations
from EPNM.models.draw_functions import draw_function as draw_function
from EPNM.data.calibration_data import get_NAI_value_added, get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand

# Start and enddate
start_date = '2020-03-01' 
end_date = '2020-09-01'

# Define production functions
prodfuncs = ['linear', 'weakly_critical', 'half_critical', 'strongly_critical', 'leontief']
# define lower and upper bound on tau
tau_list = [1, 30]
# Define enddates
end_lockdown_list = ['2020-05-01', '2020-09-01']

fig,ax=plt.subplots(ncols=2, sharey=True)

for i,end_lockdown in enumerate(end_lockdown_list):

    # Loop over production functions
    out_lower_list=[]
    out_upper_list=[]
    for prodfunc in prodfuncs:
        # Initialize model
        params, model = initialize_model(shocks='alleman', prodfunc=prodfunc)
        # Change lockdown enddate to end of simulation
        model.parameters.update({'t_end_lockdown_1': pd.Timestamp(end_lockdown)})
        # Set lower tau
        model.parameters.update({'tau': tau_list[0]})
        # Slower lockdown release
        model.parameters.update({'l2': 8*7})
        # Simulate model
        out_lower_list.append(model.sim([start_date, end_date], method='RK45', rtol=1e-4))
        # Set upper tau
        model.parameters.update({'tau': tau_list[1]})
        # Simulate model
        out_upper_list.append(model.sim([start_date, end_date], method='RK45', rtol=1e-4))

    # Load data
    data_GDP = get_synthetic_GDP(relative=True)
    # List of colors
    color_list = ['black', 'blue', 'red', 'green', 'orange']
    label_list = ['linear', 'weakly critical', 'half critical', 'strongly critical', 'leontief']
    # Visualize result
    ax[i].scatter(data_GDP.loc[slice(None,end_date), 'BE'].index.get_level_values('date').unique(), data_GDP.loc[slice(None,end_date), 'BE']*100,
                color='black', alpha=0.8, linestyle='None', facecolors='none', s=80, linewidth=3, label='synthetic GDP (NBB)')
    for j,out_lower in enumerate(out_lower_list):
        #ax.plot(out['date'].values, out['x'].sum(dim='NACE64')/out['x'].sum(dim='NACE64').sel(date=out['date'].values[0])*100,
        #        color=color_list[i], label=label_list[i])
        ax[i].fill_between(out_lower['date'].values,
                            out_lower['x'].sum(dim='NACE64')/out_lower['x'].sum(dim='NACE64').sel(date=out_lower['date'].values[0])*100,
                            out_upper_list[j]['x'].sum(dim='NACE64')/out_upper_list[i]['x'].sum(dim='NACE64').sel(date=out_upper_list[j]['date'].values[0])*100,
                            color=color_list[j], alpha=0.25, label=label_list[j])

    # Set max_n ticks
    ax[i].xaxis.set_major_locator(plt.MaxNLocator(5))
    for tick in ax[i].get_xticklabels():
        tick.set_rotation(45)
    # Set y axis limit
    ax[i].set_ylim([0,105])
    # Horizontal line for lockdown release
    ax[i].axvline(x=pd.Timestamp('2020-05-01'), color='black', linewidth=1.5, linestyle='--')

# Set labels
ax[0].set_ylabel('GDP change (%)')
# Legend
ax[1].legend()

# Print to screen
plt.tight_layout()
plt.show()
plt.close()

# PART II:
start_calibration = '2020-03-01'
end_calibration = '2021-10-01'
params, model = initialize_model(shocks='alleman', prodfunc='half_critical')

from pySODM.optimization.objective_functions import log_posterior_probability, ll_gaussian

import xarray as xr
def aggregate_NACE21(simulation_in):
    """ A function to convert a simulation of the economic IO model on the NACE64 level to the NACE21 level
    
    Input
    =====
    simulation_in: xarray.DataArray
        Simulation result (NACE64 level). Obtained from a pySODM xarray.Dataset simulation result by using: xarray.Dataset[state_name]
    
    Output
    ======
    simulation_out: xarray.DataArray
        Simulation result (NACE21 level)
    """

    simulation_out = xr.DataArray(np.matmul(np.matmul(simulation_in.values, np.transpose(get_sectoral_conversion_matrix('NACE64_NACE38'))), np.transpose(get_sectoral_conversion_matrix('NACE38_NACE21'))),
                                    dims = ['date', 'NACE21'],
                                    coords = dict(NACE21=(['NACE21'], get_sector_labels('NACE21')),
                                             date=simulation_in.coords['date']))
    return simulation_out.resample(date='Q').mean()



def aggregate_dummy(simulation_in):
    """
    Does nothing
    """
    return simulation_in

def aggregate_quarterly(simulation_in):
    """
    Aggregates data temporily to quarters
    """

    aggregated_simulation = simulation_in.resample(date='Q').mean()
    simulation_out = xr.DataArray(aggregated_simulation.values,
                                    dims = ['date', 'NACE64'],
                                    coords = dict(NACE64=(['NACE64'], get_sector_labels('NACE64')),
                                                  date=aggregated_simulation.coords['date']))
    return simulation_out

# Get data
data_employment = get_employment_survey(relative=False)
data_revenue = get_revenue_survey(relative=False)
data_GDP = get_synthetic_GDP(relative=False)
data_B2B_demand = get_B2B_demand(relative=False)

# Temporal aggregation NACE64 data to quarters
data_employment_quarterly = get_employment_survey(relative=False).groupby([pd.Grouper(freq='Q', level='date'),] + [data_employment.index.get_level_values('NACE64')]).mean()
data_revenue_quarterly = get_revenue_survey(relative=False).groupby([pd.Grouper(freq='Q', level='date'),] + [data_revenue.index.get_level_values('NACE64')]).mean()
data_GDP_quarterly = get_synthetic_GDP(relative=False).groupby([pd.Grouper(freq='Q', level='date'),] + [data_GDP.index.get_level_values('NACE64')]).mean()
data_B2B_demand = get_B2B_demand(relative=False).groupby([pd.Grouper(freq='Q', level='date'),] + [data_B2B_demand.index.get_level_values('NACE21')]).mean()

# Define dataset
data = [
        # NACE 64 sectoral data
        data_employment_quarterly.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
        data_revenue_quarterly.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
        data_GDP_quarterly.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
        # National data
        data_employment.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
        data_revenue.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
        data_GDP.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
        # NACE 21 B2B Demand data
        data_B2B_demand.drop('U', level='NACE21', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
        ]

# Assign a higher weight to the national data
weights = [
           1/len(data_employment_quarterly.index.get_level_values('NACE64').unique())/len(data_employment_quarterly.index.get_level_values('date').unique()),
           1/len(data_revenue_quarterly.index.get_level_values('NACE64').unique())/len(data_revenue_quarterly.index.get_level_values('date').unique()),
           1/len(data_GDP_quarterly.index.get_level_values('NACE64').unique())/len(data_GDP_quarterly.index.get_level_values('date').unique()),
           1/len(data_employment.index.get_level_values('date').unique()),
           1/len(data_revenue.index.get_level_values('date').unique()),
           1/len(data_GDP.index.get_level_values('date').unique()),
           1/len(data_B2B_demand.index.get_level_values('NACE21').unique())/len(data_B2B_demand.index.get_level_values('date').unique()),
           ]
# States to calibrate
states = ["l", "x", "x", "l", "x", "x", "O"]  
# Log likelihood functions and arguments
log_likelihood_fnc = [ll_gaussian, ll_gaussian,ll_gaussian, ll_gaussian,ll_gaussian, ll_gaussian, ll_gaussian]
sigma = 0.05
log_likelihood_fnc_args = [
        sigma*data_employment_quarterly.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
        sigma*data_revenue_quarterly.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
        sigma*data_GDP_quarterly.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
        sigma*data_employment.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
        sigma*data_revenue.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
        sigma*data_GDP.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
        sigma*data_B2B_demand.drop('U', level='NACE21', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
        ]
# Aggregation functions
aggregation_functions = [
        aggregate_quarterly,
        aggregate_quarterly,
        aggregate_quarterly,
        aggregate_dummy,
        aggregate_dummy,
        aggregate_dummy,
        aggregate_NACE21,
        ]

# Consumer demand/Exogeneous demand shock during summer of 2020
pars = ['tau',]
bounds=((1,100),)

objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args, aggregation_function=aggregation_functions, weights=weights)

tau_lst = np.linspace(start=1, stop=100, num=30)
prodfunc_lst = ['linear', 'weakly_critical', 'half_critical', 'strongly_critical', 'leontief']

WSSE_global = []
for prodfunc in prodfunc_lst:
    print(prodfunc)
    WSSE = []
    # Initialize model
    params, model = initialize_model(shocks='alleman', prodfunc=prodfunc)
    # Initialize WSSE
    objective_function.model = model
    # Compute WSSE vs. tau
    for tau in tau_lst:
        theta = np.array([tau,])
        WSSE.append(objective_function(theta))
    WSSE_global.append(WSSE)

# Convert to relative deviation
WSSE_relative = []
for WSSE in WSSE_global:
    WSSE_relative.append(np.abs(np.array(WSSE))/np.abs(np.array(WSSE_global[-2]))*100-100)

# Visualize
colors = ['black', 'blue', 'red', 'green', 'orange']
labels = prodfunc_lst

fig,ax=plt.subplots()
for i,WSSE in enumerate(WSSE_relative):
    if i!=3:
        # Add to plot
        ax.plot(tau_lst, WSSE, color=colors[i], label=labels[i])
ax.set_ylabel('$\Delta$ WSSE "strongly critical" (%)')
ax.set_xlabel('Average restocking time $\\tau$ (days)')
#ax.set_ylim([-5,5])
ax.legend()
plt.show()
plt.close()

# Visualize
colors = ['black', 'blue', 'red', 'green', 'orange']
labels = prodfunc_lst

fig,ax=plt.subplots()
for i,WSSE in enumerate(WSSE_global):
    # Add to plot
    ax.plot(tau_lst, WSSE, color=colors[i], label=labels[i])
ax.set_ylabel('WSSE (-)')
ax.set_xlabel('Average restocking time $\\tau$ (days)')
#ax.set_ylim([-5,5])
ax.legend()
plt.show()
plt.close()