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

# # Start and enddate
# start_date = '2020-03-01' 
# end_date = '2020-09-01'

# # Define production functions
# prodfuncs = ['linear', 'weakly_critical', 'half_critical', 'strongly_critical', 'leontief']
# # define lower and upper bound on tau
# tau_list = [1, 30]
# # Define enddates
# end_lockdown_list = ['2020-05-01', '2020-09-01']

# fig,ax=plt.subplots(ncols=2, sharey=True)

# for i,end_lockdown in enumerate(end_lockdown_list):

#     # Loop over production functions
#     out_lower_list=[]
#     out_upper_list=[]
#     for prodfunc in prodfuncs:
#         # Initialize model
#         params, model = initialize_model(shocks='alleman', prodfunc=prodfunc)
#         # Change lockdown enddate to end of simulation
#         model.parameters.update({'t_end_lockdown_1': pd.Timestamp(end_lockdown)})
#         # Set lower tau
#         model.parameters.update({'tau': tau_list[0]})
#         # Slower lockdown release
#         model.parameters.update({'l2': 8*7})
#         # Simulate model
#         out_lower_list.append(model.sim([start_date, end_date], method='RK45', rtol=1e-4))
#         # Set upper tau
#         model.parameters.update({'tau': tau_list[1]})
#         # Simulate model
#         out_upper_list.append(model.sim([start_date, end_date], method='RK45', rtol=1e-4))

#     # Load data
#     data_GDP = get_synthetic_GDP(relative=True)
#     # List of colors
#     color_list = ['black', 'blue', 'red', 'green', 'orange']
#     label_list = ['linear', 'weakly critical', 'half critical', 'strongly critical', 'leontief']
#     # Visualize result
#     ax[i].scatter(data_GDP.loc[slice(None,end_date), 'BE'].index.get_level_values('date').unique(), data_GDP.loc[slice(None,end_date), 'BE']*100,
#                 color='black', alpha=0.8, linestyle='None', facecolors='none', s=80, linewidth=3, label='synthetic GDP (NBB)')
#     for j,out_lower in enumerate(out_lower_list):
#         #ax.plot(out['date'].values, out['x'].sum(dim='NACE64')/out['x'].sum(dim='NACE64').sel(date=out['date'].values[0])*100,
#         #        color=color_list[i], label=label_list[i])
#         ax[i].fill_between(out_lower['date'].values,
#                             out_lower['x'].sum(dim='NACE64')/out_lower['x'].sum(dim='NACE64').sel(date=out_lower['date'].values[0])*100,
#                             out_upper_list[j]['x'].sum(dim='NACE64')/out_upper_list[i]['x'].sum(dim='NACE64').sel(date=out_upper_list[j]['date'].values[0])*100,
#                             color=color_list[j], alpha=0.25, label=label_list[j])

#     # Set max_n ticks
#     ax[i].xaxis.set_major_locator(plt.MaxNLocator(5))
#     for tick in ax[i].get_xticklabels():
#         tick.set_rotation(45)
#     # Set y axis limit
#     ax[i].set_ylim([0,105])
#     # Horizontal line for lockdown release
#     ax[i].axvline(x=pd.Timestamp('2020-05-01'), color='black', linewidth=1.5, linestyle='--')

# # Set labels
# ax[0].set_ylabel('GDP change (%)')
# # Legend
# ax[1].legend()

# # Print to screen
# plt.tight_layout()
# plt.show()
# plt.close()

# PART II:
tau_lst = np.linspace(start=1, stop=50, num=10)
prodfunc_lst = ['linear', 'weakly_critical', 'half_critical', 'strongly_critical', 'leontief']

# Aggregation functions
import xarray as xr
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

def aggregate_NACE21(simulation_in):
    """ A function to convert a simulation of the economic IO model on the NACE64 level to the NACE21 level
        Also aggregates data to quarters temporily
    
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
    return simulation_out

# Start- and enddate visualization
start_vis = '2020-04-01'
end_vis = '2021-04-01'
# Load (relative) data
data_employment = get_employment_survey().loc[slice(start_vis,end_vis), slice(None)]
data_revenue = get_revenue_survey().loc[slice(start_vis,end_vis), slice(None)]
data_GDP = get_synthetic_GDP().loc[slice(start_vis,end_vis), slice(None)]
data_B2B = get_B2B_demand().loc[slice(start_vis,end_vis), slice(None)]
# Aggregate to quarters
data_employment = data_employment.groupby([pd.Grouper(freq='Q', level='date'),] + [data_employment.index.get_level_values('NACE64')]).mean()
data_revenue = data_revenue.groupby([pd.Grouper(freq='Q', level='date'),] + [data_revenue.index.get_level_values('NACE64')]).mean()
data_GDP = data_GDP.groupby([pd.Grouper(freq='Q', level='date'),] + [data_GDP.index.get_level_values('NACE64')]).mean()
data_B2B = data_B2B.groupby([pd.Grouper(freq='Q', level='date'),] + [data_B2B.index.get_level_values('NACE21')]).mean()

# Define function to compute euclidian distance
def compute_euclidian_distance(tau, model):
    # Assign parameter
    model.parameters['tau'] = tau
    # Simulate model
    out = model.sim([start_vis, end_vis], method='RK45', rtol=1e-4)
    # Pre-allocate metric
    hyperdist_abs = []
    hyperdist = []

    # B2B Weighted Euclidian distance
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sectors = data_B2B.index.get_level_values('NACE21').unique()
    dates = data_B2B.index.get_level_values('date').unique()
    out_NACE21 = aggregate_NACE21(out['O'])
    out_NACE21_quart = out_NACE21.resample(date='Q').mean()
    B2B_demand = np.matmul(params['O_j'], np.transpose(get_sectoral_conversion_matrix('NACE64_NACE21')))
    dist_abs=np.zeros(4)
    dist=np.zeros(4)
    for i,date in enumerate(dates):
        dist_abs_temp=[]
        dist_temp=[]
        for j,sector in enumerate(sectors):
            if sector!='U':
                x=data_B2B.loc[date, sector]-100
                y=out_NACE21_quart.sel(NACE21=sector).sel(date=date)/out_NACE21.sel(NACE21=sector).isel(date=0)*100-100
                # Weighted euclidian distance in plane
                dist_abs_temp.append(B2B_demand[j]/sum(B2B_demand)*abs(abs(x)-abs(y.values)) )
                dist_temp.append(B2B_demand[j]/sum(B2B_demand)*(abs(x)-abs(y.values)) )
        dist_abs[i] = np.sum(dist_abs_temp)
        dist[i] = np.sum(dist_temp)
    hyperdist_abs.append(np.mean(dist_abs))
    hyperdist.append(np.mean(dist))

    # GDP, revenue, employment
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    states = ['x', 'x', 'l']
    sizes = [params['x_0'], params['x_0'], params['l_0']]

    dist_abs=np.zeros(4)
    dist=np.zeros(4)
    for k, data in enumerate([data_GDP, data_revenue, data_employment]):
        dates = data.index.get_level_values('date').unique()
        sectors = data.index.get_level_values('NACE64').unique()
        out_quart = out[states[k]].resample(date='Q').mean()
        for i,date in enumerate(dates):
            cumsize=[]
            dist_abs_temp=[]
            dist_temp=[]
            for j,sector in enumerate(sectors):
                if sector != 'BE':
                    x=data.loc[date, sector]*100-100
                    y=out_quart.sel(NACE64=sector).sel(date=date)/out[states[k]].sel(NACE64=sector).isel(date=0)*100-100
                    # Weighted euclidian distance in plane
                    dist_abs_temp.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k])*abs(abs(x)-abs(y.values)) )
                    dist_temp.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k])*(abs(x)-abs(y.values)) )
                    cumsize.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k]))

            # Weighted euclidian distance in plane
            x=data.loc[date, 'BE']*100-100
            y=out_quart.sum(dim='NACE64').sel(date=date)/out[states[k]].sum(dim='NACE64').isel(date=0)*100-100
            dist_abs_temp.append(abs(abs(x)-abs(y.values)))
            dist_temp.append((abs(x)-abs(y.values)))
            # Average
            dist_abs[i] = 1/(1+sum(cumsize))*np.sum(dist_abs_temp)
            dist[i] = 1/(1+sum(cumsize))*np.sum(dist_temp)
        hyperdist_abs.append(np.mean(dist_abs))
        hyperdist.append(np.mean(dist))


    print(np.mean(hyperdist_abs), np.mean(hyperdist))
    return np.mean(hyperdist_abs), np.mean(hyperdist)

# Perform computation
dist_abs_global = []
dist_global = []
for prodfunc in prodfunc_lst:
    print(prodfunc)
    dist_abs = []
    dist = []
    # Initialize model
    params, model = initialize_model(shocks='alleman', prodfunc=prodfunc)
    # Compute WSSE vs. tau
    for tau in tau_lst:
        theta = np.array([tau,])
        d_a, d = compute_euclidian_distance(theta, model)
        dist_abs.append(d_a)
        dist.append(d)
    dist_abs_global.append(dist_abs)
    dist_global.append(dist)

# Visualize
colors = ['black', 'blue', 'red', 'green', 'orange']
labels = prodfunc_lst

fig,ax=plt.subplots()
for i,dist_abs in enumerate(dist_abs_global):
    # Add to plot
    ax.plot(tau_lst, dist_abs, color=colors[i], label=labels[i])
    #ax[1].plot(tau_lst, dist_global[i], color=colors[i], label=labels[i])
ax.set_ylabel('|Euclidian distance| (%)')
ax.set_xlabel('Average restocking time $\\tau$ (days)')
ax.legend()
plt.show()
plt.close()

fig,ax=plt.subplots()
for i,dist_abs in enumerate(dist_abs_global):
    # Add to plot
    ax.plot(tau_lst, dist_global[i], color=colors[i], label=labels[i])
ax.set_ylabel('Euclidian distance (%)')
ax.set_xlabel('Average restocking time $\\tau$ (days)')
ax.legend()
plt.show()
plt.close()