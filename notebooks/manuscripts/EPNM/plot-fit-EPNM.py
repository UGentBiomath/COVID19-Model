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

# Start- and enddate simulation
start_sim = '2020-03-01'
end_sim = '2021-03-31'
# Start- and enddate visualization
start_vis = '2020-04-01'
end_vis = '2021-03-31'
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

# Initialize model
params, model = initialize_model(shocks='alleman', prodfunc='half_critical')

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

# Draw function
# TODO

# Simulate
out = model.sim([start_sim, end_sim], method='RK45', rtol=1e-4)

###################
## Make the plot ##
###################

# B2B demand
dates = data_B2B.index.get_level_values('date').unique()
sectors = data_B2B.index.get_level_values('NACE21').unique()
out_NACE21 = aggregate_NACE21(out['O'])
out_NACE21_quart = out_NACE21.resample(date='Q').mean()
B2B_demand = np.matmul(params['O_j'], np.transpose(get_sectoral_conversion_matrix('NACE64_NACE21')))
model_dates = out_NACE21_quart.coords['date']

titles = ['2020Q2', '2020Q3', '2020Q4', '2021Q1']
grouping_sectors = ['Agriculture; Mining;\nManufacturing;\nUtilities; Construction', 'Wholesale & Retail;\nTransport; Accomodation;\nRecreation; Services',
                    'Information & Communication;\nInsurance & Finance;\nPrivate administration', 'Public administration;\nEducation; Defence;\nHuman health']

fig,ax=plt.subplots(ncols=4, sharey=True, figsize=(0.75*21,4))
for i,date in enumerate(dates):
    for j,sector in enumerate(sectors):
        if sector!='U':
            if sector in ['A', 'B', 'C', 'D', 'E', 'F']:
                color = 'black'
                label = 'Agriculture; Mining; Manufacturing; Utilities; Construction'
            elif sector in ['G', 'H', 'I', 'R', 'S', 'T']:
                color = 'red'
                label = 'Wholesale & Retail; Transport; Accomodation; Recreation; Services'
            elif sector in ['J', 'K', 'L', 'M', 'N']:
                color = 'blue'
                label = 'Information & Communication; Insurance & Finance; Private administration'
            elif sector in ['O', 'P', 'Q']:
                color = 'green'
                label = 'Public administration; Education; Defence; Human health'

            x=data_B2B.loc[date, sector]-100
            y=out_NACE21_quart.sel(NACE21=sector).sel(date=date)/out_NACE21.sel(NACE21=sector).isel(date=0)*100-100
            ax[i].scatter(x, y, s=B2B_demand[j], color=color, alpha=0.5)

            # Sector label
            if sector in ['I', 'R', 'S']:
                ax[i].annotate(sector,  xy=(x - 1, y + 2), fontsize=9)

    ax[i].set_title(titles[i])
    ax[i].set_xlabel('Observed decline (%)')
    ax[i].plot(np.linspace(start=-100, stop=10), np.linspace(start=-100, stop=10), color='black', linewidth=1, linestyle='--')
    ax[i].set_xlim([-100,10])
    ax[i].set_ylim([-100,10])

# Label y axis
ax[0].set_ylabel('Predicted decline (%)')
# Custom legend
from matplotlib.lines import Line2D
custom_circles = [Line2D([0], [0], marker='o', markersize=10, color='w', markerfacecolor='black', alpha=0.5),
                    Line2D([0], [0], marker='o', markersize=10, color='w', markerfacecolor='red', alpha=0.5),
                    Line2D([0], [0], marker='o', markersize=10, color='w', markerfacecolor='blue', alpha=0.5),
                    Line2D([0], [0], marker='o', markersize=10, color='w', markerfacecolor='green', alpha=0.5)
                 ]        
ax[3].legend(custom_circles, grouping_sectors, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
leg = ax[3].get_legend()
leg.get_frame().set_linewidth(0.0)

# Show figure
plt.tight_layout()
plt.show()
plt.close()