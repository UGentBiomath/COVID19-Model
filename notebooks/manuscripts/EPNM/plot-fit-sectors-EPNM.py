# General packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# EPNM functions
from EPNM.models.utils import initialize_model
from EPNM.data.utils import get_sector_labels, get_sector_names, aggregate_simulation, get_sectoral_conversion_matrix
from EPNM.data.calibration_data import get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand
# Nicer colors
colors = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
          "green" : "#009E73", "yellow" : "#F0E442",
          "blue" : "#0072B2", "red" : "#D55E00",
          "pink" : "#CC79A7", "black" : "#000000"}

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
params, model = initialize_model(shocks='pichler', prodfunc='half_critical')

# Pichler parameters
model.parameters.update({'gamma_F': 15, 'gamma_H': 30, 'tau': 10, 'rho': 1-(1-0.60)/90})

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

# Simulate
out = model.sim([start_sim, end_sim], tau=1)

###################
## Make the plot ##
###################

titles = ['2020Q2', '2020Q3', '2020Q4', '2021Q1']
grouping_sectors = ['Agriculture; Mining;\nManufacturing;\nUtilities; Construction', 'Wholesale & Retail;\nTransport; Accomodation;\nRecreation; Services',
                    'Information & Communication;\nInsurance & Finance;\nPrivate administration', 'Public administration;\nEducation; Defence;\nHuman health']

hyperdist_abs = []
hyperdist = []

fig,ax=plt.subplots(nrows=4, ncols=4, sharey=True, sharex=True, figsize=(11.69,8.27))

################
## B2B demand ##
################

dates = data_B2B.index.get_level_values('date').unique()
sectors = data_B2B.index.get_level_values('NACE21').unique()
out_NACE21 = aggregate_NACE21(out['O'])
out_NACE21_quart = out_NACE21.resample(date='Q').mean()
model_dates = out_NACE21_quart.coords['date']
B2B_demand = np.matmul(params['O_j'], np.transpose(get_sectoral_conversion_matrix('NACE64_NACE21')))

dist_abs=np.zeros(4)
dist=np.zeros(4)
for i,date in enumerate(dates):
    dist_abs_temp=[]
    dist_temp=[]
    for j,sector in enumerate(sectors):
        if sector!='U':
            if sector in ['A', 'B', 'C', 'D', 'E', 'F']:
                color = 'black'
            elif sector in ['G', 'H', 'I', 'R', 'S', 'T']:
                color = colors['red']
            elif sector in ['J', 'K', 'L', 'M', 'N']:
                color = colors['blue']
            elif sector in ['O', 'P', 'Q']:
                color = colors['green']

            # Plot
            x=data_B2B.loc[date, sector]*100-100
            y=out_NACE21_quart.sel(NACE21=sector).sel(date=date)/out_NACE21.sel(NACE21=sector).isel(date=0)*100-100
            if color==colors['black']:
                alpha = 0.30
            else:
                alpha = 0.50
            ax[0,i].scatter(x, y, s=B2B_demand[j]/sum(B2B_demand)*1000, color=color, alpha=alpha)
            # Weighted euclidian distance in plane
            dist_abs_temp.append(B2B_demand[j]/sum(B2B_demand)*abs(abs(x)-abs(y.values)) )
            dist_temp.append(B2B_demand[j]/sum(B2B_demand)*(abs(x)-abs(y.values)) )
            # Sector label
            if sector in ['I', 'R', 'S']:
                ax[0,i].annotate(sector,  xy=(x - 2, y + 2), fontsize=7)
            # circle around transport
            if sector == 'H':
                from matplotlib import pyplot as plt, patches
                circle = patches.Circle((x, y), radius=12, color='black', linewidth=0.75, linestyle='--', alpha=0.9, fill=False)
                ax[0,i].add_patch(circle)

    dist_abs[i] = np.sum(dist_abs_temp)
    dist[i] = np.sum(dist_temp)

    # text box with average euclidian distance in plane
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    ax[0,i].text(0.05, 0.95, f"{dist_abs[i]:.1f}; {dist[i]:.1f} %", transform=ax[0,i].transAxes, fontsize=7,
                 verticalalignment='top', bbox=props)

    ax[0,i].set_title(titles[i])
    ax[0,i].plot(np.linspace(start=-100, stop=10), np.linspace(start=-100, stop=10), color='black', linewidth=1, linestyle='--')
    ax[0,i].set_xlim([-100,10])
    ax[0,i].set_ylim([-100,10])

hyperdist_abs.append(np.mean(dist_abs))
hyperdist.append(np.mean(dist))

# Label y axis
ax[0,0].set_ylabel('B2B transactions\nprediction (%)')

datasets = [data_GDP, data_revenue, data_employment]
states = ['x', 'x', 'l']
sizes = [params['x_0'], params['x_0'], params['l_0']]
print_label = [['G45',],['G45','N79', 'R93'],['G45','N79', 'R93']]
offset = [[-4,5],[-4,5],[-4,5]]
ylabels = ['Synthetic GDP\nprediction (%)', 'Revenue\nprediction (%)', 'Employment\nprediction (%)']

#########
## GDP ##
#########

dist_abs=np.zeros(4)
dist=np.zeros(4)

for k, data in enumerate(datasets):
    dates = data.index.get_level_values('date').unique()
    sectors = data.index.get_level_values('NACE64').unique()
    out_quart = out[states[k]].resample(date='Q').mean()
    for i,date in enumerate(dates):
        cumsize=[]
        dist_abs_temp=[]
        dist_temp=[]
        for j,sector in enumerate(sectors):
            if sector != 'BE':
                if sector[0] in ['A', 'B', 'C', 'D', 'E', 'F']:
                    color = 'black'
                elif sector[0] in ['G', 'H', 'I', 'R', 'S', 'T']:
                    color = colors['red']
                elif sector[0] in ['J', 'K', 'L', 'M', 'N']:
                    color = colors['blue']
                elif sector[0] in ['O', 'P', 'Q']:
                    color = colors['green']

                # Plot
                x=data.loc[date, sector]*100-100
                y=out_quart.sel(NACE64=sector).sel(date=date)/out[states[k]].sel(NACE64=sector).isel(date=0)*100-100
                if color==colors['black']:
                    alpha = 0.30
                else:
                    alpha = 0.50
                ax[k+1,i].scatter(x, y, s=sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k])*2000, color=color, alpha=alpha)
                # Weighted euclidian distance in plane
                dist_abs_temp.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k])*abs(abs(x)-abs(y.values)) )
                dist_temp.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k])*(abs(x)-abs(y.values)) )
                cumsize.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k]))
                # Sector label
                if sector in print_label[k]:
                    ax[k+1,i].annotate(sector,  xy=(x + offset[k][0], y + offset[k][1]), fontsize=7)
                # Circle around transport
                if ((sector == 'H49')):
                    from matplotlib import pyplot as plt, patches
                    circle = patches.Circle((x, y), radius=12, color='black', linewidth=0.75, linestyle='--', alpha=0.9, fill=False)
                    ax[k+1,i].add_patch(circle)

        # Weighted euclidian distance in plane
        x=data.loc[date, 'BE']*100-100
        y=out_quart.sum(dim='NACE64').sel(date=date)/out[states[k]].sum(dim='NACE64').isel(date=0)*100-100
        dist_abs_temp.append(abs(abs(x)-abs(y.values)))
        dist_temp.append((abs(x)-abs(y.values)))
        # Average
        dist_abs[i] = 1/(1+sum(cumsize))*np.sum(dist_abs_temp)
        dist[i] = 1/(1+sum(cumsize))*np.sum(dist_temp)

        # text box with average euclidian distance in plane
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        ax[k+1,i].text(0.05, 0.95, f"{dist_abs[i]:.1f} %; {dist[i]:.1f} %", transform=ax[k+1,i].transAxes, fontsize=7,
                        verticalalignment='top', bbox=props)

        ax[k+1,i].plot(np.linspace(start=-100, stop=10), np.linspace(start=-100, stop=10), color='black', linewidth=1, linestyle='--')
        ax[k+1,i].set_xlim([-100,10])
        ax[k+1,i].set_ylim([-100,10])

        ax[k+1,0].set_ylabel(ylabels[k])
        ax[3,i].set_xlabel('observation (%)')

    hyperdist_abs.append(np.mean(dist_abs))
    hyperdist.append(np.mean(dist))

# Custom legend
from matplotlib.lines import Line2D
custom_circles = [Line2D([0], [0], marker='o', markersize=10, color='w', markerfacecolor='black', alpha=0.4),
                    Line2D([0], [0], marker='o', markersize=10, color='w', markerfacecolor=colors['red'], alpha=0.5),
                    Line2D([0], [0], marker='o', markersize=10, color='w', markerfacecolor=colors['blue'], alpha=0.5),
                    Line2D([0], [0], marker='o', markersize=10, color='w', markerfacecolor=colors['green'], alpha=0.5),
                 ]   

# Arrow and text on first plot
ax[0,0].annotate('optimistic',  xytext=(-95, -20), xy=(-50,-50), fontsize=9, arrowprops=dict(arrowstyle="<-"))
ax[0,0].annotate('pessimistic',  xytext=(-45, -85), xy=(-50,-50), fontsize=9, arrowprops=dict(arrowstyle="<-"))

# To the right
plt.legend(custom_circles, grouping_sectors, loc='upper right', bbox_to_anchor=(2.25, 1.04), ncol=1, fancybox=True, fontsize=8)
# Below
#plt.legend(custom_circles, grouping_sectors, loc='lower center', bbox_to_anchor=(-1.25, -0.75), ncol=4, fancybox=True, fontsize=8)

print(np.mean(hyperdist_abs), np.mean(hyperdist))
print(np.mean(hyperdist_abs[1:]), np.mean(hyperdist[1:]))

# Show figure
plt.tight_layout()
plt.show()
fig.savefig('plot-fit-sectors.pdf')
plt.close()