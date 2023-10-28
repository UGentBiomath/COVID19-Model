# general packages
import itertools
import numpy as np
import pandas as pd
# EPNM functions
from EPNM.models.utils import initialize_model
from EPNM.data.parameters import get_model_parameters
from EPNM.data.utils import get_sector_labels, get_sector_names, aggregate_simulation, get_sectoral_conversion_matrix
from EPNM.models.TDPF import household_demand_shock, compute_income_expectations
from EPNM.models.draw_functions import draw_function as draw_function
from EPNM.data.calibration_data import get_NAI_value_added, get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand

# start- and enddate simulation
start_sim = '2020-03-01'
end_sim = '2021-03-31'

###############
## load data ##
###############

# load (relative) data
data_employment = get_employment_survey().loc[slice('2020-04-01',end_sim), slice(None)]
data_revenue = get_revenue_survey().loc[slice('2020-04-01',end_sim), slice(None)]
data_GDP = get_synthetic_GDP().loc[slice('2020-04-01',end_sim), slice(None)]
data_B2B = get_B2B_demand().loc[slice('2020-04-01',end_sim), slice(None)]
# aggregate to quarters
data_employment = data_employment.groupby([pd.Grouper(freq='Q', level='date'),] + [data_employment.index.get_level_values('NACE64')]).mean()
data_revenue = data_revenue.groupby([pd.Grouper(freq='Q', level='date'),] + [data_revenue.index.get_level_values('NACE64')]).mean()
data_GDP = data_GDP.groupby([pd.Grouper(freq='Q', level='date'),] + [data_GDP.index.get_level_values('NACE64')]).mean()
data_B2B = data_B2B.groupby([pd.Grouper(freq='Q', level='date'),] + [data_B2B.index.get_level_values('NACE21')]).mean()
# load objective function
#from EPNM.data.objective_function import compute_AAD
# pre-allocate a dataframe for the output
iterables = 3*[['Pichler et al.', 'Alleman et al.'],]
index = pd.MultiIndex.from_product(iterables, names=["labor supply", "household demand", "other demand"])
df = pd.Series(0, index=index)

# Aggregation function
import xarray as xr
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

# function to compute AAD
def compute_AAD(out, params, data_B2B, data_GDP, data_revenue, data_employment, weighted=True):
    """Computes the Average Absolute Deviation between model prediction and data
    """

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
                x=data_B2B.loc[date, sector]*100-100
                y=out_NACE21_quart.sel(NACE21=sector).sel(date=date)/out_NACE21.sel(NACE21=sector).isel(date=0)*100-100
                # Weighted euclidian distance in plane
                if weighted==True:
                    dist_abs_temp.append(B2B_demand[j]/sum(B2B_demand)*abs(abs(x)-abs(y.values)))
                    dist_temp.append(B2B_demand[j]/sum(B2B_demand)*(abs(x)-abs(y.values)))
                else:
                    dist_abs_temp.append(abs(abs(x)-abs(y.values)))
                    dist_temp.append((abs(x)-abs(y.values)))
        
        if weighted == True:
            dist_abs[i] = np.sum(dist_abs_temp)
            dist[i] = np.sum(dist_temp)
        else:
            dist_abs[i] = np.mean(dist_abs_temp)
            dist[i] = np.mean(dist_temp)

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
                    if weighted == True:
                        dist_abs_temp.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k])*abs(abs(x)-abs(y.values)))
                        dist_temp.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k])*(abs(x)-abs(y.values)))
                        cumsize.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k]))
                    else:
                        dist_abs_temp.append(abs(abs(x)-abs(y.values)))
                        dist_temp.append((abs(x)-abs(y.values)))

            # Weighted euclidian distance in plane
            x=data.loc[date, 'BE']*100-100
            y=out_quart.sum(dim='NACE64').sel(date=date)/out[states[k]].sum(dim='NACE64').isel(date=0)*100-100
            dist_abs_temp.append(abs(abs(x)-abs(y.values)))
            dist_temp.append((abs(x)-abs(y.values)))
            # Average
            if weighted==True:
                dist_abs[i] = 1/(1+sum(cumsize))*np.sum(dist_abs_temp)
                dist[i] = 1/(1+sum(cumsize))*np.sum(dist_temp)
            else:
                dist_abs[i] = np.mean(dist_abs_temp)
                dist[i] = np.mean(dist_temp)

        hyperdist_abs.append(np.mean(dist_abs))
        hyperdist.append(np.mean(dist))

    return np.mean(hyperdist_abs), np.mean(hyperdist)

#########################
## load different DoFs ##
#########################

## DoF 1: labor supply shocks + DoF 2: household demand and parameters
# pichler 
params, model = initialize_model(shocks='pichler', prodfunc='half_critical')
parameters_pichler = {
    'l_s_1': params['l_s_1'],
    'l_s_2': params['l_s_2'],
    'c_s': params['c_s'],
    'ratio_c_s': 0.5,
    'tau': 10,
    'gamma_F': 15,
    'gamma_H': 2*15,
}
# alleman
params, model = initialize_model(shocks='alleman', prodfunc='half_critical')
parameters_alleman = {
    'l_s_1': params['l_s_1'],
    'l_s_2': params['l_s_2'],
    'c_s': params['c_s'],
    'ratio_c_s': 0.5,
    'tau': 1,
    'gamma_F': 28,
    'gamma_H': 2*28,
}
## DoF 3: other demand shock
# pichler
params_pichler, model_pichler = initialize_model(shocks='pichler', prodfunc='half_critical')
# alleman
params_alleman, model_alleman = initialize_model(shocks='alleman', prodfunc='half_critical')

########################################
## Compute accuracy of 8 combinations ##
########################################

### Alleman
### ~~~~~~~

## pure alleman
out = model_alleman.sim([start_sim, end_sim], tau=1)
accuracy, __ = compute_AAD(out, params_alleman, data_B2B, data_GDP, data_revenue, data_employment)
df.loc['Alleman et al.', 'Alleman et al.', 'Alleman et al.'] = accuracy
print(compute_AAD(out, params_alleman, data_B2B, data_GDP, data_revenue, data_employment))
## alleman + labor supply shock pichler + parameters pichler
# update parameters
model_alleman.parameters.update(parameters_pichler)
# simulate and compute accuracy
out = model_alleman.sim([start_sim, end_sim], tau=1)
accuracy, __ = compute_AAD(out, params_alleman, data_B2B, data_GDP, data_revenue, data_employment)
df.loc['Pichler et al.', 'Pichler et al.', 'Alleman et al.'] = accuracy

## alleman + labor supply shock pichler
# reset paramaters
model_alleman.parameters.update(parameters_alleman)
# update parameters
model_alleman.parameters.update({
    'l_s_1': parameters_pichler['l_s_1'],
    'l_s_2': parameters_pichler['l_s_2'],
})
# simulate and compute accuracy
out = model_alleman.sim([start_sim, end_sim], tau=1)
accuracy, __ = compute_AAD(out, params_alleman, data_B2B, data_GDP, data_revenue, data_employment)
df.loc['Pichler et al.', 'Alleman et al.', 'Alleman et al.'] = accuracy

## alleman + parameters pichler
# set pichler's parameters
model_alleman.parameters.update(parameters_pichler)
# set alleman's labor supply shock
model_alleman.parameters.update({
    'l_s_1': parameters_alleman['l_s_1'],
    'l_s_2': parameters_alleman['l_s_2'],
})
# simulate and compute accuracy
out = model_alleman.sim([start_sim, end_sim], tau=1)
accuracy, __ = compute_AAD(out, params_alleman, data_B2B, data_GDP, data_revenue, data_employment)
df.loc['Alleman et al.', 'Pichler et al.', 'Alleman et al.'] = accuracy

print('\n')

### Pichler
### ~~~~~~~

## pure pichler
out = model_pichler.sim([start_sim, end_sim], tau=1)
accuracy, __ = compute_AAD(out, params_pichler, data_B2B, data_GDP, data_revenue, data_employment)
df.loc['Pichler et al.', 'Pichler et al.', 'Pichler et al.'] = accuracy

## pichler + labor supply shock alleman + parameters alleman
# update parameters
model_pichler.parameters.update(parameters_alleman)
# simulate and compute accuracy
out = model_pichler.sim([start_sim, end_sim], tau=1)
accuracy, __ = compute_AAD(out, params_pichler, data_B2B, data_GDP, data_revenue, data_employment)
df.loc['Alleman et al.', 'Alleman et al.', 'Pichler et al.'] = accuracy

## pichler + labor supply shock alleman
# reset paramaters
model_pichler.parameters.update(parameters_pichler)
# update parameters
model_pichler.parameters.update({
    'l_s_1': parameters_alleman['l_s_1'],
    'l_s_2': parameters_alleman['l_s_2'],
})
# simulate and compute accuracy
out = model_pichler.sim([start_sim, end_sim], tau=1)
accuracy, __ = compute_AAD(out, params_pichler, data_B2B, data_GDP, data_revenue, data_employment)
df.loc['Alleman et al.', 'Pichler et al.', 'Pichler et al.'] = accuracy

## pichler + parameters alleman
# set alleman's parameters
model_pichler.parameters.update(parameters_alleman)
# set pichler's labor supply shock
model_pichler.parameters.update({
    'l_s_1': parameters_pichler['l_s_1'],
    'l_s_2': parameters_pichler['l_s_2'],
})
# simulate and compute accuracy
out = model_pichler.sim([start_sim, end_sim], tau=1)
accuracy, __ = compute_AAD(out, params_pichler, data_B2B, data_GDP, data_revenue, data_employment)
df.loc['Pichler et al.', 'Alleman et al.', 'Pichler et al.'] = accuracy

# save result
df.to_csv('comparison_accuracy.csv')