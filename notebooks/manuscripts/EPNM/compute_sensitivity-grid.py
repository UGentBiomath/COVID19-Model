# General packages
import itertools
from math import comb
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
# Load (relative) data
data_employment = get_employment_survey().loc[slice('2020-04-01',end_sim), slice(None)]
data_revenue = get_revenue_survey().loc[slice('2020-04-01',end_sim), slice(None)]
data_GDP = get_synthetic_GDP().loc[slice('2020-04-01',end_sim), slice(None)]
data_B2B = get_B2B_demand().loc[slice('2020-04-01',end_sim), slice(None)]
# Aggregate to quarters
data_employment = data_employment.groupby([pd.Grouper(freq='Q', level='date'),] + [data_employment.index.get_level_values('NACE64')]).mean()
data_revenue = data_revenue.groupby([pd.Grouper(freq='Q', level='date'),] + [data_revenue.index.get_level_values('NACE64')]).mean()
data_GDP = data_GDP.groupby([pd.Grouper(freq='Q', level='date'),] + [data_GDP.index.get_level_values('NACE64')]).mean()
data_B2B = data_B2B.groupby([pd.Grouper(freq='Q', level='date'),] + [data_B2B.index.get_level_values('NACE21')]).mean()

# Initialize model
params, model = initialize_model(shocks='alleman_exogenous', prodfunc='half_critical')

## Calibration: Sector reductions
# production functions
prodfuncs = ['leontief','strongly_critical','half_critical','weakly_critical','linear']
# shocks
consumer_facing = [0.70,0.80,0.90,0.99,]
industry = [0.10, 0.20, 0.30, 0.40]
f_s = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
ratio_c_s = [0.20, 0.40, 0.60, 0.80]
# parameters
tau = [7, 14, 21, 28]
hiring_firing = [7, 14, 21, 28, 35]
# excluded
l2 = [6*7,]
retail = [0,]
combinations = list(itertools.product(*[consumer_facing, industry, f_s, ratio_c_s, tau, hiring_firing, l2, retail]))

# make an output dataframe
iterables = [prodfuncs, consumer_facing, industry, f_s, ratio_c_s, tau, hiring_firing, l2, retail]
index = pd.MultiIndex.from_product(iterables, names=["production_function", "consumer_facing", "industry", "f_s", "ratio_c_s", "tau", "hiring_firing", "l2", "retail"])
df = pd.DataFrame(0, index=index, columns=['weighted', 'unweighted'])

# Sector labels
industry_labels = []
for sector in get_sector_labels('NACE64'):
    if sector[0] in ['A', 'B', 'C', 'D', 'E', 'F']:
        industry_labels.append(sector)
consumer_facing_labels = ['I55-56', 'N77', 'N79', 'R90-92', 'R93', 'S94', 'S96']
retail_labels = ['G46', 'G47', 'S95']
labels = [industry_labels, consumer_facing_labels, retail_labels]

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

def compute_AAD(out, weighted=True):
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

def mp_run_sensitivity(combinations, model, params):
    # Unpack arguments
    cons, ind, f_s, ratio_c_s, tau, hiring_firing, l2, ret = combinations
    # Construct c_s
    c_s = params['c_s']
    c_s[[get_sector_labels('NACE64').index(lab) for lab in industry_labels]] = ind
    c_s[[get_sector_labels('NACE64').index(lab) for lab in consumer_facing_labels]] = cons
    c_s[[get_sector_labels('NACE64').index(lab) for lab in retail_labels]] = ret    
    # Update parameters
    model.parameters['ratio_c_s'] = ratio_c_s
    model.parameters['l2'] = l2
    model.parameters['c_s'] = c_s
    model.parameters['f_s'] = f_s
    model.parameters['tau'] = tau
    model.parameters['gamma_F'] = hiring_firing
    model.parameters['gamma_H'] = 2*hiring_firing
    # Simulate model (discrete)
    out = model.sim([start_sim, end_sim], tau=1)
    return compute_AAD(out, weighted=False)[0], compute_AAD(out, weighted=True)[0]


from functools import partial
import multiprocessing as mp
processes = 36
print(f'\nTotal number of simulations per core: {len(prodfuncs)*len(combinations)/processes:.0f}')
w_res = []
uw_res = []
for i, prodfunc in enumerate(prodfuncs):
    print(f'\nInitializing new model: {prodfunc}\n')
    params, model = initialize_model(shocks='alleman', prodfunc=prodfunc)
    with mp.Pool(processes) as pool:
        res = pool.map(partial(mp_run_sensitivity, model=model, params=params), combinations)
    # Extract results
    uw=[]
    w=[]
    for r in res:
        uw.append(r[0])
        w.append(r[1])
    uw_res.append(np.reshape(uw, [len(consumer_facing), len(industry), len(f_s), len(ratio_c_s), len(tau), len(hiring_firing), len(l2), len(retail)]))
    w_res.append(np.reshape(w, [len(consumer_facing), len(industry), len(f_s), len(ratio_c_s), len(tau), len(hiring_firing), len(l2), len(retail)]))
# stack production functions
unweighted = np.stack(uw_res, axis=0)
weighted = np.stack(w_res, axis=0)
# save results to dataframe
df = pd.concat([pd.Series(unweighted.flatten(), index=index, name='unweighted'),pd.Series(weighted.flatten(), index=index, name='weighted')], axis=1)
df.to_csv('sensitivity-grid.csv')



