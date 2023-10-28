# General packages
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
from EPNM.data.objective_function import compute_AAD

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
params, model = initialize_model(shocks='alleman', prodfunc='half_critical')

## Calibration: Sector reductions
# production functions
prodfuncs = ['leontief','strongly_critical','half_critical','weakly_critical','linear']
# shocks
industry = [0, 0.15, 0.30, 0.45, 0.60]
retail = [0, 0.10, 0.20, 0.30, 0.40]
transport = [0, 0.20, 0.40, 0.60, 0.80, 1]
consumer_facing = [0.95, 0.99,]
ratio_c_s = [0, 0.25, 0.50, 0.75, 1]
# parameters
tau = [1, 7, 14, 21, 28]
hiring_firing = [1, 7, 14, 21, 28,]
# make all possible combinations
combinations = list(itertools.product(*[industry, retail, transport, consumer_facing, ratio_c_s, tau, hiring_firing,]))

# make an output dataframe
iterables = [prodfuncs, industry, retail, transport, consumer_facing, ratio_c_s, tau, hiring_firing,]
index = pd.MultiIndex.from_product(iterables, names=["production_function", "industry", "retail", "transport", "consumer-facing", "ratio_c_s", "tau", "hiring_firing",])
df = pd.DataFrame(0, index=index, columns=['weighted', 'unweighted'])

# Sector labels
industry_labels = []
for sector in get_sector_labels('NACE64'):
    if sector[0] in ['C', 'F']:
        industry_labels.append(sector)
consumer_facing_labels = ['I55-56', 'N77', 'N79', 'R90-92', 'R93', 'S94', 'S96', 'T97-98']
retail_labels = ['G46', 'G47', 'S95']
transport_labels = ['H49', 'H50', 'H51']
labels = [industry_labels, retail_labels, transport_labels, consumer_facing_labels]

def mp_run_sensitivity(combinations, model, params):
    # Unpack arguments
    ind, ret, tran, cons, ratio_c_s, tau, hiring_firing = combinations
    # Construct c_s
    c_s = params['c_s']
    c_s[[get_sector_labels('NACE64').index(lab) for lab in industry_labels]] = ind
    c_s[[get_sector_labels('NACE64').index(lab) for lab in retail_labels]] = ret    
    c_s[[get_sector_labels('NACE64').index(lab) for lab in transport_labels]] = tran
    c_s[[get_sector_labels('NACE64').index(lab) for lab in consumer_facing_labels]] = cons
    c_s[[get_sector_labels('NACE64').index(lab) for lab in retail_labels]] = ret    
    # Update parameters
    model.parameters['ratio_c_s'] = ratio_c_s
    model.parameters['c_s'] = c_s
    model.parameters['tau'] = tau
    model.parameters['gamma_F'] = hiring_firing
    model.parameters['gamma_H'] = 2*hiring_firing
    # Simulate model (discrete)
    out = model.sim([start_sim, end_sim], tau=1)
    print('done')
    return compute_AAD(out, params, data_B2B, data_GDP, data_revenue, data_employment, weighted=False)[0], \
                compute_AAD(out, params, data_B2B, data_GDP, data_revenue, data_employment, weighted=True)[0]

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
    uw_res.append(np.reshape(uw, [len(industry), len(retail), len(transport), len(consumer_facing), len(ratio_c_s), len(tau), len(hiring_firing),]))
    w_res.append(np.reshape(w, [len(industry), len(retail), len(transport), len(consumer_facing), len(ratio_c_s), len(tau), len(hiring_firing),]))
# stack production functions
unweighted = np.stack(uw_res, axis=0)
weighted = np.stack(w_res, axis=0)
# save results to dataframe
df = pd.concat([pd.Series(unweighted.flatten(), index=index, name='unweighted'),pd.Series(weighted.flatten(), index=index, name='weighted')], axis=1)
df.to_csv('sensitivity-grid.csv')