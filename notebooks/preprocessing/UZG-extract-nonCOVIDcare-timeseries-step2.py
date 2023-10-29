"""
This script converts the intermediate dataset on the postponement of non-COVID-19 care, located in `~/data/covid19_DTM/interim/QALY_model/postponement_non_covid_care/UZG/MZG_2016_2021.xlsx`
folder (not on github, contact tijs.alleman@ugent.be) into the final format (postponement of care vs. prepandemic baseline)

The intermediate format is a pd.DataFrame, which, for each MDC, age and type of stay contains the total number of patients on a given date (from 2016-2021).
The intermediate dataframe is not on Github because of its size (100 Mb).

The final timeseries to be used in the analysis is a pd.DataFrame containing the data from 2020-2021 (pandemic data), given as a percentage reduction compared to a prepandemic baseline (2017-2019).
The computation was split because step 1 takes much longer than step 2.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime as datetime
from tqdm import tqdm

###############
## Load data ##
###############

print('\n(1) Loading intermediate dataset\n')

# Name and location of intermediate dataset
abs_dir = os.getcwd()
rel_dir = '../../data/QALY_model/interim/postponed_healthcare/MZG_2016_2021.csv'
# Name and location of saved dataframe
result_normdata =  '../../data/QALY_model/interim/postponed_healthcare/MZG_2020_2021_normalized.csv'
result_baseline = '../../data/QALY_model/interim/postponed_healthcare/MZG_baseline.csv'
# Load data
df = pd.read_csv(os.path.join(abs_dir, rel_dir), index_col=[0,1,2,3], parse_dates=True,
                    dtype = {'APR_MDC_key': str, 'age_group': str, 'stay_type': str, 'n_patients': int})
# Sum to weekly frequency to reduce noise in the dataset
df = df.reset_index().dropna().groupby(by=['APR_MDC_key', 'date']).sum().sort_index()

# Immediately pop the age groups and hospitalisation types as we won't use them
del df['age_group']
del df['stay_type']

############################################################
## Construct baseline dataframe using data from 2017-2019 ##
############################################################

bootstrap_repeats = 500
subset_size=3

print('\n(2) Constructing baseline dataframe using data from 2016-2020\n')

# Define a target dataframe containing the week and day number instead of the date
week_numbers = list(range(1,53+1))
day_numbers = list(range(1,7+1))
iterables=[]
names=[]
for index_name in df.index.names:
    if index_name != 'date':
        iterables += [df.index.get_level_values(index_name).unique()]
        names += [index_name,]
iterables.append(week_numbers)
iterables.append(day_numbers)
names.append('week_number')
names.append('day_number')
# Add bootstrap axis
iterables.append(list(range(bootstrap_repeats)))
names.append('bootstrap_sample')
index = pd.MultiIndex.from_product(iterables, names=names)
baseline_df = pd.Series(index=index, name='n_patients', data=np.zeros(len(index), dtype=int))
# Use all data from the jan. 2016 until jan. 2020 as baseline
baseline = df[((df.index.get_level_values('date')<datetime(2020,1,1))&(df.index.get_level_values('date')>=datetime(2016,1,1)))]

# compute
names=['week_number', 'day_number']
iterables=[baseline_df.index.get_level_values('week_number').unique(), baseline_df.index.get_level_values('day_number').unique()]
index = pd.MultiIndex.from_product(iterables, names=names)
merge_df = pd.Series(index=index, name='n_patients', data=np.zeros(len(index), dtype=float))
# Loop over all possible indices, convert date to day of year, take average of values with same day-of-year number
with tqdm(total=len(baseline.index.get_level_values('APR_MDC_key').unique())*bootstrap_repeats) as pbar:
    for APR_MDC_key in baseline.index.get_level_values('APR_MDC_key').unique():
        for idx in baseline_df.index.get_level_values('bootstrap_sample').unique():
            # Extract dataseries
            data = baseline.loc[(APR_MDC_key,),:]
            # Reset index to 'unlock' the date
            data.reset_index(inplace=True)
            # Convert the date to week and day number
            data['week_number'] = pd.to_datetime(data['date'].values).isocalendar().week.values
            data['day_number'] = pd.to_datetime(data['date'].values).isocalendar().day.values
            # pop the date
            del data['date']
            # Perform a groupby 'date' operation with mean() to take the mean of all values with similar daynumber
            d = data.groupby(by=['week_number','day_number']).apply(lambda x: np.mean(x.sample(n=subset_size, replace=True)))
            d.name = 'n_patients'
            baseline_df.loc[APR_MDC_key, slice(None), slice(None), idx] = pd.merge(d, merge_df, how='right', on=['week_number','day_number']).ffill()['n_patients_x'].values   
            pbar.update(1)

# Save baseline
baseline_df.to_csv(os.path.join(abs_dir, result_baseline))

#####################################################################
## Normalizing pandemic data (2020-2021) with prepandemic baseline ##
#####################################################################

print('\n(3) Normalizing pandemic data (2020-2021) with prepandemic baseline\n')

# Consider all data from the beginning of 2020 as the actual 'data'
data_df = df[df.index.get_level_values('date')>=datetime(2020,1,1)]
# Initialize target dataframe
iterables=[data_df.index.get_level_values('APR_MDC_key').unique(), data_df.index.get_level_values('date').unique(), list(range(bootstrap_repeats))]
names=['APR_MDC_key', 'date', 'bootstrap_sample']
index = pd.MultiIndex.from_product(iterables, names=names)
target_df = pd.Series(index=index, name='rel_hospitalizations', data=np.zeros(len(index), dtype=int))

# Loop over all possible indices, convert date to day of year, take average of values with same day-of-year number
with tqdm(total=len(data_df.index.get_level_values('APR_MDC_key').unique())*bootstrap_repeats) as pbar:
    for APR_MDC_key in data_df.index.get_level_values('APR_MDC_key').unique():
        for idx in baseline_df.index.get_level_values('bootstrap_sample').unique():
            # Extract dataseries
            data = data_df.loc[(APR_MDC_key,),:]
            # Reset index to 'unlock' the date
            data.reset_index(inplace=True)
            # Convert the date to week and day number
            data['week_number'] = pd.to_datetime(data['date'].values).isocalendar().week.values
            data['day_number'] = pd.to_datetime(data['date'].values).isocalendar().day.values
            # Extract baseline
            baseline = baseline_df.loc[(APR_MDC_key, slice(None), slice(None), idx)]
            # Perform computation
            tmp=np.zeros(len(data['date'].values))
            for jdx,date in enumerate(data['date'].values):
                week_number = data.iloc[jdx]['week_number']
                day_number = data.iloc[jdx]['day_number']
                if baseline.loc[week_number, day_number] != 0:
                    tmp[jdx] = data.iloc[jdx]['n_patients']/baseline.loc[week_number, day_number]
                else:
                    tmp[jdx] = 1
            # Assign result
            target_df.loc[(APR_MDC_key, slice(None), idx)] = tmp
            pbar.update(1)

#########################################################
## Convert bootstrap samples to meaningfull statistics ##
#########################################################

new_df = target_df.groupby(by=['APR_MDC_key', 'date']).median().to_frame()
new_df.rename(columns={'rel_hospitalizations': 'median'}, inplace=True)
new_df['mean'] = target_df.groupby(by=['APR_MDC_key', 'date']).mean()
new_df['std'] = target_df.groupby(by=['APR_MDC_key', 'date']).std()
new_df['q0.025'] = target_df.groupby(by=['APR_MDC_key', 'date']).quantile(q=0.025)
new_df['q0.975'] = target_df.groupby(by=['APR_MDC_key', 'date']).quantile(q=0.975)

############################################
## Convert to weekly data to reduce noise ##
############################################

level_values = new_df.index.get_level_values
new_df = new_df.groupby([level_values(i) for i in [0,]]
                       +[pd.Grouper(freq='W', level=-1)]).mean()

#################
## Save result ##
#################

print('\n(4) Saving result\n')

new_df.to_csv(os.path.join(abs_dir, result_normdata))