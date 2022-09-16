"""
This script converts the intermediate dataset on the postponement of non-COVID-19 care, located in `~/data/interim/QALY_model/postponement_non_covid_care/UZG/MZG_2016_2021.xlsx`
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
from tqdm import tqdm

###############
## Load data ##
###############

print('\n(1) Loading intermediate dataset\n')

# Name and location of intermediate dataset
abs_dir = os.getcwd()
rel_dir = '../../data/interim/QALY_model/postponement_non_covid_care/UZG/MZG_2016_2021.csv'
# Name and location of saved dataframe
result_folder =  '../../data/interim/QALY_model/postponement_non_covid_care/UZG/2020_2021_normalized.csv'
# Load data
df = pd.read_csv(os.path.join(abs_dir, rel_dir), index_col=[0,1,2,3], parse_dates=True)
# Protection against nans
df = df.reset_index().dropna()
df = df.set_index(['APR_MDC_key', 'age_group', 'stay_type', 'date']).sort_index()

############################################################
## Construct baseline dataframe using data from 2017-2019 ##
############################################################

print('\n(2) Constructing baseline dataframe using data from 2016-2020\n')

# Define a target dataframe containing the day-of-year number instead of the date
week_numbers = list(range(1,53+1))
day_numbers = list(range(1,7+1))
#day_of_year = np.linspace(1,365,num=364+1, dtype=int)
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
index = pd.MultiIndex.from_product(iterables, names=names)
baseline_df = pd.Series(index=index, name='n_patients', data=np.zeros(len(index), dtype=int))
# Use all data from the jan. 2016 until jan. 2020 as baseline
baseline = df[((df.index.get_level_values('date')<pd.Timestamp('2020-01-01'))&(df.index.get_level_values('date')>=pd.Timestamp('2016-01-01')))]

# Loop over all possible indices, convert date to day of year, take average of values with same day-of-year number
with tqdm(total=len(baseline.index.get_level_values('APR_MDC_key').unique())*len(baseline.index.get_level_values('age_group').unique())) as pbar:
    for APR_MDC_key in baseline.index.get_level_values('APR_MDC_key').unique():
        for age_group in baseline.index.get_level_values('age_group').unique():
            for stay_type in baseline.index.get_level_values('stay_type').unique():
                # Extract dataseries
                data = baseline.loc[(APR_MDC_key, age_group, stay_type),:]
                # Reset index to 'unlock' the date
                data.reset_index(inplace=True)
                # Convert the date to week and day number
                data['week_number'] = pd.to_datetime(data['date'].values).isocalendar().week.values
                data['day_number'] = pd.to_datetime(data['date'].values).isocalendar().day.values
                # Perform a groupby 'date' operation with mean() to take the mean of all values with similar daynumber
                d = data.groupby(by=['week_number','day_number']).mean().squeeze()
                b = baseline_df.loc[APR_MDC_key, age_group, stay_type, slice(None), slice(None)]
                baseline_df.loc[APR_MDC_key, age_group, stay_type, slice(None), slice(None)] = pd.merge(d, b, how='right', on=['week_number','day_number']).fillna(method='ffill')['n_patients_x'].values   
            pbar.update(1)

#####################################################################
## Normalizing pandemic data (2020-2021) with prepandemic baseline ##
#####################################################################

print('\n(3) Normalizing pandemic data (2020-2021) with prepandemic baseline\n')

# Consider all data from the beginning of 2020 as the actual 'data'
data_df = df[df.index.get_level_values('date')>=pd.Timestamp('2020-01-01')]
# Initialize target dataframe
target_df = data_df
# Add a column for versus_baseline
target_df.loc[(),'versus_baseline']=0

# Loop over all possible indices, convert date to day of year, take average of values with same day-of-year number
with tqdm(total=len(data_df.index.get_level_values('APR_MDC_key').unique())*len(data_df.index.get_level_values('age_group').unique())) as pbar:
    for APR_MDC_key in data_df.index.get_level_values('APR_MDC_key').unique():
        for age_group in data_df.index.get_level_values('age_group').unique():
            for stay_type in data_df.index.get_level_values('stay_type').unique():
                # Extract dataseries
                data = data_df.loc[(APR_MDC_key, age_group, stay_type),:]
                # Reset index to 'unlock' the date
                data.reset_index(inplace=True)
                # Convert the date to week and day number
                data.loc[slice(None), 'week_number'] = pd.to_datetime(data['date'].values).isocalendar().week.values
                data.loc[slice(None), 'day_number'] = pd.to_datetime(data['date'].values).isocalendar().day.values
                # Extract baseline
                baseline = baseline_df.loc[(APR_MDC_key, age_group, stay_type, slice(None), slice(None))]
                # Perform computation
                tmp=np.zeros(len(data['date'].values))
                for idx,date in enumerate(data['date'].values):
                    week_number = data.iloc[idx]['week_number']
                    day_number = data.iloc[idx]['day_number']
                    if baseline.loc[week_number, day_number] != 0:
                        tmp[idx] = data.iloc[idx]['n_patients']/baseline.loc[week_number, day_number]
                    else:
                        tmp[idx] = 1
                # Assign result
                target_df['versus_baseline'].loc[(APR_MDC_key, age_group, stay_type, slice(None))] = tmp

            pbar.update(1)

#################
## Save result ##
#################

print('\n(4) Saving result\n')

target_df.to_csv(os.path.join(abs_dir, result_folder))