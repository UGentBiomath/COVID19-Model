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

############################################################
## Construct baseline dataframe using data from 2017-2019 ##
############################################################

print('\n(2) Constructing baseline dataframe using data from 2017-2019\n')

# Define a target dataframe containing the day-of-year number instead of the date
day_of_year = np.linspace(1,365,num=364+1, dtype=int)
iterables=[]
names=[]
for index_name in df.index.names:
    if index_name != 'date':
        iterables += [df.index.get_level_values(index_name).unique()]
        names += [index_name,]
iterables.append(day_of_year)
names.append('day_of_year')
index = pd.MultiIndex.from_product(iterables, names=names)
baseline_df = pd.Series(index=index, name='n_patients', data=np.zeros(len(index), dtype=int))

# Use all data from the second week of 2017 until beginning of 2020 as baseline
baseline = df[((df.index.get_level_values('date')<pd.Timestamp('2020-01-01'))&(df.index.get_level_values('date')>=pd.Timestamp('2017-02-01')))]

# Loop over all possible indices, convert date to day of year, take average of values with same day-of-year number
with tqdm(total=len(baseline.index.get_level_values('APR_MDC_key').unique())*len(baseline.index.get_level_values('age_group').unique())) as pbar:
    for APR_MDC_key in baseline.index.get_level_values('APR_MDC_key').unique():
        for age_group in baseline.index.get_level_values('age_group').unique():
            for stay_type in baseline.index.get_level_values('stay_type').unique():
                # Extract dataseries
                data = baseline.loc[(APR_MDC_key, age_group, stay_type),:]
                # Reset index to 'unlock' the date
                data.reset_index(inplace=True)
                # Convert the date to a daynumber
                data['date'] = data['date'].apply(lambda x: x.day_of_year).values
                # In a leap year, the method returns 366 days (and we'll simply throw it out since 2016 will not be in the final baseline)
                data['date'] = data['date'][data['date']!=366]
                # Perform a groupby 'date' operation with mean() to take the mean of all values with similar daynumber
                baseline_df.loc[APR_MDC_key, age_group, stay_type, slice(None)] = data.groupby(by='date').mean().values.flatten()
            pbar.update(1)

#####################################################################
## Normalizing pandemic data (2020-2021) with prepandemic baseline ##
#####################################################################

print('\n(3) Normalizing pandemic data (2020-2021) with prepandemic baseline\n')

# Consider all data from the beginning of 2020 as the actual 'data'
data_df = df[df.index.get_level_values('date')>=pd.Timestamp('2020-01-01')]

# Initialize target dataframe
target_df = data_df
# Remove leap year extra day
tmp=target_df.reset_index()
tmp=tmp[tmp['date'] != '2020-02-29']
target_df=tmp.set_index(['APR_MDC_key','age_group','stay_type','date'])
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
                # Convert the date to a daynumber
                data['date'] = data['date'].apply(lambda x: x.day_of_year).values
                # In a leap year, the method returns 366 days
                # Important: Check if year is a leap year, delete entry 60, subtract 1 from all entries above 60
                data = data[data['date']!=366]
                # Extract baseline
                baseline = baseline_df.loc[(APR_MDC_key, age_group, stay_type, slice(None))]
                # Perform computation
                tmp=np.zeros(len(data['date'].values))
                for idx,date in enumerate(data['date'].values):
                    if baseline[date] != 0:
                        tmp[idx] = data.iloc[idx]['n_patients']/baseline[date]
                    else:
                        tmp[idx] = 1
                # Assign result
                target_df.loc[(APR_MDC_key, age_group, stay_type, slice(None)), 'versus_baseline'] = tmp
            pbar.update(1)

#################
## Save result ##
#################

print('\n(4) Saving result\n')

target_df.to_csv(os.path.join(abs_dir, result_folder))