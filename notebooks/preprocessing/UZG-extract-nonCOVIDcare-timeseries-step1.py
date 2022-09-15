"""
This script converts the data on the postponement of non-COVID-19 care, provided by the Ghent University Hospital and located in the `~/data/raw/QALY_model/postponement_non_covid_care/UZG/`
folder (confidential, contact tijs.alleman@ugent.be) into an intermediate format.

The raw data provided has the following format: Major Diagnostic Group (MDC), Patient age (bins, 5 years), Type of hospitalization, Date of hospital intake, Date of hospital discharge.
The intermediate format is a pd.DataFrame, which, for each MDC, age and type of stay contains the total number of patients on a given date (from 2017-2021).
The intermediate dataframe is not on Github because of its size (70 Mb). The computation is demanding, taking roughly 9 hrs on an Intel® Xeon(R) W-2295 CPU @ 3.00GHz.
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

print('\n(1) Loading data\n')

# Names and location of datasets
abs_dir = os.getcwd()
rel_dir = '../../data/raw/QALY_model/postponement_non_covid_care/UZG/'
name_raw_list=['MZG_2016.xlsx', 'MZG_2017.xlsx','MZG_2018.xlsx', 'MZG_2019.xlsx', 'MZG_2020.xlsx', 'MZG_2021.xlsx']
# Construct list of locations
path_list=[]
for idx,name in enumerate(name_raw_list):
    path_list.append(os.path.join(abs_dir,rel_dir,name))
# Location of results
result_folder='../../data/interim/QALY_model/postponement_non_covid_care/UZG/'
result_name='MZG_2016_2021.csv'
result_path = os.path.join(abs_dir,result_folder,result_name)
if not os.path.exists(os.path.join(abs_dir,result_folder)):
    os.makedirs(os.path.join(abs_dir,result_folder))
# Load datasets and merge them togheter
df = pd.concat(
    map(pd.read_excel, path_list))

#################
## Format data ##
#################

print('(2) Formatting data\n')

# Throw out names of MDC (can look this up online)
del df['Pathologie - APR-MDC naam']
del df['Pathologie - APR-MDC naam (en)']
df.rename(columns={'Pathologie - APR-MDC sleutel': 'APR_MDC_key',
                   'MZG Opnamedatum - tekst': 'intake_date',
                   'MZG Ontslagdatum - tekst': 'discharge_date',
                   'Bezoek Opnameleeftijd per 5': 'age_group',
                   'Verblijf type - label': 'stay_type'}, inplace=True)
# Remove missing discharge dates
print('\tRemoved {0}/{1} entries due to missing discharge date'.format(len(df['discharge_date'][df['discharge_date']=='No Date']), len(df)))
df = df[df.discharge_date != 'No Date']
# Parse dates
df['intake_date'] = pd.to_datetime(df.intake_date, format='%d/%m/%Y')
df['discharge_date'] = pd.to_datetime(df.discharge_date, format='%d/%m/%Y')
# Replace stay types with their key
mapping = {'H | Klassieke hospitalisatie': 'H',
           'D | Andere daghospitalisatie': 'D',
           'C | Chirurgisch dagziekenhuis': 'C',
           'M | Tussentijdse registratie langdurig verblijf': 'M',
           'F | Eerste registratie langdurig verblijf': 'F',
           'L | Laatste registratie langdurig verblijf': 'L'}
df.stay_type=df.stay_type.map(lambda x: mapping.get(x) if x else None)
# Throw out categories M, F, L
print("\tRemoved {0}/{1} entries with stay_type 'M'".format(len(df['stay_type'][df['stay_type']=='M']), len(df)))
print("\tRemoved {0}/{1} entries with stay_type 'F'".format(len(df['stay_type'][df['stay_type']=='F']), len(df)))
print("\tRemoved {0}/{1} entries with stay_type 'L'".format(len(df['stay_type'][df['stay_type']=='L']), len(df)))
df = df[( (df.stay_type != 'L') & (df.stay_type != 'F') & (df.stay_type != 'M'))]
# Replace age categories with a simpler format
mapping={'Cat: 0 - 4': '0-4', 'Cat: 5 - 9': '5-9', 'Cat: 10 - 14': '10-14', 'Cat: 15 - 19': '15-19',
         'Cat: 20 - 24': '20-24', 'Cat: 25 - 29': '25-29', 'Cat: 30 - 34': '30-34', 'Cat: 35 - 39': '35-39',
         'Cat: 40 - 44': '40-44', 'Cat: 45 - 49': '45-49', 'Cat: 50 - 54': '50-54', 'Cat: 55 - 59': '55-59',
         'Cat: 60 - 64': '60-64', 'Cat: 65 - 69': '65-69', 'Cat: 70 - 74': '70-74', 'Cat: 75 - 79': '75-79',
         'Cat: 80 - 84': '80-84', 'Cat: 85 - 89': '85-89', 'Cat: 90 - 94': '90-94', 'Cat: 95+': '95-120'}
df.age_group=df.age_group.map(lambda x: mapping.get(x) if x else None)
# Throw out missing APR-MDC classifications
print("\tRemoved {0}/{1} entries because APR-MDC classification was missing\n".format(len(df.APR_MDC_key[df.APR_MDC_key=='.']), len(df)))
df = df[df.APR_MDC_key!='.']
# Index on intake_date and sort
df = df.set_index(['APR_MDC_key', 'age_group', 'stay_type']).sort_index()
# Save lowest and highest intake date
intake_min = df.intake_date.min()
intake_max = df.intake_date.max()

#############################################
## Define dataframe containing the results ##
#############################################

dates = pd.date_range(start=intake_min, end=intake_max)
# Make a dataframe with desired output format
iterables=[]
for index_name in df.index.names:
    iterables += [df.index.get_level_values(index_name).unique()]
iterables.append(dates)
names=list(df.index.names)
names.append('date')
index = pd.MultiIndex.from_product(iterables, names=names)
target_df = pd.Series(index=index, name='n_patients', data=np.zeros(len(index), dtype=int))

#########################
## Perform computation ##
#########################

print('(3) Performing computation\n')

with tqdm(total=len(df.index.get_level_values('APR_MDC_key').unique())*len(df.index.get_level_values('age_group').unique())) as pbar:
    for APR_MDC_key in df.index.get_level_values('APR_MDC_key').unique():
        for age_group in df.index.get_level_values('age_group').unique():
            for stay_type in df.index.get_level_values('stay_type').unique():
                # Extract series (Use try/except structure to skip over missing entries)
                try:
                    data = df.loc[(APR_MDC_key, age_group, stay_type),:]
                    # Loop over every entry
                    for i in range(len(data)):
                        # Generate daterange from intake and discharge date
                        date_range = pd.date_range(start=data['intake_date'].iloc[i], end=data['discharge_date'].iloc[i])
                        # Add 1 in target dataframe between intake and discharge date
                        target_df.loc[APR_MDC_key, age_group, stay_type, date_range] += 1
                        
                except:
                    pass
            pbar.update(1)

#################
## Save result ##
#################

print('(4) Saving result\n')

target_df.to_csv(result_path)