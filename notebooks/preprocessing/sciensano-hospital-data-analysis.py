"""
This script contains all necessary code to extract and convert the patients data from the Sciensano hospital survey into parameters usable by the BIOMATH COVID-19 SEIRD model.
You must place the super secret detailed hospitalization dataset `COVID19BE_CLINIC.csv`in the same folder as this script in order to run it.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

# -------------------------------------------------
# Load and format Sciensano hospital survey dataset
# -------------------------------------------------

df = pd.read_csv('COVID19BE_CLINIC.csv')
print(df.shape[0])

# Drop the columns on admission_data and discharge_data --> do this myself
df=df.drop(columns=['admission_data','discharge_data'])

# Only if admission data, discharge data, status of discharge and ICU transfer is known, the data can be used by our model
df.dropna(subset=['dt_admission'], inplace=True)
print(df.shape[0])
print(df['dt_admission'].isnull().sum())

df.dropna(subset=['dt_discharge'], inplace=True)
print(df.shape[0])

df.dropna(subset=['status_discharge'], inplace=True)
print(df.shape[0])

df.drop(df[df.status_discharge == 'Autre'].index, inplace=True)
print(df.shape[0])

df.dropna(subset=['ICU_transfer'], inplace=True)
print(df.shape[0])

df.drop(df[df.ICU_transfer == 'Inconnu'].index, inplace=True)
print(df.shape[0])

# Convert dates
df['dt_admission'] = pd.to_datetime(df['dt_admission'])
df['dt_admission'] = df['dt_admission'].dt.date

df['dt_discharge'] = pd.to_datetime(df['dt_discharge'])
df['dt_discharge'] = df['dt_discharge'].dt.date

df['dt_onset'] = pd.to_datetime(df['dt_onset'])
df['dt_onset'] = df['dt_onset'].dt.date

df['dt_icu_transfer'] = pd.to_datetime(df['dt_icu_transfer'])
df['dt_icu_transfer'] = df['dt_icu_transfer'].dt.date

# Add column with the age classes
age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], 
                                           closed='left')
df['age_class'] = pd.cut(df.age, bins=age_classes)


# --------------------------------
# Initialize dataframe for results
# --------------------------------
hospital_parameters_age = pd.DataFrame(index=age_classes, 
                                   columns=['sample_size','admission_propensity','c','d_transfer','m0','m0_{ICU}','m0_{C}','dC','dC_R','dC_D','dICU','dICU_R','dICU_D'])

# ------------------
# Compute parameters
# ------------------

# Sample size
hospital_parameters_age['sample_size']=df.groupby(by='age_class').apply(lambda x: x.age.count())

# Hospitalization propensity
hospital_parameters_age['admission_propensity']=df.groupby(by='age_class').apply(lambda x: x.age.count())/df.shape[0]

# Distribution cohort/icu
hospital_parameters_age['c'] = df.groupby(by='age_class').apply(
                                lambda x: x[x.ICU_transfer=='Non'].age.count()/
                                          x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count())

# Days in cohort before ICU transfer
# Must be checked further
df['d_transfer'] = np.nan
for i in range(len(df['d_transfer'])):
    if ((df['ICU_transfer'].iloc[i] == 'Oui') & (not pd.isnull(df['dt_icu_transfer'].iloc[i]))):
        val = (df['dt_icu_transfer'].iloc[i] - df['dt_admission'].iloc[i])/datetime.timedelta(days=1)
        if val >= 0:
            df['d_transfer'].iloc[i] = val

hospital_parameters_age['d_transfer'] = df.groupby(by='age_class').apply(
                                lambda x: x[x.ICU_transfer=='Oui'].d_transfer.mean())

# Mortalities
hospital_parameters_age['m0']=df.groupby(by='age_class').apply(
                                lambda x: x[( (x.status_discharge=='D'))].age.count()/
                                            x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count())

hospital_parameters_age['m0_{ICU}']= df.groupby(by='age_class').apply(
                                lambda x: x[((x.ICU_transfer=='Oui') & (x.status_discharge=='D'))].age.count()/
                                          x[x.ICU_transfer.isin(['Oui'])].age.count())

hospital_parameters_age['m0_{C}']= df.groupby(by='age_class').apply(
                                lambda x: x[((x.ICU_transfer=='Non') & (x.status_discharge=='D'))].age.count()/
                                          x[x.ICU_transfer.isin(['Non'])].age.count())

# Hospitalization lengths
hospital_parameters_age['dC']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1)).mean())

hospital_parameters_age['dC_R']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).mean())

hospital_parameters_age['dC_D']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()).fillna(1)

hospital_parameters_age['dICU']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).mean())

hospital_parameters_age['dICU_R']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).mean())

hospital_parameters_age['dICU_D']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()).fillna(1)

# Averages and totals
averages = pd.DataFrame(index=['average'],columns=['sample_size', 'admission_propensity', 'd_transfer', 'c','dC','dC_R', 'dC_D', 'dICU',
                                            'dICU_R', 'dICU_D',
                                            'm0', 'm0_{C}', 'm0_{ICU}'])

averages['sample_size'] = hospital_parameters_age['sample_size'].sum()
averages['admission_propensity'] = sum(((hospital_parameters_age['sample_size']*hospital_parameters_age['admission_propensity']).values)/(np.ones(9)*hospital_parameters_age['sample_size'].sum()))
averages['c'] = df[df.ICU_transfer=='Non'].age.count()/df[df.ICU_transfer.isin(['Oui', 'Non'])].age.count()
averages['m0'] = df[((df.status_discharge=='D'))].age.count()/df[df.ICU_transfer.isin(['Oui', 'Non'])].age.count()
averages['m0_{ICU}'] = df[((df.ICU_transfer=='Oui') & (df.status_discharge=='D'))].age.count()/df[df.ICU_transfer.isin(['Oui'])].age.count()
averages['m0_{C}'] = df[((df.ICU_transfer=='Non') & (df.status_discharge=='D'))].age.count()/df[df.ICU_transfer.isin(['Non'])].age.count()
averages['dC']=((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1)).mean()
averages['dC_R']=((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).mean()
averages['dC_D']=((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()
averages['dICU']=((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).mean()
averages['dICU_R']=((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).mean()
averages['dICU_D']=((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()

hospital_parameters_age = pd.concat([hospital_parameters_age, averages])

# ----------------------------------------------------------------
# Write age-stratified parameters to data/interim/model_parameters
# ----------------------------------------------------------------

hospital_parameters_age.to_csv('../../data/interim/model_parameters/COVID19_SEIRD/sciensano_hospital_parameters.csv')

