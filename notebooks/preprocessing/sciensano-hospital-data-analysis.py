"""
This script contains all necessary code to extract and convert the patients data from the Sciensano hospital survey into parameters usable by the BIOMATH COVID-19 SEIRD model.
You must place the super secret detailed hospitalization dataset `COVID19BE_CLINIC.csv` in the same folder as this script in order to run it.
Further, you must MANUALLY replace décédé and rétabli in the file `COVID19BE_CLINIC.csv` with D and R.

To load the resulting .xlsx into a pandas dataframe use:
    dataframe = pd.read_excel('../../data/interim/model_parameters/COVID19_SEIRD/sciensano_hospital_parameters.xlsx', sheet_name='residence_times', index_col=0, header=[0,1])
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
from scipy.stats import gamma

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

# Remove the negative residence times
df.drop(df[((df['dt_discharge'] - df['dt_admission'])/datetime.timedelta(days=1)) < 0].index, inplace=True)
print(df.shape[0])

# Remove all residence times larger than 90 days
df.drop(df[((df['dt_discharge'] - df['dt_admission'])/datetime.timedelta(days=1)) >= 90].index, inplace=True)
print(df.shape[0])

#dC_vector=((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1))
#fig,ax=plt.subplots()
#ax.hist(dC_vector,bins=10, density=True, color='blue', alpha=0.4)
#ax.axvline(np.mean(dC_vector),color='black')
#ax.axvline(np.median(dC_vector),color='red')
#plt.show()

# --------------------------------
# Initialize dataframe for results
# --------------------------------

samples = pd.DataFrame(index=age_classes, columns=[])
samples_total  = pd.DataFrame(index=['total'], columns=[])

fractions = pd.DataFrame(index=age_classes, columns=['total_sample_size', 'admission_propensity', 'c', 'm0', 'm0_{ICU}', 'm0_{C}'])

columns = [[],[]]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["parameter", "quantity"])
residence_times = pd.DataFrame(index=age_classes, columns=columns)

# ---------------------------
# Compute fraction parameters
# ---------------------------

# Sample size
fractions['total_sample_size']=df.groupby(by='age_class').apply(lambda x: x.age.count())
# Hospitalization propensity
fractions['admission_propensity']=df.groupby(by='age_class').apply(lambda x: x.age.count())/df.shape[0]
# Distribution cohort/icu
fractions['c'] = df.groupby(by='age_class').apply(
                                lambda x: x[x.ICU_transfer=='Non'].age.count()/
                                          x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count())
# Mortalities
fractions['m0']=df.groupby(by='age_class').apply(
                                lambda x: x[( (x.status_discharge=='D'))].age.count()/
                                            x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count())
fractions['m0_{ICU}']= df.groupby(by='age_class').apply(
                                lambda x: x[((x.ICU_transfer=='Oui') & (x.status_discharge=='D'))].age.count()/
                                          x[x.ICU_transfer.isin(['Oui'])].age.count())
fractions['m0_{C}']= df.groupby(by='age_class').apply(
                                lambda x: x[((x.ICU_transfer=='Non') & (x.status_discharge=='D'))].age.count()/
                                          x[x.ICU_transfer.isin(['Non'])].age.count())

# Bootstrap the mortalities
subset_size = 120
n = 500

m0_lst = []
for idx in range(n):
    samples = df.groupby(by='age_class').apply(lambda x: x.sample(n=subset_size,replace=True))
    samples=samples.drop(columns='age_class')
    m0 = samples.groupby(by='age_class').apply(lambda x: x[( (x.status_discharge=='D'))].age.count()/x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count())
    m0_lst.append(m0[1])
    #samples = df.sample(n=subset_size,replace=True)
    #m0_lst.append(samples.groupby(by='age_class').apply(lambda x: x[( (x.status_discharge=='D'))].age.count()/x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count()).values[1])

fig,ax=plt.subplots()
ax.hist(m0_lst,bins=15,density=True)
ax.set_xlim([0,1])
plt.show()


# -----------------------
# Compute residence times
# -----------------------

# Days in cohort before ICU transfer
df['d_transfer'] = np.nan
values=[]
for i in range(len(df['d_transfer'])):
    if ((df['ICU_transfer'].iloc[i] == 'Oui') & (not pd.isnull(df['dt_icu_transfer'].iloc[i]))):
        val = (df['dt_icu_transfer'].iloc[i] - df['dt_admission'].iloc[i])/datetime.timedelta(days=1)
        if ((val >= 0) & (val <= 21)):
            df['d_transfer'].iloc[i] = val
            if val == 0:
                values.append(0.010)
            else:
                values.append(val)


residence_times['d_transfer','mean'] = df.groupby(by='age_class').apply(
                                lambda x: x[x.ICU_transfer=='Oui'].d_transfer.mean())
residence_times['d_transfer','median'] = df.groupby(by='age_class').apply(lambda x: x[x.ICU_transfer=='Oui'].d_transfer.median())                     
residence_times['d_transfer','median']=residence_times['d_transfer','median'].replace(0,0.1)

# Append samples
samples['d_transfer'] = df.groupby(by='age_class').d_transfer.agg(lambda x: list(x.dropna()))
samples_total['d_transfer'] = [df.d_transfer.agg(lambda x: list(x.dropna()))]

sample_size_lst=[]
shape_lst=[]
loc_lst=[]
scale_lst=[]
v = df.groupby(by='age_class').apply(lambda x: x[x.ICU_transfer=='Oui'].d_transfer)
for age_group in v.index.get_level_values(0).unique().values:
    v[age_group][v[age_group]==0] = 0.01
    v = v.dropna()
    shape, loc, scale = gamma.fit(v[age_group].values, floc=0)
    sample_size_lst.append(len(v[age_group].values))
    shape_lst.append(shape)
    loc_lst.append(loc)
    scale_lst.append(scale)

residence_times['d_transfer','sample_size'], residence_times['d_transfer','shape'], residence_times['d_transfer','loc'], residence_times['d_transfer', 'scale'] = sample_size_lst, shape_lst, loc_lst, scale_lst

# Hospitalization lengths
def fit_gamma(values):
    sample_size_lst=[]
    shape_lst=[]
    loc_lst=[]
    scale_lst=[]
    for age_group in v.index.get_level_values(0).unique().values:
        v[age_group][v[age_group]==0] = 0.01
        shape, loc, scale = gamma.fit(v[age_group].values, floc=0)
        sample_size_lst.append(len(v[age_group].values))
        shape_lst.append(shape)
        loc_lst.append(loc)
        scale_lst.append(scale)
    return sample_size_lst, shape_lst, loc_lst, scale_lst

########
## dC ##
########

v = df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1)))
residence_times['dC','mean']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1)).mean())
residence_times['dC','median']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1)).median())
residence_times['dC','sample_size'], residence_times['dC','shape'],residence_times['dC','loc'],residence_times['dC','scale'] = fit_gamma(v)
# Append samples
samples['dC'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dC'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1))).agg(lambda x: list(x))]

##########
## dC_R ##
##########

v = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)))
residence_times['dC_R', 'mean']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).mean())
residence_times['dC_R', 'median']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).median())
residence_times['dC_R','sample_size'], residence_times['dC_R','shape'],residence_times['dC_R','loc'],residence_times['dC_R','scale'] = fit_gamma(v)
# Append samples
samples['dC_R'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dC_R'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1))).agg(lambda x: list(x))]

##########
## dC_D ##
##########

v = df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)))
residence_times['dC_D', 'mean']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()).fillna(1)
residence_times['dC_D', 'median']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).median()).fillna(1)
sample_size, shape, loc, scale = fit_gamma(v)
for i in range(2):
    sample_size.insert(0,0)
    shape.insert(0,1)
    loc.insert(0,0)
    scale.insert(0,1)
residence_times['dC_D','sample_size'], residence_times['dC_D','shape'],residence_times['dC_D','loc'],residence_times['dC_D','scale'] = sample_size, shape, loc, scale

# Append samples
samples['dC_D'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dC_D'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1))).agg(lambda x: list(x))]
samples['dC_D'].loc[residence_times.index.get_level_values(0).unique().values[0]] = [1]
samples['dC_D'].loc[residence_times.index.get_level_values(0).unique().values[1]] = [1]

##########
## dICU ##
##########

v = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1)))
residence_times['dICU','mean']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).mean())
residence_times['dICU','median']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).median())
residence_times['dICU','sample_size'], residence_times['dICU','shape'],residence_times['dICU','loc'],residence_times['dICU','scale'] = fit_gamma(v)

# Append samples
samples['dICU'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dICU'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1))).agg(lambda x: list(x))]

############
## dICU_R ##
############

residence_times['dICU_R','mean']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).mean())
residence_times['dICU_R','median']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).median())
v = df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)))
residence_times['dICU_R','sample_size'], residence_times['dICU_R','shape'],residence_times['dICU_R','loc'],residence_times['dICU_R','scale'] = fit_gamma(v)

# Append samples
samples['dICU_R'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dICU_R'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1))).agg(lambda x: list(x))]

############
## dICU_D ##
############

residence_times['dICU_D','mean']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()).fillna(1)
residence_times['dICU_D','median']=df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).median()).fillna(1)
v = df.groupby(by='age_class').apply(
                                lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)))
sample_size, shape, loc, scale = fit_gamma(v)
for i in range(1):
    sample_size.insert(0,0)
    shape.insert(0,1)
    loc.insert(0,0)
    scale.insert(0,1)
residence_times['dICU_D','sample_size'], residence_times['dICU_D','shape'],residence_times['dICU_D','loc'],residence_times['dICU_D','scale'] = sample_size, shape, loc, scale

# Append samples
samples['dICU_D'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dICU_D'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1))).agg(lambda x: list(x))]
samples['dICU_D'].loc[residence_times.index.get_level_values(0).unique().values[0]] = [1]
samples = pd.concat([samples, samples_total])

# Averages and totals
averages = pd.DataFrame(index=['average'],columns=['total_sample_size', 'admission_propensity', 'c','m0', 'm0_{C}', 'm0_{ICU}'])
averages['total_sample_size'] = fractions['total_sample_size'].sum()
averages['admission_propensity'] = sum(((fractions['total_sample_size']*fractions['admission_propensity']).values)/(np.ones(9)*fractions['total_sample_size'].sum()))
averages['c'] = df[df.ICU_transfer=='Non'].age.count()/df[df.ICU_transfer.isin(['Oui', 'Non'])].age.count()
averages['m0'] = df[((df.status_discharge=='D'))].age.count()/df[df.ICU_transfer.isin(['Oui', 'Non'])].age.count()
averages['m0_{ICU}'] = df[((df.ICU_transfer=='Oui') & (df.status_discharge=='D'))].age.count()/df[df.ICU_transfer.isin(['Oui'])].age.count()
averages['m0_{C}'] = df[((df.ICU_transfer=='Non') & (df.status_discharge=='D'))].age.count()/df[df.ICU_transfer.isin(['Non'])].age.count()
fractions = pd.concat([fractions, averages])

columns = [[],[]]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["parameter", "quantity"])
averages = pd.DataFrame(index=['averages'], columns=columns)

averages['dC','mean'] = ((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1)).mean()
averages['dC','median'] = ((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1)).median()
v = ((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dC','sample_size'] = len(v)
averages['dC','shape'],averages['dC','loc'],averages['dC','scale'] = gamma.fit(v, floc=0)

averages['dC_R','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).mean()
averages['dC_R','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).median()
v = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dC_R','sample_size'] = len(v)
averages['dC_R','shape'],averages['dC_R','loc'],averages['dC_R','scale'] = gamma.fit(v, floc=0)

averages['dC_D','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()
averages['dC_D','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).median()
v = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dC_D','sample_size'] = len(v)
averages['dC_D','shape'],averages['dC_D','loc'],averages['dC_D','scale'] = gamma.fit(v, floc=0)

averages['dICU','mean'] = ((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).mean()
averages['dICU','median'] = ((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).median()
v = ((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dICU','sample_size'] = len(v)
averages['dICU','shape'],averages['dICU','loc'],averages['dICU','scale'] = gamma.fit(v, floc=0)

averages['dICU_R','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).mean()
averages['dICU_R','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).median()
v = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dICU_R','sample_size'] = len(v)
averages['dICU_R','shape'],averages['dICU_R','loc'],averages['dICU_R','scale'] = gamma.fit(v, floc=0)

averages['dICU_D','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()
averages['dICU_D','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).median()
v = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dICU_D','sample_size'] = len(v)
averages['dICU_D','shape'],averages['dICU_D','loc'],averages['dICU_D','scale'] = gamma.fit(v, floc=0)

averages['d_transfer','mean'] = np.mean(values)
averages['d_transfer','median'] = np.median(values)
averages['d_transfer','shape'], averages['d_transfer','loc'], averages['d_transfer', 'scale'] = gamma.fit(values, floc=0)
averages['d_transfer','sample_size'] = len(values)

residence_times = pd.concat([residence_times, averages])

# ----------------------------------------------------------------
# Write age-stratified parameters to data/interim/model_parameters
# ----------------------------------------------------------------
with pd.ExcelWriter('../../data/interim/model_parameters/COVID19_SEIRD/sciensano_hospital_parameters.xlsx') as writer:  
    fractions.to_excel(writer,sheet_name='fractions')
    residence_times.to_excel(writer,sheet_name='residence_times')
    samples.to_excel(writer,sheet_name='residence_times_samples')