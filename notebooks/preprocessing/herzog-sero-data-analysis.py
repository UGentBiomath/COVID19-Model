"""
This script contains all necessary code to extract and convert the serological data from Herzog into a timeseries for model calibration.
The raw data are located in the `~/data/raw/sero/` folder and were downloaded from https://zenodo.org/record/4665373#.YH13QHUzaV4.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

# -------------------------------------------------
# Load and format Sciensano hospital survey dataset
# -------------------------------------------------

# Load data
df = pd.read_csv('../../data/raw/sero/serology_covid19_belgium_round_1_to_7_v20210406.csv', parse_dates=True)

# Reindex data and drop unused columns
df['collection_midpoint'] = (pd.to_datetime(df['collection_start']) + (pd.to_datetime(df['collection_end']) - pd.to_datetime(df['collection_start']))/2)
df = df.drop(columns=['collection_start', 'collection_end', 'sex', 'collection_round', 'code'])
df.index = df['collection_midpoint']
df = df.drop(columns='collection_midpoint')

# -------------------------------------------
# Compute fraction of positives per age group
# -------------------------------------------

columns = ['fraction_pos_direct', 'fraction_pos_mean', 'fraction_pos_LL', 'fraction_pos_UL']

vals = df.groupby(by=['age_cat',df.index]).apply(lambda x: (x[((x.igg_cat == 'positive') | (x.igg_cat == 'borderline'))].count())/(x.igg_cat.count()))
new_df = pd.DataFrame(index=vals.index, columns=columns)
new_df['fraction_pos_direct'] = df.groupby(by=['age_cat',df.index]).apply(lambda x: (x[((x.igg_cat == 'positive') | (x.igg_cat == 'borderline'))].count())/(x.igg_cat.count()))

# -----------------------------------------------------------------
# Bootstrap the positives per age group to obtain confidence bounds
# -----------------------------------------------------------------

subset_size = 1000
n = 1000

data = np.zeros([10,7,n])
for idx in range(n):
    samples = df.groupby(by=['age_cat',df.index]).apply(lambda x: x.sample(n=subset_size,replace=True))
    samples = samples.drop(columns=['age_cat'])
    samples.index = samples.index.droplevel(2)
    pos = samples.groupby(by=['age_cat', 'collection_midpoint']).apply(lambda x: (x[((x.igg_cat == 'positive') | (x.igg_cat == 'borderline'))].count())/(x.igg_cat.count())).values[:,0]
    pos = pos.reshape((10,7))
    data[:,:,idx] = pos

#fig,ax=plt.subplots()
#ax.hist(data[0,0,:],bins=10,density=True,alpha=0.4)
#ax.hist(data[0,1,:],bins=10,density=True,alpha=0.4)
#ax.set_xlim([0,0.3])
#plt.show()

# --------------------------------------
# Compute fraction of positives in total
# --------------------------------------

#new_df['fraction_pos_mean']=0
#new_df['fraction_pos_LL']=0
#new_df['fraction_pos_UL']=0
# Append the percentiles
for idx,age in enumerate(new_df.index.unique(level=0)):
    for jdx,wave in enumerate(new_df.index.unique(level=1)):
        new_df['fraction_pos_mean'].loc[age,wave] = np.mean(data[idx,jdx,:])
        new_df['fraction_pos_LL'].loc[age,wave] = np.quantile(data[idx,jdx,:],q=0.025)
        new_df['fraction_pos_UL'].loc[age,wave] = np.quantile(data[idx,jdx,:],q=0.975)

# Totals
arrays = [['total','total','total','total','total','total','total'],vals.index.unique(level=1)]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=["age_cat", "collection_midpoint"])
total_df = pd.DataFrame(index=index, columns=columns)
total_df['fraction_pos_direct'] = df.groupby(by=df.index).apply(lambda x: (x[((x.igg_cat == 'positive') )].count())/(x.igg_cat.count())).values

# -----------------------------------------------------------------
# Bootstrap the positives in total to obtain confidence bounds
# -----------------------------------------------------------------

subset_size = 1000
n = 1000

data = np.zeros([7,n])
for idx in range(n):
    samples = df.groupby(by=[df.index]).apply(lambda x: x.sample(n=subset_size,replace=True))
    samples.index = samples.index.droplevel(1)
    samples = samples.drop(columns=['age_cat'])
    pos = samples.groupby(by=['collection_midpoint']).apply(lambda x: (x[((x.igg_cat == 'positive') )].count())/(x.igg_cat.count())).values[:,0]
    data[:,idx] = pos

#fig,ax=plt.subplots()
#ax.hist(data[0,:],bins=10,density=True,alpha=0.4)
#ax.hist(data[1,:],bins=10,density=True,alpha=0.4)
#ax.hist(data[2,:],bins=10,density=True,alpha=0.4)
#ax.set_xlim([0,0.3])
#plt.show()

#total_df['fraction_pos_LL']=0
#total_df['fraction_pos_UL']=0
# Append the percentiles
for idx,wave in enumerate(new_df.index.unique(level=1)):
    total_df['fraction_pos_mean'].loc['total', wave] = np.mean(data[idx,:])
    total_df['fraction_pos_LL'].loc['total', wave] = np.quantile(data[idx,:],q=0.025)
    total_df['fraction_pos_UL'].loc['total', wave] = np.quantile(data[idx,:],q=0.975)

# -----------------
# Merge the results
# -----------------

merged = pd.concat([new_df, total_df])

# -----------------
# Save the results
# -----------------

merged.to_csv('../../data/interim/sero/herzog_serodata_national.csv')

