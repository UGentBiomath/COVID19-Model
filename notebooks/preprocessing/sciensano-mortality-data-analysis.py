"""
This script contains all necessary code to extract and convert the mortality data from Sciensano into a seperate timeseries for hospitals and nursing homes.
You must place the super secret detailed hospitalization dataset `COVID19BE_MORT_RAW.csv` in the same folder as this script in order to run it.
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

df_raw = pd.read_csv('COVID19BE_MORT_RAW.csv', parse_dates=['DATE'])

# Add age classes to the dataset
labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80+']
df_raw['age_class'] = pd.cut(df_raw.AGE, bins=[0,10,20,30,40,50,60,70,80,120], labels=labels)

df = df_raw.groupby(by=['age_class','DATE']).age_class.count()
df = df.to_frame(name='incidence_total')
df.index = df.index.set_names(['age_class','date'])

columns = [['total','total','hospital','hospital','nursing','nursing','others','others'],['incidence','cumsum','incidence','cumsum','incidence','cumsum','incidence','cumsum']]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["place", "quantity"])
df_new = pd.DataFrame(index=df.index, columns=columns)
df_new['total','incidence'] = df['incidence_total'].fillna(0).astype(int)
df = df_new

df['hospital','incidence'] = df_raw[((df_raw.PLACE=='Hospital'))].groupby(by=['age_class','DATE']).age_class.count()
df['hospital','incidence'] = df['hospital','incidence'].fillna(0).astype(int)

df['nursing','incidence'] = df_raw[((df_raw.PLACE=='Nursing home'))].groupby(by=['age_class','DATE']).age_class.count()
df['nursing','incidence'] = df['nursing','incidence'].fillna(0).astype(int)

df['others','incidence'] = df_raw[((df_raw.PLACE=='Unknown')|(df_raw.PLACE=='Domicile or other places'))].groupby(by=['age_class','DATE']).age_class.count() 
df['others','incidence'] = df['others','incidence'].fillna(0).astype(int)

df['total','cumsum']=0
df['hospital','cumsum'] = 0
df['nursing','cumsum'] = 0
df['others','cumsum'] = 0
for age_group in labels:
    df.loc[age_group]['total','cumsum'] = df.loc[age_group]['total','incidence'].cumsum().values
    df.loc[age_group]['hospital','cumsum'] = df.loc[age_group]['hospital','incidence'].cumsum().values
    df.loc[age_group]['nursing','cumsum'] = df.loc[age_group]['nursing','incidence'].cumsum().values
    df.loc[age_group]['others','cumsum'] = df.loc[age_group]['others','incidence'].cumsum().values

# ----------------------
# Add data over all ages
# ----------------------

# Initialize new dataframe
iterables = [["all"], df.index.get_level_values(1).unique().values]
index_overall = pd.MultiIndex.from_product(iterables, names=["age_class", "date"])
df_overall = pd.DataFrame(index=index_overall, columns=columns)

# Loop over all columns
for idx, column in enumerate(df_overall.columns):
    for jdx, age_group in enumerate(df.index.get_level_values(0).unique().values):
        if jdx == 0:
            total = df.xs(key=age_group, level="age_class", drop_level=True)[column].values
        else:
            total = total + df.xs(key=age_group, level="age_class", drop_level=True)[column].values
    df_overall[column] = pd.Series(data=np.squeeze(total), index=df.index.get_level_values(1).unique()).values

# Concatenate result
df = pd.concat([df_overall,df])

# -----------------------------
# Make a graphic representation
# -----------------------------

fig,ax=plt.subplots(figsize=(12,4))
data = df.xs(key='80+', level="age_class", drop_level=True)
ax.scatter(x=data.index,y=data['total','cumsum'], color='black', alpha=0.3, linestyle='None', facecolors='none', s=40, linewidth=1)
ax.scatter(x=data.index,y=data['hospital','cumsum'], color='red', alpha=0.3, linestyle='None', facecolors='none', s=40, linewidth=1)
ax.scatter(x=data.index,y=data['nursing','cumsum'], color='blue', alpha=0.3, linestyle='None', facecolors='none', s=40, linewidth=1)
plt.legend(['total','hospital','nursing homes'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
plt.tight_layout()
plt.show()

# ------------------------
# Save resulting dataframe
# ------------------------

df.to_csv('../../data/interim/sciensano/sciensano_detailed_mortality.csv')
