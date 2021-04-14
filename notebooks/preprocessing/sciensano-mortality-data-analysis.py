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
df['incidence_total'] = df['incidence_total'].fillna(0)
df['incidence_hospital'] = df_raw[((df_raw.PLACE=='Hospital'))].groupby(by=['age_class','DATE']).age_class.count()
df['incidence_hospital'] = df['incidence_hospital'].fillna(0)
df['incidence_nursing'] = df_raw[((df_raw.PLACE=='Nursing home'))].groupby(by=['age_class','DATE']).age_class.count()
df['incidence_nursing'] = df['incidence_nursing'].fillna(0)
df['incidence_others'] = df_raw[((df_raw.PLACE=='Unknown')|(df_raw.PLACE=='Domicile or other places'))].groupby(by=['age_class','DATE']).age_class.count() 
df['incidence_others'] = df['incidence_others'].fillna(0)

df['cumsum_total']=0
df['cumsum_hospital'] = 0
df['cumsum_nursing'] = 0
df['cumsum_others'] = 0
for age_group in labels:
    df.loc[age_group,'cumsum_total'] = df.loc[age_group,'incidence_total'].cumsum().values
    df.loc[age_group,'cumsum_hospital'] = df.loc[age_group,'incidence_hospital'].cumsum().values
    df.loc[age_group,'cumsum_nursing'] = df.loc[age_group,'incidence_nursing'].cumsum().values
    df.loc[age_group,'cumsum_others'] = df.loc[age_group,'incidence_others'].cumsum().values


# -----------------------------
# Make a graphic representation
# -----------------------------

# Can be used as an example on how to index this dataset
data = df.xs(key='80+', level="age_class", drop_level=True)[['cumsum_total','cumsum_hospital','cumsum_nursing']]
fig,ax=plt.subplots(figsize=(12,4))
ax.scatter(x=data.index,y=data['cumsum_total'], color='black', alpha=0.3, linestyle='None', facecolors='none', s=40, linewidth=1)
ax.scatter(x=data.index,y=data['cumsum_hospital'], color='red', alpha=0.3, linestyle='None', facecolors='none', s=40, linewidth=1)
ax.scatter(x=data.index,y=data['cumsum_nursing'], color='blue', alpha=0.3, linestyle='None', facecolors='none', s=40, linewidth=1)
plt.legend(['total','hospital','nursing homes'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
plt.tight_layout()
plt.show()

# ------------------------
# Save resulting dataframe
# ------------------------

df.to_csv('../../data/interim/sciensano/sciensano_detailed_mortality.csv')
