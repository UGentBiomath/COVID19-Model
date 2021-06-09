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
from scipy.stats import mannwhitneyu, ttest_ind, gamma, exponweib, weibull_min
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

# -----------------------------
# Helper functions and settings
# -----------------------------

plot_fit=False

colorscale_okabe_ito = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
                        "green" : "#009E73", "yellow" : "#F0E442",
                        "blue" : "#0072B2", "red" : "#D55E00",
                        "pink" : "#CC79A7", "black" : "#000000"}

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

def fit_weibull(v):
    sample_size_lst=[]
    shape_lst=[]
    loc_lst=[]
    scale_lst=[]
    for age_group in v.index.get_level_values(0).unique().values:
        v[age_group][v[age_group]==0] = 0.01
        v = v.dropna()
        shape, loc, scale = weibull_min.fit(v[age_group].values,floc=0)
        #shape, loc, scale = gamma.fit(v[age_group].values,floc=0)
        sample_size_lst.append(len(v[age_group].values))
        shape_lst.append(shape)
        loc_lst.append(loc)
        scale_lst.append(scale)
    return sample_size_lst, shape_lst, loc_lst, scale_lst

def plot_weibull_fit(v,par,max_val):
    fig,axes = plt.subplots(nrows=3,ncols=3,sharex=True,figsize=(12,12))
    axes = axes.flatten()
    for idx,age_group in enumerate(v.index.get_level_values(0).unique().values):
        bins = np.linspace(0, max_val, 10)
        axes[idx].hist(v[age_group], bins=bins, density=True)
        x = np.linspace (0.5, max_val, 1000)
        #y = gamma.pdf(x, a=residence_times[par,'shape'][age_group], loc=residence_times[par,'loc'][age_group], scale=residence_times[par,'scale'][age_group])  
        y = weibull_min.pdf(x, c=residence_times[par,'shape'][age_group], loc=residence_times[par,'loc'][age_group], scale=residence_times[par,'scale'][age_group]) 
        axes[idx].plot(x,y)        
        axes[idx].text(x=0.70,y=0.82,s='Shape: '+"{:.2f}".format(residence_times[par,'shape'][age_group]) + '\nScale: ' + "{:.2f}".format(residence_times[par,'scale'][age_group]) + '\nLoc: '+ "{:.2f}".format(residence_times[par,'loc'][age_group]), transform=axes[idx].transAxes, fontsize=8) 
        axes[idx].set_title('Age group: ' + str(age_group), fontsize=12)
        axes[idx].set_xlim([0,max_val])
    fig.suptitle(par,fontsize=16)
    plt.show()
    plt.close()

#######################################################
## Load and format Sciensano hospital survey dataset ##
#######################################################

df = pd.read_csv('COVID19BE_CLINIC.csv')
n_orig = df.shape[0]
print('The original dataset contains ' + str(n_orig) + ' entries.')

# Drop the columns on admission_data and discharge_data --> do this myself
df=df.drop(columns=['admission_data','discharge_data'])

# Drop the columns with missing age
df.dropna(subset=['age'], inplace=True)
n_filtering_age = df.shape[0]
print(str(n_orig-n_filtering_age) + ' entries were removed because the age was missing.')

# Only if admission data, discharge data, status of discharge and ICU transfer is known, the data can be used by our model
df.dropna(subset=['dt_admission'], inplace=True)
df.dropna(subset=['dt_discharge'], inplace=True)
df.dropna(subset=['status_discharge'], inplace=True)
df.dropna(subset=['ICU_transfer'], inplace=True)
df.drop(df[df.status_discharge == 'Autre'].index, inplace=True)
df.drop(df[df.status_discharge == 'Inconnu'].index, inplace=True)
df.drop(df[df.status_discharge == 'Transfert'].index, inplace=True)

n_filtering_dates = df.shape[0]
print(str(n_filtering_age-n_filtering_dates) + ' entries were removed because the admission date, discharge date, status at discharge or ICU transfer was missing.')

# Convert dates to pd.datetimes
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

# Remove the negative admission to onset times
df.drop(df[((df['dt_admission'] - df['dt_onset'])/datetime.timedelta(days=1)) < 0].index, inplace=True)

# Remove all residence times larger than 180 days
df.drop(df[((df['dt_discharge'] - df['dt_admission'])/datetime.timedelta(days=1)) >= 180].index, inplace=True)
n_filtering_times = df.shape[0]
print(str(n_filtering_dates-n_filtering_times) + ' entries were removed because the residence time or onset time were negative.')

# Print a summary of the filtering
print(str(n_orig-n_filtering_times)+' entries were removed during filtering. '+str(n_filtering_times)+' entries remained.')

###################################################
## Compute fractions: c, m0, m0_{ICU} and m0_{C} ##
###################################################

quantiles = [25,75,2.5,97.5]

# ------------------------------------------------------
# Initialize dataframe for results and population totals
# ------------------------------------------------------

columns = [[],[]]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["parameter", "quantity"])
fractions = pd.DataFrame(index=age_classes, columns=columns)
averages = pd.DataFrame(index=['population'],columns=columns)

# -------------------------------------------
# Compute fraction parameters point estimates
# -------------------------------------------

# Sample size
fractions['total_sample_size','point estimate']=df.groupby(by='age_class').apply(lambda x: x.age.count())
# Hospitalization propensity
fractions['admission_propensity','point estimate']=df.groupby(by='age_class').apply(lambda x: x.age.count())/df.shape[0]
# Distribution cohort/icu
fractions['c','point estimate'] = df.groupby(by='age_class').apply(lambda x: x[x.ICU_transfer=='Non'].age.count()/x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count())
# Mortalities
fractions['m0','point estimate']=df.groupby(by='age_class').apply(
                                lambda x: x[( (x.status_discharge=='D'))].age.count()/
                                            x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count())
fractions['m0_{ICU}','point estimate']= df.groupby(by='age_class').apply(
                                lambda x: x[((x.ICU_transfer=='Oui') & (x.status_discharge=='D'))].age.count()/
                                          x[x.ICU_transfer.isin(['Oui'])].age.count())
fractions['m0_{C}','point estimate']= df.groupby(by='age_class').apply(
                                lambda x: x[((x.ICU_transfer=='Non') & (x.status_discharge=='D'))].age.count()/
                                          x[x.ICU_transfer.isin(['Non'])].age.count())

# -----------------------------
# Bootstrap fraction parameters
# -----------------------------

subset_size = 1000
n = 2000

# First initialize a numpy array for the results
# First axis: parameter: c, m0, m0_C, m0_ICU
# Second axis: age group
# Third axis: bootstrap sample
bootstrap_fractions_age = np.zeros([4, len(age_classes), n])

# Loop over parameters
for idx in range(4):
    for jdx in range(n):
        smpl = df.groupby(by='age_class').apply(lambda x: x.sample(n=subset_size,replace=True))
        smpl=smpl.drop(columns='age_class')
        if idx == 0:
            bootstrap_fractions_age[idx,:,jdx] = smpl.groupby(by='age_class').apply(lambda x: x[x.ICU_transfer=='Non'].age.count()/
                                                                                            x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count()).values
        elif idx == 1:
            bootstrap_fractions_age[idx,:,jdx] = smpl.groupby(by='age_class').apply(lambda x: x[( (x.status_discharge=='D'))].age.count()/
                                                                                            x[x.ICU_transfer.isin(['Oui', 'Non'])].age.count()).values
        elif idx == 2:
            bootstrap_fractions_age[idx,:,jdx] = smpl.groupby(by='age_class').apply(lambda x: x[((x.ICU_transfer=='Non') & (x.status_discharge=='D'))].age.count()/
                                                                                            x[x.ICU_transfer.isin(['Non'])].age.count()).values
        elif idx == 3:
            bootstrap_fractions_age[idx,:,jdx] = smpl.groupby(by='age_class').apply(lambda x: x[((x.ICU_transfer=='Oui') & (x.status_discharge=='D'))].age.count()/
                                                                                            x[x.ICU_transfer.isin(['Oui'])].age.count()).values
# Compute summary statistics
for idx,par in enumerate(['c', 'm0', 'm0_{C}', 'm0_{ICU}']):
    fractions[par,'bootstrap mean'] = np.median(bootstrap_fractions_age[idx,:,:], axis=1)
    fractions[par,'bootstrap median'] = np.median(bootstrap_fractions_age[idx,:,:], axis=1)
    for quantile in quantiles:
        fractions[par,'bootstrap Q'+str(quantile)] = np.quantile(bootstrap_fractions_age[idx,:,:], q=quantile/100, axis=1)

# Save raw samples as a .npy
with open('../../data/interim/model_parameters/COVID19_SEIRD/sciensano_bootstrap_fractions.npy', 'wb') as f:
    np.save(f,bootstrap_fractions_age)

# Compute population average/total point estimate
averages['total_sample_size','point estimate'] = fractions['total_sample_size','point estimate'].sum()
averages['admission_propensity', 'point estimate'] = sum(((fractions['total_sample_size','point estimate']*fractions['admission_propensity', 'point estimate']).values)/(np.ones(9)*fractions['total_sample_size', 'point estimate'].sum()))
averages['c', 'point estimate'] = df[df.ICU_transfer=='Non'].age.count()/df[df.ICU_transfer.isin(['Oui', 'Non'])].age.count()
averages['m0', 'point estimate'] = df[((df.status_discharge=='D'))].age.count()/df[df.ICU_transfer.isin(['Oui', 'Non'])].age.count()
averages['m0_{ICU}', 'point estimate'] = df[((df.ICU_transfer=='Oui') & (df.status_discharge=='D'))].age.count()/df[df.ICU_transfer.isin(['Oui'])].age.count()
averages['m0_{C}', 'point estimate'] = df[((df.ICU_transfer=='Non') & (df.status_discharge=='D'))].age.count()/df[df.ICU_transfer.isin(['Non'])].age.count()

# Bootstrap total population
bootstrap_fractions = np.zeros([4, n])
# Loop over parameters
for idx in range(4):
    for jdx in range(n):
        smpl = df.sample(n=subset_size,replace=True)
        if idx == 0:
            bootstrap_fractions[idx,jdx] = smpl[smpl.ICU_transfer=='Non'].age.count()/smpl[smpl.ICU_transfer.isin(['Oui', 'Non'])].age.count()
        elif idx == 1:
            bootstrap_fractions[idx,jdx] = smpl[((smpl.status_discharge=='D'))].age.count()/smpl[smpl.ICU_transfer.isin(['Oui', 'Non'])].age.count()
        elif idx == 2:
            bootstrap_fractions[idx,jdx] = smpl[((smpl.ICU_transfer=='Non') & (smpl.status_discharge=='D'))].age.count()/smpl[smpl.ICU_transfer.isin(['Non'])].age.count()
        elif idx == 3:
            bootstrap_fractions[idx,jdx] = smpl[((smpl.ICU_transfer=='Oui') & (smpl.status_discharge=='D'))].age.count()/smpl[smpl.ICU_transfer.isin(['Oui'])].age.count()

# Compute summary statistics
for idx,par in enumerate(['c', 'm0', 'm0_{C}', 'm0_{ICU}']):
    averages[par,'bootstrap mean'] = np.median(bootstrap_fractions[idx,:])
    averages[par,'bootstrap median'] = np.median(bootstrap_fractions[idx,:])
    for quantile in quantiles:
        averages[par,'bootstrap Q'+str(quantile)] = np.quantile(bootstrap_fractions[idx,:], q=quantile/100)

# -------------------------------------------
# Perform Mann-Whitney U-tests on mortalities
# -------------------------------------------

# Difference in mortality, ICU vs. Cohort
# Boxplot
x = bootstrap_fractions[2,:]
y = bootstrap_fractions[3,:]
stat, p_tt = ttest_ind(x, y)
stat, p_mwu = mannwhitneyu(x, y)
fig, ax = plt.subplots(figsize=(8,6))
bp = ax.boxplot([x, y], positions=[1,2])
plt.setp(bp['medians'], color='k')
ax.set_ylabel('mortality (-)')
ax.set_ylim(0,1)
ax.set_xticklabels(['Cohort mortality (N={}) \n median = {:.2f} \n mean = {:.2f}'.format(len(x), np.median(x), np.mean(x)),
                    'ICU mortality (N={}) \n median = {:.2f} \n mean = {:.2f}'.format(len(y), np.median(y), np.mean(y))])
ax.set_title('Difference in overall mortality, \ntwo-sided t-test: p={:.2e} \nMann-Withney U-test: p={:.2e}'.format(p_tt,p_mwu))
plt.savefig('../../results/analysis/hospital/SCIENSANO_test_mortalities.pdf', dpi=600, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()

# -----------------------------------------------------------------
# Make a violin plot of mortalities in ICU and cohort per age group
# -----------------------------------------------------------------

data = []
for idx,age_class in enumerate(age_classes):
    data.append(bootstrap_fractions_age[2,idx,:])

# Violin plot
fig,ax = plt.subplots(figsize=(12,4))

parts = ax.violinplot(
        data, positions=range(1,len(age_classes)+1), vert=False,showmeans=False, showmedians=False,
        showextrema=False)
for idx,pc in enumerate(parts['bodies']):
    pc.set_facecolor(colorscale_okabe_ito['green'])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartiles = [25, 50, 75]    
quartile1 = np.zeros(len(data))
medians = np.zeros(len(data))
quartile3 = np.zeros(len(data))
for i,x in enumerate(data):
    quartile1[i],medians[i],quartile3[i]  = np.percentile(x, quartiles)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
inds = np.arange(1, len(medians)+1)
ax.scatter( medians, inds, marker='o', color='white', s=30, zorder=3)

ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.hlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

data = []
for idx,age_class in enumerate(age_classes):
    data.append(bootstrap_fractions_age[3,idx,:])

parts = ax.violinplot(
        data, positions=range(1,len(age_classes)+1), vert=False,showmeans=False, showmedians=False,
        showextrema=False)
for idx,pc in enumerate(parts['bodies']):
    pc.set_facecolor(colorscale_okabe_ito['red'])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartiles = [25, 50, 75]    
quartile1 = np.zeros(len(data))
medians = np.zeros(len(data))
quartile3 = np.zeros(len(data))
for i,x in enumerate(data):
    quartile1[i],medians[i],quartile3[i]  = np.percentile(x, quartiles)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
inds = np.arange(1, len(medians)+1)
ax.scatter( medians, inds, marker='o', color='white', s=30, zorder=3)
ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.hlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)


ax.set_xlabel('mortality (-)')
ax.set_xlim(0,1)
ax.set_ylim(0,10)
ax.set_yticks(inds)
ax.set_yticklabels(age_classes.values,fontsize=10)
plt.tight_layout()
plt.savefig('../../results/analysis/hospital/SCIENSANO_violin_mortalities.pdf', dpi=300, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()


# Concatenate dataframes
fractions = pd.concat([fractions, averages])

###################################################################################
## Compute residence times: d_{hospital}, d_{C,R}, d_{C,D}, d_{ICU,R}, d_{ICU,D} ##
###################################################################################

# --------------------------------
# Initialize dataframe for results
# --------------------------------

columns = [[],[]]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["parameter", "quantity"])
residence_times = pd.DataFrame(index=age_classes, columns=columns)
samples = pd.DataFrame(index=age_classes, columns=[])
samples_total = pd.DataFrame(index=['total'], columns=[])

# ----------
# d_hospital
# ----------

# Summary statistics
residence_times['d_hospital','mean'] = df.groupby(by='age_class').apply(lambda x: (x['dt_admission'] - x['dt_onset']).mean()/datetime.timedelta(days=1))
residence_times['d_hospital','median']  = df.groupby(by='age_class').apply(lambda x: (x['dt_admission'] - x['dt_onset']).median()/datetime.timedelta(days=1))
for quantile in quantiles:
    residence_times['d_hospital','Q'+str(quantile)] = df.groupby(by='age_class').apply(lambda x: (x['dt_admission'] - x['dt_onset']).quantile(q=quantile/100)/datetime.timedelta(days=1))
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: (x['dt_admission'] - x['dt_onset'])/datetime.timedelta(days=1))
residence_times['d_hospital','sample_size'], residence_times['d_hospital','shape'],residence_times['d_hospital','loc'],residence_times['d_hospital','scale'] = fit_weibull(v)
if plot_fit:
    plot_weibull_fit(v,'d_hospital',30)

# ----------------------------
# Transfer time cohort --> ICU
# ----------------------------

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
values_d_transfer = values

# Summary statistics
residence_times['d_transfer','mean'] = df.groupby(by='age_class').apply(lambda x: x[x.ICU_transfer=='Oui'].d_transfer.mean())
residence_times['d_transfer','median'] = df.groupby(by='age_class').apply(lambda x: x[x.ICU_transfer=='Oui'].d_transfer.median())
for quantile in quantiles:
    residence_times['d_transfer','Q'+str(quantile)] = df.groupby(by='age_class').apply(lambda x: x[x.ICU_transfer=='Oui'].d_transfer.quantile(q=quantile/100)) 
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: x[x.ICU_transfer=='Oui'].d_transfer)
residence_times['d_transfer','sample_size'], residence_times['d_transfer','shape'], residence_times['d_transfer','loc'], residence_times['d_transfer', 'scale'] = fit_weibull(v)
if plot_fit:
    plot_weibull_fit(v,'d_transfer',30)
# Append samples
samples['d_transfer'] = df.groupby(by='age_class').d_transfer.agg(lambda x: list(x.dropna()))
samples_total['d_transfer'] = [df.d_transfer.agg(lambda x: list(x.dropna()))]

# ---
# dC
# ---

# Summary statistics
residence_times['dC','mean']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1)).mean())
residence_times['dC','median']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1)).median())
for quantile in quantiles:
    residence_times['dC','Q'+str(quantile)]=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1)).quantile(q=quantile/100))
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1)))
residence_times['dC','sample_size'], residence_times['dC','shape'],residence_times['dC','loc'],residence_times['dC','scale'] = fit_weibull(v)
if plot_fit:
    plot_weibull_fit(v,'dC',90)
# Append samples
samples['dC'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dC'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1))).agg(lambda x: list(x))]

# -----
# dC_R
# -----

# Summary statistics
residence_times['dC_R', 'mean']= df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).mean())
residence_times['dC_R', 'median']= df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).median())
for quantile in quantiles:
    residence_times['dC_R', 'Q'+str(quantile)] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).quantile(q=quantile/100))
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)))

residence_times['dC_R','sample_size'], residence_times['dC_R','shape'],residence_times['dC_R','loc'],residence_times['dC_R','scale'] = fit_weibull(v)
if plot_fit:
    plot_weibull_fit(v,'dC_R',90)
# Append samples
samples['dC_R'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dC_R'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Non'] - x['dt_admission'][x.ICU_transfer=='Non'])/datetime.timedelta(days=1))).agg(lambda x: list(x))]

# -----
# dC_D
# -----

# Summary statistics
residence_times['dC_D', 'mean']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()).fillna(1)
residence_times['dC_D', 'median']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).median()).fillna(1)
for quantile in quantiles:
    residence_times['dC_D', 'Q'+str(quantile)]=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).quantile(q=quantile/100)).fillna(1)
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)))
sample_size, shape, loc, scale = fit_weibull(v)
for i in range(2):
    sample_size.insert(0,0)
    shape.insert(0,1)
    loc.insert(0,0)
    scale.insert(0,1)
residence_times['dC_D','sample_size'], residence_times['dC_D','shape'],residence_times['dC_D','loc'],residence_times['dC_D','scale'] = sample_size, shape, loc, scale
if plot_fit:
    plot_weibull_fit(v,'dC_D',90)
# Append samples
samples['dC_D'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dC_D'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1))).agg(lambda x: list(x))]
samples['dC_D'].loc[residence_times.index.get_level_values(0).unique().values[0]] = [1]
samples['dC_D'].loc[residence_times.index.get_level_values(0).unique().values[1]] = [1]

#------
# dICU
# -----

# Summary statistics
residence_times['dICU','mean']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).mean())
residence_times['dICU','median']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).median())
for quantile in quantiles:
    residence_times['dICU','Q'+str(quantile)]=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).quantile(q=quantile/100))
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1)))
residence_times['dICU','sample_size'], residence_times['dICU','shape'],residence_times['dICU','loc'],residence_times['dICU','scale'] = fit_weibull(v)
if plot_fit:
    plot_weibull_fit(v,'dICU',90)
# Append samples
samples['dICU'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dICU'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][x.ICU_transfer=='Oui'] - x['dt_admission'][x.ICU_transfer=='Oui'])/datetime.timedelta(days=1))).agg(lambda x: list(x))]

# -------
# dICU_R 
# -------

# Summary statistics
residence_times['dICU_R','mean']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).mean())
residence_times['dICU_R','median']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).median())
for quantile in quantiles:
    residence_times['dICU_R','Q'+str(quantile)]=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)).quantile(q=quantile/100))
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)))
residence_times['dICU_R','sample_size'], residence_times['dICU_R','shape'],residence_times['dICU_R','loc'],residence_times['dICU_R','scale'] = fit_weibull(v)
if plot_fit:
    plot_weibull_fit(v,'dICU_R',90)
# Append samples
samples['dICU_R'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dICU_R'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1))).agg(lambda x: list(x))]

# -------
# dICU_D
# -------

# Summary statistics
residence_times['dICU_D','mean']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()).fillna(1)
residence_times['dICU_D','median']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).median()).fillna(1)
for quantile in quantiles:
    residence_times['dICU_D','Q'+str(quantile)]=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).quantile(q=quantile/100)).fillna(1)
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)))
sample_size, shape, loc, scale = fit_weibull(v)
for i in range(1):
    sample_size.insert(0,0)
    shape.insert(0,1)
    loc.insert(0,0)
    scale.insert(0,1)
residence_times['dICU_D','sample_size'], residence_times['dICU_D','shape'],residence_times['dICU_D','loc'],residence_times['dICU_D','scale'] = sample_size, shape, loc, scale
if plot_fit:
    plot_weibull_fit(v,'dICU_D',90)
# Append samples
samples['dICU_D'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dICU_D'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])/datetime.timedelta(days=1))).agg(lambda x: list(x))]
samples['dICU_D'].loc[residence_times.index.get_level_values(0).unique().values[0]] = [1]
samples = pd.concat([samples, samples_total])

#################################
## Compute averages and totals ##
#################################

columns = [[],[]]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["parameter", "quantity"])
averages = pd.DataFrame(index=['averages'], columns=columns)

# ---
# dC
# ---

# Summary statistics
averages['dC','mean'] = ((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1)).mean()
averages['dC','median'] = ((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1)).median()
for quantile in quantiles:
    averages['dC','Q'+str(quantile)] = ((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1)).quantile(q=quantile/100)
# Gamma fit
v = ((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dC','sample_size'] = len(v)
averages['dC','shape'],averages['dC','loc'],averages['dC','scale'] = gamma.fit(v, floc=0)

# ----
# dC,R
# ----

# Summary statistics
averages['dC_R','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).mean()
averages['dC_R','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).median()
for quantile in quantiles:
    averages['dC_R','Q'+str(quantile)] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).quantile(q=quantile/100)
# Gamma fit
v = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dC_R','sample_size'] = len(v)
averages['dC_R','shape'],averages['dC_R','loc'],averages['dC_R','scale'] = gamma.fit(v, floc=0)

# ----
# dC,D
# ----

# Summary statistics
averages['dC_D','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()
averages['dC_D','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).median()
for quantile in quantiles:
    averages['dC_D','Q'+str(quantile)] = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).quantile(q=quantile/100)
# Gamma fit
v = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dC_D','sample_size'] = len(v)
averages['dC_D','shape'],averages['dC_D','loc'],averages['dC_D','scale'] = gamma.fit(v, floc=0)

# ----
# dICU
# ----

# Summary statistics
averages['dICU','mean'] = ((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).mean()
averages['dICU','median'] = ((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).median()
for quantile in quantiles:
    averages['dICU','Q'+str(quantile)] = ((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).quantile(q=quantile/100)
# Gamma fit
v = ((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dICU','sample_size'] = len(v)
averages['dICU','shape'],averages['dICU','loc'],averages['dICU','scale'] = gamma.fit(v, floc=0)

# ------
# dICU,R
# ------

# Summary statistics
averages['dICU_R','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).mean()
averages['dICU_R','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).median()
for quantile in quantiles:
    averages['dICU_R','Q'+str(quantile)] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).quantile(q=quantile/100)
# Gamma fit
v = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dICU_R','sample_size'] = len(v)
averages['dICU_R','shape'],averages['dICU_R','loc'],averages['dICU_R','scale'] = gamma.fit(v, floc=0)

# ------
# dICU,D
# ------

# Summary statistics
averages['dICU_D','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).mean()
averages['dICU_D','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).median()
for quantile in quantiles:
    averages['dICU_D','Q'+str(quantile)] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).quantile(q=quantile/100)
# Gamma fit
v = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1))
v[v==0] = 0.01
averages['dICU_D','sample_size'] = len(v)
averages['dICU_D','shape'],averages['dICU_D','loc'],averages['dICU_D','scale'] = gamma.fit(v, floc=0)

# ----------
# d_transfer
# ----------

averages['d_transfer','mean'] = np.mean(values_d_transfer)
averages['d_transfer','median'] = np.median(values_d_transfer)
for quantile in quantiles:
    averages['d_transfer','Q'+str(quantile)] = np.quantile(values_d_transfer,q=quantile/100)
averages['d_transfer','shape'], averages['d_transfer','loc'], averages['d_transfer', 'scale'] = gamma.fit(values, floc=0)
averages['d_transfer','sample_size'] = len(values)

# ----------
# d_hospital
# ----------

# Summary statistics
averages['d_hospital','mean'] = (df['dt_admission'] - df['dt_onset']).mean()/datetime.timedelta(days=1)
averages['d_hospital','median']  = (df['dt_admission'] - df['dt_onset']).median()/datetime.timedelta(days=1)
for quantile in quantiles:
    averages['d_hospital','Q'+str(quantile)] = (df['dt_admission'] - df['dt_onset']).quantile(q=quantile/100)/datetime.timedelta(days=1)
# Gamma fit
v = ((df['dt_admission'] - df['dt_onset'])/datetime.timedelta(days=1)).dropna()
v[v==0] = 0.01
averages['d_hospital','sample_size'] = len(v)
averages['d_hospital','shape'],averages['d_hospital','loc'],averages['d_hospital','scale'] = gamma.fit(v, floc=0)

residence_times = pd.concat([residence_times, averages])

# -----------------------------------------------
# Perform Mann-Whitney U-tests on residence times
# -----------------------------------------------

# Difference in hospital residence time Cohort vs. ICU
x = ((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)) # ICU
y = ((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1)) # Cohort
stat, p_tt = ttest_ind(x, y)
stat, p_mwu = mannwhitneyu(x, y)
fig, ax = plt.subplots(figsize=(8,6))
bp = ax.boxplot([x, y], positions=[1,2])
plt.setp(bp['medians'], color='k')
ax.set_ylabel('length of stay (days)')
ax.set_ylim(0,200)
ax.set_xticklabels(['ICU patients (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(x), x.median(), x.mean()),
                    'Cohort only patients (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(y), y.median(), y.mean())])
ax.set_title('Difference in hospital residence Cohort vs ICU, \ntwo-sided t-test: p={:.2e} \nMann-Withney U-test: p={:.2e}'.format(p_tt,p_mwu))
plt.savefig('../../results/analysis/hospital/SCIENSANO_test_residence_ICU_Cohort.pdf', dpi=300, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()

# Difference in ICU residence time in case of recovery and death
x = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1))
y = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1))
stat, p_tt = ttest_ind(x, y)
stat, p_mwu = mannwhitneyu(x, y)
fig, ax = plt.subplots(figsize=(8,6))
bp = ax.boxplot([x, y], positions=[1,2])
plt.setp(bp['medians'], color='k')
ax.set_ylabel('length of stay (days)')
ax.set_ylim(0,200)
ax.set_xticklabels(['ICU recovered (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(x), x.median(), x.mean()),
                    'ICU deceased (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(y), y.median(), y.mean())])
ax.set_title('Difference in ICU residence time in case of recovery and death, \ntwo-sided t-test: p={:.1e} \nMann-Withney U-test: p={:.1e}'.format(p_tt,p_mwu))
plt.savefig('../../results/analysis/hospital/SCIENSANO_test_residence_ICU_R_D.pdf', dpi=300, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()

# Difference in Cohort residence time in case of recovery and death
x = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1))
y = ((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1))
stat, p_tt = ttest_ind(x, y)
stat, p_mwu = mannwhitneyu(x, y)
fig, ax = plt.subplots(figsize=(8,6))
bp = ax.boxplot([x, y], positions=[1,2])
plt.setp(bp['medians'], color='k')
ax.set_ylabel('length of stay (days)')
ax.set_ylim(0,200)
ax.set_xticklabels(['Cohort only recovered (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(x), x.median(), x.mean()),
                    'Cohort only deceased (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(y), y.median(), y.mean())])
ax.set_title('Difference in Cohort residence time in case of recovery and death, \ntwo-sided t-test: p={:.1e} \nMann-Withney U-test: p={:.1e}'.format(p_tt,p_mwu))
plt.savefig('../../results/analysis/hospital/SCIENSANO_test_residence_Cohort_R_D.pdf', dpi=600, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()

# ------------------------------------------------------
# Make a violin plot of the residence time distributions
# ------------------------------------------------------

# dC, dC_r, dC_d, ICU, ICU_R, ICU_D
dC = np.array(((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1)).values)
dC_R = np.array(((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).values)
dC_D = np.array(((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).values)
dICU = np.array(((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).values)
dICU_R = np.array(((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).values)
dICU_D = np.array(((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).values)
data = [dC,dC_R,dC_D,dICU,dICU_R,dICU_D]
colors = [colorscale_okabe_ito['black'],colorscale_okabe_ito['green'],colorscale_okabe_ito['red'],colorscale_okabe_ito['black'],colorscale_okabe_ito['green'],colorscale_okabe_ito['red']]
alphas = [0.4,1,1,0.4,1,1]
labels = ['Cohort (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(data[0]), np.median(data[0]), np.mean(data[0])),
        'Cohort recovered (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(data[1]), np.median(data[1]), np.mean(data[1])),
        'Cohort deceased (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(data[2]), np.median(data[2]), np.mean(data[2])),
        'IC (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(data[3]), np.median(data[3]), np.mean(data[3])),
        'IC recovered (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(data[4]), np.median(data[4]), np.mean(data[4])),
        'IC deceased (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(data[5]), np.median(data[5]), np.mean(data[5]))
        ]

# Violin plot
fig,ax = plt.subplots(figsize=(8,6))

parts = ax.violinplot(
        data, vert=False,showmeans=False, showmedians=False,
        showextrema=False)
for idx,pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[idx])
    pc.set_edgecolor('black')
    pc.set_alpha(alphas[idx])

quartiles = [25, 50, 75]    
quartile1 = np.zeros(len(data))
medians = np.zeros(len(data))
quartile3 = np.zeros(len(data))
for i,x in enumerate(data):
    quartile1[i],medians[i],quartile3[i]  = np.percentile(x, quartiles)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
inds = np.arange(1, len(medians) + 1)
ax.scatter( medians, inds, marker='o', color='white', s=30, zorder=3)
ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.hlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

ax.set_xlabel('length of stay (days)')
ax.set_xlim(0,60)
ax.set_yticks(inds)
ax.set_yticklabels(labels,fontsize=10)
plt.tight_layout()
plt.savefig('../../results/analysis/hospital/SCIENSANO_violin_residence_times.pdf', dpi=600, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()

# ----------------------------------------------------------------
# Write age-stratified parameters to data/interim/model_parameters
# ----------------------------------------------------------------
with pd.ExcelWriter('../../data/interim/model_parameters/COVID19_SEIRD/sciensano_hospital_parameters.xlsx') as writer:  
    fractions.to_excel(writer,sheet_name='fractions')
    residence_times.to_excel(writer,sheet_name='residence_times')
    #samples.to_excel(writer,sheet_name='residence_times_samples')
