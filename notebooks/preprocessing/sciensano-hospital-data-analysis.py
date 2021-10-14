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
import math
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, gamma, exponweib, weibull_min
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import argparse

# ----------------
# Script arguments
# ----------------

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subset_size", help="Size of subset drawn from total population during bootstrapping", default=1000, type=int)
parser.add_argument("-n", "--number_iterations", help="Total number of bootstraps", default=100, type=int)
parser.add_argument("-a", "--age_stratification_size", help="Total number of age groups", default=9, type=int)

# Save as dict
args = parser.parse_args()

# Set correct age_classes
if args.age_stratification_size == 3:
    age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
    age_path = '0_20_60/'
elif args.age_stratification_size == 9:
    age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
    age_path = '0_10_20_30_40_50_60_70_80/'
elif args.age_stratification_size == 10:
    age_classes =pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')
    age_path = '0_12_18_25_35_45_55_65_75_85/'
else:
    raise ValueError(
        "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(args.age_stratification_size)
        )

# -----
# Paths
# -----

fig_path = '../../results/analysis/hospital/'+age_path
data_path = '../../data/interim/model_parameters/COVID19_SEIRD/hospitals/' + age_path

# Verify that the paths exist and if not, generate them
for directory in [fig_path, data_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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
        if isinstance(v[age_group],list):
            values = [x for x in v[age_group] if (math.isnan(x) == False)]
            shape, loc, scale = weibull_min.fit(values,floc=0)
            sample_size_lst.append(len(v[age_group]))
        else:
            v[age_group][v[age_group]==0] = 0.01
            v = v.dropna()
            shape, loc, scale = weibull_min.fit(v[age_group].values,floc=0)
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

df['age_class'] = pd.cut(df.age, bins=age_classes)

# Remove the negative residence times
df.drop(df[((df['dt_discharge'] - df['dt_admission'])/datetime.timedelta(days=1)) < 0].index, inplace=True)

# Remove the negative admission to onset times
df.drop(df[((df['dt_admission'] - df['dt_onset'])/datetime.timedelta(days=1)) < 0].index, inplace=True)

# Remove all residence times larger than 180 days
df.drop(df[((df['dt_discharge'] - df['dt_admission'])/datetime.timedelta(days=1)) >= 180].index, inplace=True)
n_filtering_times = df.shape[0]
print(str(n_filtering_dates-n_filtering_times) + ' entries were removed because the residence time or onset time were negative.')

# Drop retirement home patients from dataset
exclude_homes = True
if exclude_homes:
    df.drop(df[df.Expo_retirement_home == 'Oui'].index, inplace=True)
    n_filtering_homes = df.shape[0]
    print(str(n_filtering_times-n_filtering_homes) + ' additional entries were removed because the patient came from a retirement home.')
    # Print a summary of the filtering
    print(str(n_orig-n_filtering_homes)+' entries were removed during filtering. '+str(n_filtering_homes)+' entries remained.')
else:
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

subset_size = args.subset_size
n = args.number_iterations

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
with open(data_path+'sciensano_bootstrap_fractions.npy', 'wb') as f:
    np.save(f,bootstrap_fractions_age)

# Compute population average/total point estimate
averages['total_sample_size','point estimate'] = fractions['total_sample_size','point estimate'].sum()
averages['admission_propensity', 'point estimate'] = sum(((fractions['total_sample_size','point estimate']*fractions['admission_propensity', 'point estimate']).values)/(np.ones(len(age_classes))*fractions['total_sample_size', 'point estimate'].sum()))
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
plt.savefig(fig_path+'SCIENSANO_test_mortalities.pdf', dpi=600, bbox_inches='tight',orientation='portrait', papertype='a4')
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
ax.set_ylim(0,len(age_classes)+1)
ax.set_yticks(inds)
ax.set_yticklabels(age_classes.values,fontsize=10)
plt.tight_layout()
plt.savefig(fig_path+'SCIENSANO_violin_mortalities.pdf', dpi=300, bbox_inches='tight',orientation='portrait', papertype='a4')
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

# --------
# dICU,rec
# --------
cutoff = 60
df['dICUrec'] = np.nan
values=[]
for i in range(len(df['d_transfer'])):
    if ((df['ICU_transfer'].iloc[i] == 'Oui') & (not pd.isnull(df['dt_icu_transfer'].iloc[i])) & (df['status_discharge'].iloc[i] == 'R') & (not pd.isnull(df['length_stay_ICU'].iloc[i]))):
        val = (df['dt_discharge'].iloc[i] - (df['dt_icu_transfer'].iloc[i] + datetime.timedelta(days=df['length_stay_ICU'].iloc[i])))/datetime.timedelta(days=1)
        if ((val >= 0) & (val <= cutoff)):
            df['dICUrec'].iloc[i] = val

# Summary statistics
residence_times['dICUrec','mean'] = df.groupby(by='age_class').apply(lambda x: x[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))].dICUrec.mean())
residence_times['dICUrec','median'] = df.groupby(by='age_class').apply(lambda x: x[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))].dICUrec.median())
for quantile in quantiles:
    residence_times['dICUrec','Q'+str(quantile)] = df.groupby(by='age_class').apply(lambda x: x[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))].dICUrec.quantile(q=quantile/100)) 
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: x[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))].dICUrec)
residence_times['dICUrec','sample_size'], residence_times['dICUrec','shape'], residence_times['dICUrec','loc'], residence_times['dICUrec', 'scale'] = fit_weibull(v)
if plot_fit:
    plot_weibull_fit(v,'dICUrec',cutoff)

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
samples['dC_R'] = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1))).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dC_R'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='R'))])/datetime.timedelta(days=1))).agg(lambda x: list(x))]

# -----
# dC_D
# -----

df['dt_discharge'] = pd.to_datetime(df['dt_discharge'])
df['dt_admission'] = pd.to_datetime(df['dt_admission'])

# Summary statistics
residence_times['dC_D', 'mean']=df.groupby(by='age_class').apply(lambda x: ((pd.to_datetime(x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))]) - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))]))/datetime.timedelta(days=1)).mean()).fillna(1)
residence_times['dC_D', 'median']=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).median()).fillna(1)
for quantile in quantiles:
    residence_times['dC_D', 'Q'+str(quantile)]=df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)).quantile(q=quantile/100)).fillna(1)
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))] - x['dt_admission'][((x.ICU_transfer=='Non')&(x.status_discharge=='D'))])/datetime.timedelta(days=1)))
sample_size, shape, loc, scale = fit_weibull(v)

if args.age_stratification_size == 3:
    append_idx = 1
elif args.age_stratification_size == 9:
    append_idx = 2
elif args.age_stratification_size == 10:
    append_idx = 2

for i in range(append_idx):
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

# -------
# dICU_R 
# -------

# Summary statistics
residence_times['dICU_R','mean']=df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x.dICUrec[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]).mean())
residence_times['dICU_R','median']=df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x.dICUrec[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]).median())
for quantile in quantiles:
    residence_times['dICU_R','Q'+str(quantile)]=df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x.dICUrec[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]).quantile(q=quantile/100))
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x.dICUrec[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]))
residence_times['dICU_R','sample_size'], residence_times['dICU_R','shape'],residence_times['dICU_R','loc'],residence_times['dICU_R','scale'] = fit_weibull(v)
if plot_fit:
    plot_weibull_fit(v,'dICU_R',90)
# Append samples
samples['dICU_R'] =df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x.dICUrec[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dICU_R'] = [df.groupby(by='age_class').apply(lambda x: ((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))])/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))] - x.dICUrec[((x.ICU_transfer=='Oui')&(x.status_discharge=='R'))]).agg(lambda x: list(x))]

# -------
# dICU_D
# -------

# Summary statistics
residence_times['dICU_D','mean']=df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]).mean()).fillna(1)
residence_times['dICU_D','median']=df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]).median()).fillna(1)
for quantile in quantiles:
    residence_times['dICU_D','Q'+str(quantile)]=df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]).quantile(q=quantile/100)).fillna(1)
# Gamma fit
v = df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]))
sample_size, shape, loc, scale = fit_weibull(v)

if args.age_stratification_size == 3:
    append_idx = 0
elif args.age_stratification_size == 9:
    append_idx = 1
elif args.age_stratification_size == 10:
    append_idx = 1

for i in range(append_idx):
    sample_size.insert(0,0)
    shape.insert(0,1)
    loc.insert(0,0)
    scale.insert(0,1)
residence_times['dICU_D','sample_size'], residence_times['dICU_D','shape'],residence_times['dICU_D','loc'],residence_times['dICU_D','scale'] = sample_size, shape, loc, scale
if plot_fit:
    plot_weibull_fit(v,'dICU_D',90)
# Append samples
samples['dICU_D'] = df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])).groupby(by='age_class').agg(lambda x: list(x))
samples_total['dICU_D'] = [df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))] - pd.to_datetime(x['dt_admission'][((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))]))/datetime.timedelta(days=1)) - x.d_transfer[((x.ICU_transfer=='Oui')&(x.status_discharge=='D'))])).agg(lambda x: list(x))]
samples['dICU_D'].loc[residence_times.index.get_level_values(0).unique().values[0]] = [1]

#------
# dICU
# -----

# Add dICU_R and dICU_D together to compute parameters of dICU
samples['dICU'] = samples['dICU_R'] + samples['dICU_D']
# Summary statistics
residence_times['dICU','mean'] = np.nan
residence_times['dICU','median'] = np.nan
for quantile in quantiles:
    residence_times['dICU','Q'+str(quantile)] = np.nan
for idx,age_group in enumerate(samples['dICU'].index.get_level_values(0).unique().values):
    residence_times['dICU','mean'].loc[age_group] = np.nanmean(samples['dICU'][age_group])
    residence_times['dICU','median'].loc[age_group] = np.nanmedian(samples['dICU'][age_group])
    for quantile in quantiles:
        residence_times['dICU','Q'+str(quantile)].loc[age_group] = np.nanquantile(samples['dICU'][age_group],q=quantile/100)
# Gamma fit
v = samples['dICU']#df.groupby(by='age_class').apply(lambda x: (((x['dt_discharge'][x.ICU_transfer=='Oui'] - pd.to_datetime(x['dt_admission'][x.ICU_transfer=='Oui']))/datetime.timedelta(days=1)) - x.d_transfer[x.ICU_transfer=='Oui']))
residence_times['dICU','sample_size'], residence_times['dICU','shape'],residence_times['dICU','loc'],residence_times['dICU','scale'] = fit_weibull(v)
if plot_fit:
    plot_weibull_fit(v,'dICU',90)
# Append samples
samples_total['dICU'] = ''
samples_total['dICU'] = samples_total['dICU'].apply(list)
total_list=[]
for idx,age_group in enumerate(samples['dICU'].index.get_level_values(0).unique().values):
    total_list.extend(samples['dICU'][age_group])
samples_total['dICU']['total'] = total_list

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

# ------
# dICU,R
# ------

# Summary statistics
averages['dICU_R','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1) - df['d_transfer'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='R'))] - df['dICUrec'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='R'))]
).mean()
averages['dICU_R','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1) - df['d_transfer'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='R'))] - df['dICUrec'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='R'))]
).median()
for quantile in quantiles:
    averages['dICU_R','Q'+str(quantile)] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1) - df['d_transfer'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='R'))] - df['dICUrec'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='R'))]
).quantile(q=quantile/100)
# Gamma fit
v = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)- df['d_transfer'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='R'))] - df['dICUrec'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='R'))])
v[v==0] = 0.01
v = [x for x in v if (math.isnan(x) == False)]
v = [x for x in v if (x > 0)]
averages['dICU_R','sample_size'] = len(v)
averages['dICU_R','shape'],averages['dICU_R','loc'],averages['dICU_R','scale'] = gamma.fit(v, floc=0)

# ------
# dICU,D
# ------

# Summary statistics
averages['dICU_D','mean'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1) - df['d_transfer'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='D'))]).mean()
averages['dICU_D','median'] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1) - df['d_transfer'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='D'))]).median()
for quantile in quantiles:
    averages['dICU_D','Q'+str(quantile)] = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1) - df['d_transfer'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='D'))]).quantile(q=quantile/100)
# Gamma fit
v = ((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1) - df['d_transfer'][((df['ICU_transfer']=='Oui')&(df['status_discharge']=='D'))])
v[v==0] = 0.01
v = [x for x in v if (math.isnan(x) == False)]
v = [x for x in v if (x > 0)]
averages['dICU_D','sample_size'] = len(v)
averages['dICU_D','shape'],averages['dICU_D','loc'],averages['dICU_D','scale'] = gamma.fit(v, floc=0)

# ----
# dICU
# ----

# Summary statistics
averages['dICU','mean'] = np.nanmean(samples_total['dICU'][0])
averages['dICU','median'] = np.nanmedian(samples_total['dICU'][0])#((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)- df['d_transfer'][df['ICU_transfer']=='Oui']).median()
for quantile in quantiles:
    averages['dICU','Q'+str(quantile)] = np.nanquantile(samples_total['dICU'][0],q=quantile/100)#((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)- df['d_transfer'][df['ICU_transfer']=='Oui']).quantile(q=quantile/100)
# Gamma fit
v = samples_total['dICU'][0]#((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)- df['d_transfer'][df['ICU_transfer']=='Oui'])
v[(v==0)] = 0.01
v = [x for x in v if (math.isnan(x) == False)]
v = [x for x in v if (x > 0)]
averages['dICU','sample_size'] = len(v)
averages['dICU','shape'],averages['dICU','loc'],averages['dICU','scale'] = gamma.fit(v, floc=0)

# --------
# dICU,rec
# --------
averages['dICUrec','mean'] = df['dICUrec'].mean()
averages['dICUrec','median'] = df['dICUrec'].median()
for quantile in quantiles:
    averages['dICUrec','Q'+str(quantile)] = df['dICUrec'].quantile(q=quantile/100)
v = df['dICUrec']
v = [x for x in v if (math.isnan(x) == False)]
v = [x for x in v if (x > 0)]
averages['dICUrec','sample_size'] = len(v)
averages['dICUrec','shape'],averages['dICUrec','loc'],averages['dICUrec','scale'] = gamma.fit(v, floc=0)

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
df['dt_onset'] = pd.to_datetime(df['dt_onset'])
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

# --------------------
# Build data variables
# --------------------

dC = samples_total['dC'][0] #np.array(((df['dt_discharge'][df.ICU_transfer=='Non'] - df['dt_admission'][df.ICU_transfer=='Non'])/datetime.timedelta(days=1)).values)
dC_R = samples_total['dC_R'][0] #np.array(((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).values)
dC_D = samples_total['dC_D'][0] #np.array(((df['dt_discharge'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Non')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).values)
dICU = samples_total['dICU'][0] #np.array(((df['dt_discharge'][df.ICU_transfer=='Oui'] - df['dt_admission'][df.ICU_transfer=='Oui'])/datetime.timedelta(days=1)).values)
dICU = np.array([x for x in dICU if (math.isnan(x) == False)])
dICU_R = samples_total['dICU_R'][0] #np.array(((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='R'))])/datetime.timedelta(days=1)).values)
dICU_R = np.array([x for x in dICU_R if (math.isnan(x) == False)])
dICU_D = samples_total['dICU_D'][0] #np.array(((df['dt_discharge'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))] - df['dt_admission'][((df.ICU_transfer=='Oui')&(df.status_discharge=='D'))])/datetime.timedelta(days=1)).values)
dICU_D = np.array([x for x in dICU_D if (math.isnan(x) == False)])

# -----------------------------------------------
# Perform Mann-Whitney U-tests on residence times
# -----------------------------------------------

# Difference in hospital residence time Cohort vs. ICU
x = dICU 
y = dC
stat, p_tt = ttest_ind(x, y)
stat, p_mwu = mannwhitneyu(x, y)
fig, ax = plt.subplots(figsize=(8,6))
bp = ax.boxplot([x, y], positions=[1,2])
plt.setp(bp['medians'], color='k')
ax.set_ylabel('length of stay (days)')
ax.set_ylim(0,200)
ax.set_xticklabels(['ICU patients (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(x), np.median(x), np.mean(x)),
                    'Cohort only patients (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(y), np.median(y), np.mean(y))])
ax.set_title('Difference in hospital residence Cohort vs ICU, \ntwo-sided t-test: p={:.2e} \nMann-Withney U-test: p={:.2e}'.format(p_tt,p_mwu))
plt.savefig(fig_path+'SCIENSANO_test_residence_ICU_Cohort.pdf', dpi=300, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()

# Difference in ICU residence time in case of recovery and death
x = dICU_R
y = dICU_D
stat, p_tt = ttest_ind(x, y)
stat, p_mwu = mannwhitneyu(x, y)
fig, ax = plt.subplots(figsize=(8,6))
bp = ax.boxplot([x, y], positions=[1,2])
plt.setp(bp['medians'], color='k')
ax.set_ylabel('length of stay (days)')
ax.set_ylim(0,200)
ax.set_xticklabels(['ICU recovered (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(x), np.median(x), np.mean(x)),
                    'ICU deceased (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(y), np.median(y), np.mean(y))])
ax.set_title('Difference in ICU residence time in case of recovery and death, \ntwo-sided t-test: p={:.1e} \nMann-Withney U-test: p={:.1e}'.format(p_tt,p_mwu))
plt.savefig(fig_path+'SCIENSANO_test_residence_ICU_R_D.pdf', dpi=300, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()

# Difference in Cohort residence time in case of recovery and death
x = dC_R
y = dC_D
stat, p_tt = ttest_ind(x, y)
stat, p_mwu = mannwhitneyu(x, y)
fig, ax = plt.subplots(figsize=(8,6))
bp = ax.boxplot([x, y], positions=[1,2])
plt.setp(bp['medians'], color='k')
ax.set_ylabel('length of stay (days)')
ax.set_ylim(0,200)
ax.set_xticklabels(['Cohort only recovered (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(x), np.median(x), np.mean(x)),
                    'Cohort only deceased (N={}) \n median = {:.1f} \n mean = {:.1f}'.format(len(y), np.median(y), np.mean(y))])
ax.set_title('Difference in Cohort residence time in case of recovery and death, \ntwo-sided t-test: p={:.1e} \nMann-Withney U-test: p={:.1e}'.format(p_tt,p_mwu))
plt.savefig(fig_path+'SCIENSANO_test_residence_Cohort_R_D.pdf', dpi=600, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()

# ------------------------------------------------------
# Make a violin plot of the residence time distributions
# ------------------------------------------------------

data = [dC,dC_R,dC_D,dICU,dICU_R,dICU_D]
colors = [colorscale_okabe_ito['black'],colorscale_okabe_ito['green'],colorscale_okabe_ito['red'],colorscale_okabe_ito['black'],colorscale_okabe_ito['green'],colorscale_okabe_ito['red']]
alphas = [0.4,1,1,0.4,1,1]
labels = ['Cohort \n median = {:.1f} \n mean = {:.1f}'.format( np.median(data[0]), np.mean(data[0])),
        'Cohort recovered \n median = {:.1f} \n mean = {:.1f}'.format(np.median(data[1]), np.mean(data[1])),
        'Cohort deceased \n median = {:.1f} \n mean = {:.1f}'.format(np.median(data[2]), np.mean(data[2])),
        'ICU \n median = {:.1f} \n mean = {:.1f}'.format(np.median(data[3]), np.mean(data[3])),
        'ICU recovered \n median = {:.1f} \n mean = {:.1f}'.format(np.median(data[4]), np.mean(data[4])),
        'ICU deceased \n median = {:.1f} \n mean = {:.1f}'.format(np.median(data[5]), np.mean(data[5]))
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
plt.savefig(fig_path+'SCIENSANO_violin_residence_times.pdf', dpi=600, bbox_inches='tight',orientation='portrait', papertype='a4')
plt.close()

# ----------------------------------------------------------------
# Write age-stratified parameters to data/interim/model_parameters
# ----------------------------------------------------------------
with pd.ExcelWriter(data_path+'sciensano_hospital_parameters.xlsx') as writer:  
    fractions.to_excel(writer,sheet_name='fractions')
    residence_times.to_excel(writer,sheet_name='residence_times')
    #samples.to_excel(writer,sheet_name='residence_times_samples')
