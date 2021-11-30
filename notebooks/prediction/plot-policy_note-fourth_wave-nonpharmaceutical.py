import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from covid19model.visualization.output import _apply_tick_locator 

######################
## Load simulations ##
######################

report_version = 'v1.2'
report_name = 'policy_note-fourth_wave-nonpharmaceutical'
# Path where figures and results should be stored
results_path = '../../results/predictions/prov/' + report_name
df = pd.read_csv(results_path+'/simulations'+'-'+report_version+'.csv', index_col = [0,1,2], header=[0,1,2], parse_dates=True)
dates = df.index.get_level_values('date').unique()
scenarios = df.index.get_level_values('scenario').unique()
#scenario_names = ['S0: No intervention', 'S1: Mandatory telework', 'S2: School closure', 'S3: -50% leisure contacts', 'S4: S1 + S2 + S3']
scenario_names = ['S0: Mandatory telework only', 'S1: S0 - 30% leisure contacts', 'S2: S0 - 60% leisure contacts', 'S3: S0 - 90% leisure contacts']
models = df.columns.get_level_values('model').unique()
end_calibration = '2021-11-12'

###############
## Load data ##
###############

from covid19model.data import sciensano, model_parameters
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_stratification_size=10, spatial='prov', vaccination=True, VOC=True)

start_visualization = '2020-09-01'
end_visualization = '2022-01-01'

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,figsize=(12,6))
# Hospitalizations
rolling_windows = df_hosp.groupby(level='date').sum()['H_in'].rolling(7, min_periods=7)
rolling_mean = rolling_windows.mean()
ax[0].plot(df_hosp.index.get_level_values('date').unique().values, rolling_mean, color='red', alpha=1, linewidth=2)
ax[0].scatter(df_hosp['H_in'].groupby(level='date').sum()[:end_calibration].index, df_hosp['H_in'].groupby(level='date').sum()[:end_calibration],color='black', alpha=0.2, linestyle='None', facecolors='none', s=60, linewidth=2)
ax[0].scatter(df_hosp['H_in'].groupby(level='date').sum()[end_calibration:].index, df_hosp['H_in'].groupby(level='date').sum()[end_calibration:],color='red', alpha=0.2, linestyle='None', facecolors='none', s=60, linewidth=2)
ax[0] = _apply_tick_locator(ax[0])
ax[0].set_xlim([start_visualization, end_visualization])
ax[0].grid(False)
ax[0].set_ylabel('New hospitalizations')
# Cases
df_cases = df_cases.groupby(level='date').sum().iloc[:-1]
rolling_windows = df_cases.groupby(level='date').sum().rolling(7, min_periods=7)
rolling_mean = rolling_windows.mean()
ax[1].plot(df_cases.index.get_level_values('date').unique().values, rolling_mean, color='red', alpha=1, linewidth=2)
ax[1].scatter(df_cases.groupby(level='date').sum()[:end_calibration].index, df_cases.groupby(level='date').sum()[:end_calibration],color='black', alpha=0.2, linestyle='None', facecolors='none', s=60, linewidth=2)
ax[1].scatter(df_cases.groupby(level='date').sum()[end_calibration:].index, df_cases.groupby(level='date').sum()[end_calibration:],color='red', alpha=0.2, linestyle='None', facecolors='none', s=60, linewidth=2)
ax[1] = _apply_tick_locator(ax[1])
ax[0].set_xlim([start_visualization, end_visualization])
ax[1].grid(False)
ax[1].set_ylabel('New cases')
# Print figure
plt.tight_layout()
plt.show()
plt.close()

####################################################
## Relative effect on cumulative hospitalizations ##
####################################################

baseline=[]
new_index = pd.IndexSlice
for idx, scenario in enumerate(scenarios):
    for jdx, model in enumerate(models):
        if idx == 0:
            baseline.append(df.loc[new_index['2021-11-17':'2022-02-01',1000,scenario], (model, 'H_tot', 'mean')].cumsum()[-1])
        val = (df.loc[new_index['2021-11-17':'2022-02-01',1000,scenario], (model, 'H_tot', 'mean')].cumsum()[-1]/baseline[jdx])*100
        print('\nCumulative hospitalizations under scenario S' + str(idx) + ' and model "' + model + '":' + str(round(val,2)))

##############
## National ##
##############

ICU_ratio = 0.20
start_visualization = '2020-03-15'
end_visualization = '2022-04-01'
maxy = [950, 8500]
states = ['H_in', 'H_tot']
state_labels = ['$H_{in}$ (-)', '$H_{tot}$ (-)']
colors = ['blue', 'green']

for kdx, state in enumerate(states):

    fig,ax = plt.subplots(nrows=len(scenarios), ncols=1, figsize=(15, 8.3), sharex=True)

    for idx, scenario in enumerate(scenarios):
        for jdx, model in enumerate(models):
            ax[idx].plot(dates, df.loc[(slice(None), 1000, scenario), (model, state, 'mean')], '--', linewidth=1.5, color = colors[jdx])
            ax[idx].fill_between(dates, df.loc[(slice(None), 1000, scenario), (model, state, 'lower')],
                                    df.loc[(slice(None), 1000, scenario), (model, state, 'upper')], alpha=0.2, color = colors[jdx])
            ax[idx].scatter(df_hosp[state].groupby(level='date').sum()[:end_calibration].index, df_hosp[state].groupby(level='date').sum()[:end_calibration],color='black', alpha=0.2, linestyle='None', facecolors='none', s=60, linewidth=2)
            ax[idx].scatter(df_hosp[state].groupby(level='date').sum()[end_calibration:].index, df_hosp[state].groupby(level='date').sum()[end_calibration:],color='red', alpha=0.2, linestyle='None', facecolors='none', s=60, linewidth=2)

        if ((state == 'H_tot') & (idx == 0)):
            ax[idx].axhline(y=1000/ICU_ratio, c='gray', linestyle='dashed', zorder=-10, linewidth=2)
            ax[idx].text(pd.Timestamp('2021-03-15'),1000/ICU_ratio+500, 'Nominal IC-capacity (1000 beds)', fontsize=13)
        elif state == 'H_tot':
            ax[idx].axhline(y=1000/ICU_ratio, c='gray', linestyle='dashed', zorder=-10, linewidth=2)

        ax[idx] = _apply_tick_locator(ax[idx])
        ax[idx].set_xlim([start_visualization, end_visualization])
        ax[idx].set_ylim([0, maxy[kdx]])
        ax[idx].set_ylabel(state_labels[kdx])
        ax[idx].grid(False)

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        ax[idx].text(0.02, 0.88, scenario_names[idx], transform=ax[idx].transAxes, fontsize=13, verticalalignment='center', bbox=props)

        #ax[idx].set_title(scenario_names[idx])

    ax[0].legend(['UGent nation-level', 'UGent provincial-level\n(aggregated)'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)

    plt.tight_layout()
    plt.show()
    plt.close()

##############
## Regional ##
##############

# Settings
start_visualization = '2020-03-15'
end_visualization = '2022-04-01'
maxy = [12, 65]

for kdx, state in enumerate(states):

    # Calculate the number of inhabitants in the regions
    title_list = ['Belgium', 'Flanders', 'Wallonia', 'Brussels']
    NIS_prov_regions = [[10000,70000,40000,20001,30000], [50000, 60000, 80000, 90000, 20002], [21000]]
    initN_regions = [np.sum(np.sum(initN,axis=0))]
    for NIS in NIS_prov_regions:
        initN_regions.append(np.sum(np.sum(initN.loc[NIS])))

    # Aggregate the data
    data_regions_calibration=[]
    data_regions_no_calibration=[]
    for idx,NIS_list in enumerate(NIS_prov_regions):
        data_calibration = 0
        data_no_calibration = 0
        for NIS in NIS_list:
            data_calibration = data_calibration + df_hosp.loc[(slice(None), NIS),state][:end_calibration].values
            data_no_calibration = data_no_calibration + df_hosp.loc[(slice(None), NIS),state][end_calibration:].values
        data_regions_calibration.append(data_calibration)
        data_regions_no_calibration.append(data_no_calibration)

    # Make visualization
    colors = ['black', 'red', 'orange', 'blue', 'green']
    fig,ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 8.3), sharex=True)

    for idx, NIS in enumerate([1000, 2000, 3000, 21000]):
        for jdx, scenario in enumerate(scenarios):
            ax[idx].plot(dates, df.loc[(slice(None), NIS, scenario), ('spatial', state, 'mean')]/initN_regions[idx]*100000, '--', linewidth=1.5, color = colors[jdx])
            ax[idx].fill_between(dates, df.loc[(slice(None), NIS, scenario), ('spatial', state, 'lower')]/initN_regions[idx]*100000,
                                    df.loc[(slice(None), NIS, scenario), ('spatial', state, 'upper')]/initN_regions[idx]*100000, alpha=0.04, color = 'black')
            if idx == 0:
                ax[idx].scatter(df_hosp[state].groupby(level='date').sum()[:end_calibration].index, df_hosp[state].groupby(level='date').sum()[:end_calibration]/initN_regions[idx]*100000,color='black', alpha=0.1, linestyle='None', facecolors='none', s=60, linewidth=2)
                ax[idx].scatter(df_hosp[state].groupby(level='date').sum()[end_calibration:].index, df_hosp[state].groupby(level='date').sum()[end_calibration:]/initN_regions[idx]*100000,color='red', alpha=0.1, linestyle='None', facecolors='none', s=60, linewidth=2)
       
            else:
                ax[idx].scatter(df_hosp[state].groupby(level='date').sum()[:end_calibration].index, data_regions_calibration[idx-1]/initN_regions[idx]*100000, color='black', alpha=0.10, linestyle='None', facecolors='none', s=60, linewidth=2)
                ax[idx].scatter(df_hosp[state].groupby(level='date').sum()[end_calibration:].index, data_regions_no_calibration[idx-1]/initN_regions[idx]*100000, color='red', alpha=0.10, linestyle='None', facecolors='none', s=60, linewidth=2)
        
        ax[idx] = _apply_tick_locator(ax[idx])
        ax[idx].set_xlim([start_visualization, end_visualization])
        ax[idx].set_ylim([0, maxy[kdx]])
        #ax[idx].set_title(title_list[idx])
        ax[idx].grid(False)
        ax[idx].set_ylabel(state_labels[kdx])

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        ax[idx].text(0.02, 0.88, title_list[idx], transform=ax[idx].transAxes, fontsize=13, verticalalignment='center', bbox=props)


    ax[0].legend(scenario_names, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.show()
    plt.close()

################
## Provincial ##
################

# Settings
start_visualization = '2020-03-15'
end_visualization = '2022-04-01'
maxy = 10
state = 'H_in'
kdx=0

title_list = ['Antwerp', 'Flemish Brabant', 'Walloon Brabant', 'Brussels', 'West Flanders', 'East Flanders', 'Hainaut Province', 'Liège Province', 'Limburg', 'Luxembourg Province', 'Namur Province']
NIS_list = [10000, 20001, 20002, 21000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]

# Make visualization
colors = ['black', 'red', 'orange', 'blue', 'green']

# Pt. I
fig,ax = plt.subplots(nrows=int(np.floor(len(NIS_list)/2)+1),ncols=1,figsize=(15,8.3), sharex=True)

for idx, NIS in enumerate(NIS_list[0:int(np.floor(len(NIS_list)/2)+1)]):
    for jdx, scenario in enumerate(scenarios):
        ax[idx].plot(dates, df.loc[(slice(None), NIS, scenario), ('spatial', state, 'mean')]/np.sum(initN.loc[NIS])*100000, '--', linewidth=1.5, color = colors[jdx])
        ax[idx].fill_between(dates, df.loc[(slice(None), NIS, scenario), ('spatial', state, 'lower')]/np.sum(initN.loc[NIS])*100000,
                                df.loc[(slice(None), NIS, scenario), ('spatial', state, 'upper')]/np.sum(initN.loc[NIS])*100000, alpha=0.04, color = 'black')
        ax[idx].scatter(df_hosp.loc[(slice(None),NIS),state][:end_calibration].index.get_level_values('date').unique().values, df_hosp.loc[(slice(None),NIS),state][:end_calibration]/np.sum(initN.loc[NIS])*100000, color='black', alpha=0.10, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax[idx].scatter(df_hosp.loc[(slice(None),NIS),state][end_calibration:].index.get_level_values('date').unique().values, df_hosp.loc[(slice(None),NIS),state][end_calibration:]/np.sum(initN.loc[NIS])*100000, color='red', alpha=0.10, linestyle='None', facecolors='none', s=60, linewidth=2)

    ax[idx] = _apply_tick_locator(ax[idx])
    ax[idx].set_xlim([start_visualization, end_visualization])
    ax[idx].set_ylim([0, maxy])
    ax[idx].grid(False)
    ax[idx].set_ylabel(state_labels[kdx])

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax[idx].text(0.02, 0.88, title_list[idx], transform=ax[idx].transAxes, fontsize=13, verticalalignment='center', bbox=props)


ax[0].legend(scenario_names, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
plt.tight_layout()
plt.show()
plt.close()

# Pt. II
fig,ax = plt.subplots(nrows=len(NIS_list) - int(np.floor(len(NIS_list)/2)+1),ncols=1,figsize=(15,8.3), sharex=True)

for idx, NIS in enumerate(NIS_list[ (len(NIS_list)-int(np.floor(len(NIS_list)/2)+1) + 1):]):
    for jdx, scenario in enumerate(scenarios):
        ax[idx].plot(dates, df.loc[(slice(None), NIS, scenario), ('spatial', state, 'mean')]/np.sum(initN.loc[NIS])*100000, '--', linewidth=1.5, color = colors[jdx])
        ax[idx].fill_between(dates, df.loc[(slice(None), NIS, scenario), ('spatial', state, 'lower')]/np.sum(initN.loc[NIS])*100000,
                                df.loc[(slice(None), NIS, scenario), ('spatial', state, 'upper')]/np.sum(initN.loc[NIS])*100000, alpha=0.04, color = 'black')
        ax[idx].scatter(df_hosp.loc[(slice(None),NIS),state][:end_calibration].index.get_level_values('date').unique().values, df_hosp.loc[(slice(None),NIS),state][:end_calibration]/np.sum(initN.loc[NIS])*100000, color='black', alpha=0.10, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax[idx].scatter(df_hosp.loc[(slice(None),NIS),state][end_calibration:].index.get_level_values('date').unique().values, df_hosp.loc[(slice(None),NIS),state][end_calibration:]/np.sum(initN.loc[NIS])*100000, color='red', alpha=0.10, linestyle='None', facecolors='none', s=60, linewidth=2)

    ax[idx] = _apply_tick_locator(ax[idx])
    ax[idx].set_xlim([start_visualization, end_visualization])
    ax[idx].set_ylim([0, maxy])
    ax[idx].grid(False)
    ax[idx].set_ylabel(state_labels[kdx])

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax[idx].text(0.02, 0.88, title_list[idx + (len(NIS_list)-int(np.floor(len(NIS_list)/2)+1)) + 1], transform=ax[idx].transAxes, fontsize=13, verticalalignment='center', bbox=props)


ax[0].legend(scenario_names, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
plt.tight_layout()
plt.show()
plt.close()

# Pt. III
state= 'H_in'
kdx=1
NIS = 40000
maxy=8

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(16,3))

for jdx, scenario in enumerate(scenarios):
    ax.plot(dates, df.loc[(slice(None), NIS, scenario), ('spatial', state, 'mean')]/np.sum(initN.loc[NIS])*100000, '--', linewidth=1.5, color = colors[jdx])
    ax.fill_between(dates, df.loc[(slice(None), NIS, scenario), ('spatial', state, 'lower')]/np.sum(initN.loc[NIS])*100000,
                            df.loc[(slice(None), NIS, scenario), ('spatial', state, 'upper')]/np.sum(initN.loc[NIS])*100000, alpha=0.04, color = 'black')
    ax.scatter(df_hosp.loc[(slice(None),NIS),state][:end_calibration].index.get_level_values('date').unique().values, df_hosp.loc[(slice(None),NIS),state][:end_calibration]/np.sum(initN.loc[NIS])*100000, color='black', alpha=0.10, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.scatter(df_hosp.loc[(slice(None),NIS),state][end_calibration:].index.get_level_values('date').unique().values, df_hosp.loc[(slice(None),NIS),state][end_calibration:]/np.sum(initN.loc[NIS])*100000, color='red', alpha=0.10, linestyle='None', facecolors='none', s=60, linewidth=2)

ax = _apply_tick_locator(ax)
ax.set_xlim([start_visualization, end_visualization])
ax.set_ylim([0, maxy])
ax.grid(False)
ax.set_ylabel(state_labels[kdx])

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.02, 0.88, 'East Flanders', transform=ax.transAxes, fontsize=13, verticalalignment='center', bbox=props)


ax.legend(scenario_names, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
plt.tight_layout()
plt.show()
plt.close()