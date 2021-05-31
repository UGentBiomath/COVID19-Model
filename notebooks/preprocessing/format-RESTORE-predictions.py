"""
This script contains all necessary code to extract and convert all raw RESTORE predictions into a more uniform format using pandas multicolumn dataframes.
The resulting dataframe is written to a .csv and contains a five-dimensional header.
To load the .csv file correctly using pandas: RESTORE_df = pd.read_csv(path_to_file+'all_RESTORE_simulations.csv', header=[0,1,2,3,4])
Indexing is performed in the following way: RESTORE_df['UGent','v7.0','S1','incidences','mean'] returns the daily hospitalizations (incidences) in scenario S1 of report v7.0 by UGent.
"""
__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

import os
import pandas as pd
import numpy as np

#########################
## Setup preliminaries ##
#########################

# Path to raw data
abs_dir = os.path.dirname(__file__)
raw_dir = os.path.join(abs_dir,'../../data/raw/RESTORE/simulations/')
iterim_dir = os.path.join(abs_dir,'../../data/interim/RESTORE/')

# Pre-allocation of results dataframe
index = pd.date_range(start='2020-09-01', end='2021-09-01')
columns = [[],[],[],[],[],[]]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["author", "report version", "scenario", "inc/load", "statistic"])
df_RESTORE = pd.DataFrame(index=index, columns=columns)

#################
## RESTORE 4.2 ##
#################
folder = 'RESTORE_4.2/'

# ---------
# Load data
# ---------

# Load SIMID model
SIMID = pd.read_csv(raw_dir+folder+'PredictionsUHasseltmodel.csv', parse_dates=['Date'])
SIMID.index = SIMID['Date']
SIMID.pop('Date')
SIMID = SIMID.rename(columns={SIMID.columns[0]:'S1_load_mean'})
SIMID = SIMID.rename(columns={SIMID.columns[1]:'S1_load_LL'})
SIMID = SIMID.rename(columns={SIMID.columns[2]:'S1_load_UL'})
SIMID = SIMID.rename(columns={SIMID.columns[3]:'S2_load_mean'})
SIMID = SIMID.rename(columns={SIMID.columns[4]:'S2_load_LL'})
SIMID = SIMID.rename(columns={SIMID.columns[5]:'S2_load_UL'})
SIMID = SIMID.rename(columns={SIMID.columns[6]:'S3_load_mean'})
SIMID = SIMID.rename(columns={SIMID.columns[7]:'S3_load_LL'})
SIMID = SIMID.rename(columns={SIMID.columns[8]:'S3_load_UL'})
# Load UNamur model
UNamur = pd.read_csv(raw_dir+folder+'PredictionsUNamurmodel.csv')
UNamur.Date = pd.to_datetime(UNamur.Date, format='%d/%m/%y')
UNamur.index = UNamur['Date']
UNamur.pop('Date')
UNamur = UNamur.rename(columns={UNamur.columns[0]:'S1_incidences_mean'})
UNamur = UNamur.rename(columns={UNamur.columns[1]:'S1_incidences_LL'})
UNamur = UNamur.rename(columns={UNamur.columns[2]:'S1_incidences_UL'})
UNamur = UNamur.rename(columns={UNamur.columns[3]:'S1_load_mean'})
UNamur = UNamur.rename(columns={UNamur.columns[4]:'S1_load_LL'})
UNamur = UNamur.rename(columns={UNamur.columns[5]:'S1_load_UL'})
UNamur = UNamur.rename(columns={UNamur.columns[6]:'S2_incidences_mean'})
UNamur = UNamur.rename(columns={UNamur.columns[7]:'S2_incidences_LL'})
UNamur = UNamur.rename(columns={UNamur.columns[8]:'S2_incidences_UL'})
UNamur = UNamur.rename(columns={UNamur.columns[9]:'S2_load_mean'})
UNamur = UNamur.rename(columns={UNamur.columns[10]:'S2_load_LL'})
UNamur = UNamur.rename(columns={UNamur.columns[11]:'S2_load_UL'})
# Load VUB model
VUB = pd.read_csv(raw_dir+folder+'PredictionsVUBmodel.csv',skiprows=1,decimal=",")
VUB.Date = pd.to_datetime(VUB.Date, format='%d/%m/%y')
VUB.index = VUB['Date']
VUB.pop('Date')
VUB = VUB.rename(columns={VUB.columns[0]:'S1_load_mean'})
VUB = VUB.rename(columns={VUB.columns[2]:'S1_load_LL'})
VUB = VUB.rename(columns={VUB.columns[3]:'S1_load_UL'})

# -----------
# Assign data
# -----------

authors = ['SIMID','UNamur','VUB']
authors_df = [SIMID, UNamur, VUB]
report_v = 'v4.2'
scenarios = ['1','2','3']
statistics = ['mean', 'LL', 'UL']

for idx, author in enumerate(authors):
    for scenario in scenarios:
        for statistic in statistics:
            if author == 'VUB':
                if scenario == '1':
                    df_RESTORE[author,report_v,"S"+scenario,"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]
            elif author == 'SIMID':
                df_RESTORE[author,report_v,"S"+scenario,"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]
            elif author == 'UNamur':
                if ((scenario == '1') | (scenario == '2')):
                    df_RESTORE[author,report_v,"S"+scenario,"incidences", statistic] = authors_df[idx]['S'+scenario+'_incidences_'+statistic]
                    df_RESTORE[author,report_v,"S"+scenario,"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]


#################
## RESTORE 5.0 ##
#################
folder = 'RESTORE_5.0/'

# ---------
# Load data
# ---------

# Load SIMID model
SIMID = pd.read_csv(raw_dir+folder+'predictions_v5_UHasselt.csv', parse_dates=['Date'])
SIMID.index = SIMID['Date']
SIMID.pop('Date')
# Load UGent model
UGent = pd.read_csv(raw_dir+folder+'predictions_UGent.csv')
UGent = UGent.rename(columns={UGent.columns[0]:'Date'})
UGent.Date = pd.to_datetime(UGent.Date)
UGent.index = UGent['Date']
UGent.pop('Date')
# Load UNamur model
UNamur = pd.read_csv(raw_dir+folder+'predictions_Unamur_2410.csv')
UNamur.Date = pd.to_datetime(UNamur.Date, format='%d/%m/%y')
UNamur.index = UNamur['Date']
UNamur.pop('Date')
# Load VUB1 model
VUB1 = pd.read_excel(raw_dir+folder+'VUB_HOSPpredictions2610Fagg.xlsx', skiprows=0)
VUB1.Date = pd.to_datetime(VUB1.Date)
VUB1.columns = ['Date','Observations','Fit', 'S1_load_median','S1_load_mean', 'S1_load_LL', 'S1_load_UL']
VUB1.index = VUB1['Date']
VUB1.pop('Date')
# Load VUB2 model
VUB2 = pd.read_excel(raw_dir+folder+'VUB_HOSPpredictions0112.xlsx', skiprows=0)
VUB2.Date = pd.to_datetime(VUB2.Date)
#VUB2.pop("Unnamed: 7")
#VUB2.pop("Unnamed: 8")
#VUB2.pop("Unnamed: 9")
#VUB2.pop("Unnamed: 10")
VUB2.columns = ['Date','Observations','Fit', 'S1_load_median','S1_load_mean', 'S1_load_LL', 'S1_load_UL']
VUB2.index = VUB2['Date']
VUB2.pop('Date')

# -----------
# Assign data
# -----------

authors = ['UGent','SIMID','UNamur','VUB']
authors_df = [UGent, SIMID, UNamur, VUB]
report_v = 'v5.0'
scenarios = ['1','2','3','4']
statistics = ['mean', 'median', 'LL', 'UL']

for idx, author in enumerate(authors):
    for scenario in scenarios:
        for statistic in statistics:
            if author == 'VUB':
                if scenario == '1':
                    df_RESTORE[author,report_v,"S"+scenario+'_2610',"load", statistic] = VUB1['S'+scenario+'_load_'+statistic]
                    df_RESTORE[author,report_v,"S"+scenario+'_0111',"load", statistic] = VUB2['S'+scenario+'_load_'+statistic]
            elif author == 'UGent':
                df_RESTORE[author,report_v,"S"+scenario,"incidences", statistic] = authors_df[idx]['S'+scenario+'_incidences_'+statistic]
                if statistic != 'median':
                    df_RESTORE[author,report_v,"S"+scenario,"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]                                                                               
            else:
                df_RESTORE[author,report_v,"S"+scenario,"incidences", statistic] = authors_df[idx]['S'+scenario+'_incidences_'+statistic]
                df_RESTORE[author,report_v,"S"+scenario,"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]

#################
## RESTORE 6.0 ##
#################
folder = 'RESTORE_6.0/'

# ---------
# Load data
# ---------

# Load SIMID model
SIMID = pd.read_csv(raw_dir+folder+'UHasselt_predictions_v6.csv', parse_dates=['Date'])
SIMID.index = SIMID['Date']
SIMID.pop('Date')
# Load UGent model
UGent = pd.read_csv(raw_dir+folder+'UGent_restore_v6.csv', parse_dates=['Date'])
#UGent = UGent.rename(columns={UGent.columns[0]:'Date'})
UGent.Date = pd.to_datetime(UGent.Date)
UGent.index = UGent['Date']
UGent.pop('Date')
# Load UNamur model
UNamur = pd.read_csv(raw_dir+folder+'Unamur_Model_allscenarios_new.csv', parse_dates=['Date'], skipinitialspace=True)
UNamur.Date = pd.to_datetime(UNamur.Date, format='%d/%m/%y')
UNamur.index = UNamur['Date']
UNamur.pop('Date')
# Load VUB model
VUB = pd.read_excel(raw_dir+folder+'VUB_Hosp1412.xlsx', skiprows=0)
VUB.Date = pd.to_datetime(VUB.Date)
VUB.columns = ['Date','Observations','Fit', 'S1a_load_median','S1a_load_mean', 'S1a_load_LL', 'S1a_load_UL'] 
VUB.index = VUB['Date']
VUB.pop('Date')
# Load ULB model
# Scenario 1a
ULB_1a = pd.read_csv(raw_dir+folder+'S1a_ULB_model_1213.csv')
ULB_1a = ULB_1a.rename(columns={ULB_1a.columns[0]:'Date'})
ULB_1a['Date'] = pd.date_range('2020-03-01', periods=len(ULB_1a.Date), freq='1D')
ULB_1a.index = ULB_1a['Date']
ULB_1a.pop('Date')
# XMas scenario 1
ULBXmas1 = pd.read_csv(raw_dir+folder+'SXmas1_ULB_model_1213.csv')
ULBXmas1 = ULBXmas1.rename(columns={ULBXmas1.columns[0]:'Date'})
ULBXmas1['Date'] = pd.date_range('2020-03-01', periods=len(ULBXmas1.Date), freq='1D')
ULBXmas1.index = ULBXmas1['Date']
ULBXmas1.pop('Date')
# XMas scenario 2
ULBXmas2 = pd.read_csv(raw_dir+folder+'SXmas2_ULB_model_1213.csv')
ULBXmas2 = ULBXmas2.rename(columns={ULBXmas2.columns[0]:'Date'})
ULBXmas2['Date'] = pd.date_range('2020-03-01', periods=len(ULBXmas2.Date), freq='1D')
ULBXmas2.index = ULBXmas2['Date']
ULBXmas2.pop('Date')
# XMas scenario 3
ULBXmas3 = pd.read_csv(raw_dir+folder+'SXmas3_ULB_model_1213.csv')
ULBXmas3 = ULBXmas3.rename(columns={ULBXmas3.columns[0]:'Date'})
ULBXmas3['Date'] = pd.date_range('2020-03-01', periods=len(ULBXmas3.Date), freq='1D')
ULBXmas3.index = ULBXmas3['Date']
ULBXmas3.pop('Date')

# -----------
# Assign data
# -----------

authors = ['UGent','SIMID','UNamur','VUB']
authors_df = [UGent, SIMID, UNamur, VUB]
report_v = 'v6.0'
scenarios = ['1a','2a','2b1','2c1','3']
scenarios_mapped = ['1','2a','2b','2c','3']
statistics = ['mean', 'median', 'LL', 'UL']

for idx, author in enumerate(authors):
    for jdx, scenario in enumerate(scenarios):
        for statistic in statistics:
            if author == 'VUB':
                if scenario == '1a':
                    df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]
            
            else:
                df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"incidences", statistic] = authors_df[idx]['S'+scenario+'_incidences_'+statistic]
                df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]

author = 'ULB'
authors_df = [ULB_1a, ULBXmas1, ULBXmas2, ULBXmas3]
scenarios = ['S1a', 'SXmas1', 'SXmas2', 'SXmas3']
scenarios_mapped = ['S1','S_Xmas1', 'S_Xmas2', 'S_Xmas3']
for idx, scenario in enumerate(scenarios):
    for statistic in statistics:
        if statistic != 'median':
            df_RESTORE[author,report_v, scenarios_mapped[idx], "incidences", statistic] = authors_df[idx][scenario + '_incidences_' +statistic].rolling(window=7, center=True).mean()

#################
## RESTORE 6.1 ##
#################
folder = 'RESTORE_6.1/'

# ---------
# Load data
# ---------

# Load SIMID model
SIMID = pd.read_csv(raw_dir+folder+'UHasselt_predictions_v6_1.csv', parse_dates=['Date'])
SIMID.index = SIMID['Date']
SIMID.pop('Date')
# Load UGent model
UGent = pd.read_csv(raw_dir+folder+'UGent_restore_v6.1.csv', parse_dates=['Date'])
#UGent = UGent.rename(columns={UGent.columns[0]:'Date'})
UGent.Date = pd.to_datetime(UGent.Date)
UGent.index = UGent['Date']
UGent.pop('Date')
# Load UNamur model
UNamur = pd.read_csv(raw_dir+folder+'Unamur_Model_allscenarios_6.1.csv', parse_dates=['Date'], skipinitialspace=True)
UNamur.Date = pd.to_datetime(UNamur.Date, format='%d/%m/%y')
UNamur.index = UNamur['Date']
UNamur.pop('Date')
# Load VUB model
VUB = pd.read_excel(raw_dir+folder+'VUB_HOSPpredictions150121_Rapport6-1.xlsx', skiprows=0)
VUB.Date = pd.to_datetime(VUB.Date)
#VUB.pop("Unnamed: 7")
#VUB.pop("Unnamed: 8")
#VUB.pop("Unnamed: 9")
#VUB.pop("Unnamed: 10")
#VUB.pop("Unnamed: 11")
VUB.columns = ['Date','Observations','Fit', 'S1_load_median','S1_load_mean', 'S1_load_LL', 'S1_load_UL'] 
VUB.index = VUB['Date']
VUB.pop('Date')

# -----------
# Assign data
# -----------

authors = ['UGent','SIMID','UNamur','VUB']
authors_df = [UGent, SIMID, UNamur, VUB]
report_v = 'v6.1'
scenarios = ['1','2a','2b','2c']
scenarios_mapped = ['1','2a','2b','2c']
statistics = ['mean', 'median', 'LL', 'UL']

for idx, author in enumerate(authors):
    for jdx, scenario in enumerate(scenarios):
        for statistic in statistics:
            if author == 'VUB':
                if scenario == '1':
                    df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]
            else:
                df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"incidences", statistic] = authors_df[idx]['S'+scenario+'_incidences_'+statistic]
                df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]

#################
## RESTORE 7.0 ##
#################
folder = 'RESTORE_7.0/'

# ---------
# Load data
# ---------

# Load SIMID model
SIMID = pd.read_csv(raw_dir+folder+'UHasselt_predictions_v7_np_full.csv', parse_dates=['Date'])
SIMID.index = SIMID['Date']
SIMID.pop('Date')
# Load UGent model
UGent = pd.read_csv(raw_dir+folder+'UGent_restore_v7.csv', parse_dates=['Date'])
UGent.Date = pd.to_datetime(UGent.Date)
UGent.index = UGent['Date']
UGent.pop('Date')
# Load UNamur model
UNamur = pd.read_csv(raw_dir+folder+'Unamur_Model_allscenarios_70.csv', parse_dates=['Date'], skipinitialspace=True)
UNamur.index = UNamur['Date']
UNamur.pop('Date')
# Load VUB model
VUB = pd.read_excel(raw_dir+folder+'VUB_prediction_March9th.xlsx',skiprows=1)
VUB.Date = pd.to_datetime(VUB.Date, format='%d/%m/%y')
VUB.index = VUB['Date']
VUB.pop('Date')
VUB.pop("Unnamed: 6")
VUB.pop("Unnamed: 7")
VUB.pop("Unnamed: 8")
VUB.pop("Unnamed: 9")
VUB.pop("Unnamed: 10")
VUB.pop("Unnamed: 11")
VUB.pop("Unnamed: 12")
VUB.pop("Unnamed: 13")
VUB = VUB.rename(columns={VUB.columns[2]:'S1b_load_mean'})
VUB = VUB.rename(columns={VUB.columns[3]:'S1b_load_LL'})
VUB = VUB.rename(columns={VUB.columns[4]:'S1b_load_UL'})

# -----------
# Assign data
# -----------

authors = ['UGent','SIMID','UNamur']
authors_df = [UGent, SIMID, UNamur]
report_v = 'v7.0'
scenarios = ['1a','1b','1c','2a','2b','2c','3a','3b','3c','4a','4b','4c']
scenarios_mapped = scenarios
statistics = ['mean', 'median', 'LL', 'UL']

for idx, author in enumerate(authors):
    for jdx, scenario in enumerate(scenarios):
        for statistic in statistics:
            df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"incidences", statistic] = authors_df[idx]['S'+scenario+'_incidences_'+statistic]
            df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]

for statistic in statistics:
    if not statistic == 'median':
        df_RESTORE['VUB','v7.0','S1b','load',statistic] = VUB['S1b_load_'+statistic]

#################
## RESTORE 8.0 ##
#################
folder = 'RESTORE_8.0/'

# ---------
# Load data
# ---------

# Load SIMID model
SIMID = pd.read_csv(raw_dir+folder+'restore8_SIMID.csv', parse_dates=['Date'])
SIMID.Date = pd.date_range(start='2020-03-01',periods=550)
SIMID.index = SIMID['Date']
SIMID.pop('Date')
# Load UGent model
UGentmodel_or = pd.read_csv(raw_dir+folder+'RESTORE8_UGent_simulations.csv', header=[0,1,2,3,4])
# Format UGent model
UGentmodel_S1 = UGentmodel_or.loc[:,('5','2021-05-01','40000','old --> young')].copy()
UGentmodel_S2 = UGentmodel_or.loc[:,('5','2021-05-01','60000','old --> young')].copy()
UGentmodel_S3 = UGentmodel_or.loc[:,('5','2021-06-01','40000','old --> young')].copy()
UGentmodel_S4 = UGentmodel_or.loc[:,('5','2021-06-01','60000','old --> young')].copy()
UGentmodel_S1.columns = ['S1_'+x for x in UGentmodel_S1.columns]
UGentmodel_S2.columns = ['S2_'+x for x in UGentmodel_S2.columns]
UGentmodel_S3.columns = ['S3_'+x for x in UGentmodel_S3.columns]
UGentmodel_S4.columns = ['S4_'+x for x in UGentmodel_S4.columns]
dates = UGentmodel_or.droplevel([1,2,3,4], axis=1)['social scenario'].rename('Date')
UGent = pd.concat([dates, UGentmodel_S1, UGentmodel_S2, UGentmodel_S3, UGentmodel_S4], axis=1)
UGent.Date = pd.to_datetime(UGent.Date)
UGent.index = UGent['Date']
UGent.pop('Date')
# Load UNamur model
UNamur = pd.read_csv(raw_dir+folder+'Unamur_Model_80.csv', parse_dates=['Date'], skipinitialspace=True)
UNamur.index = UNamur['Date']
UNamur.pop('Date')
# Load VUB model
VUB = pd.read_excel(raw_dir+folder+'restore8_VUB.xlsx',skiprows=1)
VUB.Date = pd.to_datetime(VUB.Date, format='%d/%m/%y')
VUB.index = VUB['Date']
VUB.pop('Date')
VUB = VUB.rename(columns={VUB.columns[6]:'current_contact_load_mean'})
VUB = VUB.rename(columns={VUB.columns[7]:'current_contact_load_LL'})
VUB = VUB.rename(columns={VUB.columns[8]:'current_contact_load_UL'})

# -----------
# Assign data
# -----------

authors = ['UGent','SIMID','UNamur']
authors_df = [UGent, SIMID, UNamur]
report_v = 'v8.0'
scenarios = ['1','2','3','4']
scenarios_mapped = scenarios
statistics = ['mean', 'median', 'LL', 'UL']

for idx, author in enumerate(authors):
    for jdx, scenario in enumerate(scenarios):
        for statistic in statistics:
            if author == 'SIMID':
                if statistic != 'median':
                    df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"incidences", statistic] = authors_df[idx]['S'+scenario+'_incidences_'+statistic]
                    df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic]
            else:
                df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"incidences", statistic] = authors_df[idx]['S'+scenario+'_incidences_'+statistic]
                df_RESTORE[author,report_v,"S"+scenarios_mapped[jdx],"load", statistic] = authors_df[idx]['S'+scenario+'_load_'+statistic] 

for statistic in statistics:
    if not statistic == 'median':
        df_RESTORE['VUB','v7.0','current_contact_behaviour','load',statistic] = VUB['current_contact_load_'+statistic]

##################
## Save results ##
##################

df_RESTORE.to_csv(iterim_dir+'all_RESTORE_simulations.csv')