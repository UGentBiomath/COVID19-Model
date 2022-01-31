import os
import numpy as np
import pandas as pd

abs_dir = os.path.dirname(__file__)
rel_dir = os.path.join(os.path.dirname(__file__), '../../../data/raw/VOCs/')

def get_abc_data():
    # Load and format alpha, beta and gamma VOC data
    filename = 'sequencing_501YV1_501YV2_501YV3.csv'
    df_raw = pd.read_csv(os.path.join(rel_dir,filename), parse_dates=True, encoding='cp1252').set_index('collection_date', drop=True).drop(columns=['sampling_week','year', 'week'])
    # Format dataframe
    df_VOC_abc = pd.DataFrame(index=pd.to_datetime(df_raw.index), columns=['abc'],data=((df_raw['baselinesurv_n_501Y.V1']+df_raw['baselinesurv_n_501Y.V2']+df_raw['baselinesurv_n_501Y.V3'])/df_raw['baselinesurv_total_sequenced']).values)
    # Rename index
    df_VOC_abc.index.names = ['date']
    return df_VOC_abc

def get_delta_data():
    # Copied from the molecular surveillance reported in the weekly Sciensano bulletins: https://covid-19.sciensano.be/sites/default/files/Covid19/COVID-19_Weekly_report_NL.pdf
    dates = pd.date_range(start = '2021-06-16', end = '2021-08-18', freq='7D')
    df_VOC_delta = pd.DataFrame(data=[0.284, 0.456, 0.675, 0.790, 0.902, 0.950, 0.960, 0.986, 0.992, 1], index=dates, columns=['delta'])
    return df_VOC_delta

def get_omicron_data():
    # Load and format omicron VOC data
    filename = 'sgtf_belgium.csv'
    df_raw = pd.read_csv(os.path.join(rel_dir,filename), parse_dates=True).set_index('date', drop=True).drop(columns=['country','pos_tests', 'sgtf', 'comment','source'])
    # Format dataframe
    df_VOC_omicron = pd.DataFrame(index=pd.to_datetime(df_raw.index), columns=['omicron'],data=df_raw.values)
    return df_VOC_omicron