import os
import datetime
import pandas as pd

def get_abc_data():
    # Load and format B1.1.7 VOC data 
    rel_dir = os.path.join(os.path.dirname(__file__), '../../../data/raw/VOCs/sequencing_501YV1_501YV2_501YV3.csv')
    df_VOC_abc = pd.read_csv(rel_dir, parse_dates=True).set_index('collection_date', drop=True).drop(columns=['sampling_week','year', 'week'])
    # Converting the index as date
    df_VOC_abc.index = pd.to_datetime(df_VOC_abc.index)
    # Extrapolate missing dates to obtain a continuous index
    df_VOC_abc['baselinesurv_f_501Y.V1_501Y.V2_501Y.V3'] = (df_VOC_abc['baselinesurv_n_501Y.V1']+df_VOC_abc['baselinesurv_n_501Y.V2']+df_VOC_abc['baselinesurv_n_501Y.V3'])/df_VOC_abc['baselinesurv_total_sequenced']
    # Assign data to class
    return df_VOC_abc

def get_delta_data():
    #Copied from the molecular surveillance reported in the weekly Sciensano bulletins: https://covid-19.sciensano.be/sites/default/files/Covid19/COVID-19_Weekly_report_NL.pdf
    dates = pd.date_range(start = '2021-06-16', end = '2021-08-18', freq='7D')
    df_VOC_delta = pd.DataFrame(data=[0.284, 0.456, 0.675, 0.790, 0.902, 0.950, 0.960, 0.986, 0.992, 1], index=dates, columns=['delta'])
    return df_VOC_delta
