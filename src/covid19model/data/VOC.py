import os
import datetime
import pandas as pd

def get_501Y_data():
    # Load and format B1.1.7 VOC data 
    rel_dir = os.path.join(os.path.dirname(__file__), '../../../data/raw/VOCs/sequencing_501YV1_501YV2_501YV3.csv')
    df_VOC_501Y = pd.read_csv(rel_dir, parse_dates=True, encoding='cp1252').set_index('collection_date', drop=True).drop(columns=['sampling_week','year', 'week'])
    # Converting the index as date
    df_VOC_501Y.index = pd.to_datetime(df_VOC_501Y.index)
    # Extrapolate missing dates to obtain a continuous index
    df_VOC_501Y = df_VOC_501Y.resample('D').interpolate('linear')
    df_VOC_501Y['baselinesurv_f_501Y.V1_501Y.V2_501Y.V3'] = (df_VOC_501Y['baselinesurv_n_501Y.V1']+df_VOC_501Y['baselinesurv_n_501Y.V2']+df_VOC_501Y['baselinesurv_n_501Y.V3'])/df_VOC_501Y['baselinesurv_total_sequenced']
    # Assign data to class
    return df_VOC_501Y
