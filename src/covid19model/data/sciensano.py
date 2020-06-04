import os
import datetime
import pandas as pd
import numpy as np

def get_sciensano_COVID19_data():
    """Download Sciensano hospitalisation cases data 

    This function returns the publically available Sciensano data on COVID-19 related hospitalisations.
    A copy of the downloaded dataset is automatically saved in the /data/raw folder.

    Returns
    -----------
    index : pd.DatetimeIndex
        datetimes for which a data point is available
    initial :  str 'YYYY-MM-DD'
        initial date of records as string
    ICU_tot : np.array
        total number of hospitalised patients in ICU
    H_tot : np.array
        total number of hospitalised patients (according to Sciensano)
    H_in : np.array
        total number of patients going to hospital on given date
    H_out: np.array
        total number of patients discharged from hospital on given data
    H_tot_cumsum: np.array
        calculated total number of patients in hospital
        calculated as by taking the cumulative sum of H_net = H_in - H_out
    
    Notes
    ----------
    The data is extracted from Sciensano database: https://epistat.wiv-isp.be/covid/
    Variables in raw dataset are documented here: https://epistat.sciensano.be/COVID19BE_codebook.pdf

    Example use
    -----------
    index, initial, H_tot, ICU_tot, H_in, H_out = get_sciensano_data()
    """

    # Data source
    url = 'https://epistat.sciensano.be/Data/COVID19BE.xlsx'


    # Extract hospitalisation data from source
    df = pd.read_excel(url, sheet_name="HOSP")
    # save a copy in the raw folder
    abs_dir = os.path.dirname(__file__)
    rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_HOSP.csv')
    df.to_csv(rel_dir,index=False)
    # Date of initial records
    initial = df.astype(str)['DATE'][0]

    # Resample data from all regions and sum all values for each date
    data = df.loc[:,['DATE','TOTAL_IN','TOTAL_IN_ICU','NEW_IN','NEW_OUT']]
    data = data.resample('D', on='DATE').sum()
    H_tot = np.array([data.loc[:,'TOTAL_IN'].tolist()]).flatten() # export as 1D array
    ICU_tot = np.array([data.loc[:,'TOTAL_IN_ICU'].tolist()]).flatten() # export as 1D array
    H_in = np.array([data.loc[:,'NEW_IN'].tolist()]).flatten() # export as 1D array
    H_out = np.array([data.loc[:,'NEW_OUT'].tolist()]).flatten() # export as 1D array
    H_tot_cumsum=np.reshape(np.cumsum(H_in-H_out),(1,np.cumsum(H_in-H_out).size)).flatten() # export as 1D array

    # List of time datapoints
    index = pd.date_range(initial, freq='D', periods=ICU_tot.size)
    return index, initial, H_tot, H_tot_cumsum, ICU_tot, H_in, H_out