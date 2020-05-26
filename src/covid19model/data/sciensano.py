import pandas as pd
import numpy as np


def get_sciensano_data():
    """Download Sciensano hospitalisation cases data 

    This function returns the publically available Sciensano data on COVID-19 related hospitalisations.

    Returns
    -----------
    index : pd.DatetimeIndex
        datetimes for which a data point is available
    initial :  str 'YYYY-MM-DD'
        initial date of records as string
    ICU : np.array
        total number of hospitalised patients in ICU
    hospital : np.array
        total number of hospitalised patients

    Notes
    ----------
    The data is extracted from Sciensano database: https://epistat.wiv-isp.be/covid/
    Data is reported as showed in: https://epistat.sciensano.be/COVID19BE_codebook.pdf

    Example use
    -----------
    index, initial, ICU, hospital = get_sciensano_data()
    """

    # Data source
    url = 'https://epistat.sciensano.be/Data/COVID19BE.xlsx'

    # Extract hospitalisation data from source
    df = pd.read_excel(url, sheet_name="HOSP")

    # Date of initial records
    initial = df.astype(str)['DATE'][0]

    # Resample data from all regions and sum all values for each date
    data = df.loc[:,['DATE','TOTAL_IN','TOTAL_IN_ICU']]
    data = data.resample('D', on='DATE').sum()
    hospital = np.array([data.loc[:,'TOTAL_IN'].tolist()]) # export as array
    ICU = np.array([data.loc[:,'TOTAL_IN_ICU'].tolist()]) # export as array

    # List of time datapoints
    index = pd.date_range(initial, freq='D', periods=ICU.size)

    return index, initial, ICU, hospital