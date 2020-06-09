import os
import datetime
import pandas as pd
import numpy as np

def get_sciensano_COVID19_data(update=True):
    """Download Sciensano hospitalisation cases data

    This function returns the publically available Sciensano data
    on COVID-19 related hospitalisations.A copy of the downloaded dataset
    is automatically saved in the /data/raw folder.

    Parameters
    ----------
    update : boolean (default True)
        True if you want to update the data,
        False if you want to read only previously saved data

    Returns
    -----------
    df : pandas.DataFrame
        DataFrame with the sciensano data on daily basis. The following columns
        are returned:

        - pd.DatetimeIndex : datetimes for which a data point is available
        - H_tot : total number of hospitalised patients (according to Sciensano)
        - ICU_tot : total number of hospitalised patients in ICU
        - H_in : total number of patients going to hospital on given date
        - H_out : total number of patients discharged from hospital on given data
        - H_tot_cumsum : calculated total number of patients in hospital,
              calculated as by taking the cumulative sum of H_net = H_in - H_out

    Notes
    ----------
    The data is extracted from Sciensano database: https://epistat.wiv-isp.be/covid/
    Variables in raw dataset are documented here: https://epistat.sciensano.be/COVID19BE_codebook.pdf

    Example use
    -----------
    >>> # download data from sciensano website and store new version
    >>> sciensano_data = get_sciensano_COVID19_data(update=True)
    >>> # load data from raw data directory (no new download)
    >>> sciensano_data = get_sciensano_COVID19_data()
    """
    # Data source
    url = 'https://epistat.sciensano.be/Data/COVID19BE.xlsx'
    abs_dir = os.path.dirname(__file__)

    if update==True:
        # Extract hospitalisation data from source
        df = pd.read_excel(url, sheet_name="HOSP")
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_HOSP.csv')
        df.to_csv(rel_dir, index=False)
    else:
        df = pd.read_csv(os.path.join(abs_dir,
        '../../../data/raw/sciensano/COVID19BE_HOSP.csv'), parse_dates=['DATE'])

    # Resample data from all regions and sum all values for each date
    df = df.resample('D', on='DATE').sum()

    variable_mapping = {"TOTAL_IN": "H_tot",
                        "TOTAL_IN_ICU": "ICU_tot",
                        "NEW_IN": "H_in",
                        "NEW_OUT": "H_out"}
    df = df.rename(columns=variable_mapping)
    df = df[list(variable_mapping.values())]
    df["H_tot_cumsum"] = (df["H_in"] - df["H_out"]).cumsum().values

    return df
