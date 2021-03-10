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
        - D_tot : total number of deaths
        - D_xx_yy: total number of deaths in the age group xx to yy years old

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
        # Extract case data from source
        df_cases = pd.read_excel(url, sheet_name="CASES_AGESEX")
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_CASES_AGESEX.csv')
        df_cases.to_csv(rel_dir, index=False)

        # Extract hospitalisation data from source
        df = pd.read_excel(url, sheet_name="HOSP")
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_HOSP.csv')
        df.to_csv(rel_dir, index=False)

        # Extract total reported deaths per day
        df_mort = pd.read_excel(url, sheet_name='MORT', parse_dates=['DATE'])
        # save a copy in the raw folder
        rel_dir_M = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_MORT.csv')
        df_mort.to_csv(rel_dir_M, index=False)

    else:
        df_cases = pd.read_csv(os.path.join(abs_dir,
        '../../../data/raw/sciensano/COVID19BE_CASES_AGESEX.csv'), parse_dates=['DATE'])

        df = pd.read_csv(os.path.join(abs_dir,
        '../../../data/raw/sciensano/COVID19BE_HOSP.csv'), parse_dates=['DATE'])

        df_mort = pd.read_csv(os.path.join(abs_dir,
        '../../../data/raw/sciensano/COVID19BE_MORT.csv'), parse_dates=['DATE'])

    # Resample data from all regions and sum all values for each date
    df = df.resample('D', on='DATE').sum()

    variable_mapping = {"TOTAL_IN": "H_tot",
                        "TOTAL_IN_ICU": "ICU_tot",
                        "NEW_IN": "H_in",
                        "NEW_OUT": "H_out"}
    df = df.rename(columns=variable_mapping)
    df = df[list(variable_mapping.values())]
    df["H_tot_cumsum"] = (df["H_in"] - df["H_out"]).cumsum().values

    df["D_tot"] = df_mort.resample('D', on='DATE')['DEATHS'].sum()

    # Extract total reported deaths per day and per age group
    df["D_25_44"] = df_mort.loc[(df_mort['AGEGROUP'] == '25-44')].resample('D', on='DATE')['DEATHS'].sum()
    df["D_45_64"] = df_mort.loc[(df_mort['AGEGROUP'] == '45-64')].resample('D', on='DATE')['DEATHS'].sum()
    df["D_65_74"] = df_mort.loc[(df_mort['AGEGROUP'] == '65-74')].resample('D', on='DATE')['DEATHS'].sum()
    df["D_75_84"] = df_mort.loc[(df_mort['AGEGROUP'] == '75-84')].resample('D', on='DATE')['DEATHS'].sum()
    df["D_85+"] = df_mort.loc[(df_mort['AGEGROUP'] == '85+')].resample('D', on='DATE')['DEATHS'].sum()

    # Extract total cases per day and cases per age group per day
    #df["C_tot"] = df_cases.resample('D', on='DATE')['CASES'].sum()
    #df["C_0_9"] = df_cases.loc[(df_cases['AGEGROUP'] == '0-9')].resample('D', on='DATE')['CASES'].sum()
    #df["C_10_19"] = df_cases.loc[(df_cases['AGEGROUP'] == '10-19')].resample('D', on='DATE')['CASES'].sum()
    #df["C_20_29"] = df_cases.loc[(df_cases['AGEGROUP'] == '20-29')].resample('D', on='DATE')['CASES'].sum()
    #df["C_30_39"] = df_cases.loc[(df_cases['AGEGROUP'] == '30-39')].resample('D', on='DATE')['CASES'].sum()
    #df["C_40_49"] = df_cases.loc[(df_cases['AGEGROUP'] == '40-49')].resample('D', on='DATE')['CASES'].sum()
    #df["C_50_59"] = df_cases.loc[(df_cases['AGEGROUP'] == '50-59')].resample('D', on='DATE')['CASES'].sum()
    #df["C_60_69"] = df_cases.loc[(df_cases['AGEGROUP'] == '60-69')].resample('D', on='DATE')['CASES'].sum()
    #df["C_70_79"] = df_cases.loc[(df_cases['AGEGROUP'] == '70-79')].resample('D', on='DATE')['CASES'].sum()
    #df["C_80_89"] = df_cases.loc[(df_cases['AGEGROUP'] == '80-89')].resample('D', on='DATE')['CASES'].sum()
    #df["C_90+"] = df_cases.loc[(df_cases['AGEGROUP'] == '90+')].resample('D', on='DATE')['CASES'].sum()
    
    return df.fillna(0)

def get_sciensano_COVID19_data_spatial(agg='arr', column='hospitalised_IN', moving_avg=True):
    """
    This function returns the spatially explicit private Sciensano data
    on COVID-19 related confirmed cases, hospitalisations, or hospital deaths.
    A copy of the downloaded dataset is automatically saved in the /data/raw folder.
    
    NOTE that this raw data is NOT automatically uploaded to the Git repository, considering it is nonpublic data.
    Instead, download the processed data from the S-drive (data/interim/sciensano) and copy it to data/interim/nonpublic_timeseries.
    The nonpublic_timeseries directory is added to the .gitignore, so it is not uploaded to the repository.
    
    Parameters
    ----------
    agg : str
        Aggregation level. Either 'prov', 'arr' or 'mun', for provinces, arrondissements or municipalities, respectively. Note that not all data is available on every aggregation level.
    column : str
        Choose which time series is to be loaded. Options are 'confirmed_cases', 'tested_cases', 'confirmed_per_tested_5days_window', 'hospitalised_IN' (default), 'recovered', 'deceased_hosp', 'ICU', 'confirmed_cases_per_100k', 'hospitalised_IN_per_100k', 'recovered_per_100k', 'deceased_hosp_per_100k', 'ICU_per_100k'. Choose 'all' to return full data.
    moving_avg : boolean
        If True (default), the 7-day moving average of the data is taken to smooth out the weekend effect.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the sciensano data of the chosen column on daily basis (with or without 7-day averaging).
    """

    # Exceptions
    if agg not in ['prov', 'arr', 'mun']:
        raise Exception(f"Aggregation level {agg} not recognised. Choose between agg = 'prov', 'arr' or 'mun'.")
    if column not in ['confirmed_cases', 'tested_cases', 'confirmed_per_tested_5days_window', 'hospitalised_IN', 'recovered', 'deceased_hosp', 'ICU', 'confirmed_cases_per_100k', 'hospitalised_IN_per_100k', 'recovered_per_100k', 'deceased_hosp_per_100k', 'ICU_per_100k', 'all']:
        raise Exception(f"Column type {column} not recognised. Choose between 'confirmed_cases', 'tested_cases', 'confirmed_per_tested_5days_window', 'hospitalised_IN', 'recovered', 'deceased_hosp', 'ICU', 'confirmed_cases_per_100k', 'hospitalised_IN_per_100k', 'recovered_per_100k', 'deceased_hosp_per_100k', or 'ICU_per_100k'. Choose 'all' to return full data.")
        

        
    
    
    return

    
    
    
    
    