import os
import datetime
import pandas as pd
import numpy as np

def get_sciensano_COVID19_data(update=True):
    """Download Sciensano hospitalisation cases data

    This function returns the publically available Sciensano data
    on COVID-19 related cases, hospitalizations, deaths and vaccinations.
    A copy of the downloaded dataset is automatically saved in the /data/raw folder.

    Parameters
    ----------
    update : boolean (default True)
        True if you want to update the data,
        False if you want to read previously saved data

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
        - D_tot : total number of deaths
        - D_xx_yy : total number of deaths in the age group xx to yy years old
        - V1_tot : total number of first dose vaccinations
        - V2_tot : total number of second dose vaccinations
        - V1_xx_yy: : total number of first dose vaccinations in the age group xx to yy years old
        - V2_xx_yy : total number of second dose vaccinations in the age group xx to yy years old

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
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_CASES.csv')
        df_cases.to_csv(rel_dir, index=False)

        # Extract vaccination data from source
        df_cases = pd.read_excel(url, sheet_name="VACC")
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_VACC.csv')
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
        '../../../data/raw/sciensano/COVID19BE_CASES.csv'), parse_dates=['DATE'])

        df_vacc = pd.read_csv(os.path.join(abs_dir,
        '../../../data/raw/sciensano/COVID19BE_VACC.csv'), parse_dates=['DATE'])

        df = pd.read_csv(os.path.join(abs_dir,
        '../../../data/raw/sciensano/COVID19BE_HOSP.csv'), parse_dates=['DATE'])

        df_mort = pd.read_csv(os.path.join(abs_dir,
        '../../../data/raw/sciensano/COVID19BE_MORT.csv'), parse_dates=['DATE'])

    # --------
    # Hospital
    # --------

    # Use hospitalization dataframe as the template
    df = df.resample('D', on='DATE').sum()

    variable_mapping = {"TOTAL_IN": "H_tot",
                        "TOTAL_IN_ICU": "ICU_tot",
                        "NEW_IN": "H_in",
                        "NEW_OUT": "H_out"}
    df = df.rename(columns=variable_mapping)
    df = df[list(variable_mapping.values())]

    # ------
    # Deaths
    # ------

    df["D_tot"] = df_mort.resample('D', on='DATE')['DEATHS'].sum()
    df["D_25_44"] = df_mort.loc[(df_mort['AGEGROUP'] == '25-44')].resample('D', on='DATE')['DEATHS'].sum()
    df["D_45_64"] = df_mort.loc[(df_mort['AGEGROUP'] == '45-64')].resample('D', on='DATE')['DEATHS'].sum()
    df["D_65_74"] = df_mort.loc[(df_mort['AGEGROUP'] == '65-74')].resample('D', on='DATE')['DEATHS'].sum()
    df["D_75_84"] = df_mort.loc[(df_mort['AGEGROUP'] == '75-84')].resample('D', on='DATE')['DEATHS'].sum()
    df["D_85+"] = df_mort.loc[(df_mort['AGEGROUP'] == '85+')].resample('D', on='DATE')['DEATHS'].sum()

    # -----
    # Cases
    # -----

    df["C_tot"] = df_cases.resample('D', on='DATE')['CASES'].sum()
    df["C_0_9"] = df_cases.loc[(df_cases['AGEGROUP'] == '0-9')].resample('D', on='DATE')['CASES'].sum()
    df["C_10_19"] = df_cases.loc[(df_cases['AGEGROUP'] == '10-19')].resample('D', on='DATE')['CASES'].sum()
    df["C_20_29"] = df_cases.loc[(df_cases['AGEGROUP'] == '20-29')].resample('D', on='DATE')['CASES'].sum()
    df["C_30_39"] = df_cases.loc[(df_cases['AGEGROUP'] == '30-39')].resample('D', on='DATE')['CASES'].sum()
    df["C_40_49"] = df_cases.loc[(df_cases['AGEGROUP'] == '40-49')].resample('D', on='DATE')['CASES'].sum()
    df["C_50_59"] = df_cases.loc[(df_cases['AGEGROUP'] == '50-59')].resample('D', on='DATE')['CASES'].sum()
    df["C_60_69"] = df_cases.loc[(df_cases['AGEGROUP'] == '60-69')].resample('D', on='DATE')['CASES'].sum()
    df["C_70_79"] = df_cases.loc[(df_cases['AGEGROUP'] == '70-79')].resample('D', on='DATE')['CASES'].sum()
    df["C_80_89"] = df_cases.loc[(df_cases['AGEGROUP'] == '80-89')].resample('D', on='DATE')['CASES'].sum()
    df["C_90+"] = df_cases.loc[(df_cases['AGEGROUP'] == '90+')].resample('D', on='DATE')['CASES'].sum()
    
    # -----------
    # Vaccination
    # -----------

    # First dose
    df["V1_tot"] = df_vacc[df_vacc['DOSE'] == 'A'].resample('D', on='DATE')['COUNT'].sum()
    df["V1_18_34"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '18-34'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_35_44"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '35-44'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_45_54"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '45-54'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_55_64"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '55-64'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_65_74"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '65-74'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_75_84"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '75-84'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_85+"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '85+'))].resample('D', on='DATE')['COUNT'].sum()
    # Second dose
    df["V2_tot"] = df_vacc[df_vacc['DOSE'] == 'B'].resample('D', on='DATE')['COUNT'].sum()
    df["V2_18_34"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '18-34'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_35_44"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '35-44'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_45_54"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '45-54'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_55_64"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '55-64'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_65_74"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '65-74'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_75_84"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '75-84'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_85+"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '85+'))].resample('D', on='DATE')['COUNT'].sum()

    return df.fillna(0)
