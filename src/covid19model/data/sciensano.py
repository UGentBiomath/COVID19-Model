import os
import datetime
import pandas as pd
import numpy as np

def get_mortality_data():
    """Load and return the detailed mortality data for Belgium

    Returns
    -------
    mortality_df : pd.dataframe
        Dataframe containing both the incidence and cumulative sums for the number of deaths in total, hospitals, nursing homes and other locations. Data available per age group and per day.
        The dataframe has two index levels: age group ('all', '10-20', ..., '80+'), and date. The dataframe has two column levels: 'place' ('total', 'hospital' ,'nursing', 'others') and 'quantity' ('incidence' or 'cumsum')

    Example use
    -----------
    mortality_df = get_mortality_data()
    # slice out 80+ age group
    slice = mortality_df.xs(key='80+', level="age_class", drop_level=True)
    # extract cumulative total in age group 80+
    data = slice['total','cumsum']
    """

    return pd.read_csv(os.path.join(os.path.dirname(__file__),'../../../data/interim/sciensano/sciensano_detailed_mortality.csv'), index_col=[0,1], header=[0,1])

def get_serological_data():
    """Load and format the available serological data for Belgium

    Returns
    -------

    df_sero_herzog: pandas.DataFrame
        DataFrame with the number of individuals (mean, 5% quantile, 95% quantile) that have anti-SARS-CoV-2 antibodies in their blood as estimated by Sereina Herzog. Seroprevelance provided as absolute figure (number of individuals) as well as relative figure (fraction of population).

    df_sero_sciensano: pandas.DataFrame
        DataFrame with the number of individuals (mean, 5% quantile, 95% quantile) that have anti-SARS-CoV-2 antibodies in their blood as estimated by Sciensano. Seroprevelance provided as absolute figure (number of individuals) as well as relative figure (fraction of population).

    Example use
    -----------
    df_sero_herzog, df_sero_sciensano = get_serological_data()
    The resulting dataframes have the same format and use pandas multicolumns.
    >> To extract the fraction of individuals with anti-SARS-CoV-2 antibodies in the blood
        df_sero_herzog['rel','mean']
    >> To extract the number of individuals with anti-SARS-CoV-2 antibodies in the blood
        df_sero_herzog['abs','mean']
    """
    
    # Load demographic data
    abs_dir = os.path.dirname(__file__)
    initN_data = "../../../data/interim/demographic/initN_arr.csv"
    initN_df = pd.read_csv(os.path.join(abs_dir, initN_data), index_col='NIS')
    initN = initN_df.values[:,:-1].sum(axis=0)
    # Load and format serodata of Herzog
    data = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/sero/sero_national_overall_herzog.csv'), parse_dates=True)
    data.index = data['collection_midpoint']
    data.index.names = ['date']
    data.index = pd.to_datetime(data.index)
    data = data.drop(columns=['collection_midpoint','age_cat'])
    columns = [[],[]]
    tuples = list(zip(*columns))
    columns = pd.MultiIndex.from_tuples(tuples, names=["abs/rel", "data"])
    df = pd.DataFrame(index=data.index, columns=columns)
    df['abs','mean'] = data['mean']*sum(initN) 
    df['abs','LL'] = data['LL']*sum(initN)
    df['abs','UL'] = data['UL']*sum(initN)
    df['rel','mean'] = data['mean']
    df['rel','LL'] = data['LL']
    df['rel','UL'] = data['UL']
    df_sero_herzog = df

    # Load and format serodata of Sciensano
    data= pd.read_csv(os.path.join(abs_dir,'../../../data/raw/sero/Belgium COVID-19 Studies - Sciensano_Blood Donors_Tijdreeks.csv'), parse_dates=True)
    data.index = data['Date']
    data.index = pd.to_datetime(data.index)
    data = data.drop(columns=['Date'])
    data.index.names = ['date']
    columns = [[],[]]
    tuples = list(zip(*columns))
    columns = pd.MultiIndex.from_tuples(tuples, names=["abs/rel", "data"])
    df = pd.DataFrame(index=data.index, columns=columns)
    df['abs','mean'] = data['mean']*sum(initN) 
    df['abs','LL'] = data['LL']*sum(initN)
    df['abs','UL'] = data['UL']*sum(initN)
    df['rel','mean'] = data['mean']
    df['rel','LL'] = data['LL']
    df['rel','UL'] = data['UL']
    df_sero_sciensano = df

    return df_sero_herzog, df_sero_sciensano

def get_sciensano_COVID19_data(update=True):
    """Download and convert public Sciensano data

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
        df_vacc = pd.read_excel(url, sheet_name="VACC")
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_VACC.csv')
        df_vacc.to_csv(rel_dir, index=False)

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

    # ----------------------
    # Vaccination (national)
    # ----------------------

    # First dose
    df["V1_tot"] = df_vacc[df_vacc['DOSE'] == 'A'].resample('D', on='DATE')['COUNT'].sum()
    df["V1_00_11"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '00-11'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_12_15"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '12-15'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_16_17"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '16-17'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_18_24"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '18-24'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_25_34"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '25-34'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_35_44"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '35-44'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_45_54"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '45-54'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_55_64"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '55-64'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_65_74"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '65-74'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_75_84"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '75-84'))].resample('D', on='DATE')['COUNT'].sum()
    df["V1_85+"] = df_vacc[((df_vacc['DOSE'] == 'A')&(df_vacc['AGEGROUP'] == '85+'))].resample('D', on='DATE')['COUNT'].sum()
    # Second dose
    df["V2_tot"] = df_vacc[df_vacc['DOSE'] == 'B'].resample('D', on='DATE')['COUNT'].sum()
    df["V2_00_11"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '00-11'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_12_15"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '12-15'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_16_17"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '16-17'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_18_24"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '18-24'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_25_34"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '25-34'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_35_44"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '35-44'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_45_54"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '45-54'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_55_64"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '55-64'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_65_74"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '65-74'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_75_84"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '75-84'))].resample('D', on='DATE')['COUNT'].sum()
    df["V2_85+"] = df_vacc[((df_vacc['DOSE'] == 'B')&(df_vacc['AGEGROUP'] == '85+'))].resample('D', on='DATE')['COUNT'].sum()
    # One-shot vaccines
    df["VJ&J_tot"] = df_vacc[df_vacc['DOSE'] == 'C'].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_00_11"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '00-11'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_12_15"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '12-15'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_16_17"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '16-17'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_18_24"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '18-24'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_25_34"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '25-34'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_35_44"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '35-44'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_45_54"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '45-54'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_55_64"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '55-64'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_65_74"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '65-74'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_75_84"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '75-84'))].resample('D', on='DATE')['COUNT'].sum()
    df["VJ&J_85+"] = df_vacc[((df_vacc['DOSE'] == 'C')&(df_vacc['AGEGROUP'] == '85+'))].resample('D', on='DATE')['COUNT'].sum()
    return df.fillna(0)

def get_public_spatial_vaccination_data(update=False, agg='arr'):
    """Download and convert public spatial vaccination data of Sciensano

    This function returns the spatial, publically available Sciensano vaccination data (first dose/one dose only)
    A copy of the downloaded raw dataset is automatically saved in the /data/raw folder.
    The formatted data on the municipality level (NUTS5) is automatically saved in the /data/interim folder.
    If update=True, the dataset is downloaded and formatted into the following format: per week, per municipality NIS code and per age group, the number of first doses is given. The dataset is then automatically saved.
    If update=False, the formatted dataset is loaded and an aggregation is performed to the desired NUTS level.

    Parameters
    ----------
    update : boolean (default True)
        True if you want to update the data,
        False if you want to read-in the previously saved and formatted data

    Returns
    -----------
    df : pandas.DataFrame
        DataFrame with the vaccination data on a weekly basis. The following columns
        are returned:

        Indices:
        - start_week : the startdate of the week the doses were administered
        - NIS : the NIS codes of the desired aggregation level (municipalities, arrondissements or provinces)
        - age : the age group in which the vaccines were administered
        Columns:
        - CUMULATIVE : cumulative number of administered doses
        - INCIDENCE : weekly administered number of doses

    Notes
    -----
    The data is extracted from Sciensano database: https://epistat.wiv-isp.be/covid/
    Variables in raw dataset are documented here: https://epistat.sciensano.be/COVID19BE_codebook.pdf


    Example use
    -----------
    >>> # download data from sciensano website and store new version
    >>> sciensano_data = get_sciensano_COVID19_data(update=True)
    >>> df.sum(level=[0]).plot() # Visualize nationally aggregated number of doses
    >>> df.loc[:,21000,:].sum(level=0).plot() # Visualize the number of administered doses in Brussels (NIS=21000)

    """
    # Data source
    url = 'https://epistat.sciensano.be/Data/COVID19BE.xlsx'
    abs_dir = os.path.dirname(__file__)

    if update==True:
        # Extract case data from source
        #df = pd.read_excel(url, sheet_name="VACC_MUNI_CUM")
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_VACC_MUNI_raw.csv')
        #df.to_csv(rel_dir, index=False)
        df = pd.read_csv(rel_dir)
        ########################################################
        ## Convert YEAR_WEEK to startdate and enddate of week ##
        ########################################################

        start=[]
        end=[]
        for year_week in df["YEAR_WEEK"].values:
            year = '20'+year_week[0:2]
            week = year_week[3:]
            if week == '53':
                week = str(int(year_week[3:])-1)
                
            startdate = "{}-{}-1".format(year, week)
            enddate = "{}-{}-6".format(year, week)
            
            dt = datetime.datetime.strptime(startdate, "%Y-%W-%w")
            start.append(dt.strftime("%Y-%m-%d"))

            dt = datetime.datetime.strptime(enddate, "%Y-%W-%w")
            end.append(dt.strftime("%Y-%m-%d"))
        df['start_week'] = start
        df['end_week'] = end

        ######################
        ## Format dataframe ##
        ######################

        df = df.drop(df[df.NIS5 == ''].index)
        df = df.dropna()
        df['CUMUL'][df['CUMUL'] == '<10'] = '0'
        df['CUMUL'] = df['CUMUL'].astype(int)
        df['NIS5'] = ((df['NIS5'].astype(float)).astype(int)).astype(str)
        df = df.rename(columns={'NIS5':'NUTS5', 'AGEGROUP':'age'})
        df.set_index('start_week')
        df.pop('YEAR_WEEK')
        df.pop('end_week')
        # Drop the second doses --> for use with one-jab vaccination model
        df = df.drop(df[df.DOSE == 'B'].index)

        ######################################################
        ## Add up the first doses and the one-shot vaccines ##
        ######################################################

        df_A = df.drop(df[df.DOSE == 'C'].index)
        df_A.pop('DOSE')
        df_C = df.drop(df[df.DOSE == 'A'].index)
        df_C.pop('DOSE')
        
        multi_df_A = df_A.set_index(['start_week','NUTS5','age'])
        multi_df_C = df_C.set_index(['start_week','NUTS5','age'])
        multi_df = multi_df_A + multi_df_C

        #################################
        ## Fill up the missing entries ##
        #################################
        # Make a dataframe containing all the desired levels
        iterables = [df['start_week'].unique(), multi_df.index.get_level_values(1).unique(), multi_df.index.get_level_values(2).unique()]
        index = pd.MultiIndex.from_product(iterables, names=["start_week", "NUTS5", "age"])
        columns = ['CUMUL']
        complete_df = pd.DataFrame(index=index, columns=columns)
        # Merge the dataframe containing no missing indices with the actual dataframe
        mergedDf = complete_df.merge(multi_df, left_index=True, right_index=True, how='outer')
        # Remove obsolete columns
        mergedDf.pop('CUMUL_x')
        mergedDf = mergedDf.fillna(0)
        mergedDf = mergedDf.rename(columns={'CUMUL_y': 'CUMULATIVE'})

        ##############################################
        ## Convert cumulative numbers to incidences ##
        ##############################################
        # Pre-allocate column
        mergedDf['INCIDENCE'] = 0
        # Loop over indices (computationally expensive)
        for idx,start_week in enumerate(mergedDf.index.get_level_values(0).unique()[:-1]):
            next_week = mergedDf.index.get_level_values(0).unique()[idx+1]
            for NIS in mergedDf.index.get_level_values(1).unique():
                incidence = mergedDf['CUMULATIVE'].loc[next_week,NIS,:].values - mergedDf['CUMULATIVE'].loc[start_week,NIS,:].values
                mergedDf['INCIDENCE'].loc[start_week,NIS,:] = incidence
        # Rename mergedDf back to df for convenience
        df = mergedDf

        ##############################
        ## Save formatted dataframe ##
        ##############################
        rel_dir = os.path.join(abs_dir, '../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format.csv')
        df.to_csv(rel_dir, index=True)

    else:
        ##############################
        ## Load formatted dataframe ##
        ##############################
        rel_dir = os.path.join(abs_dir, '../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format.csv')
        df = pd.read_csv(rel_dir, index_col=[0,1,2], parse_dates=['start_week'])

    ##########################################
    ## Perform aggregation to desired level ##
    ##########################################

    if agg == 'mun':
        return df
    elif ((agg == 'arr') | (agg == 'prov')):
        # Extract arrondissement's NIS codes
        NIS_arr = read_coordinates_nis(spatial='arr')
        # Make a new dataframe
        iterables = [df.index.get_level_values(0).unique(), NIS_arr, ['0-17','18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+']]
        index = pd.MultiIndex.from_product(iterables, names=["start_week", "NIS", "age"])
        columns = ['CUMULATIVE','INCIDENCE']
        arr_df = pd.DataFrame(index=index, columns=columns)
        arr_df['CUMULATIVE'] = 0
        arr_df['INCIDENCE'] = 0
        # Loop over indices (computationally expensive)
        for idx,start_week in enumerate(df.index.get_level_values(0).unique()):
            som = np.zeros([df.index.get_level_values(2).unique().shape[0],2])    
            for NIS in df.index.get_level_values(1).unique():
                arr_NIS = int(str(NIS)[0:2] + '000')
                arr_df.loc[start_week, arr_NIS, :] = arr_df.loc[start_week, arr_NIS, :].values + df.loc[start_week, NIS, :].values

        if agg == 'prov':
            # Extract provincial NIS codes
            NIS_prov = read_coordinates_nis(spatial='prov')
            # Make a new dataframe
            iterables = [df.index.get_level_values(0).unique(), NIS_prov, ['0-17','18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+']]
            index = pd.MultiIndex.from_product(iterables, names=["start_week", "NIS", "age"])
            columns = ['CUMULATIVE','INCIDENCE']
            prov_df = pd.DataFrame(index=index, columns=columns)
            prov_df['CUMULATIVE'] = 0
            prov_df['INCIDENCE'] = 0
            # Loop over indices (computationally expensive)
            for idx,start_week in enumerate(arr_df.index.get_level_values(0).unique()):
                som = np.zeros([arr_df.index.get_level_values(2).unique().shape[0],2])    
                for NIS in arr_df.index.get_level_values(1).unique():
                    if NIS == 21000:
                        prov_df.loc[start_week, NIS, :] = arr_df.loc[start_week, NIS, :].values
                    elif ((NIS == 23000) | (NIS == 24000)):
                        prov_df.loc[start_week, 20001, :] = prov_df.loc[start_week, 20001, :].values + arr_df.loc[start_week, NIS, :].values
                    elif NIS == 25000:
                        prov_df.loc[start_week, 20002, :] = arr_df.loc[start_week, NIS, :].values
                    else:
                        prov_NIS = int(str(NIS)[0:1] + '0000')
                        prov_df.loc[start_week, prov_NIS, :] = prov_df.loc[start_week, prov_NIS, :].values + arr_df.loc[start_week, NIS, :].values
            return prov_df
        else:
            return arr_df


def get_sciensano_COVID19_data_spatial(agg='arr', values='hospitalised_IN', moving_avg=True):
    """
    This function returns the spatially explicit private Sciensano data
    on COVID-19 related confirmed cases, hospitalisations, or hospital deaths.
    A copy of the downloaded dataset is automatically saved in the /data/raw folder.
    
    NOTE that this raw data is NOT automatically uploaded to the Git repository, considering it is nonpublic data.
    Instead, download the processed data from the S-drive (data/interim/sciensano) and copy it to data/interim/nonpublic_timeseries.
    The nonpublic_timeseries directory is added to the .gitignore, so it is not uploaded to the repository.
    
    TO DO: function currently does *not* support loading data at level of postal codes (from all_nonpublic_timeseries_Postcode.csv).
    
    TO DO: currently gives a pandas copy warning if values='all' and moving_avg=True.
    
    Parameters
    ----------
    agg : str
        Aggregation level. Either 'prov', 'arr' or 'mun', for provinces, arrondissements or municipalities, respectively. Note that not all data is available on every aggregation level.
    values : str
        Choose which time series is to be loaded. Options are 'confirmed_cases', 'tested_cases', 'confirmed_per_tested_5days_window', 'hospitalised_IN' (default), 'recovered', 'deceased_hosp', 'ICU', 'confirmed_cases_per_100k', 'hospitalised_IN_per_100k', 'recovered_per_100k', 'deceased_hosp_per_100k', 'ICU_per_100k'. Choose 'all' to return full data.
    moving_avg : boolean
        If True (default), the 7-day moving average of the data is taken to smooth out the weekend effect.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the sciensano data of the chosen values on daily basis (with or without 7-day averaging).
    """

    # Exceptions
    if agg not in ['prov', 'arr', 'mun']:
        raise Exception(f"Aggregation level {agg} not recognised. Choose between agg = 'prov', 'arr' or 'mun'.")
        
    accepted_values=['confirmed_cases', 'tested_cases', 'confirmed_per_tested_5days_window', 'hospitalised_IN', 'recovered', 'deceased_hosp', 'ICU', 'confirmed_cases_per_100k', 'hospitalised_IN_per_100k', 'recovered_per_100k', 'deceased_hosp_per_100k', 'ICU_per_100k']
    if values not in (accepted_values + ['all']):
        raise Exception(f"Value type {values} not recognised. Choose between 'confirmed_cases', 'tested_cases', 'confirmed_per_tested_5days_window', 'hospitalised_IN', 'recovered', 'deceased_hosp', 'ICU', 'confirmed_cases_per_100k', 'hospitalised_IN_per_100k', 'recovered_per_100k', 'deceased_hosp_per_100k', or 'ICU_per_100k'. Choose 'all' to return full data.")

    # Data location
    abs_dir = os.path.dirname(__file__)
    nonpublic_dir_rel = f"../../../data/interim/nonpublic_timeseries/all_nonpublic_timeseries_{agg}.csv"
    nonpublic_dir_abs = os.path.join(abs_dir, nonpublic_dir_rel)
    
    # Load data with or without moving average
    if values=='all':
        nonpublic_df = pd.read_csv(nonpublic_dir_abs, parse_dates=['DATE']).pivot_table(index='DATE', columns=f'NIS_{agg}').fillna(0)
        if moving_avg:
            from covid19model.visualization.utils import moving_avg
            for value in accepted_values:
                for NIS in nonpublic_df[value].columns:
                    nonpublic_df[value][[NIS]] = moving_avg(nonpublic_df[value][[NIS]])
            nonpublic_df.dropna(inplace=True) # remove first and last 3 days (NA due to averaging)
    else:
        nonpublic_df = pd.read_csv(nonpublic_dir_abs, parse_dates=['DATE']).pivot_table(index='DATE', columns=f'NIS_{agg}', values=values).fillna(0)
        if moving_avg:
            from covid19model.visualization.utils import moving_avg
            for NIS in nonpublic_df.columns:
                nonpublic_df[[NIS]] = moving_avg(nonpublic_df[[NIS]])
            nonpublic_df.dropna(inplace=True) # remove first and last 3 days (NA due to averaging)

    return nonpublic_df

    