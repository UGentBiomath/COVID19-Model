import os
import datetime
import pandas as pd
import numpy as np
from covid19model.data.model_parameters import construct_initN

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
    
    # Load national demographic data
    abs_dir = os.path.dirname(__file__)
    # Provided age_classes don't matter, only sum is used
    initN = construct_initN(age_classes=pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)]))
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
    -------

    df_hosp : pd.DataFrame
        DataFrame with the sciensano hospital data. Contains the number of:
            new hospitalisations: H_in
            new hospital discharges: H_out
            total patients in hospitals: H_tot
            total patients in ICU : ICU_tot
        per region and per age group.

    df_mort : pd.DataFrame
        DataFrame with the sciensano hospital data. Contains the number of deaths per region and per age group.
    
    df_cases: pd.DataFrame
        DataFrame with the sciensano case data. Contains the number of cases per province and per age group.

    df_vacc : pd.DataFrame
        DataFrame with the sciensano public vaccination data.
        Contains the number of vaccines per region, per age group and per dose (first dose: A, second dose: B, one-shot: C, booster shot: E).

    Notes
    -----
    The data is extracted from Sciensano database: https://epistat.wiv-isp.be/covid/
    Variables in raw dataset are documented here: https://epistat.sciensano.be/COVID19BE_codebook.pdf
    All returned pd.DataFrame exploit multiindex capabilities. Indexing is done using: df.loc[(value_index_1, ... value_index_n), column_names].
    Summing over axes is done using: df.groupby(by=[index_name_1, ..., index_name_n]).sum()

    Example use
    -----------
    >>> # download data from sciensano website and store new version
    >>> df_hosp, df_mort, df_cases, df_vacc = get_sciensano_COVID19_data(update=True)
    >>> # load data from raw data directory (no new download)
    >>> df_hosp, df_mort, df_cases, df_vacc = get_sciensano_COVID19_data()
    >>> # extract the total number of daily hospitalisations, over all regions and ages use:
    >>> df_hosp.groupby(by='age').sum()
    >>> # extract the total number of first vaccination doses, summed over all regions and ages:
    >>> df_vacc.loc[(slice(None), slice(None), slice(None), 'A')].groupby(by=['date']).sum()
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
        df_hosp = pd.read_excel(url, sheet_name="HOSP")
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_HOSP.csv')
        df_hosp.to_csv(rel_dir, index=False)

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

        df_hosp = pd.read_csv(os.path.join(abs_dir,
        '../../../data/raw/sciensano/COVID19BE_HOSP.csv'), parse_dates=['DATE'])

        df_mort = pd.read_csv(os.path.join(abs_dir,
        '../../../data/raw/sciensano/COVID19BE_MORT.csv'), parse_dates=['DATE'])

    # --------
    # Hospital
    # --------
    
    # Format provinces to NIS codes
    data_provinces = df_hosp['PROVINCE'].unique()
    data_provinces = [x for x in data_provinces if pd.notnull(x)]
    corresponding_NIS = [10000, 21000, 50000, 70000, 60000, 80000, 90000, 40000, 20001, 20002, 30000]
    for idx, province in enumerate(data_provinces):
        df_hosp.loc[df_hosp['PROVINCE']==province, 'PROVINCE'] = corresponding_NIS[idx]
    # Rename variables of interest
    variable_mapping = {"TOTAL_IN": "H_tot",
                        "TOTAL_IN_ICU": "ICU_tot",
                        "NEW_IN": "H_in",
                        "NEW_OUT": "H_out"}
    df_hosp = df_hosp.rename(columns=variable_mapping)
    # Group data by date and province
    df_hosp = df_hosp.groupby(by=['DATE', 'PROVINCE']).sum()
    df_hosp.index.names = ['date','province']
    # Retain only columns of interest
    df_hosp = df_hosp[list(variable_mapping.values())]

    # ------
    # Deaths
    # ------

    # Format provinces to NIS codes
    data_regions = df_mort['REGION'].unique()
    data_regions = [x for x in data_regions if pd.notnull(x)]
    corresponding_NIS = [4000, 2000, 3000]
    for idx, region in enumerate(data_regions):
        df_mort.loc[df_mort['REGION']==region, 'REGION'] = corresponding_NIS[idx]

    # Define desired multiindexed pd.Series format
    interval_index = pd.IntervalIndex.from_tuples([(25,45),(45,65),(65,75),(75,85),(85,120)], closed='left')
    iterables = [df_mort['DATE'].unique(), corresponding_NIS, interval_index]
    index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age"])
    df = pd.Series(index=index)

    for idx, age_group in enumerate(['25-44', '45-64','65-74', '75-84', '85+']):
        for jdx, NIS in enumerate(corresponding_NIS):
            # Resample data: A problem occurs: only dates on which date is available are returned
            series = df_mort.loc[((df_mort['AGEGROUP'] == age_group)&(df_mort['REGION'] == NIS))].resample('D',on='DATE')['DEATHS'].sum()
            # Solution: define a dummy df with all desired dates, perform a join operation and extract the right column
            dummy = pd.Series(index = df_mort['DATE'].unique())
            C = dummy.to_frame().join(series.to_frame()).fillna(0)['DEATHS']
            # Assign data
            df.loc[(slice(None), NIS, interval_index[idx])] = C.values
    df_mort = df

    # -----
    # Cases
    # -----

    # Format provinces to NIS codes
    data_provinces = df_cases['PROVINCE'].unique()
    data_provinces = [x for x in data_provinces if pd.notnull(x)]
    corresponding_NIS = [10000, 21000, 60000, 70000, 40000, 20001, 20002, 30000, 50000, 90000, 80000]
    for idx, province in enumerate(data_provinces):
        df_cases.loc[df_cases['PROVINCE']==province, 'PROVINCE'] = corresponding_NIS[idx]

    # Define desired multiindexed pd.Series format
    interval_index = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,120)], closed='left')
    iterables = [df_cases['DATE'].unique()[:-1], corresponding_NIS, interval_index]
    index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age"])
    df = pd.Series(index=index)

    # Loop over age groups and NIS codes in dataframe
    for idx, age_group in enumerate(['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']):
        for jdx, NIS in enumerate(corresponding_NIS):
            series = df_cases.loc[((df_cases['AGEGROUP'] == age_group) & (df_cases['PROVINCE'] == NIS))].resample('D',on='DATE')['CASES'].sum()
            # Solution: define a dummy df with all desired dates, perform a join operation and extract the right column
            dummy = pd.Series(index = df_cases['DATE'].unique()[:-1])
            C = dummy.to_frame().join(series.to_frame()).fillna(0)['CASES']
            # Assign data
            df.loc[(slice(None), corresponding_NIS[jdx], interval_index[idx])] = C.values
    df_cases = df

    # ----------------------
    # Vaccination (national)
    # ----------------------

    # Format provinces to NIS codes
    df_vacc.loc[df_vacc['REGION']=='Ostbelgien', 'REGION'] = 'Wallonia'
    data_regions = df_vacc['REGION'].unique()
    data_regions = [x for x in data_regions if pd.notnull(x)]
    corresponding_NIS = [4000, 2000, 3000]
    for idx, region in enumerate(data_regions):
        df_vacc.loc[df_vacc['REGION']==region, 'REGION'] = corresponding_NIS[idx]

    # Define desired multiindexed pd.Series format
    interval_index = pd.IntervalIndex.from_tuples([(0,12),(12,16),(16,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')
    iterables = [df_vacc['DATE'].unique(), df_vacc['REGION'].unique(), interval_index, ['A', 'B', 'C']] # Leave dose 'E' out
    index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age", "dose"])
    df = pd.Series(index=index)

    for idx, age_group in enumerate(['00-11', '12-15', '16-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84','85+']):
        for jdx, NIS in enumerate(corresponding_NIS):
            for kdx, dose in enumerate(['A', 'B', 'C']):
                series = df_vacc.loc[((df_vacc['AGEGROUP'] == age_group) & (df_vacc['REGION'] == NIS) & (df_vacc['DOSE'] == dose))].resample('D',on='DATE')['COUNT'].sum()
                # Solution: define a dummy df with all desired dates, perform a join operation and extract the right column
                dummy = pd.Series(index = df_vacc['DATE'].unique())
                C = dummy.to_frame().join(series.to_frame()).fillna(0)['COUNT']
                # Assign data
                df.loc[(slice(None), corresponding_NIS[jdx], interval_index[idx], dose)] = C.values
    df_vacc = df
    
    return df_hosp, df_mort, df_cases, df_vacc

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
    
    # Load necessary functions
    from ..models.utils import read_coordinates_nis

    if update==True:
        # Extract case data from source
        df = pd.read_excel(url, sheet_name="VACC_MUNI_CUM")
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_VACC_MUNI_raw.csv')
        df.to_csv(rel_dir, index=False)
        #df = pd.read_csv(rel_dir)
        
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
        
        ############################
        # Save *municipality* data #
        ############################
        rel_dir = os.path.join(abs_dir, '../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_mun.csv')
        iterables = [df.index.get_level_values(0).unique(), df.index.get_level_values(1).unique(), pd.IntervalIndex.from_tuples([(0,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')]
        index = pd.MultiIndex.from_product(iterables, names=["date", "NUTS5", "age"])
        desired_formatted_df = pd.DataFrame(index=index, columns=df.columns)
        for col_name in df.columns:
            desired_formatted_df[col_name] = df[col_name].values
        mun_df =  desired_formatted_df
        mun_df.to_csv(rel_dir, index=True)
        
        # Save *arrondissement* data
        # Extract arrondissement's NIS codes
        NIS_arr = read_coordinates_nis(spatial='arr')
        # Make a new dataframe
        iterables = [df.index.get_level_values(0).unique(), NIS_arr, pd.IntervalIndex.from_tuples([(0,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')]
        index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age"])
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
                
        rel_dir = os.path.join(abs_dir, '../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_arr.csv')
        arr_df.to_csv(rel_dir, index=True)
        
        ########################
        # Save *province* data #
        ########################
        
        # Extract provincial NIS codes
        NIS_prov = read_coordinates_nis(spatial='prov')
        # Make a new dataframe
        iterables = [df.index.get_level_values(0).unique(), NIS_prov, pd.IntervalIndex.from_tuples([(0,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')]
        index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age"])
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
        rel_dir = os.path.join(abs_dir, '../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_prov.csv')
        prov_df.to_csv(rel_dir, index=True)
        
        #############################
        # Return relevant output df #
        #############################
        
        if agg=='mun':
            df = mun_df
        elif agg=='arr':
            df = arr_df
        elif agg=='prov':
            df = prov_df

    else:
        ##############################
        ## Load formatted dataframe ##
        ##############################
        
        rel_dir = os.path.join(abs_dir, f'../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_{agg}.csv')
        df = pd.read_csv(rel_dir, index_col=[0,1,2], parse_dates=['date'])
        # pd.read_csv cannot read an IntervalIndex so we need to set this manually
        iterables = [df.index.get_level_values(0).unique(),
                     df.index.get_level_values(1).unique(),
                     pd.IntervalIndex.from_tuples([(0,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')]
        index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age"])
        columns = df.columns
        desired_df = pd.DataFrame(index=index, columns=columns)
        for col_name in df.columns:
            desired_df[col_name] = df[col_name].values
        df = desired_df

    return df


def get_sciensano_COVID19_data_spatial(agg='arr', values='hospitalised_IN', public=False, moving_avg=True):
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
    public : Boolean
        If agg=='prov', public data is also available. This data is more complete in the most recent two weeks.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the sciensano data of the chosen values on daily basis (with or without 7-day averaging).
    """

    # Exceptions
    if agg not in ['prov', 'arr', 'mun']:
        raise Exception(f"Aggregation level {agg} not recognised. Choose between agg = 'prov', 'arr' or 'mun'.")
    if (agg!='prov') and public:
        raise Exception(f"Public data is only available on provincial and national level, not for agg={agg}.")
    
    if not public:
        accepted_values=['confirmed_cases', 'tested_cases', 'confirmed_per_tested_5days_window', 'hospitalised_IN', 'recovered', 'deceased_hosp', 'ICU', 'confirmed_cases_per_100k', 'hospitalised_IN_per_100k', 'recovered_per_100k', 'deceased_hosp_per_100k', 'ICU_per_100k']
        if values not in (accepted_values + ['all']):
            raise Exception(f"Value type {values} not recognised. Choose between 'confirmed_cases', 'tested_cases', 'confirmed_per_tested_5days_window', 'hospitalised_IN', 'recovered', 'deceased_hosp', 'ICU', 'confirmed_cases_per_100k', 'hospitalised_IN_per_100k', 'recovered_per_100k', 'deceased_hosp_per_100k', or 'ICU_per_100k'. Choose 'all' to return full data.")

        # Data location
        abs_dir = os.path.dirname(__file__)
        nonpublic_dir_rel = f"../../../data/interim/nonpublic_timeseries/all_nonpublic_timeseries_{agg}.csv"
        nonpublic_dir_abs = os.path.join(abs_dir, nonpublic_dir_rel)

        # Load data with or without moving average
        if values=='all':
            df = pd.read_csv(nonpublic_dir_abs, parse_dates=['DATE']).pivot_table(index='DATE', columns=f'NIS_{agg}').fillna(0)
            if moving_avg:
                from covid19model.visualization.utils import moving_avg
                for value in accepted_values:
                    for NIS in df[value].columns:
                        df[value][[NIS]] = moving_avg(df[value][[NIS]])
                df.dropna(inplace=True) # remove first and last 3 days (NA due to averaging)
        else:
            df = pd.read_csv(nonpublic_dir_abs, parse_dates=['DATE']).pivot_table(index='DATE', columns=f'NIS_{agg}', values=values).fillna(0)
            if moving_avg:
                from covid19model.visualization.utils import moving_avg
                for NIS in df.columns:
                    df[[NIS]] = moving_avg(df[[NIS]])
                df.dropna(inplace=True) # remove first and last 3 days (NA due to averaging)
    
    else:
        accepted_values=['hospitalised_IN', 'ICU']
        if values not in accepted_values:
            raise Exception(f"Value type {values} not recognised for public data. Choose between 'hospitalised_IN' or 'ICU'.")
        
        # use other conventions for this data
        if values=='hospitalised_IN':
            values = 'H_in'
        elif values=='ICU':
            values = 'ICU_tot'

        # get data from existing framework
        df = get_sciensano_COVID19_data(update=False)[0][[values]]
        
        # rename columns to fit the rest of the conventions
        df = df.reset_index().rename(columns={'date':'DATE', 'province':'NIS_prov'})
        
        # Put it in the same shape as the rest of the conventions
        df = df.pivot_table(index='DATE', columns='NIS_prov')[values]
        
        # moving average
        if moving_avg:
            from covid19model.visualization.utils import moving_avg
            for NIS in df.columns:
                df[[NIS]] = moving_avg(df[[NIS]])
            df=df.dropna() # remove first and last 3 days (NA due to averaging)
                
    return df

    