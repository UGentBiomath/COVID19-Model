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
    df = pd.DataFrame(index=data.index, columns=columns, dtype=float)
    df['abs','mean'] = data['Seroprevalence']*sum(initN) 
    df['abs','LL'] = data['Lower CI']*sum(initN)
    df['abs','UL'] = data['Upper CI']*sum(initN)
    df['rel','mean'] = data['Seroprevalence']
    df['rel','LL'] = data['Lower CI']
    df['rel','UL'] = data['Upper CI']
    df_sero_sciensano = df.dropna()

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
    data_provinces = ['Antwerpen', 'Brussels', 'Hainaut', 'Limburg', 'Liège', 'Luxembourg', 'Namur', 'OostVlaanderen', 'VlaamsBrabant', 'BrabantWallon', 'WestVlaanderen'] #df_hosp['PROVINCE'].unique()
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
    df_hosp.index.names = ['date','NIS']
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
    df = pd.Series(index=index, dtype=float)

    for idx, age_group in enumerate(['25-44', '45-64','65-74', '75-84', '85+']):
        for jdx, NIS in enumerate(corresponding_NIS):
            # Resample data: A problem occurs: only dates on which date is available are returned
            series = df_mort.loc[((df_mort['AGEGROUP'] == age_group)&(df_mort['REGION'] == NIS))].resample('D',on='DATE')['DEATHS'].sum()
            # Solution: define a dummy df with all desired dates, perform a join operation and extract the right column
            dummy = pd.Series(index = df_mort['DATE'].unique(), dtype=float)
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
    df = pd.Series(index=index, dtype=float)

    # Loop over age groups and NIS codes in dataframe
    for idx, age_group in enumerate(['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']):
        for jdx, NIS in enumerate(corresponding_NIS):
            series = df_cases.loc[((df_cases['AGEGROUP'] == age_group) & (df_cases['PROVINCE'] == NIS))].resample('D',on='DATE')['CASES'].sum()
            # Solution: define a dummy df with all desired dates, perform a join operation and extract the right column
            dummy = pd.Series(index = df_cases['DATE'].unique()[:-1], dtype=float)
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
    iterables = [df_vacc['DATE'].unique(), df_vacc['REGION'].unique(), interval_index, ['A', 'B', 'C','E']] # Leave dose 'E' out
    index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age", "dose"])
    df = pd.Series(index=index, dtype=float)

    for idx, age_group in enumerate(['00-11', '12-15', '16-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84','85+']):
        for jdx, NIS in enumerate(corresponding_NIS):
            for kdx, dose in enumerate(['A', 'B', 'C','E']):
                series = df_vacc.loc[((df_vacc['AGEGROUP'] == age_group) & (df_vacc['REGION'] == NIS) & (df_vacc['DOSE'] == dose))].resample('D',on='DATE')['COUNT'].sum()
                # Solution: define a dummy df with all desired dates, perform a join operation and extract the right column
                dummy = pd.Series(index = df_vacc['DATE'].unique(), dtype=float)
                C = dummy.to_frame().join(series.to_frame()).fillna(0)['COUNT']
                # Assign data
                df.loc[(slice(None), corresponding_NIS[jdx], interval_index[idx], dose)] = C.values
    df_vacc = df
    # Sum over regions to go to national level
    df_vacc = df_vacc.groupby(by=['date','age', 'dose']).sum()

    return df_hosp, df_mort, df_cases, df_vacc

def get_public_spatial_vaccination_data(update=False, agg=None):
    """Download and convert public spatial vaccination data of Sciensano

    This function returns the spatial, publically available Sciensano vaccination data
    A copy of the downloaded raw dataset is automatically saved in the /data/raw folder.
    The formatted data on the municipality level (NUTS5) is automatically saved in the /data/interim folder.
    If update=True, the dataset is downloaded and formatted into the following format: per week, per municipality NIS code, per age group and per dose, the incidence is given. The dataset is then automatically saved.
    If update=False, the formatted dataset is loaded and an aggregation is performed to the desired spatial aggregation level.

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
    # Actually, the first age group is 0-18 but no jabs were given < 12 yo before Jan. 2022
    age_groups_data = pd.IntervalIndex.from_tuples([(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')
    age_groups_model = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')

    if update==True:
        # Extract case data from source
        df1 = pd.read_excel(url, sheet_name="VACC_MUNI_CUM_1").dropna()
        df2 = pd.read_excel(url, sheet_name="VACC_MUNI_CUM_2").dropna()
        df = pd.concat([df1, df2])

        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/sciensano/COVID19BE_VACC_MUNI_raw.csv')
        df.to_csv(rel_dir, index=False)

        ########################################################
        ## Convert YEAR_WEEK to startdate and enddate of week ##
        ########################################################

        start=[]
        end=[]
        # pretty inefficient loop
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
        df.set_index(['start_week','NUTS5','age','DOSE'], inplace=True)

        #################################
        ## Fill up the missing entries ##
        #################################
        
        # Make a dataframe containing all the desired levels
        iterables = [df.index.get_level_values('start_week').unique(),df.index.get_level_values('NUTS5').unique(), df.index.get_level_values('age').unique(), df.index.get_level_values('DOSE').unique()]
        index = pd.MultiIndex.from_product(iterables, names=["start_week", "NUTS5", "age", "DOSE"])
        columns = ['CUMUL']
        complete_df = pd.DataFrame(index=index, columns=columns)
        # Merge the dataframe containing no missing indices with the actual dataframe
        mergedDf = complete_df.merge(df, left_index=True, right_index=True, how='outer')
        # Remove obsolete columns
        mergedDf.pop('CUMUL_x')
        mergedDf = mergedDf.fillna(0)
        mergedDf = mergedDf.rename(columns={'CUMUL_y': 'CUMULATIVE'})

        ##############################################
        ## Convert cumulative numbers to incidences ##
        ##############################################

        # Pre-allocate column
        mergedDf['INCIDENCE'] = 0
        # Loop over weeks and differentiate manually (is there a builtin function for this?)
        for idx,start_week in enumerate(mergedDf.index.get_level_values('start_week').unique()[:-1]):
            next_week = mergedDf.index.get_level_values('start_week').unique()[idx+1]
            incidence = mergedDf['CUMULATIVE'].loc[next_week, slice(None), slice(None), slice(None)] - mergedDf['CUMULATIVE'].loc[start_week, slice(None), slice(None), slice(None)]
            mergedDf['INCIDENCE'].loc[start_week, slice(None), slice(None), slice(None)] = incidence.values
        # Rename mergedDf back to df for convenience
        df = mergedDf

        ####################################################
        ## Convert to the 10 age groups used in the model ##
        ####################################################

        # First convert string-based age groups to corresponding pd.IntervalIndex
        rel_dir = os.path.join(abs_dir, '../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_mun.csv')
        iterables = [df.index.get_level_values('start_week').unique(), df.index.get_level_values('NUTS5').unique(), age_groups_data, df.index.get_level_values('DOSE').unique()]
        index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age", "dose"])
        desired_formatted_df = pd.DataFrame(index=index, columns=df.columns)
        for col_name in df.columns:
            desired_formatted_df[col_name] = df[col_name].values
        df = desired_formatted_df   

        # Extend pd.IntervalIndex of dataset with age group 0-12
        iterables = [df.index.get_level_values('date').unique(), df.index.get_level_values('NIS').unique(), age_groups_model, df.index.get_level_values('dose').unique()]
        index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age", "dose"])
        desired_formatted_df = pd.DataFrame(index=index, columns=df.columns)
        mergedDf = desired_formatted_df.merge(df, left_index=True, right_index=True, how='outer')
        mergedDf.pop('CUMULATIVE_x')
        mergedDf.pop('INCIDENCE_x')
        mergedDf = mergedDf.rename(columns={'CUMULATIVE_y': 'CUMULATIVE', 'INCIDENCE_y': 'INCIDENCE'})
        mergedDf = mergedDf.fillna(0)
        df = mergedDf
        df = df.index.set_levels(df.index.get_level_values('NIS').unique().astype(int), level='NIS')
        print(mergedDf)

        ##############################################################################
        ## Fix assignment of all (0,18[ vaccines into age group of (12,18[ in model ##
        ##############################################################################

        # For West-Flanders and East-Flanders province, cumulative vaccination degree surpasses 100% in age group [12,18( in the model by September 2021
        # This is the longest loop of the update function
        
        # Loop over NIS
        for NIS in df.index.get_level_values('NIS').unique():
            # Compute n_individuals (12,18) in NIS
            n_1218 = construct_initN(age_groups_model, 'mun').loc[NIS,pd.IntervalIndex.from_arrays([12,], [18,], closed='left')[0]]
            # Compute n_individuals (6,12) in NIS    
            n_612 = construct_initN(pd.IntervalIndex.from_arrays([0,6,12], [6, 12, 120], closed='left'), 'mun').loc[NIS,pd.IntervalIndex.from_arrays([6,], [12,], closed='left')[0]]
            # Compute fraction in (12,18)
            f_1218 = n_1218/(n_1218+n_612)   
            # Loop over dose
            for dose in df.index.get_level_values('dose').unique():
                # Extract dataseries (12,18)
                data = df.loc[(slice(None),NIS,pd.IntervalIndex.from_arrays([12,], [18,], closed='left')[0], dose)].values
                # Take both INC and CUM dataseries of (12,18), assign to (0,12) and (12,18) using fractions
                df.loc[(slice(None),NIS,pd.IntervalIndex.from_arrays([0,], [12,], closed='left')[0], dose),:] = (1-f_1218)*data
                df.loc[(slice(None),NIS,pd.IntervalIndex.from_arrays([12,], [18,], closed='left')[0], dose),:] = f_1218*data
                
        ##############################
        ## Save formatted dataframe ##
        ##############################
        
        ############################
        # Save *municipality* data #
        ############################

        mun_df = df
        mun_df.to_csv(rel_dir, index=True)

        ##############################
        # Save *arrondissement* data #
        ############################## 

        # Extract arrondissement's NIS codes
        NIS_arr = read_coordinates_nis(spatial='arr')
        # Make a new dataframe
        iterables = [df.index.get_level_values('date').unique(), NIS_arr, df.index.get_level_values('age').unique(), df.index.get_level_values('dose').unique()]
        index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age", "dose"])
        arr_df = pd.DataFrame(index=index, columns=df.columns)
        arr_df['CUMULATIVE'] = 0
        arr_df['INCIDENCE'] = 0
        # Perform aggregation
        for NIS in df.index.get_level_values('NIS').unique():
            arr_NIS = int(str(NIS)[0:2] + '000')
            arr_df.loc[slice(None), arr_NIS, slice(None), slice(None)] = arr_df.loc[slice(None), arr_NIS, slice(None), slice(None)].values + df.loc[slice(None), NIS, slice(None), slice(None)].values
        # Save result
        rel_dir = os.path.join(abs_dir, '../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_arr.csv')
        arr_df.to_csv(rel_dir, index=True)

        ########################
        # Save *province* data #
        ########################
        
        # Extract provincial NIS codes
        NIS_prov = read_coordinates_nis(spatial='prov')
        # Make a new dataframe
        iterables = [df.index.get_level_values('date').unique(), NIS_prov, df.index.get_level_values('age').unique(), df.index.get_level_values('dose').unique()]
        index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age", "dose"])
        prov_df = pd.DataFrame(index=index, columns=df.columns)
        prov_df['CUMULATIVE'] = 0
        prov_df['INCIDENCE'] = 0
        # Perform aggregation
        for NIS in arr_df.index.get_level_values('NIS').unique():
            if NIS == 21000:
                prov_df.loc[slice(None), NIS, slice(None), slice(None)] = arr_df.loc[slice(None), NIS, slice(None), slice(None)].values
            elif ((NIS == 23000) | (NIS == 24000)):
                prov_df.loc[slice(None), 20001, slice(None), slice(None)] = prov_df.loc[slice(None), 20001, slice(None), slice(None)].values + arr_df.loc[slice(None), NIS, slice(None), slice(None)].values
            elif NIS == 25000:
                prov_df.loc[slice(None), 20002, slice(None), slice(None)] = arr_df.loc[slice(None), NIS, slice(None), slice(None)].values
            else:
                prov_NIS = int(str(NIS)[0:1] + '0000')
                prov_df.loc[slice(None), prov_NIS, slice(None), slice(None)] = prov_df.loc[slice(None), prov_NIS, slice(None), slice(None)].values + arr_df.loc[slice(None), NIS, slice(None), slice(None)].values
        # Save result
        rel_dir = os.path.join(abs_dir, '../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_prov.csv')
        prov_df.to_csv(rel_dir, index=True)

        ##########################
        ## Save *national* data ##
        ##########################

        nat_df = mun_df.groupby(by=['date', 'age', 'dose']).sum()
        rel_dir = os.path.join(abs_dir, '../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_nat.csv')
        nat_df.to_csv(rel_dir, index=True)

        #############################
        # Return relevant output df #
        #############################
        
        if agg=='mun':
            df = mun_df
        elif agg=='arr':
            df = arr_df
        elif agg=='prov':
            df = prov_df
        elif agg==None:
            df = nat_df

    else:

        ##############################
        ## Load formatted dataframe ##
        ##############################
        if agg:
            rel_dir = os.path.join(abs_dir, f'../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_{agg}.csv')
            df = pd.read_csv(rel_dir, index_col=[0,1,2,3], parse_dates=['date'])
            # pd.read_csv cannot read an IntervalIndex so we need to set this manually
            iterables = [df.index.get_level_values('date').unique(),
                        df.index.get_level_values('NIS').unique(),
                        age_groups_model,
                        df.index.get_level_values('dose').unique(),]
            index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "age", "dose"])
            columns = df.columns
            desired_df = pd.DataFrame(index=index, columns=columns)
            for col_name in df.columns:
                desired_df[col_name] = df[col_name].values
            df = desired_df
        else:
            rel_dir = os.path.join(abs_dir, f'../../../data/interim/sciensano/COVID19BE_VACC_MUNI_format_nat.csv')
            df = pd.read_csv(rel_dir, index_col=[0,1,2], parse_dates=['date'])
            # pd.read_csv cannot read an IntervalIndex so we need to set this manually
            iterables = [df.index.get_level_values('date').unique(),
                        age_groups_model,
                        df.index.get_level_values('dose').unique(),]
            index = pd.MultiIndex.from_product(iterables, names=["date", "age", "dose"])
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
        df = df.reset_index().rename(columns={'date':'DATE'})
        
        # Put it in the same shape as the rest of the conventions
        df = df.pivot_table(index='DATE', columns='NIS')[values]
        
        # moving average
        if moving_avg:
            from covid19model.visualization.utils import moving_avg
            for NIS in df.columns:
                df[[NIS]] = moving_avg(df[[NIS]])
            df=df.dropna() # remove first and last 3 days (NA due to averaging)
                
    return df

def get_vaccination_rescaling_values(spatial=False, update=False, df_inc=None, initN=None, VOC_params=None, VOC_logistic_growth_parameters=None):
    """
    Loads vaccination dataframe, which is manually created in the Notebook preprocessing/MR-calculate-effective-rescalings.ipynb.
    Note: updating takes a long time (over an hour for non-spatial, up to 10 minutes for spatial)
    Note: Omicron variant currently *not* included
    Note: only province-level vaccinations have been included (not arr or mun)
    Note: PerformanceWarning: indexing past lexsort depth may impact performance.
    
    Input
    -----
    spatial : Boolean
        Returns provincially stratified vaccination data if spatial==True. False by default.
    update : Boolean
        If True, update with new rescaling values
    df_inc : pd.Series
        Multiindex Series with indices resp. date, (NIS), age, vaccination dose
    initN : pd.Series or pd.DataFrame
        Series (if not spatial) or DataFrame (if spatial) with population per age class and (if spatial) per NIS. First output of model_parameters.get_COVID19_SEIQRD_parameters()
    VOC_params : dict
        dict with keys e_s, e_i, e_h (reduced effectivity due to vaccination), and initN (matrix with population per age and region)
    VOC_logistic_growth_parameters: pd.DataFrame
        DataFrame with parameters that determine the logistic growth of the fraction of VOCs
        
    Output
    ------
    df : pd.DataFrame
        Dataframe with three (four) levels of indices: date, (NIS), age, dose. Dates start at 28 December 2020.
        
    Example use
    ------------
    
    initN, Nc_dict, params, CORE_samples_dict = model_parameters.get_COVID19_SEIQRD_parameters(spatial='prov')
    VOCs = ['WT', 'abc', 'delta']
    VOC_logistic_growth_parameters, VOC_params = model_parameters.get_COVID19_SEIQRD_VOC_parameters(initN, params['h'], VOCs=VOCs)
    public_spatial_vaccination_data = sciensano.get_public_spatial_vaccination_data(update=False,agg='prov')
    df_inc = make_vaccination_function(public_spatial_vaccination_data['INCIDENCE']).df
    rescaling_df = get_vaccination_rescaling_values(spatial=True, update=True, df_inc=df_inc, initN=initN, VOC_params=VOC_params, VOC_logistic_growth_parameters=VOC_logistic_growth_parameters)
    """
    
    # Data location
    abs_dir = os.path.dirname(__file__)
    
    if update:
        # Input validation
        if not set(['e_i', 'e_h', 'e_s']).issubset(set(VOC_params.keys())):
            raise Exception(f"The provided parameter dict should contain keys 'e_i', 'e_h', 'e_s'.")
            
        # rescaling values from VOC_params. Values [none, first, full, waned, booster]
        e_s = 1-VOC_params['e_s']
        e_i = 1-VOC_params['e_i']
        e_h = 1-VOC_params['e_h']

        ### Wild type values
        # hard-coded
        onset_days_WT = dict({'E_susc' : {'first' : 21, 'full' : 21, 'booster' : 21},
                       'E_inf' : {'first' : 14, 'full' : 14, 'booster' : 14},
                       'E_hosp' : {'first' : 21, 'full' : 21, 'booster' : 21}})
        # E_init is the value of the previous waned vaccine
        E_init_WT = dict({'E_susc' : {'first' : e_s[0,0], 'full' : e_s[0,1], 'booster' : e_s[0,3]},
                       'E_inf' : {'first' : e_i[0,0], 'full' : e_i[0,1], 'booster' : e_s[0,3]},
                       'E_hosp' : {'first' : e_h[0,0], 'full' : e_h[0,1], 'booster' : e_h[0,3]}})
        E_best_WT = dict({'E_susc' : {'first' : e_s[0,1], 'full' : e_s[0,2], 'booster' : e_s[0,4]},
                       'E_inf' : {'first' : e_i[0,1], 'full' : e_i[0,2], 'booster' : e_i[0,4]},
                       'E_hosp' : {'first' : e_h[0,1], 'full' : e_h[0,2], 'booster' : e_h[0,4]}})
        E_waned_WT = dict({'E_susc' : {'first' : e_s[0,1], 'full' : e_s[0,3], 'booster' : e_s[0,4]},
                       'E_inf' : {'first' : e_i[0,1], 'full' : e_i[0,3], 'booster' : e_i[0,4]},
                       'E_hosp' : {'first' : e_h[0,1], 'full' : e_h[0,3], 'booster' : e_h[0,4]}})
        
        ### alpha/beta/gamma values
        # hard-coded
        onset_days_abc = dict({'E_susc' : {'first' : 21, 'full' : 21, 'booster' : 21},
                       'E_inf' : {'first' : 14, 'full' : 14, 'booster' : 14},
                       'E_hosp' : {'first' : 21, 'full' : 21, 'booster' : 21}})
        # E_init is the value of the previous waned vaccine
        E_init_abc = dict({'E_susc' : {'first' : e_s[1,0], 'full' : e_s[1,1], 'booster' : e_s[1,3]},
                       'E_inf' : {'first' : e_i[1,0], 'full' : e_i[1,1], 'booster' : e_s[1,3]},
                       'E_hosp' : {'first' : e_h[1,0], 'full' : e_h[1,1], 'booster' : e_h[1,3]}})
        E_best_abc = dict({'E_susc' : {'first' : e_s[1,1], 'full' : e_s[1,2], 'booster' : e_s[1,4]},
                       'E_inf' : {'first' : e_i[1,1], 'full' : e_i[1,2], 'booster' : e_i[1,4]},
                       'E_hosp' : {'first' : e_h[1,1], 'full' : e_h[1,2], 'booster' : e_h[1,4]}})
        E_waned_abc = dict({'E_susc' : {'first' : e_s[1,1], 'full' : e_s[1,3], 'booster' : e_s[1,4]},
                       'E_inf' : {'first' : e_i[1,1], 'full' : e_i[1,3], 'booster' : e_i[1,4]},
                       'E_hosp' : {'first' : e_h[1,1], 'full' : e_h[1,3], 'booster' : e_h[1,4]}})

        ### delta values
        # hard-coded
        onset_days_delta = dict({'E_susc' : {'first' : 21, 'full' : 21, 'booster' : 21},
                       'E_inf' : {'first' : 14, 'full' : 14, 'booster' : 14},
                       'E_hosp' : {'first' : 21, 'full' : 21, 'booster' : 21}})
        # E_init is the value of the previous waned vaccine
        E_init_delta = dict({'E_susc' : {'first' : e_s[2,0], 'full' : e_s[2,1], 'booster' : e_s[2,3]},
                       'E_inf' : {'first' : e_i[2,0], 'full' : e_i[2,1], 'booster' : e_s[2,3]},
                       'E_hosp' : {'first' : e_h[2,0], 'full' : e_h[2,1], 'booster' : e_h[2,3]}})
        E_best_delta = dict({'E_susc' : {'first' : e_s[2,1], 'full' : e_s[2,2], 'booster' : e_s[2,4]},
                       'E_inf' : {'first' : e_i[2,1], 'full' : e_i[2,2], 'booster' : e_i[2,4]},
                       'E_hosp' : {'first' : e_h[2,1], 'full' : e_h[2,2], 'booster' : e_h[2,4]}})
        E_waned_delta = dict({'E_susc' : {'first' : e_s[2,1], 'full' : e_s[2,3], 'booster' : e_s[2,4]},
                       'E_inf' : {'first' : e_i[2,1], 'full' : e_i[2,3], 'booster' : e_i[2,4]},
                       'E_hosp' : {'first' : e_h[2,1], 'full' : e_h[2,3], 'booster' : e_h[2,4]}})

        # Combine them
        onset_days = dict({'WT' : onset_days_WT, 'abc' : onset_days_abc, 'delta' : onset_days_delta})
        E_init = dict({'WT' : E_init_WT, 'abc' : E_init_abc, 'delta' : E_init_delta})
        E_best = dict({'WT' : E_best_WT, 'abc' : E_best_abc, 'delta' : E_best_delta})
        E_waned = dict({'WT' : E_waned_WT, 'abc' : E_waned_abc, 'delta' : E_waned_delta})
        
        # Make VOC fraction function. VOC_function(date)[0] is the fraction of each variant at date
        from covid19model.models.time_dependant_parameter_fncs import make_VOC_function
        VOC_function = make_VOC_function(VOC_logistic_growth_parameters)
        
        # Make new DataFrame
        verbose=True
        df = make_rescaling_dataframe(df_inc, initN, VOC_function, onset_days, E_init, E_best, E_waned, verbose=verbose)
        df = df.drop(columns=['INCIDENCE', 'CUMULATIVE', 'E_susc_WT', 'E_inf_WT', 'E_hosp_WT', \
                      'E_susc_abc', 'E_inf_abc', 'E_hosp_abc', \
                      'E_susc_delta', 'E_inf_delta', 'E_hosp_delta'])
        df = df.fillna(1)
        
        # Save DataFrame as pickle (to retain intervalindex on 'age' dimension) and as csv (to have something human-readable)
        if spatial:
            dir_rel = f"../../../data/interim/sciensano/vacc_rescaling_values_provincial"
        else:
            dir_rel = f"../../../data/interim/sciensano/vacc_rescaling_values_national"
        df.to_csv(os.path.join(abs_dir, dir_rel+'.csv'))
        df.to_pickle(os.path.join(abs_dir, dir_rel+'.pkl'))
        
    # no update
    else:    
        if spatial==False:
            dir_rel = f"../../../data/interim/sciensano/vacc_rescaling_values_national.csv"
            dir_abs = os.path.join(abs_dir, dir_rel)

            # Load and format data
            df = pd.read_csv(dir_abs, parse_dates=["date"]).groupby(['date', 'age', 'dose']).first()

        if spatial==True:
            dir_rel = f"../../../data/interim/sciensano/vacc_rescaling_values_provincial.csv"
            dir_abs = os.path.join(abs_dir, dir_rel)

            # Load and format data
            df = pd.read_csv(dir_abs, parse_dates=["date"]).groupby(['date', 'NIS', 'age', 'dose']).first()

    return df
        
    # help function for get_vaccination_rescaling_values if update==True (1/2)
def waning_exp_delay(days, onset_days, E_init, E_best, E_waned):
    """
    Function that implements time-dependence of vaccine effect based on the time it takes for the vaccine to linearly reach full efficacy, and a subsequent asymptotic waning of the vaccine.

    Input
    -----
    days : float
        number of days after the novel vaccination
    onset_days : float
        number of days it takes for the vaccine to take full effect
    E_init : float
        vaccine-related rescaling value right before vaccination
    E_best : float
        rescaling value related to the best possible protection by the currently injected vaccine
    E_waned : float
        rescaling value related to the vaccine protection after a waning period.

    Output
    ------
    E_eff : float
        effective rescaling value associated with the newly administered vaccine

    """
    waning_days = 183 # hard-coded to half a year
    if days <= 0:
        return E_init
    elif days < onset_days:
        E_eff = (E_best - E_init)/onset_days*days + E_init
        return E_eff
    else:
        if E_best == E_waned:
            return E_best
        halftime_days = waning_days - onset_days
        A = 1-E_best
        beta = -np.log((1-E_waned)/A)/halftime_days
        E_eff = -A*np.exp(-beta*(days-onset_days))+1
    return E_eff

# help function for get_vaccination_rescaling_values if update==True (2/2)
def make_rescaling_dataframe(vacc_data, initN, VOC_func, onset_days, E_init, E_best, E_waned, verbose=False):
    """
    Ouputs DataFrame per week with fractions of vaccination stage per age class and per province NIS
    Note: the parameters onset_days, E_init, E_best, E_waned are currently quasi-hardcoded in get_vaccination_rescaling_values
    
    Input
    -----
    
    vacc_data : pd.DataFrame
        Output of make_vaccination_function(df_vacc).df (non-spatial) or make_vaccination_function(vacc_data['INCIDENCE']).df (spatial) with the default age intervals. The former has daily data, the latter has weekly data at the provincial level.
    initN : pd.DataFrame
        Output of model_parameters.get_COVID19_SEIQRD_parameters(spatial='prov')
    VOC_func : covid19model.models.time_dependant_parameter_fncs.make_VOC_function
        Function whose first element provides the national fraction of every VOC at a particular time
    onset_days : dict
        Dict of dicts with elements onset_days[VOC_type][rescaling_type][dose], where VOC_type in {'WT', 'abc', 'delta'}, rescaling_type in {'E_susc', 'E_inf', 'E_hosp'}, and dose in {'first', 'full', 'booster'}
    E_init : dict
        Dict of dicts with elements E_init[VOC_type][rescaling_type][dose]. Represents the rescaling value at the exact moment of vaccination (so before vaccination takes effect). Typically E_init[VOC_type][rescaling_type][dose] = E_waned[VOC_type][rescaling_type][dose-1]
    E_best : dict
        Dict of dicts with elements E_best[VOC_type][rescaling_type][dose]. Represents the lowest rescaling value due to vaccination (so the best-possible vaccination effect).
    E_waned : dict
        Dict of dicts with elements E_waned[VOC_type][rescaling_type][dose]. Represents the rescaling value six months after vaccination (so the waned vaccination effect)
        
    Output
    ------
    
    df_new : pd.DataFrame
        MultiIndex DataFrame with indices date (weekly), NIS, age, dose.
        Values are 'fraction', which always sum to unity (a subject is in only one of four vaccination stages)
    
    """
    spatial=False
    if 'NIS' in vacc_data.reset_index().columns:
        spatial=True
    
    # set initN column names to pd.Interval objects
    intervals = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')
    intervals_str = np.array(['[0, 12)', '[12, 18)', '[18, 25)', '[25, 35)', '[35, 45)', '[45, 55)', '[55, 65)', '[65, 75)', '[75, 85)', '[85, 120)'])
    intervals_dict = dict({intervals_str[i] : intervals[i] for i in range(len(intervals))})
    if spatial:
        initN = initN.rename(columns=intervals_dict)
        initN = initN.unstack().reset_index().rename(columns={0 : 'population'})
    else:
        initN = initN.rename(index=intervals_dict)
        initN = pd.DataFrame(initN).rename(columns={0 : 'population'})
    
    # Name nameless column to 'INCIDENCE' and add 'CUMULATIVE' column
    df_new = pd.DataFrame(vacc_data).rename(columns={0 : 'INCIDENCE'})
    if spatial:
        df_new['CUMULATIVE'] = df_new.groupby(level=[1,2,3]).cumsum()
    else:
        df_new['CUMULATIVE'] = df_new.groupby(level=[1, 2]).cumsum()
    
    # Make cumulative fractions by comparing with relevant initN
    df_new = pd.DataFrame(df_new).reset_index()
    if spatial:
        df_new = df_new.merge(initN, left_on=['NIS', 'age'], right_on=['NIS', 'age_class'])
    else:
        df_new = df_new.merge(initN, left_on=['age'], right_on=['age_class'])
    df_new['fraction'] = df_new['CUMULATIVE'] / df_new['population']
    
    # Start redefining vaccination stage fractions
    df_new = df_new.set_index('date') # make sure we don't get NaN values because of mismatching indices
    df_new_copy = df_new.copy()

    # first-only: dose A (first) - dose B (second)
    df_new.loc[df_new['dose']=='B','fraction'] = (df_new_copy.loc[df_new_copy['dose']=='A','fraction'] \
        - df_new_copy.loc[df_new_copy['dose']=='B','fraction']).clip(lower=0, upper=1)

    # full: dose B (second) + dose C (Jansen) - dose E (booster)
    df_new.loc[df_new['dose']=='C','fraction'] = (df_new_copy.loc[df_new_copy['dose']=='B','fraction'] \
        + df_new_copy.loc[df_new_copy['dose']=='C','fraction'] - df_new_copy.loc[df_new_copy['dose']=='E','fraction']).clip(lower=0, upper=1)
    
    # booster: clip between 0 and 1. This is currently the latest stage
    df_new.loc[df_new['dose']=='E','fraction'] = df_new_copy.loc[df_new_copy['dose']=='E', 'fraction'].clip(lower=0, upper=1)
    
    # none. Rest category. Make sure all exclusive categories adds up to 1.
    df_new.loc[df_new['dose']=='A','fraction'] = 1 - df_new.loc[df_new['dose']=='B','fraction'] \
        - df_new.loc[df_new['dose']=='C','fraction'] - df_new.loc[df_new['dose']=='E','fraction']
    
    ### Make sure all incidence and cumulative data are in the right columns
    # full = second + janssen
    df_new.loc[df_new['dose']=='C', 'INCIDENCE'] += df_new.loc[df_new['dose']=='B', 'INCIDENCE']
    df_new.loc[df_new['dose']=='C', 'CUMULATIVE'] += df_new.loc[df_new['dose']=='B', 'CUMULATIVE']
    # first is moved from 'A' to 'B'
    df_new.loc[df_new['dose']=='B', 'INCIDENCE'] = df_new.loc[df_new['dose']=='A', 'INCIDENCE']
    df_new.loc[df_new['dose']=='B', 'CUMULATIVE'] = df_new.loc[df_new['dose']=='A', 'CUMULATIVE']
    # 'A' becomes the empty category and is nullified
    df_new.loc[df_new['dose']=='A', 'INCIDENCE'] = 0
    df_new.loc[df_new['dose']=='A', 'CUMULATIVE'] = 0
    
    
    # Initialise rescaling parameter columns
    all_rescaling = ['E_susc_WT', 'E_inf_WT', 'E_hosp_WT', \
                      'E_susc_abc', 'E_inf_abc', 'E_hosp_abc', \
                      'E_susc_delta', 'E_inf_delta', 'E_hosp_delta', \
                      'E_susc',  'E_inf', 'E_hosp'] # averaged values
    for rescaling in all_rescaling:
        df_new[rescaling] = 0
        df_new.loc[df_new['dose']=='A', rescaling] = 1
    

    # Return to multiindex
    df_new = df_new.reset_index()
    if spatial:
        df_new = df_new.drop(columns=['age_class', 'population'])
        df_new = df_new.set_index(['date', 'NIS', 'age', 'dose'])
    else:
        df_new = df_new.drop(columns=['population'])
        df_new = df_new.set_index(['date',        'age', 'dose'])

    # rename indices to clearly understandable categories
    rename_indices = dict({'A' : 'none', 'B' : 'first', 'C' : 'full', 'E' : 'booster'})
    df_new = df_new.rename(index=rename_indices)
    
    # reset indices for next line of calculations
    df_new = df_new.reset_index()
    all_available_dates = df_new.date.unique()
    df_new = df_new.set_index(['date', 'dose']).sort_index()
    for rescaling in ['E_susc', 'E_inf', 'E_hosp']:
        for VOC in ['WT', 'abc', 'delta']:
            for dose in ['first', 'full', 'booster']:
                # Calculate E values for this rescaling type and dose
                onset_days_temp = onset_days[VOC][rescaling][dose]
                E_init_temp = E_init[VOC][rescaling][dose]
                E_best_temp = E_best[VOC][rescaling][dose]
                E_waned_temp = E_waned[VOC][rescaling][dose]
                for date in all_available_dates:
                    if verbose:
                        print(f"working on rescaling {rescaling}, VOC {VOC}, dose {dose}, date {pd.Timestamp(date).date()}.")
                    # run over all dates before this date
                    for d in all_available_dates[all_available_dates<=date]:
                        # Calculate how many days there are in between
                        delta_days = pd.Timedelta(date - d).days
                        # Sum over previous days with a weight depending on incidence, dose type, and waning of vaccines
                        weight = waning_exp_delay(delta_days, onset_days_temp, E_init_temp, E_best_temp, E_waned_temp)
                        df_new.loc[(date, dose), f'{rescaling}_{VOC}'] += df_new.loc[(d, dose),'INCIDENCE'].to_numpy() * weight
                    # normalise over total number of vaccinated subjects up to that point
                    df_new.loc[(date,dose), f'{rescaling}_{VOC}'] /= df_new.loc[(date,dose), 'CUMULATIVE']
            # Get rid of all division-by-zero results
            df_new.loc[df_new[f'{rescaling}_{VOC}']==np.inf, rescaling] = 1
            df_new[f'{rescaling}_{VOC}'].fillna(1, inplace=True)
        
    if spatial:
        df_new = df_new.groupby(['date', 'NIS', 'age', 'dose']).first()
    else:
        df_new = df_new.groupby(['date', 'age', 'dose']).first()
        
    # Calculate weighted average from fractions and current rescaling factor
    if spatial:
        all_available_NIS = initN.NIS.unique()
        for date in all_available_dates:
            if verbose:
                print("working on day ", pd.Timestamp(date).date())
            for NIS in all_available_NIS:
                for interval in intervals:
                    for rescaling in ['E_susc', 'E_inf', 'E_hosp']:
                        for VOC in ['WT', 'abc', 'delta']:
                            df_new.loc[(date,NIS,interval, 'weighted_sum'), f'{rescaling}_{VOC}'] = \
                                (df_new.loc[(date,NIS,interval), f'{rescaling}_{VOC}'] * \
                                 df_new.loc[(date,NIS,interval), 'fraction']).sum()
                        df_new.loc[(date,NIS,interval, 'weighted_sum'), rescaling] = 0
                        for i, VOC in enumerate(['WT', 'abc', 'delta']):
                            df_new.loc[(date,NIS,interval, 'weighted_sum'), rescaling] += \
                                (df_new.loc[(date,NIS,interval, 'weighted_sum'), f'{rescaling}_{VOC}']) * \
                                VOC_func(pd.Timestamp(date), 0, 0)[0][i]

    else:
        # Note: takes much longer because there is at least 7 times more data available
        for date in all_available_dates:
            if verbose:
                print("working on day ", pd.Timestamp(date).date())
            for interval in intervals:
                for rescaling in ['E_susc', 'E_inf', 'E_hosp']:
                    for VOC in ['WT', 'abc', 'delta']:
                        df_new.loc[(date,interval, 'weighted_sum'), f'{rescaling}_{VOC}'] = \
                            (df_new.loc[(date,interval), f'{rescaling}_{VOC}'] * \
                             df_new.loc[(date,interval), 'fraction']).sum()
                    df_new.loc[(date,interval, 'weighted_sum'), rescaling] = 0
                    for i, VOC in enumerate(['WT', 'abc', 'delta']):
                        df_new.loc[(date,interval, 'weighted_sum'), rescaling] += \
                            (df_new.loc[(date,interval, 'weighted_sum'), f'{rescaling}_{VOC}']) * \
                            VOC_func(pd.Timestamp(date), 0, 0)[0][i]  
                    
    if spatial:
        df_new = df_new.groupby(['date', 'NIS', 'age', 'dose']).first()
    else:
        df_new = df_new.groupby(['date', 'age', 'dose']).first()
    
    return df_new

    