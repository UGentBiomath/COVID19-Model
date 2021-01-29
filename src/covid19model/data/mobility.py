import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from ..visualization.data import google_mobility

def get_apple_mobility_data(update=True):
    """Download Apple Mobility Trends

    This function downloads, formats and returns the available Apple Mobility Trends for the Provinces of Belgium.
    A copy of the downloaded dataset is automatically saved in the /data/raw folder.

    Parameters
    -----------
    update : boolean (default True)
        True if you want to update the data,
        False if you want to read only previously saved data

    Returns
    -------
    data : pandas.DataFrame
        DataFrame containing the Apple mobility data on daily basis, starting February 15th 2020.
        The dataframe has two indices: the spatial unit (f.e. 'Belgium' or 'Namur Province') and the mobility type ('walking', 'transit', 'driving').
        Example indexing: df_apple.loc[('Belgium','transit')]

    Notes
    -----
    Transport types (walking, transit, driving) are not available for every province.    
    Data before February 15th 2020 was omitted to match the start date of the Google community reports.
    Baseline was defined by Apple as the mobility on January 13th 2020. Data was rescaled using the mean mobility from February 15th 2020 until March 10th, 2020.
    Project homepage: https://covid19.apple.com/mobility
    Data extracted from: https://covid19-static.cdn-apple.com/covid19-mobility-data/2024HotfixDev12/v3/en-us/applemobilitytrends-2021-01-10.csv

    Example use
    -----------
    df_apple = mobility.get_apple_mobility_data(update=False)
    # Extracting transit mobility data for Belgium
    df_apple.loc[('Belgium','transit')]
    # Visualizing mobility data
    plt.plot(df_apple.loc[('Belgium','transit')])
    """

    # Data source
    url = 'https://covid19-static.cdn-apple.com/covid19-mobility-data/2024HotfixDev12/v3/en-us/applemobilitytrends-2021-01-10.csv'
    abs_dir = os.path.dirname(__file__)

    if update:
        # download raw data
        df_raw = pd.read_csv(url)
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/mobility/apple/apple_mobility_trends.csv')
        df_raw.to_csv(rel_dir, index=False)
    else:
        df_raw = pd.read_csv(os.path.join(abs_dir,
            '../../../data/raw/mobility/apple/apple_mobility_trends.csv'))

    # Start by extracting the overall data for Belgium
    df = df_raw[df_raw['region']=='Belgium']
    columns = pd.to_datetime(df_raw.columns[6:].values)
    data = df.values[:,6:]
    arrays = [
        np.array(["Belgium","Belgium","Belgium"]),
        np.array(["driving", "transit", "walking"]),
    ]
    df_apple = pd.DataFrame(data,index=arrays,columns=columns)

    # Loop over all available provinces and transit types
    df = df_raw[((df_raw['country']=='Belgium')&(df_raw['geo_type']=='sub-region'))]
    for province in set(df['region']):
        for mobility_type in df['transportation_type'][df['region'] == province]:
            data = df[((df['transportation_type']==mobility_type)&(df['region']==province))].values[:,6:]
            df_entry = pd.DataFrame(data,index=[[province],[mobility_type]],columns=columns)
            df_apple=df_apple.append(df_entry)

    # Re-define baseline and cut away data before February 15th 2020:
    for spatial_unit,mobility_type in df_apple.index:
        base=np.mean(df_apple.loc[(spatial_unit,mobility_type)][pd.to_datetime('2020-02-15'):pd.to_datetime('2020-03-10')])
        df_apple.loc[(spatial_unit,mobility_type)] = -(100-df_apple.loc[(spatial_unit,mobility_type)]/base*100)
        df_apple.loc[(spatial_unit,mobility_type)] = df_apple.loc[(spatial_unit,mobility_type)][pd.to_datetime('2020-02-15'):]

    return df_apple

def get_google_mobility_data(update=True, plot=False, filename_plot=None):
    """Download Google Community mobility report data

    This function downloads, formats and returns the available Belgian Google Community mobility report data.
    A copy of the downloaded dataset is automatically saved in the /data/raw folder.

    Parameters
    -----------
    update : boolean (default True)
        True if you want to update the data,
        False if you want to read only previously saved data
    plot : boolean (default False)
        If True, return a preformatted plot of the data
    filename_viz: string
        filename and extension to automatically save the generated visualisation of the data
        The argument has no effect when plot is False

    Returns
    -----------
    data : pandas.DataFrame
        DataFrame with the google mobility data on daily basis. The following columns
        are returned:

        - retail_recreation : Mobility trends for places such as restaurants, cafés, shopping centres, theme parks, museums, libraries and cinemas.
        - grocery : Mobility trends for places such as grocery shops, food warehouses, farmers markets, specialty food shops and pharmacies.
        - parks: Mobility trends for places such as local parks, national parks, public beaches, marinas, dog parks, plazas and public gardens.
        - transport: Mobility trends for places that are public transport hubs, such as underground, bus and train stations.
        - work: Mobility trends for places of work.
        - residential: Mobility trends for places of residence.

    Notes
    ----------
    Mobility data can be extracted as a report for any country from: https://www.google.com/covid19/mobility/
    Dataset was downloaded from: 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
    Documentation by Google on data collection can be found here : https://www.google.com/covid19/mobility/data_documentation.html?hl=nl

    Example use
    -----------
    data = get_google_mobility_data(filename='community_report.svg')
    >>> # download google mobility data and store new version (no viz)
    >>> data = get_google_mobility_data()
    >>> # load google mobility data from raw data directory (no new download)
    >>> data = get_google_mobility_data(update=False)
    >>> # load google mobility data from raw data directory and create viz
    >>> data = get_google_mobility_data(update=False, plot=True)
    >>> # load google mobility data from raw data directory, create viz and save viz
    >>> data = get_google_mobility_data(update=False, plot=True, filename_plot="my_viz.png")
    """

    # Data source
    url = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
    abs_dir = os.path.dirname(__file__)
    dtypes = {'sub_region_1': str, 'sub_region_2': str}

    if update:
        # download raw data
        df = pd.read_csv(url, parse_dates=['date'], dtype=dtypes)
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/raw/mobility/google/google_community_mobility_data_BE.csv')
        # Extract only Belgian data: full file is over 300 Mb
        df=df[df['country_region']=='Belgium']
        # Save data
        df.to_csv(rel_dir, index=False)

    else:
        df = pd.read_csv(os.path.join(abs_dir,
            '../../../data/raw/mobility/google/google_community_mobility_data_BE.csv'),
            parse_dates=['date'], dtype=dtypes)

    # Extract values
    data=df[df['sub_region_1'].isnull().values]

    # Assign data to output variables
    variable_mapping = {
        'retail_and_recreation_percent_change_from_baseline': 'retail_recreation',
        'grocery_and_pharmacy_percent_change_from_baseline': 'grocery',
        'parks_percent_change_from_baseline': 'parks',
        'transit_stations_percent_change_from_baseline': 'transport',
        'workplaces_percent_change_from_baseline': 'work',
        'residential_percent_change_from_baseline': 'residential'
    }
    data = data.rename(columns=variable_mapping)
    data = data.set_index("date")
    data.index.freq = 'D'
    data = data[list(variable_mapping.values())]

    if filename_plot and not plot:
        print("Filename plot has no effect, plot is not activated. Set `plot=True` to create plot.")

    if plot:
        fig, ax = google_mobility(data)
        
        if filename_plot:
            plt.savefig(filename_plot, dpi=600, bbox_inches='tight',
                        orientation='portrait', papertype='a4')
        else:
            plt.show()

    return data



####################################
# Proximus mobility data functions #
####################################

def date_to_YYYYMMDD(date, inverse=False):
    """
    Simple function to convert a datetime object to a string representing the date in the shape YYYYMMDD
    
    Input
    -----
    date: datetime.date object or str
        datetime.date object if inverse=False, str if inverse=True
    inverse: boolean
        False if date is converted to YYYYMMDD, True if YYYYMMDD string is converted to datetime.date object
    
    Return
    ------
    YYYYMMDD: str
    """
    if not inverse:
#         Something is wrong with the exception below, not sure what
#         if isinstance(date, datetime.date):
#             raise Exception("First argument in function should be of type datetime.date. If type str (YYYYMMDD), set inverse=True.")
        YYYYMMDD = date.strftime("%Y%m%d")
        return YYYYMMDD
    if inverse:
        if not isinstance(date, str) or (len(date) != 8):
            raise Exception("First argument in function should str in form YYYYMMDD. If type is datetime.time, set inverse=False")
        datetime_object = datetime.strptime(date, "%Y%m%d")
        return datetime_object
        

def week_to_date(week_nr, day = 1, year=2020):
    """
    Function that takes a week number between 1 and 53 (or 53 for 2021) and returns the corresponding dates
    
    Input
    -----
    week_nr: int
        Value from 1 to 54
    day: int
        Day of the week to return date for: 1 is first day, 7 is last day. Default is first day.
    year: int
        Year the week is in. Defaults to 2020.
    
    Returns
    -------
    date: datetime object
        Date of the requested day in format YYYY-MM-DD hh:mm:ss
    """
    
    if ((1 > week_nr) or (week_nr > 53)) and (year==2020):
        raise Exception("'week_nr' parameter must be an integer value from 1 to 53 for the year 2020.")
    if ((1 > week_nr) or (week_nr > 52)) and (year==2021):
        raise Exception("'week_nr' parameter must be an integer value from 1 to 52 for the year 2021.")
    if (day < 1) or (day > 7):
        raise Exception("'day' parameter must be an integer value from 1 to 7.")
    if day == 7:
        day = 0
    if year == 2020:
        d = str(year) + "-W" + str(week_nr - 1)
    if year == 2021:
        d = str(year) + "-W" + str(week_nr)
    date = datetime.strptime(d + '-' + str(day), "%Y-W%W-%w")
    return date

def make_date_list(week_nr, year=2020):
    """
    Makes list of dates in week 'week_nr' in the format needed for the identification of the Proximus data
    
    Input
    -----
    week_nr: int
        Value from 1 to 54
    year: int
        Year the week is in. Defaults to 2020.
        
    Returns
    -------
    date_list: list of str
        List of all 7 dates in week 'week_nr' of year 'year', in the format ['YYYYMMDD', ...]
    """
    if (1 > week_nr) or (week_nr > 54):
        raise Exception("'week_nr' parameter must be an integer value from 1 to 54.")
    date_list=[]
    for day in range(1,8):
        start_date = week_to_date(week_nr, day=day, year=year)
        YYYYMMDD = date_to_YYYYMMDD(start_date)
        date_list.append(YYYYMMDD)
    return date_list

def proximus_mobility_suffix():
    """
    Definition of the suffix in the mobility data files of Proximus.
    """
    suffix = "AZUREREF001.csv"
    return suffix

def check_missing_dates(dates, data_location):
    """
    Checks whether the requested dates correspond to existing Proximus mobility data and returns the missing dates
    
    Input
    -----
    dates: list of str
        Output of make_date_list. List for which the user wants to request data
    data_location: str
        Name of directory (relative or absolute) that contains all Proximus data files

    Returns
    -------
    missing: set of str
        All dates that do not correspond with a data file. Ideally, this set is empty.
    """
    suffix = proximus_mobility_suffix()
    full_list = []
    for f in os.listdir(data_location):
        date = f[:-len(suffix)][-8:]
        full_list.append(date)
    missing = set(dates).difference(set(full_list))
    return missing

def load_datafile_proximus(date, data_location):
    """
    Load an entire raw Proximus data file
    
    Input
    -----
    date: str
        Single date in the shape YYYYMMDD
    data_location: str
        Name of directory (relative or absolute) that contains all Proximus data files
        
    Output
    ------
    datafile: pandas DataFrame
    """
    suffix = proximus_mobility_suffix()
    datafile_name = data_location + 'outputPROXIMUS122747corona' + date + suffix
    # Note: the dtypes must not be int, because some int values are > 2^31 and this cannot be handled for int32
    datafile = pd.read_csv(datafile_name, sep=';', decimal=',', dtype={'mllp_postalcode' : str,
                                                                                         'postalcode' : str,
                                                                                         'imsisinpostalcode' : float,
                                                                                         'habitatants' : float,
                                                                                         'nrofimsi' : float,
                                                                                         'visitors' : float,
                                                                                         'est_staytime' : float,
                                                                                         'total_est_staytime' : float,
                                                                                         'est_staytime_perc' : float})
    return datafile    
    
    
def load_mobility_proximus(dates, data_location, values='nrofimsi', complete=False, verbose=True, return_missing=False):
    """
    Load Proximus mobility data (number of visitors or visitor time) corresponding to the requested dates
    
    Input
    -----
    dates: str or list of str
        Single date in YYYYMMDD form or list of these (output of make_date_list function): requested date(s)
    data_location: str
        Name of directory (relative or absolute) that contains all Proximus data files
    values: str
        Choose between absolute visitor count ('nrofimsi', default) or other values of interest (e.g. 'est_staytime') on one day.
        Special case for values='est_staytime' or 'total_est_staytime': the absolute value is taken
    complete: boolean
        If True, this function raises an exception when 'dates' contains a date that does not correspond to a data file.
    verbose: boolean
        If True, print statement every time data for a date is loaded.
    return_missing: boolean
        If True, return array of missing dates in form YYYYMMDD as second return. False by default.
    
    Returns
    -------
    mmprox_dict: dict of pandas DataFrames
        Dictionary with YYYYMMDD dates as keys and pandas DataFrames with visit counts or estimated staytime between postal codes.
    """
    # Check dates type and change to single-element list if needed
    single_date = False
    if isinstance(dates,str):
        dates = [dates]
        single_date = True
    
    missing = check_missing_dates(dates, data_location)
    load_dates = set(dates).difference(missing)
    dates_left = len(load_dates)
    if dates_left == 0:
        raise Exception("None of the requested dates correspond to a Proximus mobility file.")
    if missing != set():
        print(f"Warning: some or all of the requested dates do not correspond to Proximus data. Dates: {sorted(missing)}")
        if complete:
            raise Exception("Some requested data is not found amongst the Proximus files. Set 'complete' parameter to 'False' and rerun if you wish to proceed with an incomplete data set (not using all requested data).")
        print(f"... proceeding with {dates_left} dates.")

    # Initiate dict for remaining dates
    mmprox_dict=dict({})
    load_dates = sorted(list(load_dates))
    for date in load_dates:
        datafile = load_datafile_proximus(date, data_location)
        mmprox_temp = datafile[['mllp_postalcode', 'postalcode', values]]
        mmprox_temp = mmprox_temp.pivot_table(values=values,
                                              index='mllp_postalcode',
                                              columns='postalcode')
        mmprox_temp = mmprox_temp.fillna(value=0)
        if values in ['est_staytime', 'total_est_staytime']:
            mmprox_dict[date] = mmprox_temp.convert_dtypes().abs()
        else:
            mmprox_dict[date] = mmprox_temp.convert_dtypes()
        if verbose==True:
            print(f"Loaded dataframe for date {date}.    ", end='\r')
    print(f"Loaded dataframe for date {date}.")
    
    if not return_missing:
        return mmprox_dict
    else:
        return mmprox_dict, sorted(missing)


def load_pc_to_nis():
    # Data source
    abs_dir = os.path.dirname(__file__)
    pc_to_nis_file = os.path.join(abs_dir, '../../../data/raw/GIS/Postcode_Niscode.xlsx')
    pc_to_nis_df = pd.read_excel(pc_to_nis_file)[['Postcode', 'NISCode']]
    return pc_to_nis_df
    
def show_missing_pc(mmprox):
    """
    Function to return the missing postal codes in Proximus mobility data frames.
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and visit counts or visit lenghts as values

    Returns
    -------
    from_PCs_missing: array of str
        Missing postal codes in the indices (missing information FROM where people are coming)
    to_PCs_missing: array of str
        Missing postal codes in the columns (missing information WHERE people are going)
    """
    pc_to_nis = load_pc_to_nis()
    all_PCs = set(pc_to_nis['Postcode'].values)
    index_list=list(mmprox.index.values)
    if 'Foreigner' in index_list:
        index_list.remove('Foreigner')
    column_list=list(mmprox.columns.values)
    if 'ABROAD' in column_list:
        column_list.remove('ABROAD')
    from_PCs = set(np.array(index_list).astype(int))
    to_PCs = set(np.array(column_list).astype(int))

    from_PCs_missing = all_PCs.difference(from_PCs)
    to_PCs_missing = all_PCs.difference(to_PCs)
    
    from_PCs_missing = np.array(sorted(from_PCs_missing)).astype(str)
    to_PCs_missing = np.array(sorted(to_PCs_missing)).astype(str)
    
    return from_PCs_missing, to_PCs_missing

def fill_missing_pc(mmprox):
    """
    Function that adds the missing postal codes to the mobility matrix and fills these with zeros.
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and visit counts or visit lenghts as values
        
    Returns
    -------
    mmprox_complete: pandas DataFrame
        Square mobility matrix with all from/to postal codes
    """
    mmprox_complete = mmprox.copy()
    from_PCs_missing, to_PCs_missing = show_missing_pc(mmprox_complete)
    # Add missing PCs as empty rows/columns (and turn the matrix square)
    for pc in from_PCs_missing:
        mmprox_complete.loc[pc] = 0
    for pc in to_PCs_missing:
        mmprox_complete[pc] = 0

    if 'Foreigner' not in mmprox_complete.index:
        mmprox_complete.loc['Foreigner'] = 0
    if 'ABROAD' not in mmprox_complete.columns:
        mmprox_complete['ABROAD'] = 0
        
    return mmprox_complete
    


# What to dot with GDPR-protected <30 values
# TO DO: this needs some work
def GDPR_exponential_pick_visits(avg=5):
    """
    Choose a value ranging from 1 to 30 with an exponential distribution. Used to change the GDPR-protected value -1 for the number of visits, with an estimated number that it actually signifies.
    
    Input
    -----
    avg: int
        Expectation value of the exponential distribution. Default: 5
        
    Returns
    -------
    number: int
        Integer value between 1 and 30, with a higher chance for lower numbers.
    """
    number = np.ceil(np.random.exponential(scale=avg))
    if number > 30:
        number = 30
    return number

# TO BE CHANGED!
def GDPR_exponential_pick_length(avg=10000):
    """
    Choose a value ranging from an exponential distribution. Used to change the GDPR-protected value -1 for the length of stay, with an estimated number that it actually signifies.
    
    Input
    -----
    avg: int
        Expectation value of the exponential distribution. Default: 10000
        
    Returns
    -------
    number: int
        Integer value, with a higher chance for lower numbers.
    """
    return

def GDPR_replace(mmprox, replace_func=GDPR_exponential_pick_visits, **kwargs):
    """
    Function to replace the -1 values that denote small values proteted by the GDPR privacy protocol
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and visit counts or visit lenghts as values
    replace_func: function
        Function used to determine the value that -1 is to be replaced with
        
c
    """
    values = mmprox.values
    for (x,y), value in np.ndenumerate(values):
        if value < 0:
            values[x,y] = replace_func(**kwargs)
    mmprox_GDPR = pd.DataFrame(values, columns=mmprox.columns, index=mmprox.index)

    return mmprox_GDPR
        

# Aggregate

def mm_aggregate(mmprox, agg='mun'):
    """
    Aggregate cleaned-up mobility dataframes at the aggregation level of municipalities, arrondissements or provinces
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and visit counts or visit lenghts as values
    agg: str
        The level at which to aggregate. Choose between 'mun', 'arr' or 'prov'. Default is 'mun'.
        
    Output
    ------
    mmprox_agg: pandas DataFrame
        Mobility dataframe aggregated at the level 'agg'
    """
    # validate
    mmprox_shape = mmprox.shape
    if mmprox_shape != (1148, 1148):
        raise Exception(f"The input dataframe is of the shape {mmprox_shape}, not (1148, 1148) which is all 1147 postal codes + destinations/origins abroad. Fix this first.")
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception("The aggregation level must be either municipality ('mun'), arrondissements ('arr') or provinces ('prov').")
    
    # copy dataframe and load the postal-code-to-NIS-value translator
    mmprox_agg = mmprox.copy()
    pc_to_nis = load_pc_to_nis()
    
    rename_abroad = 'ABROAD'
    rename_foreigner = 'Foreigner'
    
    # initiate renaming dictionaries
    rename_col_dict = dict({})
    rename_idx_dict = dict({})
    for pc in mmprox_agg.columns:
        if pc != 'ABROAD':
            NIS = str(pc_to_nis[pc_to_nis['Postcode']==int(pc)]['NISCode'].values[0])
            rename_col_dict[pc] = NIS
    rename_col_dict['ABROAD'] = rename_abroad
    for pc in mmprox_agg.index:
        if pc != 'Foreigner':
            NIS = str(pc_to_nis[pc_to_nis['Postcode']==int(pc)]['NISCode'].values[0])
            rename_idx_dict[pc] = NIS
    rename_idx_dict['Foreigner'] = rename_foreigner
    
    # Rename the column names and indices to prepare for merging
    mmprox_agg = mmprox_agg.rename(columns=rename_col_dict, index=rename_idx_dict)
    
    mmprox_agg = mmprox_agg.groupby(level=0, axis=1).sum()
    mmprox_agg = mmprox_agg.groupby(level=0, axis=0).sum().astype(int)
    
    if agg in ['arr', 'prov']:
        # Rename columns
        for nis in mmprox_agg.columns:
            if nis != 'ABROAD':
                new_nis = nis[:-3] + '000'
                mmprox_agg = mmprox_agg.rename(columns={nis : new_nis})

        # Rename rows
        for nis in mmprox_agg.index:
            if nis != 'Foreigner':
                new_nis = nis[:-3] + '000'
                mmprox_agg = mmprox_agg.rename(index={nis : new_nis})

        # Collect rows and columns with the same NIS code, and automatically order column/row names
        mmprox_agg = mmprox_agg.groupby(level=0, axis=1).sum()
        mmprox_agg = mmprox_agg.groupby(level=0, axis=0).sum().astype(int)
        
        if agg == 'prov':
            # Rename columns
            for nis in mmprox_agg.columns:
                if nis not in ['ABROAD', '21000', '23000', '24000', '25000']: # Brussels is '11th province'
                    new_nis = nis[:-4] + '0000'
                    mmprox_agg = mmprox_agg.rename(columns={nis : new_nis})
                if nis in ['23000', '24000']:
                    new_nis = '20001'
                    mmprox_agg = mmprox_agg.rename(columns={nis : new_nis})
                if nis == '25000':
                    new_nis = '20002'
                    mmprox_agg = mmprox_agg.rename(columns={nis : new_nis})

            # Rename rows
            for nis in mmprox_agg.index:
                if nis not in ['Foreigner', '21000', '23000', '24000', '25000']:
                    new_nis = nis[:-4] + '0000'
                    mmprox_agg = mmprox_agg.rename(index={nis : new_nis})
                if nis in ['23000', '24000']:
                    new_nis = '20001'
                    mmprox_agg = mmprox_agg.rename(index={nis : new_nis})
                if nis == '25000':
                    new_nis = '20002'
                    mmprox_agg = mmprox_agg.rename(index={nis : new_nis})

            # Collect rows and columns with the same NIS code, and automatically order column/row names
            mmprox_agg = mmprox_agg.groupby(level=0, axis=1).sum()
            mmprox_agg = mmprox_agg.groupby(level=0, axis=0).sum().astype(int)
    
    return mmprox_agg
    
def complete_data_clean(mmprox, agg='mun'):
    """
    Execute several standard functions at the same time
    """
    mmprox_clean = mm_aggregate( fill_missing_pc( GDPR_replace(mmprox) ), agg=agg)
    return mmprox_clean



# Temporal aggregation/averaging
def average_mobility(mmprox_dict):
    """
    Calculates the average mobility over all dates in the mmprox_dict dictionary
    
    Input
    -----
    mmprox_dict: dict
        Dictionary with YYYYMMDD strings as keys and pandas DataFrames with clean Proximus data as values
        
    Output
    ------
    mmprox_avg: pandas DataFrame
        Mobility DataFrame with average values of all DataFrames in input
    """
    # Access first dict element and demand all are of the same shape
    matrix_shape = list(mmprox_dict.values())[0].shape
    for date in mmprox_dict:
        matrix_shape_next = mmprox_dict[date].shape
        if matrix_shape_next != matrix_shape:
            raise Exception("All shapes of the mobility matrices must be the same.")
        matrix_shape = matrix_shape_next

    first=True
    for date in mmprox_dict:
        if not first:
            mmprox_avg = mmprox_avg.add(mmprox_dict[date])
        if first:
            mmprox_avg = mmprox_dict[date]
            first=False
            
    mmprox_avg = mmprox_avg / len(mmprox_dict)
    return mmprox_avg

