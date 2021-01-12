import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
