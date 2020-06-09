import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..visualization.data import google_mobility


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
        rel_dir = os.path.join(abs_dir, '../../../data/raw/google/community_mobility_data.csv')
        df.to_csv(rel_dir, index=False)
    else:
        df = pd.read_csv(os.path.join(abs_dir,
            '../../../data/raw/google/community_mobility_data.csv'),
            parse_dates=['date'], dtype=dtypes)

    # Extract only Belgian data
    df=df[df['country_region']=='Belgium']
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
