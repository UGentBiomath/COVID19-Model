import pandas as pd
import matplotlib.pyplot as plt


def get_google_mobility_data(filename=None):
    """Download Google Community mobility report data

    This function downloads, formats and returns the available Belgian Google Community mobility report data.

    Parameters
    -----------
    filename: string
        filename and extension to automatically save the generated visualisation of the data
        argument is optional

    Returns
    -----------
    dates : pd.DatetimeIndex
        datetimes for which a data point is available
    retail_recreation : np.array
        Mobility trends for places such as restaurants, cafés, shopping centres, theme parks, museums, libraries and cinemas.
    grocery : np.array
        Mobility trends for places such as grocery shops, food warehouses, farmers markets, specialty food shops and pharmacies.
    parks: np.array
        Mobility trends for places such as local parks, national parks, public beaches, marinas, dog parks, plazas and public gardens.
    transport: np.array
        Mobility trends for places that are public transport hubs, such as underground, bus and train stations.
    work: np.array
        Mobility trends for places of work.
    residential: np.array
        Mobility trends for places of residence.

    Notes
    ----------
    Mobility data can be extracted as a report for any country from: https://www.google.com/covid19/mobility/
    Dataset was downloaded from: 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=2dcf78defb92930a'
    Documentation by Google on data collection can be found here : https://www.google.com/covid19/mobility/data_documentation.html?hl=nl

    Example use
    -----------
    dates, retail_recreation, grocery, parks, transport, work, residential = get_google_mobility_data(filename='community_report.svg')
    """

    # Data source
    url = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=2dcf78defb92930a'

    # Extract only Belgian data
    raw = pd.read_csv(url)
    raw=raw[raw['country_region']=='Belgium']
    data=raw[raw['sub_region_1'].isnull().values]

    # Assign data to output variables
    retail_recreation=numpy.array(data.loc[:,'retail_and_recreation_percent_change_from_baseline'].tolist())
    grocery=numpy.array(data.loc[:,'grocery_and_pharmacy_percent_change_from_baseline'].tolist())
    parks=numpy.array(data.loc[:,'parks_percent_change_from_baseline'].tolist())
    transport=numpy.array(data.loc[:,'transit_stations_percent_change_from_baseline'].tolist())
    work=numpy.array(data.loc[:,'workplaces_percent_change_from_baseline'].tolist())
    residential=numpy.array(data.loc[:,'residential_percent_change_from_baseline'].tolist())
    dates = pd.date_range(data.astype(str)['date'].tolist()[0], freq='D', periods=residential.size)
    data_lst=[[retail_recreation,grocery],[parks,transport],[work,residential]]
    titleText=[['Retail and recreation','Groceries and pharmacy'],['Parks','Transit stations'],['Workplaces','Residential']]

    # using the variable axs for multiple Axes
    fig, ax = plt.subplots(3,2,figsize=(15,12))
    for i in range(3):
        for j in range(2):
            ax[i,j].plot(dates,data_lst[i][j])
            ax[i,j].axvline(x='13-03-2020',color='k',linestyle='--')
            ax[i,j].set_ylabel('% compared to baseline')
            # Hide the right and top spines
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            ax[i,j].yaxis.set_ticks_position('left')
            ax[i,j].xaxis.set_ticks_position('bottom')
            # enable the grid
            ax[i,j].grid(True)
            # Set title
            ax[i,j].set_title(titleText[i][j],{'fontsize':18})
            # Format dateticks
            ax[i,j].xaxis.set_major_locator(mdates.MonthLocator())
            ax[i,j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%Y'))
            ax[i,j].autoscale(enable=True)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename,dpi=600,bbox_inches='tight',orientation='portrait',papertype='a4')
    else:
        plt.show()

    return dates,retail_recreation,grocery,parks,transport,work,residential