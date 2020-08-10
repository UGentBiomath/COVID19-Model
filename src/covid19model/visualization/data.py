
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def google_mobility(data):
    """Create plot of google mobility data

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the columns 'retail_recreation',
        'grocery', 'parks', 'transport', 'work', 'residential'
        as provided by the :meth:`~covid19model.data.google.get_google_mobility_data`
    """
    data_lst=[['retail_recreation', 'grocery'],
              ['parks', 'transport'],
              ['work', 'residential']]
    titleText=[['Retail and recreation', 'Groceries and pharmacy'],
               ['Parks', 'Transit stations'],
               ['Workplaces', 'Residential']]

    # using the variable axs for multiple Axes
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    for i in range(3):
        for j in range(2):
            ax[i,j].plot(data.index, data[data_lst[i][j]])
            ax[i,j].axvline(x=pd.Timestamp('2020-03-13'), color='k', linestyle='--')
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
            ax[i,j].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            ax[i,j].autoscale(enable=True)
            # Maximum number of ticks should be four
            ax[i,j].xaxis.set_major_locator(plt.MaxNLocator(4))
            
    plt.tight_layout()

    return fig, ax