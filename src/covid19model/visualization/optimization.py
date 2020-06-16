import datetime
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from .utils import colorscale_okabe_ito
from .output import _apply_tick_locator

def plot_fit(y_model,data,start_date,lag_time,states,filename=None,data_mkr=['o','v','s','*','^'],clr=['green','orange','red','black','blue'],
                legend_text=None,titleText=None,ax=None):

    """Plot model fit to user provided data 

    Parameters
    -----------
    model: model object
        correctly initialised model to be fitted to the dataset
    data: array
        list containing dataseries
    start_date: string, format DD-MM-YYY
        date corresponding to first entry of dataseries
    states: array
        list containg the names of the model states that correspond to the data
  
    filename: string, optional
        Filename + extension to save a copy of the plot_fit
    ax : matplotlib.axes.Axes, optional
        If provided, will use the axis to add the lines.

    Returns
    -----------

    Notes
    -----------

    Example use
    -----------


    """

    # Initialize figure and visualize data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # check if ax object is provided by user
    if ax is None:
        fig, ax = plt.subplots()
    # Create shifted index vector using self.extraTime
    idx = pd.date_range(start_date,freq='D',periods=data[0].size + lag_time) - datetime.timedelta(days=lag_time)
    # Plot model prediction
    y_model = y_model.sum(dim="stratification")
    for i in range(len(data)):
        data2plot = y_model[states[i]].to_array(dim="states").values.ravel()
        lines = ax.plot(idx,data2plot,)    
    # Plot data
    for i in range(len(data)):
        ax.scatter(idx[lag_time:],data[i],color="black",marker=data_mkr[i])


    # Attributes
    if legend_text is not None:
        ax.legend(legend_text, loc="upper left", bbox_to_anchor=(1,1))
    if titleText is not None:
        ax.set_title(titleText,{'fontsize':18})

    # Format axes
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    fig.autofmt_xdate(rotation=90)
    ax.set_xlim( idx[lag_time-3], pd.to_datetime(idx[-1]+ datetime.timedelta(days=1)))
    ax.set_ylabel('number of patients')

    # limit the number of ticks on the axis
    ax = _apply_tick_locator(ax)

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')

    return lines
