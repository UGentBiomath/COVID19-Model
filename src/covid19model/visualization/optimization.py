import datetime
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from .utils import colorscale_okabe_ito
from .output import _apply_tick_locator

def traceplot(samples,labels,plt_kwargs={}):
    """Make a visualization of sampled parameters

    Parameters
    ----------
    samples: np.array
        A 3-D numpy array containing the sampled parameters. 
        The x-dimension must be the number of samples, the y-dimension the number of parallel chains and the z-dimension the number of sampled parameters.
    labels: list
        A list containing the names of the sampled parameters. Must be the same length as the z-dimension of the samples np.array.
    plt_kwargs: dictionary
        A dictionary containing arguments for the plt.plot function.

    Returns
    -------
    ax
    """
    # extract dimensions of sampler output
    nsamples,nwalkers, ndim = samples.shape
    # input check    
    if len(labels) != ndim:
        raise ValueError(
        "The length of label list is not equal to the length of the z-dimension of the samples.\n"
        "The list of label is of length: {0}. The z-dimension of the samples of length: {1}".format(len(labels), ndim)
        )
    # initialise figure
    fig, axes = plt.subplots(len(labels))
    # plot data
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], **plt_kwargs)
        ax.set_xlim(0, nsamples)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("step number")

    return ax

def plot_fit(y_model,data,start_date,lag_time,states,T=1,data_mkr=['o','v','s','*','^'],clr=['green','orange','red','black','blue'],
                legend_text=None,titleText=None,ax=None,plt_kwargs={},sct_kwargs={}):

    """Plot model fit to user provided data 

    Parameters
    -----------
    y_model: xarray
        model output to be visualised
    data: array
        list containing dataseries
    start_date: string, format DD-MM-YYY
        date corresponding to first entry of dataseries
    lag_time: float or int
        time between start of simulation and start of data recording
    states: array
        list containg the names of the model states that correspond to the data
    filename: string, optional
        Filename + extension to save a copy of the plot_fit
    ax : matplotlib.axes.Axes, optional
        If provided, will use the axis to add the lines.

    Returns
    -----------
    ax

    Example use
    -----------
    data = [[71,  90, 123, 183, 212, 295, 332]]
    start_date = '15-03-2020'
    lag_time = int(42)
    states = [["H_in"]]
    T = data[0].size+lag_time-1

    y_model = model.sim(int(T))
    ax = plot_fit(y_model,data,start_date,lag_time,states)

    for i in range(100):
        y_model = model.sim(T)
        ax = plot_fit(y_model,data,start_date,lag_time,states,ax=ax)

    """

    # check if ax object is provided by user
    if ax is None:
        #fig, ax = plt.subplots()
        ax = plt.gca()
    # Create shifted index vector 
    idx = pd.date_range(start_date,freq='D',periods=data[0].size + lag_time + T) - datetime.timedelta(days=lag_time)

    # Plot model prediction
    y_model = y_model.sum(dim="stratification")
    for i in range(len(data)):
        data2plot = y_model[states[i]].to_array(dim="states").values.ravel()
        lines = ax.plot(idx,data2plot,color=clr[i],**plt_kwargs)    
    # Plot data
    for i in range(len(data)):
        lines=ax.scatter(idx[lag_time:-T],data[i],color="black",marker=data_mkr[i],**sct_kwargs)

    # Attributes
    if legend_text is not None:
        ax.legend(legend_text, loc="upper left", bbox_to_anchor=(1,1))
    if titleText is not None:
        ax.set_title(titleText,{'fontsize':18})

    # Format axes
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.setp(plt.gca().xaxis.get_majorticklabels(),
        'rotation', 90)
    #fig.autofmt_xdate(rotation=90)
    ax.set_xlim( idx[lag_time-3], pd.to_datetime(idx[-1]+ datetime.timedelta(days=1)))
    ax.set_ylabel('number of patients')

    # limit the number of ticks on the axis
    ax = _apply_tick_locator(ax)

    return ax
