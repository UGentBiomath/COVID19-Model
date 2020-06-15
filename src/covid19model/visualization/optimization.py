import datetime
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from .utils import colorscale_okabe_ito
from .output import _apply_tick_locator

def plot_fit(model,data,start_date,states,checkpoints=None,samples=None,filename=None,dataMkr=['o','v','s','*','^'],
            modelClr=['green','orange','red','black','blue'],legendText=None,titleText=None,ax=None):

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
    checkpoints : dict, optional
        A dictionary with a "time" key and additional parameter keys,in the form of
        ``{"time": [t1, t2, ..], "param": [param1, param2, ..], ..}``
        indicating new parameter values at the corresponding timestamps.
    samples: dict, optional
        A dictionary containing parameter values obtained from a sampling algorithm (f.i. MCMC)
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
    idx = pd.date_range(start_date,freq='D',periods=data[0].size + model.extraTime) - datetime.timedelta(days=model.extraTime)
    # Plot data
    for i in range(len(data)):
        ax.scatter(idx[model.extraTime:],data[i],color="black")

    # Perform simulation(s) and plot model prediction
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute number of dataseries
    n = len(data)
    # Compute simulation time
    T = data[0].size+model.extraTime-1
    # Perform simulation(s)
    if samples is None:
        k = 0
        while k < 200:
            out = model.sim(T,checkpoints=checkpoints)
            out = out.sum(dim="stratification")
            for i in range(len(data)):
                data2plot = out[states[i]].to_array(dim="states").values.ravel()
                lines = ax.plot(idx,data2plot,linewidth=0.25,alpha=0.2,color="blue")
            k = k +1
    else:
        original_parameters = model.parameters.copy()
        k=0
        while k < 100:
            for key,value in samples:
                # do random draw and assign to model
                model.parameters[key] = random.choice(value)
            # run simulation
            out = model.sim(T,checkpoints=checkpoints)
            out = out.sum(dim="stratification")
            for i in range(len(data)):
                data2plot = out[states[i]].to_array(dim="states").values.ravel()
                lines = ax.plot(idx,data2plot)
            k = k+1
        # reset parameters
        model.parameters=original_parameters
    

    # Attributes
    if legendText is not None:
        ax.legend(legendText, loc="upper left", bbox_to_anchor=(1,1))
    if titleText is not None:
        ax.set_title(titleText,{'fontsize':18})
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%Y'))
    plt.setp(plt.gca().xaxis.get_majorticklabels(),
        'rotation', 90)
    ax.set_xlim( idx[model.extraTime-3], pd.to_datetime(idx[-1]+ datetime.timedelta(days=1)))
    ax.set_ylabel('number of patients')

    # limit the number of ticks on the axis
    ax = _apply_tick_locator(ax)

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')

    return lines
