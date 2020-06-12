import datetime
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


# From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
colorscale_okabe_ito = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
                        "green" : "#009E73", "yellow" : "#F0E442",
                        "blue" : "#0072B2", "red" : "#D55E00",
                        "pink" : "#CC79A7", "black" : "#000000"}

# covid 19 specific parameters
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler('color',
                                  list(colorscale_okabe_ito.values())),
    "font.size": 15,
    "lines.linewidth" : 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "ytick.major.left": True,
    "axes.grid": True
})


def _apply_tick_locator(ax):
    """support function to apply default ticklocator settings"""
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    return ax

def plot_fit(model,data,startDate,states,checkpoints=None,samples=None,filename=None,dataMkr=['o','v','s','*','^'],
            modelClr=['green','orange','red','black','blue'],legendText=None,titleText=None,getfig=False,ax=None):

    """Plot model fit to user provided data 

    Parameters
    -----------
    model: model object
        correctly initialised model to be fitted to the dataset
    data: array
        list containing dataseries
    startDate: string, format DD-MM-YYY
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
    idx = pd.date_range(startDate,freq='D',periods=data[0].size + model.extraTime) - datetime.timedelta(days=model.extraTime)
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
                ax.plot(idx,data2plot,linewidth=0.25,alpha=0.2,color="blue")
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
                ax.plot(idx,data2plot)
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
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # enable the grid
    plt.grid(True)

    # limit the number of ticks on the axis
    ax = _apply_tick_locator(ax)

    if filename is not None:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    if getfig:
        return fig, ax
    else:
        plt.show()
