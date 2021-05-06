import datetime
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import emcee
from .utils import colorscale_okabe_ito
from .output import _apply_tick_locator

def autocorrelation_plot(samples):
    """Make a visualization of autocorrelation of each chain
    
    Parameters
    ----------
    samples: np.array
        A 3-D numpy array containing the sampled parameters.
        The x-dimension must be the number of samples, the y-dimension the number of parallel chains and the z-dimension the number of sampled parameters.

    Returns
    -------
    ax
    """
    # Compute autocorrelation/chain
    ndim = samples.shape[2]
    step_autocorr = math.ceil(samples.shape[0]/100)
    tau_vect = []
    index = 0
    for i in range(step_autocorr, samples.shape[0], step_autocorr):
        tau_vect.append(emcee.autocorr.integrated_time(samples[:i], tol = 0))
        index += 1
    n = step_autocorr * np.arange(1, index + 1)
    # Make figure
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(n, n / step_autocorr, "--k")
    ax.plot(n, tau_vect)
    ax.set_xlim(0, n.max())
    ax.set_xlabel("number of steps")
    ax.set_ylabel(r"$\hat{\tau}$");

    return ax

def traceplot(samples,labels,plt_kwargs={},filename=None):
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
    # Error when only one parameter is calibrated: axes not suscribable
    if ndim == 1:
        axes = [axes]
    # set size
    fig.set_size_inches(10, len(labels)*7/3)
    # plot data

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], **plt_kwargs)
        ax.set_xlim(0, nsamples)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("step number")

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight',
                    orientation='portrait')

    return ax

def plot_fit(y_model,data,start_date,warmup,states,end_date=None,with_ints=True,T=1,
                    data_mkr=['o','v','s','*','^'],plt_clr=['blue','red','green','orange','black'],
                    legend_text=None,titleText=None,ax=None,ylabel='number of patients',
                    plt_kwargs={},sct_kwargs={}):

    """Plot model fit to user provided data

    Parameters
    -----------
    y_model : xarray
        model output to be visualised
    data : array
        list containing dataseries
    start_date : string, format YYYY-MM-DD
        date corresponding to first entry of dataseries
    warmup : float or int
        time between start of simulation and start of data recording
    states : array
        list containg the names of the model states that correspond to the data
    end_date : string, format YYYY-MM-DD, optional
        end date of simulation
    with_ints : boolean, optional
        if True: use ints; if False: use datetime strings
    filename : string, optional
        Filename + extension to save a copy of the plot_fit
    ax : matplotlib.axes.Axes, optional
        If provided, will use the axis to add the lines.
    ylabel : string, optional
        label for y-axis, default 'number of patients'

    Returns
    -----------
    ax

    Example use
    -----------
    data = [[71,  90, 123, 183, 212, 295, 332]]
    start_date = '15-03-2020'
    warmup = int(42)
    states = [["H_in"]]
    T = data[0].size+warmup-1

    y_model = model.sim(int(T))
    ax = plot_fit(y_model,data,start_date,warmup,states)

    for i in range(100):
        y_model = model.sim(T)
        ax = plot_fit(y_model,data,start_date,warmup,states,ax=ax)

    """
    # Make sure to use pandas plot settings
    pd.plotting.register_matplotlib_converters()

    # check if ax object is provided by user
    if ax is None:
        #fig, ax = plt.subplots()
        ax = plt.gca()
    # Create shifted index vector
    if with_ints==True:
        idx = pd.date_range(start_date,freq='D',periods=data[0].size + warmup + T) - datetime.timedelta(days=warmup)
    else:
        idx_model = pd.date_range(pd.to_datetime(start_date)-pd.to_timedelta(warmup, unit='days'),
                                  pd.to_datetime(end_date))

        idx_data = pd.date_range(pd.to_datetime(start_date),
                                  pd.to_datetime(end_date))

    # Plot model prediction
    y_model = y_model.sum(dim="Nc")
    for i in range(len(data)):
        # dummy lines for legend
        lines = ax.plot([],[],plt_clr[i],alpha=1)

    for i in range(len(states)):
        data2plot = y_model[states[i]].to_array(dim="states").values.ravel()
        if with_ints==True:
            lines = ax.plot(idx,data2plot,plt_clr[i],**plt_kwargs)
        else:
            lines = ax.plot(idx_model,data2plot,plt_clr[i],**plt_kwargs)
    # Plot data
    for i in range(len(data)):
        if with_ints==True:
            lines=ax.scatter(idx[warmup:-T],data[i],color="black",facecolors='none',**sct_kwargs)
        else:
            if len(data[i]) < len(idx_data):
                idx_data_short = pd.date_range(pd.to_datetime(start_date),
                                               pd.to_datetime(start_date)+pd.to_timedelta(len(data[i])-1, unit='days'))
                lines=ax.scatter(idx_data_short,data[i],color="black",facecolors='none',**sct_kwargs)
            else:
                lines=ax.scatter(idx_data,data[i],color="black",facecolors='none',**sct_kwargs)


    # Attributes
    if legend_text is not None:
        leg=ax.legend(legend_text, loc="upper left", bbox_to_anchor=(1,1))
    if titleText is not None:
        ax.set_title(titleText,{'fontsize':18})

    # Format axes
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.setp(plt.gca().xaxis.get_majorticklabels(),
        'rotation', 90)
    #fig.autofmt_xdate(rotation=90)
    if with_ints==True:
        ax.set_xlim( idx[warmup-3], pd.to_datetime(idx[-1]+ datetime.timedelta(days=1)))
    else:
        #breakpoint()
        ax.set_xlim('2020-03-12', end_date)
    ax.set_ylabel(ylabel)

    # limit the number of ticks on the axis
    ax = _apply_tick_locator(ax)

    return ax

# def plot_calibration_fit(out, df_sciensano, state, start_date, end_date, conf_int=0.05, ax=None, start_calibration=None, end_calibration=None, **kwargs):
#     """
#     """

#     # Find requested range from simulation output
#     if 'place' in out.keys():
#         all_ts = out[state].sum(dim='Nc').sum(dim='place').values
#     else:
#         all_ts = out[state].sum(dim='Nc').values
        
#     # Compute mean and median
#     ts_median = np.median(all_ts,axis=1)
#     # Compute quantiles
#     LL = conf_int/2
#     UL = 1-conf_int/2
#     ts_LL = np.quantile(all_ts, q = LL, axis = 1)
#     ts_UL = np.quantile(all_ts, q = UL, axis = 1)
    
#     # Plot
#     if not ax:
#         fig,ax = plt.subplots(figsize=(10,5))
        
#     # Simulation
#     ax.fill_between(pd.to_datetime(out['time'].values),ts_LL, ts_UL,alpha=0.20, color = 'blue')
#     ax.plot(out['time'],H_in_mean,'--', color='blue')
    
#     if start_calibration:
        
        
#     # Plot result for sum over all places. Black dots for data used for calibration, red dots if not used for calibration.
#     ax.scatter(df_sciensano[start_calibration:end_calibration_beta].index, df_sciensano[start_calibration:end_calibration_beta].sum(axis=1), color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
#     ax.scatter(df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].index, df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].sum(axis=1), color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
#     ax = _apply_tick_locator(ax)
#     ax.set_xlim(start_calibration,end_sim)
#     ax.set_ylabel('$H_{in}$ (-)')
#     fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_BETAs_prelockdown_SUM_'+run_date+'.pdf', dpi=400, bbox_inches='tight')
#     plt.close()
    
    
    
    
    
    
    
    
