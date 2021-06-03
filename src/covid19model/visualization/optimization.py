import datetime
import random
import math
import corner
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import emcee
from .utils import colorscale_okabe_ito
from .output import _apply_tick_locator

def checkplots(sampler, discard, thin, fig_path, spatial_unit, figname, labels):

    samples = sampler.get_chain(discard=discard,thin=thin,flat=False)
    flatsamples = sampler.get_chain(discard=discard,thin=thin,flat=True)

    # Traceplots of samples
    traceplot(samples,labels=labels,plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
    plt.savefig(fig_path+'traceplots/'+str(spatial_unit)+'_TRACE_'+figname+'_'+str(datetime.date.today())+'.pdf',
                dpi=400, bbox_inches='tight')

    # Autocorrelation plots of chains
    autocorrelation_plot(samples)
    plt.savefig(fig_path+'autocorrelation/'+str(spatial_unit)+'_AUTOCORR_'+figname+'_'+str(datetime.date.today())+'.pdf',
                dpi=400, bbox_inches='tight')

    # Cornerplots of samples
    fig = corner.corner(flatsamples,labels=labels)
    plt.savefig(fig_path+'cornerplots/'+str(spatial_unit)+'_CORNER_'+figname+'_'+str(datetime.date.today())+'.pdf',
                dpi=400, bbox_inches='tight')

    return

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

def plot_calibration_fit(out, df_sciensano, state, start_date, end_date, conf_int=0.05, show_all=False, ax=None, NIS=None, savename=None, **kwargs):
    """
    Plot the data as well as the calibration results with added binomial uncertainty.

    Input
    -----
    out : covid19model simulation output
        Output of the simulation, either spatially explicit or not
    df_sciensano : pandas DataFrame
        If spatial model: states as columns. If spatial: NIS codes as columns.
    state : str
        Choose the output state (e.g. 'H_in')
    start_date : first date to plot
        Format YYYY-MM-DD
    end_date : last dat to plot
        Format YYYY-MM-DD
    conf_int : float
        Confidence interval. 0.05 by default
    show_all : boolean
        If True, plot all simulation draws over each other.
    ax : matplotlib ax to plot on
    NIS : int
        NIS code to plot if spatial. None by default (plot national aggregation)
    savename : str
        Complete path and name under which to save the resulting fit figure
    kwargs : dict
        Dictionary with keyword arguments for the matplotlib make-up.

    Output
    ------
    ax : matplotlib ax object
    """
    # Find requested range from simulation output
    spatial=False
    if 'place' in out.keys():
        spatial=True
        # Plot nationally aggregated result
        if not NIS:
            all_ts = out[state].sum(dim='Nc').sum(dim='place').values
        # Plot one particular region
        else:
            all_ts = out[state].sel(place=NIS).sum(dim='Nc').values
    else:
        # There is no 'place' dimension
        all_ts = out[state].sum(dim='Nc').values

    # Compute mean and median over all draws
    # Note that the resulting values DO NOT correspond to a single simulation!
    if not show_all:
        ts_median = np.median(all_ts,axis=0)
        # Compute quantiles
        LL = conf_int/2
        UL = 1-conf_int/2
        ts_LL = np.quantile(all_ts, q=LL, axis=0)
        ts_UL = np.quantile(all_ts, q=UL, axis=0)
        ts_mean = np.mean(all_ts, axis=0)
    
    # Plot all lines and highlight simulation whose maximum value is the median value at the point where the overall highest simulation is
    # Now the resulting values DO correspond to a single simulation
    # Coordinates (draw, day) of highest simulated value
    elif show_all:
        top_draw, top_day = np.unravel_index(np.argmax(all_ts), all_ts.shape)
        # All simulated values at the time where the overall maximum is
        max_values = all_ts[:,top_day]
        # Draw for which the value is the median in these max_values
        median_draw = np.argsort(max_values)[len(max_values)//2]
        ts_median = all_ts[median_draw]
        # Create series of average values. Note that this does not correspond to a single simulation!
        ts_mean = np.mean(all_ts, axis=0)
        # Series of variances
        ts_var = np.var(all_ts, axis=0)
        # Series of resulting errors (sampling + Poisson)
        ts_err = np.sqrt(ts_var + ts_mean)

    # Plot
    if not ax:
        fig,ax = plt.subplots(figsize=(10,5))

    # Simulation
    # plot mean and CI
    if not show_all:
        ax.fill_between(pd.to_datetime(out['time'].values),ts_LL, ts_UL,alpha=0.20, color = 'blue')
        ax.plot(out['time'],ts_mean,'--', color='blue')
    # ... or plot all values with alpha
    elif show_all:
        number_of_draws = len(out.draws)
        err_min = np.clip(np.array(ts_mean) - ts_err, 0, None)
        err_max = np.array(all_ts[median_draw]) + ts_err
        ax.fill_between(pd.to_datetime(out['time'].values), err_min, err_max, alpha=0.10, color='red')
        for draw in range(number_of_draws):
            ax.plot(out['time'], all_ts[draw], alpha=2/number_of_draws, color='blue')
        # and overplot the 'median' single simulation
        ax.plot(out['time'], ts_median, '--', color='red', linewidth=1.5)

    # Plot result for sum over all places. Black dots for data used for calibration, red dots if not used for calibration.
    if not spatial:
        ax.scatter(df_sciensano[start_date:end_date].index, df_sciensano[state][start_date:end_date], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax = _apply_tick_locator(ax)
        ax.set_xlim(start_date,end_date)
        ax.set_ylabel('$H_{in}$ (-)') # Hard-coded
        if savename:
            fig.savefig(savename, dpi=400, bbox_inches='tight')
        return ax
    else:
        if not NIS:
            ax.scatter(df_sciensano[start_date:end_date].index, df_sciensano.sum(axis=1)[start_date:end_date], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
            ax = _apply_tick_locator(ax)
            ax.set_xlim(start_date,end_date)
            ax.set_ylabel('$H_{in}$ (national)') # Hard-coded
            if savename:
                fig.savefig(savename, dpi=400, bbox_inches='tight')
            return ax
        else:
            ax.scatter(df_sciensano[start_date:end_date].index, df_sciensano[NIS][start_date:end_date], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
            ax = _apply_tick_locator(ax)
            ax.set_xlim(start_date,end_date)
            ax.set_ylabel('$H_{in}$ (NIS ' + str(NIS) + ')') # Hard-coded
            if savename:
                fig.savefig(savename, dpi=400, bbox_inches='tight')
            return ax
