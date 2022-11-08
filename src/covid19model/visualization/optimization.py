import math
import matplotlib.pyplot as plt
import numpy as np
import emcee
from .output import _apply_tick_locator

def autocorrelation_plot(samples, labels=None, filename=None, plt_kwargs={}):
    """Make a visualization of autocorrelation of each chain

    Parameters
    ----------
    samples: np.array
        A 3-D numpy array containing the sampled parameters.
        The x-dimension must be the number of samples, the y-dimension the number of parallel chains and the z-dimension the number of sampled parameters.

    Returns
    -------
    ax: matplotlib axis object

    tau_vect[-1]: list
        Autocorrelation of last step
    """

    # Extract dimensions of sampler output
    nsamples,nwalkers, ndim = samples.shape
    # Generate list of lables if none provided
    if not labels:
        labels = [f"$\\theta_{i}$" for i in range(ndim)]
    else:
        # input check
        if len(labels) != ndim:
            raise ValueError(
            "The length of label list is not equal to the length of the z-dimension of the samples.\n"
            "The list of label is of length: {0}. The z-dimension of the samples of length: {1}".format(len(labels), ndim)
            )

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
    # Autocorrelation
    ax.plot(n, np.array(tau_vect))
    ax.plot(n, n/50, "--k")
    ax.set_xlim(0, n.max())
    ax.set_ylabel(r"$\hat{\tau}$");    
    ax.grid(False)
    ax.legend(labels)

    # Save result if desired
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight', orientation='portrait')

    return ax, tau_vect[-1]

def traceplot(samples, labels=None, filename=None, plt_kwargs={}):
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
    # Extract dimensions of sampler output
    nsamples,nwalkers, ndim = samples.shape
    # Generate list of lables if none provided
    if not labels:
        labels = [f"$\\theta_{i}$" for i in range(ndim)]
    else:
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
        ax.grid(False)
    axes[-1].set_xlabel("step number")

    # Save result if desired
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight', orientation='portrait')

    return ax

def plot_PSO(output, data, states, start_calibration, end_calibration):
    """
    A generic function to visualize a PSO estimate on multiple dataseries
    Automatically ums over all available dimensions in the xarray model output.

    Parameters
    ----------

    output : xr.DataArray
        Model simulation

    data : list
        List containing dataseries to compare model output to in calibration objective function

    states :
        List containing the names of the model states that must be matched to the corresponding dataseries in 'data'

    start_calibration : string
        Startdate of calibration, 'YYYY-MM-DD'

    end_calibration : string
        Enddate of calibration, 'YYYY-MM-DD'

    Returns
    -------

    fig: plt figure

    Example use
    -----------

    # run optimisation
    theta = pso.fit_pso(model, data, pars, states, weights, bounds, maxiter=maxiter, popsize=popsize,start_date=start_calibration, processes=processes)
    # Assign estimates to model parameters dictionary
    warmup, model.parameters = assign_PSO(model.parameters, pars, theta)
    # Perform simulation
    out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup,draw_fcn=draw_fcn,samples={})
    # Plot result
    plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)

    """

    # Visualize fit
    if len(states) == 1:
        idx = 0
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,4))
        # Reduce dimensions
        new_xarray = output[states[idx]].copy(deep=True)
        for dimension in output.dims:
            if (dimension != 'time') :
                new_xarray = new_xarray.sum(dim=dimension)
        # Plot data
        ax.plot(output['time'],new_xarray,'--', color='blue')
        try: 
            ax.scatter(data[idx].index,data[idx].sum(axis=1), color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        except:
            ax.scatter(data[idx].index,data[idx], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.set_xlim([start_calibration,end_calibration])
    else:
        fig,axes = plt.subplots(nrows=len(states),ncols=1,figsize=(12,4*len(states)),sharex=True)
        for idx,ax in enumerate(axes):
            # Reduce dimensions
            new_xarray = output[states[idx]].copy(deep=True)
            for dimension in output.dims:
                if (dimension != 'time') :
                    new_xarray = new_xarray.sum(dim=dimension)
            # Plot data
            ax.plot(output['time'],new_xarray,'--', color='blue')
            try:
                ax.scatter(data[idx].index,data[idx].sum(axis=1), color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
            except:
                ax.scatter(data[idx].index,data[idx], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)   
            ax.set_xlim([start_calibration,end_calibration])
    ax = _apply_tick_locator(ax)
    return ax

def plot_PSO_spatial(output, df_sciensano, start_calibration, end_calibration, agg, desired_agg):

    """
    A tailored function to visualize a PSO estimate on the H_in data for the spatial model, can aggregate arrondissment/provincial simulations to provincial and regional level and visualize them.

    Parameters
    ----------

    output : xr.DataArray
        Model simulation

    df_sciensano : pd.DataFrame
        Daily hospitalisation data

    start_calibration : string
        Startdate of calibration, 'YYYY-MM-DD'

    end_calibration : string
        Enddate of calibration, 'YYYY-MM-DD'

    agg : string
        Spatial aggregation level, either 'reg' or 'prov'

    Returns
    -------

    fig: plt figure
    """
    
    # Reduce all dimensions except time and space
    for dimension in output.dims:
        if ((dimension != 'time') & (dimension != 'NIS')):
            output = output.sum(dim=dimension)
    

    agg_prov_reg = [[10000, 20001, 30000, 40000, 70000],
                    [21000],
                    [20002, 50000, 60000, 80000, 90000]]

    agg_arr_prov_simple = [
                        [11000, 12000, 13000],
                        [23000, 24000],
                        [31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000],
                        [41000, 42000, 43000, 44000, 45000, 46000],
                        [71000, 72000, 73000],
                        [21000,],
                        [25000,],
                        [51000, 52000, 53000, 55000, 56000, 57000, 58000],
                        [61000, 62000, 63000, 64000],
                        [81000, 82000, 83000, 84000, 85000],
                        [91000, 92000, 93000]
    ]

    agg_arr_prov = [
                    [[11000, 12000, 13000],
                    [23000, 24000],
                    [31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000],
                    [41000, 42000, 43000, 44000, 45000, 46000],
                    [71000, 72000, 73000]],

                    [[21000,],],

                    [[25000,],
                    [51000, 52000, 53000, 55000, 56000, 57000, 58000],
                    [61000, 62000, 63000, 64000],
                    [81000, 82000, 83000, 84000, 85000],
                    [91000, 92000, 93000]]
    ]
    title_list_reg = ['Flanders', 'Brussels', 'Wallonia']
    title_list_prov = ['Antwerpen','Vlaams Brabant','West-Vlaanderen','Oost-Vlaanderen','Limburg','Brussels','Brabant Wallon','Hainaut','Liege','Luxembourg','Namur']
    color_list = ['blue', 'blue', 'blue']

    if agg == 'prov':
        if desired_agg == 'reg':
            fig,ax=plt.subplots(nrows=len(title_list_reg),ncols=1, figsize=(12,12), sharex=True)
            for idx,NIS_list in enumerate(agg_prov_reg):
                model_vals = 0
                data_vals= 0
                for NIS in NIS_list:
                    model_vals = model_vals + output['H_in'].sel(NIS=NIS).values
                    data_vals = data_vals + df_sciensano.loc[slice(None), NIS].values

                ax[idx].plot(output['time'].values, model_vals, '--', color='blue')
                ax[idx].scatter(df_sciensano.index.get_level_values('date').unique(), data_vals, color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
                ax[idx].set_title(title_list_reg[idx])
                ax[idx].set_xlim([start_calibration, end_calibration])
                ax[idx].set_ylim([0, 380])
                ax[idx].grid(False)
                ax[idx].set_ylabel('$H_{in}$ (-)')
                ax[idx] = _apply_tick_locator(ax[idx])
        
        elif desired_agg == 'prov':
            fig,ax = plt.subplots(nrows=len(title_list_prov),ncols=1,figsize=(12,16), sharex=True)
            for idx,NIS in enumerate(df_sciensano.index.get_level_values('NIS').unique()):
                ax[idx].plot(output['time'], output['H_in'].sel(NIS=NIS),'--', color='blue')
                ax[idx].set_title(title_list_prov[idx])
                ax[idx].scatter(df_sciensano.index.get_level_values('date').unique(), df_sciensano.loc[slice(None), NIS].values, color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
                ax[idx].set_xlim([start_calibration, end_calibration])
                ax[idx].set_ylim([0, 150])
                ax[idx].grid(False)
                ax[idx].set_ylabel('$H_{in}$ (-)')
                ax[idx] = _apply_tick_locator(ax[idx])

    elif agg == 'arr':
        if desired_agg == 'prov':
            fig,ax=plt.subplots(nrows=len(title_list_prov),ncols=1, figsize=(12,12), sharex=True)
            for idx,NIS_list in enumerate(agg_arr_prov_simple):
                model_vals = 0
                data_vals= 0
                for NIS in NIS_list:
                    model_vals = model_vals + output['H_in'].sel(NIS=NIS).values
                    data_vals = data_vals + df_sciensano.loc[slice(None), NIS].values

                ax[idx].plot(output['time'].values, model_vals, '--', color='blue')
                ax[idx].scatter(df_sciensano.index.get_level_values('date').unique(), data_vals, color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
                ax[idx].set_title(title_list_prov[idx])
                ax[idx].set_xlim([start_calibration, end_calibration])
                ax[idx].set_ylim([0, 150])
                ax[idx].grid(False)
                ax[idx].set_ylabel('$H_{in}$ (-)')
                ax[idx] = _apply_tick_locator(ax[idx])
        
        elif desired_agg == 'reg':
            fig,ax=plt.subplots(nrows=len(title_list_reg),ncols=1, figsize=(12,12), sharex=True)
            for idx,NIS_list_reg in enumerate(agg_prov_reg):
                model_vals_list=[]
                data_vals_list=[]
                for jdx,NIS_prov in enumerate(NIS_list_reg):
                    model_vals = 0
                    data_vals= 0
                    for NIS in agg_arr_prov[idx][jdx]:
                        model_vals += output['H_in'].sel(NIS=NIS).values
                        data_vals += df_sciensano.loc[slice(None), NIS].values
                    model_vals_list.append(model_vals)
                    data_vals_list.append(data_vals)
                # Sum
                model_vals=0
                data_vals=0
                for i, mat in enumerate(model_vals_list):
                    model_vals += mat
                    data_vals += data_vals_list[i]
                # Plot
                ax[idx].plot(output['time'].values, model_vals, '--', color='blue')
                ax[idx].scatter(df_sciensano.index.get_level_values('date').unique(), data_vals, color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
                ax[idx].set_title(title_list_reg[idx])
                ax[idx].set_xlim([start_calibration, end_calibration])
                ax[idx].set_ylim([0, 380])
                ax[idx].grid(False)
                ax[idx].set_ylabel('$H_{in}$ (-)')
                ax[idx] = _apply_tick_locator(ax[idx])
    return ax