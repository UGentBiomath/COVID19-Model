import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # subfigure structuring
from mpl_toolkits.axes_grid1 import make_axes_locatable # for plot aesthetics
import matplotlib.colors as colors
from .utils import colorscale_okabe_ito
from .utils import _apply_tick_locator

def population_status(data, filename=None, *, ax=None, **kwargs):
    """Plot evolution of the population as function of time

    The function is currently tailor-made for the COVID19_SEIRD model,
    (but could be more generalized in the future). As such, it requires
    the following variable names from the COVID19_SEIRD model:
    S, E, R, C, Cicurec, I, ICU, A, M.

    # TODO - should work with MC-based version as well, using either percentiles or plotting every iteration as one line

    Parameters
    ----------
    data : xarray DataSet
        Model output
    ax : matplotlib.axes.Axes, optional
        If provided, will use the axis to add the lines.
    **kwargs :
        Keyword arguments are passed to xarray/matplotlib line function

    Returns
    -------
    lines : list of 4 Line Artists
        Each of the lines can be furhter customized by the user
    """
    # check the required variable names to let function work
    required_var_names = ["S", "E", "R",
                          "C", "C_icurec",
                          "I", "ICU", "A", "M"]
    if not (set(required_var_names)).issubset(set(data.variables.keys())):
        raise Exception(f"population_status plot function"
                        f"requires the variables {required_var_names}")

    # create the 'combined variables'
    # TODO abstract away 'observed' or 'states of interest' to higher level
    data["C_total"] = data["C"] + data["C_icurec"]
    data["I_total"] = (data["ICU"] + data["I"] +
                       data["A"] + data["M"] + data["C_total"])

    # stratified models are summarized over stratification dimension
    if "Nc" in data.dims:
        data = data.sum(dim="Nc")

    # check if ax object is provided by user
    if ax is None:
        fig, ax = plt.subplots()

    # create plot using xarray interface
    data2plot = data[["S", "E", "I_total", "R"]].to_array(dim="states")
    lines = data2plot.plot.line(x='time', hue="states", ax=ax, **kwargs)
    ax.set_xlabel('days')
    ax.set_ylabel('number of patients')

    # use custom defined colors
    colors = ["black", "orange", "red", "green"]
    for color, line in zip(colors, lines):
        line.set_color(colorscale_okabe_ito[color])

    # add custom legend
    ax.legend(('susceptible', 'exposed',
               'total infected', 'immune'),
              loc="upper left", bbox_to_anchor=(1,1))

    # limit the number of ticks on the axis
    ax = _apply_tick_locator(ax)

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')

    return lines


def infected(data, asymptomatic=False, mild=False, filename=None, *, ax=None, **kwargs):
    """Plot evolution of the infected people as function of time

    The function is currently tailor-made for the COVID19_SEIRD model,
    (but could be more generalized in the future). As such, it requires
    the following variable names from the COVID19_SEIRD model:
    C, Cicurec, ICU, D, and optionally A and/or M.

    # TODO - should work with MC-based version as well, using percentiles

    Parameters
    ----------
    data : xarray DataSet
        Model output
    ax : matplotlib.axes.Axes, optional
        If provided, will use the axis to add the lines.
    **kwargs :
        Keyword arguments are passed to xarray/matplotlib line function

    Returns
    -------
    lines : list of  Line Artists
        Each of the lines can be further customized by the user
    """

    required_var_names = ["C", "C_icurec",
                          "ICU", "D"]
    variables = ["H", "ICU", "D"]
    legend_labels = ['hospitalised','ICU','deceased']
    colors = ["orange", "red", "black"]

    if mild:
        required_var_names.append("M")
        variables.insert(0, "M")
        legend_labels.insert(0, 'mild')
        colors.insert(0, "green")

    if asymptomatic:
        required_var_names.append("A")
        variables.insert(0, "A")
        legend_labels.insert(0, 'asymptomatic')
        colors.insert(0, "blue")

    # check the required variable names to let function work
    if not (set(required_var_names)).issubset(set(data.variables.keys())):
        raise Exception(f"infected plot function"
                        f"requires the variables {required_var_names}")

    # create the 'combined variables'
    # TODO abstract away 'observed' or 'states of interest' to higher level
    data["C_total"] = data["C"] + data["C_icurec"]
    data["H"] = data["C_total"] + data["ICU"]

    # stratified models are summarized over stratification dimension
    if "Nc" in data.dims:
        data = data.sum(dim="Nc")

    # check if ax object is provided by user
    if ax is None:
        fig, ax = plt.subplots()

    # create plot using xarray interface
    data2plot = data[variables].to_array(dim="states")
    lines = data2plot.plot.line(x="time", hue="states", ax=ax, **kwargs)
    ax.set_xlabel('days')
    ax.set_ylabel('number of patients')

    for color, line in zip(colors, lines):
        line.set_color(colorscale_okabe_ito[color])

    ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1,1))

    # limit the number of ticks on the axis
    ax = _apply_tick_locator(ax)

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')

    return lines


def show_map(data, geo, ts_geo='E', day=0, lin=False, rel=False, cmap='Oranges', nis=None,
                 ts_graph=['E', 'H_in', 'ICU', 'D'], figname=None, dpi=200, verbose=False):
    """Plot a snapshot of the median evolution of the infection on a Geopandas map
    
    Parameters
    ----------
    data : xarray DataSet
        Model output. 'place' dimension values are integers.
    geo : GeoDataFrame
        Geopandas dataframe from Belgian shapefiles whose entries correspond to the values in the model output's 'place' dimension. NISCode values are strings.
    ts_geo : string
        The SEIR compartment time series that is plotted into the color map on the chosen day. Either S, E (default), I, A, M, ER, C, C_icurec, ICU, R, D, H_in, H_out, or H_tot.
    day : int
        The simulated day that is to be plotted in the color map. Iterate over this value to create an animation. Default is start date.
    lin : Boolean
        Plots a linear representation of the values in the map if True. Otherwise the values are shown with a symlog representation (default).
    rel : Boolean
        Plot the values relative to the population if True. Default is False: absolute numbers.
    cmap : string
        The color used in the color map of the Geopandas plot. Default is 'Oranges'. Best to use sequential coloring.
    nis : int or array of int
        NIS value or array of NIS values of the region(s) whose time series will appear in the graphs. Maximum number of graphs per plot is 6, as showing too many will make the graph illegible. Default is None: sum of all regions.
    ts_graph : string or list of strings or None
        Choose which time series are to be shown in the graphs on the side of the map. List contains one or more of S, E (default), I, A, M, ER, C, C_icurec, ICU, R, D, H_in, H_out, or H_tot. Default: ['E', 'H_in', 'ICU', 'D']. Maximal length is 5. If None is chosen only the geopandas map is shown
    figname : string
        Directory and name for the figure when saved. Include data extension type (e.g. .jpg). Default is None (image is not saved). Make sure to add an iterator name when iterating over days, in order not to overwrite images.
    dpi : int
        Resolution of the saved image. Only applicable if figname is not None.
    verbose : Boolean
        Print progress and additional information. Default is False.
        
    Returns
    -------
    maps : AxesSubplot
        Plot of the geopandas color map. The user can customize this map further.
    graphs : array of AxesSubplot objects (optional, only if ts_graphs != None)
        Plots of the entire time series
        
    TO DO
    -----
    Implement rel=True for relative values (import population per NIS value, change vmax value and method to show the sum of results)
    """
    ####################
    # CHECK PARAMETERS #
    ####################
    
    # Raise exceptions in case of wrong input
    # Verify that NIS values correspond
    nis_geo = geo.NISCode.values.astype(int)
    nis_data = data.place.values.astype(int)
    if set(nis_geo) != set(nis_data):
        raise Exception(f"The NIS values in the model output and the geopandas dataframe do not correspond (function parameters 'data' and 'geo')")
        
    # Check whether the model output has recognisable compartments
    full_comp_set = {'S', 'E', 'I', 'A', 'M', 'ER', 'C', 'C_icurec', 'ICU', 'R', 'D', 'H_in', 'H_out', 'H_tot'}
    data_comp_set = set(data.data_vars)
    if not data_comp_set.issubset(full_comp_set):
        diff = data_comp_set.difference(full_comp_set)
        raise Exception(f"Unrecognised compartments in data: {diff}")
    if ts_geo not in data_comp_set:
        raise Exception(f"'{ts_geo}' is not an acceptable compartment time series to plot in a map. Choose a ts_geo value from {data_comp_set}")
    if verbose:
        print(f"Compartments whose time series are included in the data: {data_comp_set}")
    
    # Check if the day is not out of reach
    maxday = data.time.values[-1]
    if (day < 0) or (day > maxday) or (type(day) != int):
        raise Exception(f"Choose an integer value day between (and including) 0 and {maxday}")
    if verbose:
        print(f"Working on day {day} of {maxday}")
    
    # Check if the chosen nis value (if any) is legitimate
    if nis:
        if type(nis) != list:
            nis = [nis]
        for value in nis:
            if type(value) != int:
                raise Exception(f"The chosen NIS value(s) must be (an) integer(s)")
        if not set(nis).issubset(nis_geo):
            raise Exception(f"The chosen NIS value(s) to plot in the graphs do(es) not correspond to data (parameter 'nis')")
        if len(nis) > 6:
            raise Exception(f"List of NIS values is longer than 6; showing more than 6 graphs per plot turns it illegible")
    
    # Check if the list of compartment whose time series are to be plotted, is in the correct format
    if ts_graph:
        if type(ts_graph) != list:
            ts_graph = [ts_graph]
        for ts in ts_graph:
            if type(ts) != str:
                raise Exception(f"Parameter 'ts_graph', the chosen compartment(s) for time series graphing, must be a (list of) string(s) from {data_comp_set}")
            if ts not in data_comp_set:
                raise Exception(f"'{ts}' is not an acceptable compartment time series to graph. Choose for parameter 'ts_graph' a (list of) string value(s) from {data_comp_set}, corresponding to the time series that are to be graphed")
            
    # Check properties of figname and dpi
    if (figname != None) & (type(figname) != str):
        raise Exception(f"Parameter 'figname' should be a string with the appropriate data extension (.jpg, .png, .svg, .pdf, ...)")
    if (type(dpi) != int) or (dpi < 1):
        raise Exception(f"Parameter 'dpi' should be an integer larger than 0.")

    #############################################
    # INITIATE INTERNAL VALUES AND ENVIRONMENTS #
    #############################################
    
    # Define time list from data
    tlist = data['time'].values
    
    # Define colors dictionary for graphs (maximally six)
    if ts_graph:
        color_list_total = ['g', 'orange', 'r', 'k', 'c', 'm']
        color_dict_total = dict(zip(ts_graph, color_list_total[:len(ts_graph)]))
    if nis:
        color_list = ['b', 'g', 'r', 'c', 'm', 'y']
        color_dict = dict(zip(nis, color_list[:len(nis)]))
    
    # Initialise new column in geopandas dataframe containing time series value of the chosen compartment
    geo.loc[:,ts_geo] = 0
    
    # Set minima/maxima dictionary based on median value
    # Note that the age classes are aggregated (this may be extended later)
    if ts_graph:
        vmax_graph = dict({})
        for ts in ts_graph:
            vmax_graph[ts]=0
            if nis: # take highest value of all places
                for nis_value in nis:
                    vmax_temp = data[ts].sum(dim='Nc').sel(place=nis_value).quantile(0.5, dim='draws').values.max()
                    if vmax_temp > vmax_graph[ts]:
                        vmax_graph[ts] = vmax_temp
            else: # Take highest value of the sum of all places (works only for absolute numbers right now!
                vmax_graph[ts] = data[ts].sum(dim='Nc').sum(dim='place').quantile(0.5, dim='draws').values.max()
            if verbose:
                print(f"Maximum value of median draw for timeseries of compartment {ts} (vmax_graph[{ts}]) set to {vmax_graph[ts]}.")
        vmin_graph = 0
    
    vmax_geo = data[ts_geo].sum(dim='Nc').quantile(0.5, dim='draws').values.max()
    vmin_geo = 0
    
    # Initiate plotting environment
    #if we are plotting graphs on the righthand side
    if ts_graph:
        fig = plt.figure(figsize=(15,7))
        gs = fig.add_gridspec(len(ts_graph), 2, width_ratios=[2,1])
        cax_side = 'left'
    #if we are only plotting the map
    else:
        fig = plt.figure(figsize=(15,10))
        gs = fig.add_gridspec(1,1)
        cax_side = 'right'
        
    # Axes for Geopandas
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_axis_off()
    cax = make_axes_locatable(ax0).append_axes(cax_side, size="5%", pad=1) # Legend properties

    # Axes for graphs
    text_size=18
    #ylabel_pos = (-0.12,0.5)
    yscale_graph = 'linear'
    if ts_graph:
        pos = 0
        ax_graph_dict = dict({})
        for ts in ts_graph:
            ax_graph_dict[ts] = fig.add_subplot(gs[pos,1])
            ax_graph_dict[ts].set_ylabel(ts, size=text_size)
            #ax_graph_dict[ts].yaxis.set_label_coords(ylabel_pos[0], ylabel_pos[1])
            ax_graph_dict[ts].set_yscale(yscale_graph)
            ax_graph_dict[ts].grid(False)
            ax_graph_dict[ts].set_xlim([0,tlist[-1]])
            ax_graph_dict[ts].set_ylim([vmin_graph,1.25*vmax_graph[ts]])
            if pos != (len(ts_graph)-1):
                ax_graph_dict[ts].set_xticks([])
            pos += 1
        ax_graph_dict[ts].set_xlabel('Days since initial exposure',size=text_size)
    
    # Set percentage edges to plot
    upper_pct = 90
    lower_pct = 100 - upper_pct
    
    # Initialise empty graphs list (for return)
    graphs = []
    
    ########################
    # Import and plot data #
    ########################
    
    # Add data to geopandas dataframe
    for nis_value in nis_data:
        ts_median_today = data.sel(place=nis_value, time=day).sum(dim='Nc').quantile(0.5, dim='draws')[ts_geo].values
        geo.loc[geo['NISCode']==str(nis_value), ts_geo] = ts_median_today
    if lin:
        maps = geo.plot(column=ts_geo, ax=ax0, cmap=cmap, legend=True, edgecolor = 'k',
                 vmin=vmin_geo, vmax=vmax_geo, cax=cax, alpha=1)
    else:
        maps = geo.plot(column=ts_geo, ax=ax0, cmap=cmap,legend=True, edgecolor = 'k',
                 norm=colors.SymLogNorm(linthresh=1, vmin=vmin_geo, vmax=vmax_geo), cax=cax, alpha=1)

    # Add (meta)data to graphs
    legend_size = 12
    figtext_pos = (.18, .12)
    if ts_graph:
        for ts in ts_graph:
            # show distinct regions in graphs
            if nis:
                for nis_value in nis:
                    ts_median = data[ts].sel(place=nis_value).sum(dim='Nc').quantile(0.5, dim='draws').values
                    ts_lower = data[ts].sel(place=nis_value).sum(dim='Nc').quantile(lower_pct/100, dim='draws').values
                    ts_upper = data[ts].sel(place=nis_value).sum(dim='Nc').quantile(upper_pct/100, dim='draws').values
                    label = str(nis_value)
                    if len(nis) > 1:
                        ax_graph_dict[ts].plot(tlist, ts_median, color=color_dict[nis_value], alpha=1, linewidth=2,label=label)
                        graph = ax_graph_dict[ts].fill_between(tlist, ts_lower, ts_upper, color=color_dict[nis_value], alpha=0.3)
                        graphs.append(graph)
                    else:
                        ax_graph_dict[ts].plot(tlist, ts_median, color=color_dict_total[ts], alpha=1, linewidth=2,label=label)
                        label2=f"percentile {upper_pct} to {lower_pct}"
                        graph = ax_graph_dict[ts].fill_between(tlist, ts_lower, ts_upper, color=color_dict_total[ts], alpha=0.3,label=label2)
                        graphs.append(graph)
            # Show national sum over all regions in graphs
            else:
                ts_median = data[ts].sum(dim='place').sum(dim='Nc').quantile(0.5, dim='draws').values
                ts_lower = data[ts].sum(dim='place').sum(dim='Nc').quantile(lower_pct/100, dim='draws').values
                ts_upper = data[ts].sum(dim='place').sum(dim='Nc').quantile(upper_pct/100, dim='draws').values
                label1='national'
                label2=f"percentile {lower_pct} to {higher_pct}"
                ax_graph_dict[ts].plot(tlist, ts_median, color=color_dict_total[ts], alpha=1, linewidth=2, label=label1)
                graph = ax_graph_dict[ts].fill_between(tlist, ts_lower, ts_upper, color=color_dict_total[ts], alpha=0.3, label=label2)
                graphs.append(graph)
            if (nis and (len(nis) == 1)) or not nis:
                ax_graph_dict[ts].legend(loc=2, prop={'size':legend_size})
            ax_graph_dict[ts].axvline(day, color='r', linewidth=2, linestyle='--')
        # Show legend in two columns if it becomes too big
        if nis and (len(nis) > 3):
            ax_graph_dict[ts].legend(loc=2, prop={'size':legend_size}, ncol=2)
        else:
            ax_graph_dict[ts].legend(loc=2, prop={'size':legend_size}, ncol=1)
        plt.figtext(figtext_pos[0], figtext_pos[1], f"People in compartment {ts_geo} at day {day}", backgroundcolor='whitesmoke', fontsize=18)
    else:
        plt.figtext(.12, .2, f"People in compartment {ts_geo} at day {day}", backgroundcolor='whitesmoke', fontsize=22)

    # Save figure
    if figname:
        plt.savefig(figname, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure '{figname}'")
        plt.close('all')
        
    # Return
    if ts_graph:
        return maps, graphs
    else:
        return maps