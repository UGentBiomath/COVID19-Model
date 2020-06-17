import matplotlib.pyplot as plt
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
    if "stratification" in data.dims:
        data = data.sum(dim="stratification")

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
    legend_labels = ['hospitalised','ICU','dead']
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
    if "stratification" in data.dims:
        data = data.sum(dim="stratification")

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