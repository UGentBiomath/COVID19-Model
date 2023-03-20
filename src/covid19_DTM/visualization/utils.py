import matplotlib.pyplot as plt
import pandas as pd

# From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
colorscale_okabe_ito = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
                        "green" : "#009E73", "yellow" : "#F0E442",
                        "blue" : "#0072B2", "red" : "#D55E00",
                        "pink" : "#CC79A7", "black" : "#000000"}

def _apply_tick_locator(ax):
    """support function to apply default ticklocator settings"""
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    return ax


def moving_avg(timeseries, days=7, dropna=False, win_type=None, params=None):
    """Takes a centred moving average of a time series over a window with user-defined width and shape. Note: when taking a N-day centred moving average, the first and last N//2 days won't return a value and are effectively lost, BUT they are returned as nans. Also note that the moving average applied over missing data will result in an effectively larger window - try to make sure there is no missing data in the range the data is averaged.
    
    Parameters
    ----------
    timeseries : pandas.DataFrame
        Pandas DataFrame with chronologically ordered datetime objects as indices and a single column with float values
    days : int
        Width of the averaging window in days. Resulting value is recorded in the centre of the window. 7 days by default. If the number is uneven, the window is asymmetric toward the past.
    dropnan : boolean
        If True: only return dates that have an averaged value - i.e. drop some values at the start and the end.
    win_type : str
        Type of window, determining the weight of the values in the window. Choice between 'boxcar', 'triang', 'blackman', 'hamming', 'bartlett', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann', 'kaiser' (needs parameter: beta), 'gaussian' (needs parameter: std), 'general_gaussian' (needs parameters: power, width), 'slepian' (needs parameter: width), 'exponential' (needs parameter: tau). If win_type=None (default) all points are evenly weighted.
    params : float or list of floats
        Parameters values used in some window types. 'kaiser': beta, 'gaussian': std, 'general_gaussian': [power, width], 'slepian': width, 'exponential': tau
        
    Returns
    ------
    timeseries_avg : pandas.DataFrame
        Pandas DataFrame with chronologically ordered datetime objects as indices and single column with moving-averaged float values
        
    TO DO
    -----
    Different win_types are not yet supported, because combining a non-integer window (such as '7D') cannot be combined with non-equal weighing. This can be worked around by first converting the datetime indices to regular indices, saving the datetime indices in a list, then applying centred moving average, and linking the resulting list back to the original dates.
    """
    col_name = timeseries.columns[0]
    ts_temp = pd.DataFrame(data=timeseries.values.copy())
    if win_type in [None, 'boxcar', 'triang', 'blackman', 'hamming', 'bartlett', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann']:
        ts_temp = ts_temp.rolling(window=days, center=True, win_type=win_type).mean()
    params = [params]
    if win_type == 'kaiser':
        ts_temp = ts_temp.rolling(window=days, center=True, win_type=win_type).mean(beta=params[0])
    if win_type == 'gaussian':
        ts_temp = ts_temp.rolling(window=days, center=True, win_type=win_type).mean(std=params[0])
    if win_type == 'general_gaussian':
        ts_temp = ts_temp.rolling(window=days, center=True, win_type=win_type).mean(power=params[0], width=params[1])
    if win_type == 'slepian':
        ts_temp = ts_temp.rolling(window=days, center=True, win_type=win_type).mean(width=params[0])
    if win_type == 'exponential':
        ts_temp = ts_temp.rolling(window=days, center=True, win_type=win_type).mean(tau=params[0])
    
    timeseries_avg = pd.DataFrame(data=ts_temp.values.copy(), index=timeseries.index.copy(), columns=[col_name])
    if dropna:
        timeseries_avg.dropna(inplace=True)
    return timeseries_avg