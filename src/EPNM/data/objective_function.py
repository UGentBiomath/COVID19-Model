import numpy as np
import xarray as xr
from EPNM.data.utils import get_sectoral_conversion_matrix, get_sector_labels

# Aggregation function
def aggregate_NACE21(simulation_in):
    """ A function to convert a simulation of the economic IO model on the NACE64 level to the NACE21 level
        Also aggregates data to quarters temporily
    
    Input
    =====
    simulation_in: xarray.DataArray
        Simulation result (NACE64 level). Obtained from a pySODM xarray.Dataset simulation result by using: xarray.Dataset[state_name]
    
    Output
    ======
    simulation_out: xarray.DataArray
        Simulation result (NACE21 level)
    """

    simulation_out = xr.DataArray(np.matmul(np.matmul(simulation_in.values, np.transpose(get_sectoral_conversion_matrix('NACE64_NACE38'))), np.transpose(get_sectoral_conversion_matrix('NACE38_NACE21'))),
                                    dims = ['date', 'NACE21'],
                                    coords = dict(NACE21=(['NACE21'], get_sector_labels('NACE21')),
                                    date=simulation_in.coords['date']))
    return simulation_out

# function to compute AAD
def compute_AAD(out, params, data_B2B, data_GDP, data_revenue, data_employment, weighted=True):
    """Computes the Average Absolute Deviation between model prediction and data
    """

    # Pre-allocate metric
    hyperdist_abs = []
    hyperdist = []

    # B2B Weighted Euclidian distance
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sectors = data_B2B.index.get_level_values('NACE21').unique()
    dates = data_B2B.index.get_level_values('date').unique()
    out_NACE21 = aggregate_NACE21(out['O'])
    out_NACE21_quart = out_NACE21.resample(date='Q').mean()
    B2B_demand = np.matmul(params['O_j'], np.transpose(get_sectoral_conversion_matrix('NACE64_NACE21')))
    dist_abs=np.zeros(4)
    dist=np.zeros(4)
    for i,date in enumerate(dates):
        dist_abs_temp=[]
        dist_temp=[]
        for j,sector in enumerate(sectors):
            if sector!='U':
                x=data_B2B.loc[date, sector]*100-100
                y=out_NACE21_quart.sel(NACE21=sector).sel(date=date)/out_NACE21.sel(NACE21=sector).isel(date=0)*100-100
                # Weighted euclidian distance in plane
                if weighted==True:
                    dist_abs_temp.append(B2B_demand[j]/sum(B2B_demand)*abs(abs(x)-abs(y.values)))
                    dist_temp.append(B2B_demand[j]/sum(B2B_demand)*(abs(x)-abs(y.values)))
                else:
                    dist_abs_temp.append(abs(abs(x)-abs(y.values)))
                    dist_temp.append((abs(x)-abs(y.values)))
        
        if weighted == True:
            dist_abs[i] = np.sum(dist_abs_temp)
            dist[i] = np.sum(dist_temp)
        else:
            dist_abs[i] = np.mean(dist_abs_temp)
            dist[i] = np.mean(dist_temp)

    hyperdist_abs.append(np.mean(dist_abs))
    hyperdist.append(np.mean(dist))

    # GDP, revenue, employment
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    states = ['x', 'x', 'l']
    sizes = [params['x_0'], params['x_0'], params['l_0']]

    dist_abs=np.zeros(4)
    dist=np.zeros(4)
    for k, data in enumerate([data_GDP, data_revenue, data_employment]):
        dates = data.index.get_level_values('date').unique()
        sectors = data.index.get_level_values('NACE64').unique()
        out_quart = out[states[k]].resample(date='Q').mean()
        for i,date in enumerate(dates):
            cumsize=[]
            dist_abs_temp=[]
            dist_temp=[]
            for j,sector in enumerate(sectors):
                if sector != 'BE':
                    x=data.loc[date, sector]*100-100
                    y=out_quart.sel(NACE64=sector).sel(date=date)/out[states[k]].sel(NACE64=sector).isel(date=0)*100-100
                    # Weighted euclidian distance in plane
                    if weighted == True:
                        dist_abs_temp.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k])*abs(abs(x)-abs(y.values)))
                        dist_temp.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k])*(abs(x)-abs(y.values)))
                        cumsize.append(sizes[k][get_sector_labels('NACE64').index(sector)]/sum(sizes[k]))
                    else:
                        dist_abs_temp.append(abs(abs(x)-abs(y.values)))
                        dist_temp.append((abs(x)-abs(y.values)))

            # Weighted euclidian distance in plane
            x=data.loc[date, 'BE']*100-100
            y=out_quart.sum(dim='NACE64').sel(date=date)/out[states[k]].sum(dim='NACE64').isel(date=0)*100-100
            dist_abs_temp.append(abs(abs(x)-abs(y.values)))
            dist_temp.append((abs(x)-abs(y.values)))
            # Average
            if weighted==True:
                dist_abs[i] = 1/(1+sum(cumsize))*np.sum(dist_abs_temp)
                dist[i] = 1/(1+sum(cumsize))*np.sum(dist_temp)
            else:
                dist_abs[i] = np.mean(dist_abs_temp)
                dist[i] = np.mean(dist_temp)

        hyperdist_abs.append(np.mean(dist_abs))
        hyperdist.append(np.mean(dist))

    return np.mean(hyperdist_abs), np.mean(hyperdist)