import numpy as np
import pandas as pd
from datetime import timedelta

####################
## Economic model ##
####################

def expand_shock(shock_in, convmat):
    return (np.transpose(convmat) @ np.expand_dims(shock_in, axis=1)).squeeze()

def household_demand_shock(t, states, param, l1, l2, t_start_lockdown_1, t_end_lockdown_1, t_start_lockdown_2, t_end_lockdown_2, t_start_final_relax, c_s, ratio_c_s, on_site):
    """
    A time-dependent function to return the household demand shock.

    Parameters
    ----------
    t : pd.timestamp
        current date
    param: np.array
        initialised value of epsilon_S
    states : dict
        Dictionary containing all states of the economic model
    l1 : float
        length of ramp in
    l2 : float
        length of ramp out  
    t_start_lockdown_1 : pd.Timestamp
        start of first COVID-19 lockdown
    t_end_lockdown_1: pd.Timestamp
        end of first COVID-19 lockdown
    t_start_lockdown_2 : pd.Timestamp
        start of first COVID-19 lockdown
    t_end_lockdown_2: pd.Timestamp
        end of first COVID-19 lockdown
    t_start_final_relax : pd.Timestamp
        start of COVID-19 lockdown relaxation    
    c_s : np.array
        consumer demand shock vector during COVID-19 lockdowns
    ratio_c_s: float
        relative size of consumer demand shock vectors between lockdowns 
    on_site : np.array
        vector containing 1 if sector output is consumed on-site and 0 if sector output is not consumed on-site

    Returns
    -------
    epsilon_D : np.array
        sectoral household demand shock
    """

    # Consumer demand shock during lockdown
    c_s_1 = c_s
    # Consumer demand summer 2020 / winter 2021
    c_s_2 = ratio_c_s*c_s_1

    # Before first lockdown
    if t < t_start_lockdown_1:
        return np.zeros(len(c_s_1))
    # First lockdown    
    elif ((t >= t_start_lockdown_1) & (t < t_start_lockdown_1 + pd.Timedelta(days=l1))):
        return ramp_datetime(np.zeros(len(c_s_1)), c_s_1, t, t_start_lockdown_1, l1)
    elif ((t >= t_start_lockdown_1 + pd.Timedelta(days=l1)) & (t < t_end_lockdown_1)):
        return c_s_1
    # First lockdown relaxation
    elif ((t >= t_end_lockdown_1) & (t < t_end_lockdown_1 + pd.Timedelta(days=l2))):
        epsilon = c_s_2 + (c_s_1-c_s_2)/np.log(100)*np.log(100 - 99*(t-t_end_lockdown_1)/(t_end_lockdown_1+pd.Timedelta(days=l2)-t_end_lockdown_1))
        epsilon[np.where(on_site == 0)] = ramp_datetime(c_s_1, c_s_2, t, t_end_lockdown_1, l2)[np.where(on_site == 0)]
        return epsilon
    # Summer 2020
    elif ((t >= t_end_lockdown_1 + pd.Timedelta(days=l2)) & (t < t_start_lockdown_2)):
        return c_s_2
    # Second lockdown
    elif ((t >= t_start_lockdown_2) & (t < t_start_lockdown_2 + pd.Timedelta(days=l1))):
        return ramp_datetime(c_s_2, c_s_1, t, t_start_lockdown_2, l1)
    elif ((t >= t_start_lockdown_2 + pd.Timedelta(days=l1)) & (t < t_end_lockdown_2)):
        return c_s_1
    # Second lockdown relaxation
    elif ((t >= t_end_lockdown_2) & (t < t_end_lockdown_2 + pd.Timedelta(days=l2))):
        epsilon = c_s_2 + (c_s_1-c_s_2)/np.log(100)*np.log(100 - 99*(t-t_end_lockdown_2)/(t_end_lockdown_2 + pd.Timedelta(days=l2)- t_end_lockdown_2))
        epsilon[np.where(on_site == 0)] = ramp_datetime(c_s_1, c_s_2, t, t_end_lockdown_2, l2)[np.where(on_site == 0)]
        return epsilon
    # Plateau winter
    elif ((t >= t_end_lockdown_2 + pd.Timedelta(days=l2)) & (t < t_start_final_relax)):
        return c_s_2
    # Final relaxation
    elif ((t >= t_start_final_relax) & (t < t_start_final_relax + pd.Timedelta(days=l2))):
        epsilon = c_s_2/np.log(100)*np.log(100 - 99*(t-t_start_final_relax)/(t_start_final_relax + pd.Timedelta(days=l2) - t_start_final_relax))
        epsilon[np.where(on_site == 0)] = ramp_datetime(c_s_2, np.zeros(len(c_s_1)), t, t_start_final_relax, l2)[np.where(on_site == 0)]
        return epsilon
    else:
        return np.zeros(len(c_s_1))

def labor_supply_shock(t, states, param, l1, l2, t_start_lockdown_1, t_end_lockdown_1, t_start_lockdown_2, t_end_lockdown_2, l_s_1, l_s_2):
    """
    A function returning the labor reduction due to lockdown measures during the first COVID-19 lockdown.

    Parameters
    ----------
    t : pd.timestamp
        current date
    param: np.array
        initialised value of epsilon_S
    states : dict
        Dictionary containing all states of the economic model
    l1 : float
        length of ramp in
    l2 : float
        length of ramp out  
    t_start_lockdown_1 : pd.Timestamp
        start of first COVID-19 lockdown
    t_end_lockdown_1: pd.Timestamp
        end of first COVID-19 lockdown
    t_start_lockdown_2 : pd.Timestamp
        start of first COVID-19 lockdown
    t_end_lockdown_2: pd.Timestamp
        end of first COVID-19 lockdown
    l_s : np.array
        number of unactive workers under lockdown measures (obtained from survey 25-04-2020)
   
    Returns
    -------
    epsilon_S : np.array
        reduction in labor force
        
    """

    # Before first lockdown
    if t < t_start_lockdown_1:
        return param
    # First lockdown
    elif ((t >= t_start_lockdown_1) & (t < t_start_lockdown_1 + pd.Timedelta(days=l1))):
        return ramp_datetime(param, l_s_1, t, t_start_lockdown_1, l1)
    elif ((t >= t_start_lockdown_1 + pd.Timedelta(days=l1)) & (t < t_end_lockdown_1)):
        return l_s_1
    elif ((t >= t_end_lockdown_1) & (t < t_end_lockdown_1 + pd.Timedelta(days=l2))):
        return ramp_datetime(l_s_1, param, t, t_end_lockdown_1, l2)
    # Second lockdown
    elif ((t >= t_start_lockdown_2) & (t < t_start_lockdown_2 + pd.Timedelta(days=l1))):
        return ramp_datetime(param, l_s_2, t, t_start_lockdown_2, l1)
    elif ((t >= t_start_lockdown_2 + pd.Timedelta(days=l1)) & (t < t_end_lockdown_2)):
        return l_s_2
    elif ((t >= t_end_lockdown_2) & (t < t_end_lockdown_2 + pd.Timedelta(days=l2))):
        return ramp_datetime(l_s_2, param, t, t_end_lockdown_2, l2)
    else:
        return param

def other_demand_shock(t, states, param, l1, l2, t_start_lockdown_1, t_end_lockdown_1, t_start_lockdown_2, t_end_lockdown_2, t_start_final_relax, ratio_c_s, on_site,
                        f_gov, f_inv, f_exp_goods, f_exp_services,                     # fraction of total demand 
                        c_s, f_s_inv, f_s_exp_goods):                                  # maximum shocks to other demand
    """
    A time-dependent function to return the exogeneous demand shock during the 2021-2021 COVID-19 pandemic.

    Parameters
    ----------
    t : pd.timestamp
        current date
    param: np.array
        initialised value of epsilon_S
    states : dict
        Dictionary containing all states of the economic model
    l1 : float
        length of ramp in
    l2 : float
        length of ramp out  
    t_start_lockdown_1 : pd.Timestamp
        start of first COVID-19 lockdown
    t_end_lockdown_1: pd.Timestamp
        end of first COVID-19 lockdown
    t_start_lockdown_2 : pd.Timestamp
        start of first COVID-19 lockdown
    t_end_lockdown_2: pd.Timestamp
        end of first COVID-19 lockdown
    t_start_final_relax : pd.Timestamp
        start of COVID-19 lockdown relaxation    
    ratio_c_s: float
        relative size of consumer demand shock vectors between lockdowns 
    on_site : np.array
        vector containing 1 if sector output is consumed on-site and 0 if sector output is not consumed on-site
    f_gov: np.ndarray
        fraction of other demand by government and non-profits (for every sector i)
    f_inv: np.ndarray
        fraction of other demand attributable to investments (for every sector i)
    f_exp_goods: np.ndarray
        fraction of other demand attributable to exports of goods (for every sector i)
    f_exp_services: np.ndarray
        fraction of other demand attributable to exports of services (for every sector i)
    c_s: np.array
        consumer demand shock vector during COVID-19 lockdowns
    f_s : float
        maximum shock to investments and the exports of goods

    Returns
    -------
    epsilon_F : np.array
        exogeneous demand shock
    """

    t_end_invexp_shock = t_end_lockdown_1 + timedelta(days=l2)

    if  t < t_start_lockdown_1:
        return np.zeros(len(f_gov))
    else:
        # Compute shock to investment and exports of goods
        if t_start_lockdown_1 <= t <= t_start_lockdown_1 + timedelta(days=l1):
            f_s_inv *= ramp_datetime(0, 1, t, t_start_lockdown_1, l1)
            f_s_exp_goods *= ramp_datetime(0, 1, t, t_start_lockdown_1, l1)
        elif t_start_lockdown_1 + timedelta(days=l1) < t <= t_end_invexp_shock:
            f_s_inv *= 1/np.log(100)*np.log(100 - 99*(t-(t_start_lockdown_1+timedelta(days=l1)))/(t_end_invexp_shock-(t_start_lockdown_1+timedelta(days=l1))))
            f_s_exp_goods *= 1/np.log(100)*np.log(100 - 99*(t-(t_start_lockdown_1+timedelta(days=l1)))/(t_end_invexp_shock-(t_start_lockdown_1+timedelta(days=l1))))
        else:
            f_s_inv = f_s_exp_goods = 0
        # Compute the magnitude of the household demand shock
        c_s = household_demand_shock(t, states, param, l1, l2, t_start_lockdown_1, t_end_lockdown_1, t_start_lockdown_2, t_end_lockdown_2,
                                        t_start_final_relax, c_s, ratio_c_s, on_site)
        # Compute total shock
        return f_gov*c_s + f_exp_services*c_s + f_inv*f_s_inv + f_exp_goods*f_s_exp_goods

def compute_income_expectations(t, states, param, l1, t_start_lockdown_1, t_end_lockdown_1, l_0, l_start_lockdown, rho, L):
    """
    A function to return the expected retained income in the long term of households.

    Parameters
    ----------
    t : pd.timestamp
        current date
    states : dict
        Dictionary containing all states of the economic model
    param : float
        current expected fraction of long term income
    l1 : float
        length of ramp in
    t_start_lockdown : pd.timestamp
        startdate of lockdown
    t_end_lockdown : pd.timestamp
        enddate of lockdown
    l_0 : np.array
        sectoral labour expenditure under business-as-usual
    l_start_lockdown : np.array
        sectoral labour expenditure at start of lockdown
    rho : float
        first order economic recovery time constant
    L : float
        fraction of households believing in an L-shaped economic recovery

    Returns
    -------
    zeta : float
        fraction (0-1) of pre-pandemic income households expect to retain in the long run
    """

    if t < t_start_lockdown_1:
        zeta = 1
    else:
        zeta_L = 1 - (sum(l_0)-l_start_lockdown)/sum(l_0)
        if ((t >= t_start_lockdown_1) & (t < t_start_lockdown_1+pd.Timedelta(days=l1))):
            zeta = ramp_datetime(1, zeta_L, t, t_start_lockdown_1, l1)
        elif ((t >= t_start_lockdown_1+pd.Timedelta(days=l1)) & (t < t_end_lockdown_1)):
            zeta = zeta_L
        else:
            # first order system
            zeta = zeta_L + (1 - np.exp(-(1-rho)*(t-t_end_lockdown_1).days))*(1-zeta_L)*L
    return zeta

def government_furloughing(t, states, param, t_start_compensation, t_end_compensation, b_s):
    """
    A function to simulate reimbursement of a fraction b of the income loss by policymakers (f.i. as social benefits, or "tijdelijke werkloosheid")

    Parameters
    ----------
    t : pd.timestamp
        current date
    param: float
        initialised value of b
    t_start_compensation : pd.timestamp
        startdate of compensation
    t_end_lockdown : pd.timestamp
        enddate of compensation
    b_s: float
        fraction of lost labor income furloughed to consumers under 'shock'

    Returns
    -------
    b: float
        fraction of lost labor income compensated
    """
    if t <= t_start_compensation:
        return param
    elif ((t > t_start_compensation) & (t <= t_end_compensation)):
        return b_s
    else:
        return param

######################
## Helper functions ##
######################

def ramp(old, new, t, t_start, l):
    """
    Ramp from old value to new value. 

    Parameters 
    ==========

    old: int/float/np.array
        old value
    new: int/float/np.array
        new value
    t : float
        current date
    t_start: float
        ramp start date
    l : int
        length of ramp
    """

    return old + (new-old)/l * (t-t_start)

def ramp_datetime(old, new, t, t_start, l):
    """
    Ramp from old value to new value. 

    Parameters 
    ==========

    old: int/float/np.array
        old value
    new: int/float/np.array
        new value
    t : timestamp
        current date
    t_start: timestamp
        ramp start date
    l : int
        length of ramp
    """
    return old + (new-old)/l * (t-t_start)/pd.Timedelta('1D')
