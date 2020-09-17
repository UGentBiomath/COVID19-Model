import os
import numpy as np
import pandas as pd

def calc_labor_restriction(x_0,l_0,l_t):
    """
    A function to compute sector output with the available labor force.

    Parameters
    ----------
    x_0 : np.array
        sector output under business-as-usual (in M€/d)
    l_0 : np.array
        number of employees per sector under business-as-usual
    l_t : np.array
        number of employees at time t

    Returns
    -------
    x_t : np.array
        sector output at time t (in M€/d)
    """
    return (l_t/l_0)*x_0

def calc_input_restriction(S_t,A,C):
    """
    A function to compute sector output under supply bottlenecks.

    Parameters
    ----------
    S_t : np.array
        stock matrix at time t    
    A : np.array
        matrix of technical coefficients
    C : np.array
        matrix of critical inputs

    Returns
    -------
    x_t : np.array
        sector output at time t (in M€/d)
    """

    # Pre-allocate sector output at time t
    x_t = np.zeros(A.shape[0])
    # Loop over all sectors
    for i in range(A.shape[0]):
        x_t[i] = np.nanmin(S_t[np.where(C[i,:] == 1),i]/A[np.where(C[i,:] == 1),i])
        if np.isnan(x_t[i]): # Q for Koen Schoors: sectors with no input dependencies, is this realistic?
            x_t[i]=np.inf
    return np.expand_dims(x_t,axis=1)

def household_demand_shock(t,t_start_lockdown,t_end_lockdown,t_end_pandemic,c_s,on_site):
    """
    A function to return the household demand shock in function of time.

    = time-dependent economic model parameter !

    Parameters
    ----------
    t : int
        current timestep
    t_start_lockdown : int
        time of start lockdown
    t_end_lockdown : int
        time of end lockdown
    t_end_pandemic : int
        expected end of the pandemic
    c_s : np.array
        shock vector
    on_site : np.array
        vector containing 1 if sector output is consumed on-site and 0 if sector output is not consumed on-site

    Returns
    -------
    epsilon : np.array
        sectoral household demand shock
    """

    if t < t_start_lockdown:
        return np.zeros([c_s.shape[0],1])
    elif ((t >= t_start_lockdown) & (t < t_end_lockdown)):
        return c_s
    elif ((t >= t_end_lockdown) & (t < t_end_pandemic)):
        epsilon = c_s/np.log(100)*np.log(100 - 99*t/t_end_pandemic)
        epsilon[np.where(on_site == 0)] = 0
        return epsilon
    else:
        return np.zeros([c_s.shape[0],1])

def household_preference_shock(epsilon, theta_0):
    """
    A function to return the preference of households for the output of a certain sector

    Parameters
    ----------
    epsilon : np.array
        sectoral household demand shock
    theta_0 : int
        household preference under business-as-usual (absence of shock epsilon)

    Returns
    -------
    theta : np.array
        household consumption preference vector
    """

    theta=np.zeros(epsilon.shape[0])
    for i in range(epsilon.shape[0]):
        theta[i] = (1-epsilon[i])*theta_0[i]/(sum((1-epsilon)*theta_0))

    return np.expand_dims(theta, axis=1)

def aggregate_demand_shock(epsilon,theta_0,delta_S,rho):
    """
    A function to return the aggregate household demand shock.

    Parameters
    ----------
    delta_S : float
        savings rate of households (delta_S = 1; households save all money they are not spending due to shock)
    rho : float
        first order recovery time constant

    Returns
    -------
    epsilon_t : np.array
        household consumption preference vector
    """
    return delta_S*(1-sum((1-epsilon)*theta_0))*(1-rho)

def household_income_expectations(t,zeta_previous,t_start_lockdown,t_end_lockdown,l_0,l_start_lockdown,rho,L):
    """
    A function to return the expected retained income in the long term of households.

    Parameters
    ----------
    t : float
        savings rate of households (delta_S = 1; households save all money they are not spending due to shock)
    zeta_previous : float
        value of zeta at previous timestep
    t_start_lockdown : int
        time of start lockdown
    t_end_lockdown : int
        time of end lockdown
    l_0 : np.array
        sectoral labour expenditure
    l_start_lockdown : np.array
        labor costs at start lockdown
    rho : float
        first order recovery time constant
    L : float
        fraction of households believing in an L-shaped economic recovery

    Returns
    -------
    zeta : float
        fraction (0-1) of pre-pandemic income households expect to retain in the long run
    """
    if t < t_start_lockdown:
        zeta = 1
    else:
        zeta_L = 1 - 0.5*(sum(l_0)-l_start_lockdown)/sum(l_0)
        if ((t >= t_start_lockdown) & (t < t_end_lockdown)):
            zeta = zeta_L
        else:
            zeta = 1 - rho + rho*zeta_previous - (1-rho)*(1-zeta_L)*L

    return zeta

def calc_household_demand(c_total_previous,l_t,l_p,epsilon_t,rho,m):
    """
    A function to calculate the total household consumption demand.
    Based on the consumption function of Muellbauer (2020).

    Parameters
    ----------
    c_t_previous : float
        total household consumption demand at time t -1
    l_t : float
        current labor income (per sector)
    l_p : float
        consumer expectations of permanent income (total)
    epsilon_t : float
        aggregate demand shock
    rho : float
        economic recovery speed
    m : float
        share of labor income used to consume final domestic goods

    Returns
    -------
    c_t : float
        total household consumption demand at time t
    """
    return np.exp((rho*np.log(c_total_previous) + 0.5*(1-rho)*np.log(m*sum(l_t)) + 0.5*(1-rho)*np.log(m*l_p) + epsilon_t))

def calc_intermediate_demand(d_previous,S,A,S_0,tau):
    """
    A function to calculate the intermediate demand between sectors (B2B demand).
    = Restocking function (1st order system with time constant tau)

    Parameters
    ----------
    d_previous : np.array
        total demand (per sector) at time t -1
    S : np.array
        stock matrix at time t
    A : np.array
        matrix of technical coefficients
    S_0 : np.array
        desired stock matrix
    tau : int
        restock speed

    Returns
    -------
    O : np.array
        matrix of B2B orders
    """
    O = np.zeros([A.shape[0],A.shape[0]])
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            O[i,j] = A[i,j]*d_previous[j] + (1/tau)*(S_0[i,j] - S[i,j])
    return O

def calc_total_demand(O,c_t,f_t):
    """
    A function to calculate the total demand for the output of every sector

    Parameters
    ----------
    O : np.array
        matrix of B2B orders
    c_t : np.array
        household demand
    f_t : np.array
        other demand

    Returns
    -------
    d_t : np.array
        total demand
    """
    return np.expand_dims(np.sum(O,axis=1),axis=1) + c_t + f_t

def rationing(x_t,d_t,O,c_t,f_t):
    """
    A function to ration the output if output doesn't meet demand.
    No prioritizing between B2B restocking, households and others (government/exports) is performed.

    Parameters
    ----------
    x_t : np.array
        total output of sector i
    d_t : np.array
        total demand for the output of sector i
    O : np.array
        matrix of B2B orders
    c_t : np.array
        total household demand for the output of sector i
    f_t : np.array
        total other demand for the output of sector i

    Returns
    -------
    Z_t : np.array
        fraction r of B2B orders received
    r*c_t : np.array
        fraction r of household demand met
    r*f_t : np.array
        fraction r of other demand met
    """
    r = x_t/d_t
    r[np.where(r > 1)] = 1 # Too much output --> r = 1
    Z_t = np.zeros([O.shape[0],O.shape[0]])
    for i in range(O.shape[0]):
            Z_t[i,:] = O[i,:]*r[i]
    return Z_t,r*c_t,r*f_t

def inventory_updating(S_old,Z_t,x_t,A):
    """
    A function to update the inventory.

    Parameters
    ----------
    S_old : np.array
        Stock matrix at time t
    Z_t : np.array
        Orders received at time t
    x_t : np.array
        Total output produced at time t
    A : np.array
        Matrix of technical coefficients (input need per unit output)

    Returns
    -------
    S_new : np.array
        
    """
    S_new = np.zeros([S_old.shape[0],S_old.shape[0]])
    for i in range(S_old.shape[0]):
        for j in range(S_old.shape[0]):
            S_new[i,j] = S_old[i,j] + Z_t[i,j] - A[i,j]*x_t[j]
    #S_new = np.minimum(S_new,0)
    S_new[np.where(S_new < 0)] = 0
    return S_new

def hiring_firing(l_old, l_0, x_0, x_t_input, x_t_labor, d_t, gamma_F, gamma_H, epsilon_S):
    """
    A function to update the labor income. (= proxy for size of workforce)

    Parameters
    ----------
    l_old : np.array
        labor income at time t
    l_0 : np.array
        labor income during business-as-usual
    x_0 : np.array
        sector output during business-as-usual
    x_t_input : np.array
        maximum output at time t due to supply bottlenecks
    x_t_labor : np.array
        maximum output at time t due to labor reduction
    d_t : np.array
        total demand at time t
    gamma_F : float
        number of days needed to fire a worker
    gamma_H : float
        number of days needed to hire a worker
    epsilon_S : np.array
        Labor supply shock

    Returns
    -------
    l_new : np.array
        labor income at time t + 1
        
    """
    # Normal hiring/firing procedure
    delta_l = (l_0/x_0)*(np.minimum(x_t_input,d_t)-x_t_labor)
    l_new=np.zeros([delta_l.shape[0]])
    for i in range(delta_l.shape[0]):
        if delta_l[i] > 0:
            l_new[i] = l_old[i] + gamma_H*delta_l[i]
        elif delta_l[i] <= 0:
            l_new[i] = l_old[i] + gamma_F*delta_l[i]
    l_new=np.expand_dims(l_new,axis=1)
    # Labor force reduction due to lockdown
    l_new[np.greater(l_new,(1-epsilon_S)*l_0)] =  ((1-epsilon_S)*l_0)[np.greater(l_new,(1-epsilon_S)*l_0)]

    return l_new

def labor_supply_shock(t,t_start_lockdown,t_end_lockdown,l_s):
    """
    A function returning the labor reduction due to lockdown measures. (= time-dependent parameter)

    Parameters
    ----------
    t : int
        current time
    t_start_lockdown : int
        start of economic lockdown
    t_end_lockdown : int
        end of economic lockdown
    l_s : np.array
        number of unactive workers under lockdown measures (obtained from survey 25-04-2020)
   
    Returns
    -------
    epsilon_S : np.array
        reduction in labor force
        
    """
    if t < t_start_lockdown:
        return np.zeros([l_s.shape[0],1])
    elif ((t >= t_start_lockdown) & (t < t_end_lockdown)):
        return l_s
    else:
        return np.zeros([l_s.shape[0],1])

def labor_compensation_intervention(t, t_start_compensation, t_end_compensation, l_t, l_0, b):
    """
    A function to simulate reimbursement of a fraction b of the income loss by policymakers (f.i. as social benefits, or "tijdelijke werkloosheid")

    Parameters
    ----------
    t : int
        current time
    t_start_compensation : int
        start of compensation
    t_end_lockdown : int
        end of compensation
    l_t : np.array
        labor compensation of workers at time t
    l_0 : np.array
        labour compensation under business-as-usual
    b : np.array
        reimbursed fraction of lost income
   
    Returns
    -------
    l_t_star : np.array
        labor compensation of workers after government intervention
        
    """
    if t < t_start_compensation:
        return l_t
    elif ((t >= t_start_compensation) & (t < t_end_compensation)):
        return l_t + b*(l_0-l_t)
    else:
        return l_t


def leontief(x_t_labor, x_t_input, d_t):
    """
    An implementation of the Leontief production function.

    Parameters
    ----------
    x_t_labor : np.array
        sectoral output at time t under labor constraints
    x_t_input : np.array
        sectoral output at time t under input constraints
    d_t : np.array
        total demand at time t

    Returns
    -------
    x_t : np.array
        sector output at time t (in M€)
    """
    return np.amin([x_t_labor, x_t_input, d_t],axis = 0)