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
    return x_t

def household_preference_shock(epsilon_D, theta_0):
    """
    A function to return the preference of households for the output of a certain sector

    Parameters
    ----------
    epsilon_D : np.array
        sectoral household demand shock
    theta_0 : int
        household preference under business-as-usual (absence of shock epsilon)

    Returns
    -------
    theta : np.array
        household consumption preference vector
    """

    theta=np.zeros(epsilon_D.shape[0])
    for i in range(epsilon_D.shape[0]):
        theta[i] = (1-epsilon_D[i])*theta_0[i]/(sum((1-epsilon_D)*theta_0))
    return theta

def aggregate_demand_shock(epsilon_D,theta_0,delta_S,rho):
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
    return delta_S*(1-sum((1-epsilon_D)*theta_0))*(1-rho)

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
    return np.sum(O,axis=1) + c_t + f_t

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
    l_0=np.expand_dims(l_0,axis=1)
    epsilon_S=np.expand_dims(epsilon_S,axis=1)
    # Labor force reduction due to lockdown
    l_new[np.greater(l_new,(1-epsilon_S)*l_0)] =  ((1-epsilon_S)*l_0)[np.greater(l_new,(1-epsilon_S)*l_0)]
    return l_new[:,0]

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