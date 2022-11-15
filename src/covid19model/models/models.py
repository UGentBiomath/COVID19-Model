# Copyright (c) 2022 by T.W. Alleman BIOMATH, Ghent University. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numba as nb
import numpy as np
from numba import jit
from scipy.stats import poisson
from pySODM.models.base import BaseModel
from .utils import stratify_beta_density, stratify_beta_regional, read_coordinates_place, construct_coordinates_Nc
from .economic_utils import *
# Register pandas formatters and converters with matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Ignore numba warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


###############
## jit utils ##
###############

@jit(nopython=True)
def vaccination_write_protection_2D(X, X_post_vacc, dX):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X_post_vacc[i,j] < 0:
                dX[i,j] = 0 - X[i,j]
                X_post_vacc[i,j] = 0
    return X_post_vacc, dX

@jit(nopython=True)
def vaccination_write_protection_3D(X, X_post_vacc, dX):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                if X_post_vacc[i,j,k] < 0:
                    dX[i,j,k] = 0 - X[i,j,k]
                    X_post_vacc[i,j,k] = 0
    return X_post_vacc, dX

@jit(nopython=True)
def jit_matmul_1D_2D(a, B):
    """A simple jitted implementation of a 1Dx2D matrix multiplication
    """
    n = B.shape[1]
    f = len(a)
    out = np.zeros(n, np.float64)
    for i in range(n):
            for k in range(f):
                out[i] += a[k]*B[k,i] 
    return out

@jit(nopython=True)
def jit_matmul_2D_1D(A, b):
    """ A simple jitted implementation of a 2D (n,m) with a 1D (m,) matrix multiplication
        Result is a 1D matrix (n,)
    """
    n = A.shape[0]
    f = A.shape[1]
    out = np.zeros(n, np.float64)
    for i in range(n):
            for k in range(f):
                out[i] += A[i, k] * b[k]
    return out

@jit(nopython=True)
def jit_matmul_2D_2D(A, B):
    """A simple jitted implementation of 2Dx2D matrix multiplication
    """
    n = A.shape[0]
    f = A.shape[1]
    m = B.shape[1]
    out = np.zeros((n,m), np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(f):
                out[i, j] += A[i, k] * B[k, j]
    return out

@jit(nopython=True)
def jit_matmul_2D_3D(A,B):
    """ A simple jitted implementation to multiply a 2D matrix of size (n,m) with a 3D matrix (n,m,m)"""
    out = np.zeros(A.shape, np.float64)
    for i in range(A.shape[0]):
        # reduce dimension
        a = A[i,:]
        b = B[i,:,:]
        # determine loop sizes
        n = b.shape[1]
        f = len(a)
        # loop
        for j in range(n):
            for k in range(f):
                out[i,j] += a[k]*b[k,j]
    return out

@jit(nopython=True)
def jit_matmul_3D_2D(A, B):
    """(n,k,m) x (n,m) --> for n: (k,m) x (m,) --> (n,k) """
    out = np.zeros(B.shape, np.float64)
    for idx in range(A.shape[0]):
        A_acc = A[idx,:,:]
        b = B[idx,:]
        n = A_acc.shape[0]
        f = A_acc.shape[1]
        for i in range(n):
                for k in range(f):
                    out[idx, i] += A_acc[i, k] * b[k]
    return out

@jit(nopython=True)
def jit_matmul_klm_m(A,b):
    out = np.zeros((A.shape[:-1]), np.float64)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                out[i,j] += A[i,j,k]*b[k]
    return out

@jit(nopython=True)
def jit_matmul_klmn_n(A,b):
    out = np.zeros((A.shape[:-1]), np.float64)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                for l in range(A.shape[3]):
                    out[i,j,k] += A[i,j,k,l]*b[l]
    return out

@jit(nopython=True)
def jit_main_function_spatial(place, S, beta, Nc, I_dens):
    """
    Function determining the infection pressure in the spatial context.
    
    Input
    -----
    place : np.array for mobility matrix P^gh. Dimensions [G,G]
    S : np.array for susceptible compartment S^g_i. Dimensions [G,N]
    beta : np.array for effective transmission coefficient beta^gh_ij (rescaled local beta with E_inf and E_susc). Dimensions [G,G,N,N]
    Nc : np.array for effective local social contact N^gh_ij. Dimensions [G,G,N,N]
    I_dens : np.array for density of infectious subjects. This is the fraction (A_eff^h_j + I_eff^h_j)/T_eff^h_j. Dimensions [G,N]
    
    Output
    ------
    Sdot: np.array for change of number of susceptibles Sdot^g_i. Dimensions [G,N]
    
    Note
    ----
    index g denotes the province of origin; index h denotes the visited province; index i denotes the subject's age class; index j denote's the contact's age class.
    
    """
    G = S.shape[0]
    N = S.shape[1]
    Sdot = np.zeros((G,N), np.float64)
    
    for i in range(N):
        for g in range(G):
            value = 0
            for h in range(G):
                for j in range(N):
                     value += place[g,h] * S[g,i] * beta[g,h,i,j] * Nc[g,h,i,j] * I_dens[h,j]
            Sdot[g,i] += value
    return Sdot
    

@jit(nopython=True)
def matmul_q_2D(A,B):
    """ A simple jitted implementation to multiply a 2D matrix of size (n,m) with a 3D matrix (m,k,q)
        Implemented as q times the matrix multiplication (n,m) x (m,k)
        Output of size (n,k,q)
    """
    out = np.zeros((A.shape[0],B.shape[1],B.shape[2]), np.float64)
    for q in range(B.shape[2]):
        b = B[:,:,q]
        n = A.shape[0]
        f = A.shape[1]
        m = b.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(f):
                    out[i, j, q] += A[i, k] * b[k, j]
    return out

class simple_stochastic_SIR(BaseModel):
    """
    A minimal example of a SIR compartmental disease model based on stochastic difference equations (SDEs)
    To be simulated with a tau-leaping Gillespie algorithm.

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states
        e.g. {'S': N, 'E': np.ones(n_stratification)} with N being the total population and n_stratifications the number of stratified layers
        initialising zeros is thus not required

        S : susceptible
        I : infectious
        R : removed

    parameters : dictionary
        containing the values of all parameters (both stratified and not)

        Non-stratified parameters
        -------------------------
        beta : probability of infection when encountering an infected person
        gamma : recovery rate (inverse of duration of infectiousness)

        Other parameters
        ----------------
        Nc : contact matrix between all age groups in stratification

    """

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['beta', 'gamma', 'Nc']
    stratification_names = ['age_groups']

    @staticmethod
    def integrate(t, l, S, I, R, beta, gamma, Nc):
        """Basic stochastic SIR model """

        # Needed in order not to generate the same samples in multiprocessing
        np.random.seed()

        # Compute total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        T = S + I + R

        # Define the rates of the transitionings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        rates = [beta*np.matmul(Nc,(S*I)/T),
                 gamma*np.ones(S.size)*I]
        limits = [[S,],[I,]]

        # Solve for number of transitionings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        N=[]
        for idx,rate in enumerate(rates):
            u = np.random.rand(*(rate.shape))
            draw = poisson.ppf(u, l*rate)
            # Apply limits to prevent negative states
            if len(limits[idx]) > 1:
                limit = np.minimum(*(limits[idx]))
            else:
                limit = limits[idx]
            N.append(np.minimum(draw, limit))

        # Update the system
        # ~~~~~~~~~~~~~~~~~

        S_new  = S - N[0]
        I_new =  I + N[0] - N[1]
        R_new  = R + N[1]

        return S_new, I_new, R_new

class simple_multivariant_SIR(BaseModel):
    """
    A minimal example of a SIR compartmental disease model with an implementation of transient multivariant dynamics
    Can be reduced to a single-variant SIR model by setting injection_ratio to zero.

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states
        e.g. {'S': N, 'E': np.ones(n_stratification)} with N being the total population and n_stratifications the number of stratified layers
        initialising zeros is thus not required

        S : susceptible
        I : infectious
        R : removed
        alpha : fraction of alternative COVID-19 variant

    parameters : dictionary
        containing the values of all parameters (both stratified and not)

        Non-stratified parameters
        -------------------------

        beta : probability of infection when encountering an infected person
        gamma : recovery rate (inverse of duration of infectiousness)
        injection_day : day at which injection_ratio of the new strain is introduced in the population
        injection_ratio : initial fraction of alternative variant

        Other parameters
        ----------------
        Nc : contact matrix between all age groups in stratification

    """

    # state variables and parameters
    state_names = ['S', 'I', 'R', 'alpha']
    parameter_names = ['beta', 'gamma', 'injection_day', 'injection_ratio', 'Nc']
    stratification_names = ['age_groups']

    @staticmethod
    def integrate(t, S, I, R, alpha, beta, gamma, injection_day, injection_ratio, K_inf, Nc):
        """Basic SIR model with multivariant capabilities"""

        # calculate total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        T = S + I + R

        # Compute infection pressure (IP) of both variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        IP_old = (1-alpha)*beta*np.matmul(Nc,(I/T))
        IP_new = alpha*K_inf*beta*np.matmul(Nc,(I/T))

        # Compute the  rates of change in every population compartment
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dS = - (IP_old + IP_new)*S
        dI = (IP_old + IP_new)*S - gamma*I
        dR = gamma*I

        # Update fraction of new COVID-19 variant
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if np.all((IP_old == 0)) and np.all((IP_new == 0)):
            dalpha = np.zeros(len(Nc))
        else:
            dalpha = IP_new/(IP_old+IP_new) - alpha

        # On injection_day, inject injection_ratio new strain to alpha
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if (t >= injection_day) & (alpha.sum().sum()==0):
            dalpha += injection_ratio

        return dS, dI, dR, dalpha

class COVID19_SEIQRD_hybrid_vacc(BaseModel):
    """
    The docstring will go here
    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M_R', 'M_H', 'C_R', 'C_D', 'C_icurec', 'ICU_R', 'ICU_D', 'R', 'D', 'M_in', 'H_in','H_tot','Inf_in','Inf_out']
    parameter_names = ['beta', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h', 'Nc']
    parameters_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    stratification_names = ['age_groups','doses']

    # ..transitions/equations
    @staticmethod
    @jit(nopython=True)
    def integrate(t, S, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, Inf_in, Inf_out,
                  beta, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm,  dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc,
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D):
        """
        Biomath extended SEIRD model for COVID-19
        *Deterministic implementation*
        """

        ###################
        ## Format inputs ##
        ###################

        # Remove negative derivatives to ease further computation (jit compatible in 1D but not in 2D!)
        f_VOC[1,:][f_VOC[1,:] < 0] = 0
        # Split derivatives and fraction
        d_VOC = f_VOC[1,:]
        f_VOC = f_VOC[0,:]        
        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)   

        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        # Hospitalization propensity
        h = np.sum(np.outer(h, f_VOC*K_hosp),axis=1)
        h[h > 1] = 1
        # Latent period
        sigma = np.sum(f_VOC*sigma)
        # Vaccination
        e_i = jit_matmul_klm_m(e_i,f_VOC) # Reduces from (n_age, n_doses, n_VOCS) --> (n_age, n_doses)
        e_s = jit_matmul_klm_m(e_s,f_VOC)
        e_h = jit_matmul_klm_m(e_h,f_VOC)
        # Seasonality
        beta *= seasonality

        ####################################################
        ## Expand dims on first stratification axis (age) ##
        ####################################################

        a = np.expand_dims(a, axis=1)
        h = np.expand_dims(h, axis=1)
        c = np.expand_dims(c, axis=1)
        m_C = np.expand_dims(m_C, axis=1)
        m_ICU = np.expand_dims(m_ICU, axis=1)
        dc_R = np.expand_dims(dc_R, axis=1)
        dc_D = np.expand_dims(dc_D, axis=1)
        dICU_R = np.expand_dims(dICU_R, axis=1)
        dICU_D = np.expand_dims(dICU_D, axis=1)
        dICUrec = np.expand_dims(dICUrec, axis=1)
        h_acc = e_h*h

        ############################################
        ## Compute the vaccination transitionings ##
        ############################################

        dS = np.zeros(S.shape, np.float64)
        dR = np.zeros(R.shape, np.float64)

        # 0 --> 1 and  0 --> 2
        # ~~~~~~~~~~~~~~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,0] + R[:,0]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,0]/VE
        f_R = R[:,0]/VE
        # Compute transisitoning in zero syringes
        dS[:,0] = - (N_vacc[:,0] + N_vacc[:,2])*f_S 
        dR[:,0] = - (N_vacc[:,0]+ N_vacc[:,2])*f_R
        # Compute transitioning in one short circuit
        dS[:,1] =  N_vacc[:,0]*f_S # 0 --> 1 dose
        dR[:,1] =  N_vacc[:,0]*f_R 
        # Compute transitioning in two shot circuit
        dS[:,2] =  N_vacc[:,2]*f_S # 0 --> 2 doses
        dR[:,2] =  N_vacc[:,2]*f_R 

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,1] + R[:,1]
        # Compute fraction of VE in states S and R
        f_S = S[:,1]/VE
        f_R = R[:,1]/VE
        # Compute transitioning in partially vaccinated circuit
        dS[:,1] = dS[:,1] - N_vacc[:,1]*f_S
        dR[:,1] = dR[:,1] - N_vacc[:,1]*f_R
        # Compute transitioning in fully vaccinated circuit
        dS[:,2] = dS[:,2] + N_vacc[:,1]*f_S
        dR[:,2] = dR[:,2] + N_vacc[:,1]*f_R

        # 2 --> B
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,2] + R[:,2]
        # Compute fraction of VE in states S and R
        f_S = S[:,2]/VE
        f_R = R[:,2]/VE
        # Compute transitioning in fully vaccinated circuit
        dS[:,2] = dS[:,2] - N_vacc[:,3]*f_S
        dR[:,2] = dR[:,2] - N_vacc[:,3]*f_R
        # Compute transitioning in boosted circuit
        dS[:,3] = dS[:,3] + N_vacc[:,3]*f_S
        dR[:,3] = dR[:,3] + N_vacc[:,3]*f_R

        # Update the S and R state
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        S_post_vacc = S + dS
        R_post_vacc = R + dR

        ################################
        ## calculate total population ##
        ################################

        T = np.expand_dims(np.sum(S_post_vacc + E + I + A + M_R + M_H + C_R + C_D + C_icurec + ICU_R + ICU_D + R_post_vacc, axis=1),axis=1) # sum over doses

        #################################
        ## Compute system of equations ##
        #################################

        # Compute infection pressure (IP) of all variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        IP = np.expand_dims( np.sum( np.outer(beta*s*jit_matmul_2D_1D(Nc, np.sum(((I+A)/T)*e_i, axis=1)), f_VOC*K_inf), axis=1), axis=1)

        # Compute the  rates of change in every population compartment
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dS  = dS - IP*e_s*S_post_vacc
        dE  = IP*e_s*S_post_vacc - (1/sigma)*E
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - (1/da)*A
        dM_R = (1-h_acc)*((1-a)/omega)*I - (1/dm)*M_R
        dM_H = h_acc*((1-a)/omega)*I - (1/dhospital)*M_H
        dC_R = (1/dhospital)*c*(1-m_C)*M_H - (1/dc_R)*C_R
        dC_D = (1/dhospital)*c*m_C*M_H - (1/dc_D)*C_D
        dICU_star_R = (1/dhospital)*(1-c)*(1-m_ICU)*M_H - (1/dICU_R)*ICU_R
        dICU_star_D = (1/dhospital)*(1-c)*m_ICU*M_H - (1/dICU_D)*ICU_D
        dC_icurec = (1/dICU_R)*ICU_R - (1/dICUrec)*C_icurec
        dR  = dR + (1/da)*A + (1/dm)*M_R  + (1/dc_R)*C_R + (1/dICUrec)*C_icurec
        dD  = (1/dICU_D)*ICU_D + (1/dc_D)*C_D

        dM_in = ((1-a)/omega)*I - M_in
        dH_in = (1/dhospital)*M_H - H_in
        dH_tot = (1/dhospital)*M_H - (1/dc_R)*C_R - (1/dc_D)*C_D - (1/dICUrec)*C_icurec - (1/dICU_D)*ICU_D

        dInf_in = IP*e_s*S_post_vacc - Inf_in
        dInf_out = (1/da)*A + (1/dm)*M_R + (1/dc_R)*C_R + (1/dc_D)*C_D + (1/dICUrec)*C_icurec + (1/dICU_D)*ICU_D - Inf_out

        # Waning of natural immunity
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        dS[:,0] = dS[:,0] + zeta*R_post_vacc[:,0] 
        dR[:,0] = dR[:,0] - zeta*R_post_vacc[:,0]

        return (dS, dE, dI, dA, dM_R, dM_H, dC_R, dC_D, dC_icurec, dICU_star_R, dICU_star_D, dR, dD, dM_in, dH_in, dH_tot, dInf_in, dInf_out)


@jit(nopython=True)
def compute_transitionings_national_jit(N, D, l, state, rate):
    T =[]
    size_dummy=np.ones((N,D),np.float64)
    trans_vals=np.zeros((N,D,len(rate)),np.float64)
    for n in range(N):
        for d in range(D):
            # Construct vector of probabilities
            p=np.zeros(len(rate),np.float64)
            for k in range(len(rate)):
                r = size_dummy*rate[k]
                p[k] = 1 - np.exp(-l*r[n,d])
            p = np.append(p, 1-np.sum(p))
            # Draw from multinomial distribution and omit the chance of not transitioning
            trans_vals[n,d,:] = np.random.multinomial(int(state[n,d]), p)[:-1]
    # Assign result to correct transitioning
    for k in range(len(rate)):
        T.append(trans_vals[:,:,k])
    return T

class COVID19_SEIQRD_hybrid_vacc_sto(BaseModel):
    """
    The docstring will go here
    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M_R', 'M_H', 'C_R', 'C_D', 'C_icurec','ICU_R', 'ICU_D', 'R', 'D', 'M_in', 'H_in','H_tot', 'Inf_in', 'Inf_out']
    parameter_names = ['beta', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h','Nc']
    parameters_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    stratification = ['age_groups','doses']

    # ..transitions/equations
    @staticmethod
    def integrate(t, l, S, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, Inf_in, Inf_out,
                  beta, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm,  dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc,
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D):
        """
        Biomath extended SEIRD model for COVID-19
        *Deterministic implementation*
        """

        ###################
        ## Format inputs ##
        ###################

        np.random.seed()
        # Remove negative derivatives to ease further computation (jit compatible in 1D but not in 2D!)
        f_VOC[1,:][f_VOC[1,:] < 0] = 0
        # Split derivatives and fraction
        d_VOC = f_VOC[1,:]
        f_VOC = f_VOC[0,:]        
        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)   
        
        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        # Hospitalization propensity
        h = np.sum(np.outer(h, f_VOC*K_hosp),axis=1)
        h[h > 1] = 1
        # Latent period
        sigma = np.sum(f_VOC*sigma)
        # Vaccination
        e_i = jit_matmul_klm_m(e_i,f_VOC) # Reduces from (n_age, n_doses, n_VOCS) --> (n_age, n_doses)
        e_s = jit_matmul_klm_m(e_s,f_VOC)
        e_h = jit_matmul_klm_m(e_h,f_VOC)
        # Seasonality
        beta *= seasonality

        ####################################################
        ## Expand dims on first stratification axis (age) ##
        ####################################################

        a = np.expand_dims(a, axis=1)
        h = np.expand_dims(h, axis=1)
        c = np.expand_dims(c, axis=1)
        m_C = np.expand_dims(m_C, axis=1)
        m_ICU = np.expand_dims(m_ICU, axis=1)
        dc_R = np.expand_dims(dc_R, axis=1)
        dc_D = np.expand_dims(dc_D, axis=1)
        dICU_R = np.expand_dims(dICU_R, axis=1)
        dICU_D = np.expand_dims(dICU_D, axis=1)
        dICUrec = np.expand_dims(dICUrec, axis=1)
        h_acc = e_h*h

        ############################################
        ## Compute the vaccination transitionings ##
        ############################################

        dS = np.zeros(S.shape, np.float64)
        dR = np.zeros(R.shape, np.float64)

        # Round the vaccination data
        N_vacc = l*N_vacc

        # 0 --> 1 and  0 --> 2
        # ~~~~~~~~~~~~~~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,0] + R[:,0]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,0]/VE
        f_R = R[:,0]/VE
        # Compute transisitoning in zero syringes
        dS[:,0] = - (N_vacc[:,0] + N_vacc[:,2])*f_S 
        dR[:,0] = - (N_vacc[:,0]+ N_vacc[:,2])*f_R
        # Compute transitioning in one short circuit
        dS[:,1] =  N_vacc[:,0]*f_S # 0 --> 1 dose
        dR[:,1] =  N_vacc[:,0]*f_R 
        # Compute transitioning in two shot circuit
        dS[:,2] =  N_vacc[:,2]*f_S # 0 --> 2 doses
        dR[:,2] =  N_vacc[:,2]*f_R 

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,1] + R[:,1]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE in states S and R
        f_S = S[:,1]/VE
        f_R = R[:,1]/VE
        # Compute transitioning in partially vaccinated circuit
        dS[:,1] = dS[:,1] - N_vacc[:,1]*f_S
        dR[:,1] = dR[:,1] - N_vacc[:,1]*f_R
        # Compute transitioning in fully vaccinated circuit
        dS[:,2] = dS[:,2] + N_vacc[:,1]*f_S
        dR[:,2] = dR[:,2] + N_vacc[:,1]*f_R

        # 2 --> B
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,2] + R[:,2]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE in states S and R
        f_S = S[:,2]/VE
        f_R = R[:,2]/VE
        # Compute transitioning in fully vaccinated circuit
        dS[:,2] = dS[:,2] - N_vacc[:,3]*f_S
        dR[:,2] = dR[:,2] - N_vacc[:,3]*f_R
        # Compute transitioning in boosted circuit
        dS[:,3] = dS[:,3] + N_vacc[:,3]*f_S
        dR[:,3] = dR[:,3] + N_vacc[:,3]*f_R

        # Update the S and R state
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        S_post_vacc = np.rint(S + dS)
        R_post_vacc = np.rint(R + dR)

        # Make absolutely sure the vaccinations don't let theses state go below zero
        S_post_vacc = np.where(S_post_vacc < 0, 0, S_post_vacc)
        R_post_vacc = np.where(R_post_vacc < 0, 0, R_post_vacc)

        ################################
        ## calculate total population ##
        ################################

        T = np.expand_dims(np.sum(S_post_vacc + E + I + A + M_R + M_H + C_R + C_D + C_icurec + ICU_R + ICU_D + R_post_vacc, axis=1),axis=1) # sum over doses

        ################################
        ## Compute the transitionings ##
        ################################

        # Compute infection pressure (IP) of all variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        IP = np.expand_dims( np.sum( np.outer(beta*s*jit_matmul_2D_1D(Nc,np.sum(((I+A)/T)*e_i, axis=1)), f_VOC*K_inf), axis=1), axis=1)

        # Define the rates of the transitionings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        states=[S_post_vacc, E, I, A, M_R, M_H, C_R, C_D, ICU_R, ICU_D, C_icurec, R_post_vacc]
        rates=[
            [IP*e_s,], # S
            [1/sigma,], # E
            [a/omega, (1-h_acc)*((1-a)/omega), h_acc*((1-a)/omega)], # I
            [1/da], # A
            [1/dm], # M_R
            [(1/dhospital)*c*(1-m_C), (1/dhospital)*c*m_C, (1/dhospital)*(1-c)*(1-m_ICU), (1/dhospital)*(1-c)*m_ICU], # M_H
            [1/dc_R,], # C_R
            [1/dc_D,], # C_D
            [1/dICU_R,], # ICU_R
            [1/dICU_D], # ICU_D
            [1/dICUrec,], # C_icurec
            [zeta,] # R
        ]

        # 0: S --> E
        # 1: E --> I
        # 2: I --> A
        # 3: I --> M_R
        # 4: I --> M_H
        # 5: A --> R
        # 6: M_R --> R
        # 7: M_H --> C_R
        # 8: M_H --> C_D
        # 9: M_H --> ICU_R
        # 10: M_H --> ICU_D
        # 11: C_R --> R
        # 12: C_D --> D
        # 13: ICU_R --> C_icurec
        # 14: ICU_D --> D
        # 15: C_icurec --> R
        # 16: R --> S

        
        T=[]
        for i, rate in enumerate(rates):
            state = states[i]
            T.extend(compute_transitionings_national_jit(S.shape[0], S.shape[1], l, state, rate))

        # Update the system
        # ~~~~~~~~~~~~~~~~~

        # Flowchart states
        S_new = S_post_vacc - T[0] + T[16]
        E_new = E + T[0] - T[1]
        I_new = I + T[1] - (T[2] + T[3] + T[4])
        A_new = A + T[2] - T[5]
        M_R_new = M_R + T[3] - T[6]
        M_H_new = M_H + T[4] - (T[7] + T[8] + T[9] + T[10])
        C_R_new = C_R + T[7] - T[11]
        C_D_new = C_D + T[8] - T[12]
        ICU_R_new = ICU_R + T[9] - T[13]
        ICU_D_new = ICU_D + T[10] - T[14]
        C_icurec_new = C_icurec + T[13] - T[15]
        R_new = R_post_vacc + T[5] + T[6] + T[11] + T[15] - T[16]
        D_new = D + T[12] + T[14]

        # Derivative states
        M_in_new = (T[3] + T[4])/l
        H_in_new = (T[7] + T[8] + T[9] + T[10])/l
        H_out_new = (T[11] + T[12] + T[14] + T[15])/l
        H_tot_new = H_tot + (H_in_new - H_out_new)*l
        Inf_in_new = T[0]/l
        Inf_out_new = (T[5] + T[6] + T[11] + T[12] + T[14] + T[15])/l

        return (S_new, E_new, I_new, A_new, M_R_new, M_H_new, C_R_new, C_D_new, C_icurec_new, ICU_R_new, ICU_D_new, R_new, D_new, M_in_new, H_in_new, H_tot_new, Inf_in_new, Inf_out_new)


class COVID19_SEIQRD_spatial_hybrid_vacc(BaseModel):

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M_R', 'M_H', 'C_R', 'C_D', 'C_icurec','ICU_R', 'ICU_D', 'R', 'D', 'M_in', 'H_in','H_tot']
    parameter_names = ['beta_R', 'beta_U', 'beta_M', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h', 'Nc', 'Nc_work', 'NIS']
    parameters_stratified_names = [['area', 'p'],['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    stratification_names = ['NIS','age_groups','doses']

    @staticmethod
    @jit(nopython=True)
    def integrate(t, S, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, # time + SEIRD classes
                  beta_R, beta_U, beta_M, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm, dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc, Nc_work, NIS, # SEIRD parameters
                  area, p, # spatially stratified parameters. 
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D): 
        
        ###################
        ## Format inputs ##
        ###################

        # Remove negative derivatives to ease further computation (jit compatible in 1D but not in 2D!)
        f_VOC[1,:][f_VOC[1,:] < 0] = 0
        # Split derivatives and fraction
        d_VOC = f_VOC[1,:]
        f_VOC = f_VOC[0,:]        
        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)   

        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        # Hospitalization propensity
        h = np.sum(np.outer(h, f_VOC*K_hosp),axis=1)
        h[h > 1] = 1
        # Latent period
        sigma = np.sum(f_VOC*sigma)
        # Vaccination
        e_i = jit_matmul_klmn_n(e_i,f_VOC) # Reduces from (n_age, n_doses, n_VOCS) --> (n_age, n_doses)
        e_s = jit_matmul_klmn_n(e_s,f_VOC)
        e_h = jit_matmul_klmn_n(e_h,f_VOC)
        # Seasonality
        beta_R *= seasonality
        beta_U *= seasonality
        beta_M *= seasonality

        ####################################################
        ## Expand dims on first stratification axis (age) ##
        ####################################################

        a = np.expand_dims(a, axis=1)
        h = np.expand_dims(h, axis=1)
        c = np.expand_dims(c, axis=1)
        m_C = np.expand_dims(m_C, axis=1)
        m_ICU = np.expand_dims(m_ICU, axis=1)
        dc_R = np.expand_dims(dc_R, axis=1)
        dc_D = np.expand_dims(dc_D, axis=1)
        dICU_R = np.expand_dims(dICU_R, axis=1)
        dICU_D = np.expand_dims(dICU_D, axis=1)
        dICUrec = np.expand_dims(dICUrec, axis=1)
        h_acc = e_h*h

        ############################################
        ## Compute the vaccination transitionings ##
        ############################################

        dS = np.zeros(S.shape, np.float64)
        dR = np.zeros(R.shape, np.float64)

        # 0 --> 1 and  0 --> 2
        # ~~~~~~~~~~~~~~~~~~~~
        # Compute vaccine eligible population
        VE = S[:,:,0] + R[:,:,0]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,0]/VE
        f_R = R[:,:,0]/VE
        # Compute transisitoning in zero syringes
        dS[:,:,0] = - (N_vacc[:,:,0] + N_vacc[:,:,2])*f_S 
        dR[:,:,0] = - (N_vacc[:,:,0]+ N_vacc[:,:,2])*f_R
        # Compute transitioning in one short circuit
        dS[:,:,1] =  N_vacc[:,:,0]*f_S # 0 --> 1 dose
        dR[:,:,1] =  N_vacc[:,:,0]*f_R 
        # Compute transitioning in two shot circuit
        dS[:,:,2] =  N_vacc[:,:,2]*f_S # 0 --> 2 doses
        dR[:,:,2] =  N_vacc[:,:,2]*f_R 

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,1] + R[:,:,1]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,1]/VE
        f_R = R[:,:,1]/VE
        # Compute transitioning in one short circuit
        dS[:,:,1] = dS[:,:,1] - N_vacc[:,:,1]*f_S
        dR[:,:,1] = dR[:,:,1] - N_vacc[:,:,1]*f_R
        # Compute transitioning in two shot circuit
        dS[:,:,2] = dS[:,:,2] + N_vacc[:,:,1]*f_S
        dR[:,:,2] = dR[:,:,2] + N_vacc[:,:,1]*f_R

        # 2 --> B
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,2] + R[:,:,2]
        # Compute fraction of VE in states S and R
        f_S = S[:,:,2]/VE
        f_R = R[:,:,2]/VE
        # Compute transitioning in fully vaccinated circuit
        dS[:,:,2] = dS[:,:,2] - N_vacc[:,:,3]*f_S
        dR[:,:,2] = dR[:,:,2] - N_vacc[:,:,3]*f_R
        # Compute transitioning in boosted circuit
        dS[:,:,3] = dS[:,:,3] + N_vacc[:,:,3]*f_S
        dR[:,:,3] = dR[:,:,3] + N_vacc[:,:,3]*f_R

        # Update the S and R state
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        S_post_vacc = S + dS
        R_post_vacc = R + dR

        ################################
        ## calculate total population ##
        ################################

        T = np.sum(S_post_vacc + E + I + A + M_R + M_H + C_R + C_D + C_icurec + ICU_R + ICU_D + R_post_vacc, axis=2) # Sum over doses

        ################################
        ## Compute infection pressure ##
        ################################

        # For total population and for the relevant compartments I and A
        G = S.shape[0] # spatial stratification

        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*NIS + np.identity(G)*(NIS @ (1-np.outer(p,p)))
        
        # Expand beta to size G
        beta = stratify_beta_density(beta_R, beta_U, beta_M, area, T.sum(axis=1))*np.sum(f_VOC*K_inf)

        # Compute populations after application of 'place' to obtain the S, I and A populations
        T_work = np.expand_dims(np.transpose(place_eff) @ T, axis=2)
        S_work = matmul_q_2D(np.transpose(place_eff), S_post_vacc)
        I_work = matmul_q_2D(np.transpose(place_eff), I)
        A_work = matmul_q_2D(np.transpose(place_eff), A)
        # The following line of code is the numpy equivalent of the above loop (verified)
        #S_work = np.transpose(np.matmul(np.transpose(S_post_vacc), place_eff))

        # Compute infectious work population (11,10)
        infpop_work = np.sum( (I_work + A_work)/T_work*e_i, axis=2)
        infpop_rest = np.sum( (I + A)/np.expand_dims(T, axis=2)*e_i, axis=2)

        # Multiply with number of contacts
        multip_work = np.expand_dims(jit_matmul_3D_2D(Nc_work, infpop_work), axis=2)
        multip_rest = np.expand_dims(jit_matmul_3D_2D(Nc-Nc_work, infpop_rest), axis=2)

        # Multiply result with beta
        multip_work *= np.expand_dims(np.expand_dims(beta, axis=1), axis=2)
        multip_rest *= np.expand_dims(np.expand_dims(beta, axis=1), axis=2)

        # Compute rates of change
        dS_inf = (S_work * multip_work + S_post_vacc * multip_rest)*e_s

        ############################
        ## Compute system of ODEs ##
        ############################

        dS  = dS - dS_inf
        dE  = dS_inf - (1/sigma)*E
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - (1/da)*A
        dM_R = (1-h_acc)*((1-a)/omega)*I - (1/dm)*M_R
        dM_H = h_acc*((1-a)/omega)*I - (1/dhospital)*M_H
        dC_R = (1/dhospital)*c*(1-m_C)*M_H - (1/dc_R)*C_R
        dC_D = (1/dhospital)*c*m_C*M_H - (1/dc_D)*C_D
        dICU_star_R = (1/dhospital)*(1-c)*(1-m_ICU)*M_H - (1/dICU_R)*ICU_R
        dICU_star_D = (1/dhospital)*(1-c)*m_ICU*M_H - (1/dICU_D)*ICU_D
        dC_icurec = (1/dICU_R)*ICU_R - (1/dICUrec)*C_icurec
        dR  = dR + (1/da)*A + (1/dm)*M_R  + (1/dc_R)*C_R + (1/dICUrec)*C_icurec
        dD  = (1/dICU_D)*ICU_D + (1/dc_D)*C_D

        dM_in = ((1-a)/omega)*I - M_in
        dH_in = (1/dhospital)*M_H - H_in
        dH_tot = (1/dhospital)*M_H - (1/dc_R)*C_R - (1/dc_D)*C_D - (1/dICUrec)*C_icurec - (1/dICU_D)*ICU_D

        ########################
        ## Waning of immunity ##
        ########################

        # Waning of natural immunity
        dS[:,:,0] = dS[:,:,0] + zeta*R_post_vacc[:,:,0] 
        dR[:,:,0] = dR[:,:,0] - zeta*R_post_vacc[:,:,0]      

        return (dS, dE, dI, dA, dM_R, dM_H, dC_R, dC_D, dC_icurec, dICU_star_R, dICU_star_D, dR, dD, dM_in, dH_in, dH_tot)

@jit(nopython=True)
def compute_transitionings_spatial_jit(G, N, D, l, states, rates):

    T=np.zeros((G,N,D,1), np.float64)
    for i in range(len(rates)):
        rate = rates[i]
        state = states[i]
        trans_vals=np.zeros((G,N,D,len(rate)),np.float64)
        for g in range(G):
            for n in range(N):
                for d in range(D):
                    # Construct vector of probabilities
                    p=np.zeros(len(rate),np.float64)
                    for k in range(len(rate)):
                        p_tmp = 1 - np.exp(-l*rate[k][g,n,d])
                        # Sometimes these rates become smaller than zero
                        # This likely has to do with Nc-Nc_work becoming negative
                        if p_tmp < 0:
                            p[k]=0
                        else:
                            p[k]=p_tmp
                    p = np.append(p, 1-np.sum(p))
                    # Draw from multinomial distribution and omit the chance of not transitioning
                    trans_vals[g,n,d,:] = np.random.multinomial(int(state[g,n,d]), p)[:-1]
        # Assign result to correct transitioning
        T = np.append(T, trans_vals, axis=3)
    T = T[:,:,:,1:]
    return T

class COVID19_SEIQRD_spatial_hybrid_vacc_sto(BaseModel):

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M_R', 'M_H', 'C_R', 'C_D', 'C_icurec','ICU_R', 'ICU_D', 'R', 'D', 'M_in', 'H_in','H_tot']
    parameter_names = ['beta_R', 'beta_U', 'beta_M', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h', 'Nc', 'Nc_work', 'NIS']
    parameters_stratified_names = [['area', 'p'],['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    stratification_names = ['NIS','age_groups','doses']

    @staticmethod
    def integrate(t, l, S, E, I, A, M_R, M_H, C_R, C_D, C_icurec, ICU_R, ICU_D, R, D, M_in, H_in, H_tot, # time + SEIRD classes
                  beta_R, beta_U, beta_M, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm, dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc, Nc_work, NIS, # SEIRD parameters
                  area, p, # spatially stratified parameters. 
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D):

        ###################
        ## Format inputs ##
        ###################

        np.random.seed()
        # Remove negative derivatives to ease further computation (jit compatible in 1D but not in 2D!)
        f_VOC[1,:][f_VOC[1,:] < 0] = 0
        # Split derivatives and fraction
        d_VOC = f_VOC[1,:]
        f_VOC = f_VOC[0,:]        
        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)   

        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        # Hospitalization propensity
        h = np.sum(np.outer(h, f_VOC*K_hosp),axis=1)
        h[h > 1] = 1
        # Latent period
        sigma = np.sum(f_VOC*sigma)
        # Vaccination
        e_i = jit_matmul_klmn_n(e_i,f_VOC) # Reduces from (n_NIS, n_age, n_doses, n_VOCS) --> (n_age, n_doses)
        e_s = jit_matmul_klmn_n(e_s,f_VOC)
        e_h = jit_matmul_klmn_n(e_h,f_VOC)
        # Seasonality
        beta_R *= seasonality
        beta_U *= seasonality
        beta_M *= seasonality

        ####################################################
        ## Expand dims on first stratification axis (age) ##
        ####################################################

        a = np.expand_dims(a, axis=1)
        h = np.expand_dims(h, axis=1)
        c = np.expand_dims(c, axis=1)
        m_C = np.expand_dims(m_C, axis=1)
        m_ICU = np.expand_dims(m_ICU, axis=1)
        dc_R = np.expand_dims(dc_R, axis=1)
        dc_D = np.expand_dims(dc_D, axis=1)
        dICU_R = np.expand_dims(dICU_R, axis=1)
        dICU_D = np.expand_dims(dICU_D, axis=1)
        dICUrec = np.expand_dims(dICUrec, axis=1)
        h_acc = e_h*h

        ############################################
        ## Compute the vaccination transitionings ##
        ############################################

        dS = np.zeros(S.shape, np.float64)
        dR = np.zeros(R.shape, np.float64)

        # Round the vaccination data
        N_vacc = l*N_vacc

        # 0 --> 1 and  0 --> 2
        # ~~~~~~~~~~~~~~~~~~~~
        # Compute vaccine eligible population
        VE = S[:,:,0] + R[:,:,0]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,0]/VE
        f_R = R[:,:,0]/VE
        # Compute transisitoning in zero syringes
        dS[:,:,0] = - (N_vacc[:,:,0] + N_vacc[:,:,2])*f_S
        dR[:,:,0] = - (N_vacc[:,:,0]+ N_vacc[:,:,2])*f_R
        # Compute transitioning in one short circuit
        dS[:,:,1] =  N_vacc[:,:,0]*f_S # 0 --> 1 dose
        dR[:,:,1] =  N_vacc[:,:,0]*f_R 
        # Compute transitioning in two shot circuit
        dS[:,:,2] =  N_vacc[:,:,2]*f_S # 0 --> 2 doses
        dR[:,:,2] =  N_vacc[:,:,2]*f_R

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,1] + R[:,:,1]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,1]/VE
        f_R = R[:,:,1]/VE
        # Compute transitioning in one short circuit
        dS[:,:,1] = dS[:,:,1] - N_vacc[:,:,1]*f_S
        dR[:,:,1] = dR[:,:,1] - N_vacc[:,:,1]*f_R
        # Compute transitioning in two shot circuit
        dS[:,:,2] = dS[:,:,2] + N_vacc[:,:,1]*f_S
        dR[:,:,2] = dR[:,:,2] + N_vacc[:,:,1]*f_R


        # 2 --> B
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,2] + R[:,:,2]
        VE = np.where(VE == 0, 1, VE)
        # Compute fraction of VE in states S and R
        f_S = S[:,:,2]/VE
        f_R = R[:,:,2]/VE
        # Compute transitioning in fully vaccinated circuit
        dS[:,:,2] = dS[:,:,2] - N_vacc[:,:,3]*f_S
        dR[:,:,2] = dR[:,:,2] - N_vacc[:,:,3]*f_R
        # Compute transitioning in boosted circuit
        dS[:,:,3] = dS[:,:,3] + N_vacc[:,:,3]*f_S
        dR[:,:,3] = dR[:,:,3] + N_vacc[:,:,3]*f_R

        # Update the S and R state
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        S_post_vacc = np.rint(S + dS)
        R_post_vacc = np.rint(R + dR)

        # Make absolutely sure the vaccinations don't let theses state go below zero
        S_post_vacc = np.where(S_post_vacc < 0, 0, S_post_vacc)
        R_post_vacc = np.where(R_post_vacc < 0, 0, R_post_vacc)

        ################################
        ## calculate total population ##
        ################################

        T = np.sum(S_post_vacc + E + I + A + M_R + M_H + C_R + C_D + C_icurec + ICU_R + ICU_D + R_post_vacc, axis=2) # Sum over doses

        ################################
        ## Compute infection pressure ##
        ################################

        # For total population and for the relevant compartments I and A
        G = S.shape[0] # spatial stratification
        N = S.shape[1] # age stratification

        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*NIS + np.identity(G)*(NIS @ (1-np.outer(p,p)))
        
        # Expand beta to size G
        beta = stratify_beta_density(beta_R, beta_U, beta_M, area, T.sum(axis=1))*np.sum(f_VOC*K_inf)
        #beta = stratify_beta_regional(beta_R, beta_U, beta_M, G)*np.sum(f_VOC*K_inf)

        # Compute populations after application of 'place' to obtain the S, I and A populations
        T_work = np.expand_dims(np.transpose(place_eff) @ T, axis=2)
        S_work = matmul_q_2D(np.transpose(place_eff), S_post_vacc)
        I_work = matmul_q_2D(np.transpose(place_eff), I)
        A_work = matmul_q_2D(np.transpose(place_eff), A)
        # The following line of code is the numpy equivalent of the above loop (verified)
        #S_work = np.transpose(np.matmul(np.transpose(S_post_vacc), place_eff))

        # Compute infectious work population (11,10)
        infpop_work = np.sum( (I_work + A_work)/T_work*e_i, axis=2)
        infpop_rest = np.sum( (I + A)/np.expand_dims(T, axis=2)*e_i, axis=2)

        # Multiply with number of contacts
        multip_work = np.expand_dims(jit_matmul_3D_2D(Nc_work, infpop_work), axis=2)
        multip_rest = np.expand_dims(jit_matmul_3D_2D(Nc-Nc_work, infpop_rest), axis=2)

        # Multiply result with beta
        multip_work *= np.expand_dims(np.expand_dims(beta, axis=1), axis=2)
        multip_rest *= np.expand_dims(np.expand_dims(beta, axis=1), axis=2)

        # Compute rates of change
        dS_inf = (S_work * multip_work + S_post_vacc * multip_rest)*e_s

        ################################
        ## Compute the transitionings ##
        ################################

        # Define the rates of the transitionings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        G = S.shape[0]
        N = S.shape[1]
        D = S.shape[2]
        
        states=[S_post_vacc, E, I, A, M_R, M_H, C_R, C_D, ICU_R, ICU_D, C_icurec, R_post_vacc, S_work]
        # Convert all rate sizes to size (11,10,4) for numba
        # --> TO DO: prettify this
        size_dummy = np.ones([G,N,D], np.float64)
        rates=[
            [multip_rest*e_s,], # S_post_vacc
            [size_dummy*(1/sigma),], # E
            [np.squeeze(a/omega)[np.newaxis, :, np.newaxis]*size_dummy,
                (1-h_acc)*((1-a)/omega),
                h_acc*((1-a)/omega)], # I
            [size_dummy*(1/da),], # A
            [size_dummy*(1/dm),], # M_R
            [np.squeeze((1/dhospital)*c*(1-m_C))[np.newaxis, :, np.newaxis]*size_dummy,
                np.squeeze((1/dhospital)*c*m_C)[np.newaxis, :, np.newaxis]*size_dummy,
                np.squeeze((1/dhospital)*(1-c)*(1-m_ICU))[np.newaxis, :, np.newaxis]*size_dummy,
                np.squeeze((1/dhospital)*(1-c)*m_ICU)[np.newaxis, :, np.newaxis]*size_dummy], # M_H
            [np.squeeze(1/dc_R)[np.newaxis, :, np.newaxis]*size_dummy,], # C_R
            [np.squeeze(1/dc_D)[np.newaxis, :, np.newaxis]*size_dummy,], # C_D
            [np.squeeze(1/dICU_R)[np.newaxis, :, np.newaxis]*size_dummy,], # ICU_R
            [np.squeeze(1/dICU_D)[np.newaxis, :, np.newaxis]*size_dummy,], # ICU_D
            [np.squeeze(1/dICUrec)[np.newaxis, :, np.newaxis]*size_dummy,], # C_icurec
            [size_dummy*zeta,], # R
            [multip_work*e_s,]] # S_work

        # 0: S --> E
        # 1: E --> I
        # 2: I --> A
        # 3: I --> M_R
        # 4: I --> M_H
        # 5: A --> R
        # 6: M_R --> R
        # 7: M_H --> C_R
        # 8: M_H --> C_D
        # 9: M_H --> ICU_R
        # 10: M_H --> ICU_D
        # 11: C_R --> R
        # 12: C_D --> D
        # 13: ICU_R --> C_icurec
        # 14: ICU_D --> D
        # 15: C_icurec --> R
        # 16: R --> S
        # 17: S_work --> E

        # Compute the transitionings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Convert states and rates to typed lists
        typed_states = nb.typed.List()
        [typed_states.append(state) for state in states]
        # Convert states and rates to typed lists
        typed_rates = nb.typed.List()
        [typed_rates.append(rate) for rate in rates]
        T = compute_transitionings_spatial_jit(G, N, D, l, typed_states, typed_rates)
        # Convert T back to list
        T = [T[:,:,:,i] for i in range(T.shape[-1])]

        # Update the system
        # ~~~~~~~~~~~~~~~~~

        # Flowchart states
        S_new = S_post_vacc - T[0] - T[17] + T[16]
        E_new = E + T[0] + T[17] - T[1]
        I_new = I + T[1] - (T[2] + T[3] + T[4])
        A_new = A + T[2] - T[5]
        M_R_new = M_R + T[3] - T[6]
        M_H_new = M_H + T[4] - (T[7] + T[8] + T[9] + T[10])
        C_R_new = C_R + T[7] - T[11]
        C_D_new = C_D + T[8] - T[12]
        ICU_R_new = ICU_R + T[9] - T[13]
        ICU_D_new = ICU_D + T[10] - T[14]
        C_icurec_new = C_icurec + T[13] - T[15]
        R_new = R_post_vacc + T[5] + T[6] + T[11] + T[15] - T[16]
        D_new = D + T[12] + T[14]

        # Derivative states
        M_in_new = (T[3] + T[4])/l
        H_in_new = (T[7] + T[8] + T[9] + T[10])/l
        H_out_new = (T[11] + T[12] + T[14] + T[15])/l
        H_tot_new = H_tot + (H_in_new - H_out_new)*l

        return (S_new, E_new, I_new, A_new, M_R_new, M_H_new, C_R_new, C_D_new, C_icurec_new, ICU_R_new, ICU_D_new, R_new, D_new, M_in_new, H_in_new, H_tot_new)


class Economic_Model(BaseModel):

    # ...state variables and parameters
    state_names = ['x', 'c', 'c_desired','f', 'd', 'l','O', 'S']
    parameter_names = ['x_0', 'c_0', 'f_0', 'l_0', 'IO', 'O_j', 'n', 'on_site', 'C', 'S_0','b','rho','delta_S','zeta','tau','gamma_F','gamma_H','A']
    parameters_stratified_names = ['epsilon_S','epsilon_D','epsilon_F']
    stratification_names = ['NACE64']

    # Bookkeeping of 2D stock matrix
    state_2d = ["S"]

     # ..transitions/equations
    @staticmethod

    def integrate(t, x, c, c_desired, f, d, l, O, S, x_0, c_0, f_0, l_0, IO, O_j, n, on_site, C, S_0, b, rho, delta_S, zeta, tau, gamma_F, gamma_H, A, epsilon_S, epsilon_D, epsilon_F):
        """
        BIOMATH production network model for Belgium

        *Based on the Oxford INET implementation*
        """

        # 1. Update exogeneous demand with shock vector
        # ---------------------------------------------
        f_desired = (1-epsilon_F)*f_0

        # 2. Compute labor income after government furloughing
        # ----------------------------------------------------
        l_star = l + b*(l_0-l)
  
        # 3. Compute productive capacity under labor constraints
        # ------------------------------------------------------
        x_cap = calc_labor_restriction(x_0,l_0,l)

        # 4. Compute productive capacity under input constraints
        # ------------------------------------------------------
        x_inp = calc_input_restriction(S,A,C)
        # 5. Compute total consumer demand
 
        # Compute consumer preference vector
        # --------------------------------
        theta_0 = c_0/sum(c_0)
        # Compute aggregate demand shock
        theta = household_preference_shock(epsilon_D, theta_0)
        epsilon_t = aggregate_demand_shock(epsilon_D,theta_0,delta_S,rho)
        # Compute expected total long term labor income (Eq. 22, 23)
        l_p = zeta*sum(l_0)
        # Compute total consumer demand (per sector)
        m = sum(c_0)/sum(l_0)
        c_desired_new = theta*calc_household_demand(sum(c_desired),l_star,l_p,epsilon_t,rho,m)

        # 6. Compute B2B demand
        O_desired = calc_intermediate_demand(d,S,A,S_0,tau) # 2D
        # ---------------------   

        # 7. Compute total demand
        # -----------------------
        d_new = calc_total_demand(O_desired,c_desired_new,f_desired)

        # 8. Leontief production function with critical inputs
        # ----------------------------------------------------
        x_new = leontief(x_cap, x_inp, d_new)
        # 9. Perform rationing

        # --------------------
        O_new, c_new, f_new = rationing(x_new,d_new,O_desired,c_desired_new,f_desired)

        # 10. Update inventories
        # ----------------------
        S_new = inventory_updating(S,O_new,x_new,A)

        # 11. Hire/fire workers
        # ---------------------
        l_new = hiring_firing(l, l_0, x_0, x_inp, x_cap, d_new, gamma_F, gamma_H, epsilon_S)

        # --------------------------------------------------------------
        # 12. Convert order matrix to total order per sector (2D --> 1D)
        O_new = np.sum(O_new,axis=1)
        return (x_new-x, c_new-c, c_desired_new-c_desired, f_new-f, d_new-d, l_new-l, O_new-O, S_new-S)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
