# Copyright (c) 2021 by T.W. Alleman, D. Van Hauwermeiren, BIOMATH, Ghent University. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numba as nb
import numpy as np
from numba import jit
from .base import BaseModel
from .utils import stratify_beta
from .economic_utils import *
from ..data.economic_parameters import read_economic_labels
# Register pandas formatters and converters with matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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
    Must be simulated in discrete timesteps by adding the argument `discrete=True` in the model initialization.

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
    parameter_names = ['beta', 'gamma']
    parameters_stratified_names = []
    stratification = ['Nc']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma, Nc):
        """Basic stochastic SIR model """

        # Define solver parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        l = 1 # length of discrete timestep (by default one day)
        n = 1 # number of draws to average in one timestep
        N = S.size

        # calculate total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        T = S + I + R

        # Make a dictionary containing the transitions and their propensities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        keys = ['StoI','ItoR']
        probabilities = [1 - np.exp( - l*beta*np.matmul(Nc,I/T) ),
                        (1 - np.exp(- l * gamma ))*np.ones(N),
                        ]
        states = [S,I]
        propensity={}
        for i in range(len(keys)):
            prop=[]
            for j in range(N):
                if states[i][j]<=0:
                    prop.append(0)
                else:
                    draw = np.mean(np.random.binomial(states[i][j],probabilities[i][j], size=n))
                    prop.append(draw)
            propensity.update({keys[i]: np.asarray(prop)})

        # calculate the states at timestep k+1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        S_new  = np.rint(S - propensity['StoI'])
        I_new =  np.rint(I + propensity['StoI'] - propensity['ItoR'])
        R_new  =  np.rint(R + propensity['ItoR'])

        # Add protection against states < 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        output = (S_new, I_new, R_new)
        for i in range(len(output)):
            output[i][output[i]<0] = 0

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
    parameter_names = ['beta', 'gamma', 'injection_day', 'injection_ratio']
    parameters_stratified_names = []
    stratification = ['Nc']

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
            dalpha = np.zeros(9)
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
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'C_icurec','ICU', 'R', 'D','H_in','H_out','H_tot']
    parameter_names = ['beta', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h']
    parameters_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    stratification = ['Nc','doses']
    coordinates = [None, ['none', 'partial', 'full', 'boosted']]

    # ..transitions/equations
    @staticmethod
    @jit(nopython=True)
    def integrate(t, S, E, I, A, M, C, C_icurec, ICU, R, D, H_in, H_out, H_tot,
                  beta, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm,  dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h,
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D,
                  Nc, doses):
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
        y=sum(f_VOC*K_hosp)
        x=(y-1)*sum(h)/sum(1-h)
        h=h+(1-h)*x
        #h = np.sum(np.outer(h, f_VOC*K_hosp),axis=1)
        #h[h > 1] = 1
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

        T = np.expand_dims(np.sum(S_post_vacc + E + I + A + M + C + C_icurec + ICU + R_post_vacc, axis=1),axis=1) # sum over doses

        #################################
        ## Compute system of equations ##
        #################################

        # Compute infection pressure (IP) of all variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        IP = np.expand_dims( np.sum( np.outer(beta*s*jit_matmul_2D_1D(Nc, np.sum(((I+A)/T)*e_i, axis=1)), f_VOC*K_inf), axis=1), axis=1)

        # Compute the  rates of change in every population compartment
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        h_acc = e_h*h

        dS  = dS - IP*S_post_vacc*e_s
        dE  = IP*S_post_vacc*e_s - E/sigma 
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - A/da
        dM = ((1-a)/omega)*I - M*((1-h_acc)/dm) - M*h_acc/dhospital
        dC = M*(h_acc/dhospital)*c - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D))
        dICUstar = M*(h_acc/dhospital)*(1-c) - (1-m_ICU)*ICU/(dICU_R) - m_ICU*ICU/(dICU_D)

        dC_icurec = (1-m_ICU)*ICU/(dICU_R) - C_icurec*(1/dICUrec)
        dR  = dR + A/da + ((1-h_acc)/dm)*M + (1-m_C)*C*(1/(dc_R)) + C_icurec*(1/dICUrec)
        dD  = (m_ICU/(dICU_D))*ICU + (m_C/(dc_D))*C 
        dH_in = M*(h_acc/dhospital) - H_in
        dH_out =  (1-m_C)*C*(1/(dc_R)) +  m_C*C*(1/(dc_D)) + m_ICU/(dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = M*(h_acc/dhospital) - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D)) - m_ICU*ICU/(dICU_D)- C_icurec*(1/dICUrec) 

        # Waning of natural immunity
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        dS[:,0] = dS[:,0] + zeta*R_post_vacc[:,0] 
        dR[:,0] = dR[:,0] - zeta*R_post_vacc[:,0]

        return (dS, dE, dI, dA, dM, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot)

class COVID19_SEIQRD_spatial_hybrid_vacc(BaseModel):

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'C_icurec','ICU', 'R', 'D','H_in','H_out','H_tot']
    parameter_names = ['beta_R', 'beta_U', 'beta_M', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital', 'seasonality', 'N_vacc', 'e_i', 'e_s', 'e_h', 'Nc_work']
    parameters_stratified_names = [['area', 'p'],['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    stratification = ['place','Nc','doses']
    coordinates = ['place', None, ['none', 'partial', 'full', 'boosted']]

    @staticmethod
    @jit(nopython=True)
    def integrate(t, S, E, I, A, M, C, C_icurec, ICU, R, D, H_in, H_out, H_tot, # time + SEIRD classes
                  beta_R, beta_U, beta_M, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm, dICUrec, dhospital, seasonality, N_vacc, e_i, e_s, e_h, Nc_work, # SEIRD parameters
                  area, p, # spatially stratified parameters. 
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D, # age-stratified parameters
                  place, Nc, doses): # stratified parameters that determine stratification dimensions
        
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
        y=sum(f_VOC*K_hosp)
        x=(y-1)*sum(h)/sum(1-h)
        h=h+(1-h)*x
        #h = np.sum(np.outer(h, f_VOC*K_hosp),axis=1)
        #h[h > 1] = 1
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

        T = np.sum(S_post_vacc + E + I + A + M + C + C_icurec + ICU + R_post_vacc, axis=2) # Sum over doses

        ################################
        ## Compute infection pressure ##
        ################################

        # For total population and for the relevant compartments I and A
        G = S.shape[0] # spatial stratification
        N = S.shape[1] # age stratification

        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*place + np.identity(G)*(place @ (1-np.outer(p,p)))
        
        # Expand beta to size G
        beta = stratify_beta(beta_R, beta_U, beta_M, area, T.sum(axis=1))*np.sum(f_VOC*K_inf)

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

        h_acc = e_h*h

        dS  = dS - dS_inf
        dE  = dS_inf - E/sigma 
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - A/da
        dM = ((1-a)/omega)*I - M*((1-h_acc)/dm) - M*h_acc/dhospital
        dC = M*(h_acc/dhospital)*c - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D))
        dICUstar = M*(h_acc/dhospital)*(1-c) - (1-m_ICU)*ICU/(dICU_R) - m_ICU*ICU/(dICU_D)

        dC_icurec = (1-m_ICU)*ICU/(dICU_R) - C_icurec*(1/dICUrec)
        dR  = dR + A/da + ((1-h_acc)/dm)*M + (1-m_C)*C*(1/(dc_R)) + C_icurec*(1/dICUrec)
        dD  = (m_ICU/(dICU_D))*ICU + (m_C/(dc_D))*C 
        dH_in = M*(h_acc/dhospital) - H_in
        dH_out =  (1-m_C)*C*(1/(dc_R)) +  m_C*C*(1/(dc_D)) + m_ICU/(dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = M*(h_acc/dhospital) - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D)) - m_ICU*ICU/(dICU_D)- C_icurec*(1/dICUrec) 

        ########################
        ## Waning of immunity ##
        ########################

        # Waning of natural immunity
        dS[:,:,0] = dS[:,:,0] + zeta*R_post_vacc[:,:,0] 
        dR[:,:,0] = dR[:,:,0] - zeta*R_post_vacc[:,:,0]      

        return (dS, dE, dI, dA, dM, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot)

class COVID19_SEIQRD_spatial_rescaling(BaseModel):
    """
    SEIQRD model identical to COVID19_SEIQRD_spatial_stratified_vacc, but built to implement the VOC, vaccination, and seasonality effects as a rescaling parameter from a time-dependent parameter function, rather than additional metapopulation compartments. Hopefully this will establish a significant speed-up. In particular in combination with the just-in-time addition.
    
    Note that in this model we cannot tinker with vaccine dosage for future scenarios. This should now be done in the time-dependent parameter function for the effective rescaling. It remains impossible to look at subjects with a particular dosage - we may only see the overall effect on society (per age and province).
    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'C_icurec', 'ICU', 'R', 'D', 'H_in', 'H_out', 'H_tot']
    parameter_names = ['beta_R', 'beta_U', 'beta_M', 'f_VOC', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta', 'da', 'dm', 'dc_R', 'dc_D', 'dICU_R', 'dICU_D', 'dICUrec', 'dhospital', 'Nc_work', 'seasonality', 'E_susc', 'E_inf', 'E_hosp']
    parameters_stratified_names = [['area', 'p'], ['s','a','h', 'c', 'm_C','m_ICU']]
    stratification = ['place','Nc'] # mobility and social interaction: name of the dimension (better names: ['nis', 'age'])
    coordinates = ['place', None] # 'place' is interpreted as a list of NIS-codes appropriate to the geography

    # ..transitions/equations
    @staticmethod
    @jit(nopython=True)
    def integrate(t, S, E, I, A, M, C, C_icurec, ICU, R, D, H_in, H_out, H_tot, # time + SEIRD classes
                  beta_R, beta_U, beta_M, f_VOC, K_inf, K_hosp, sigma, omega, zeta, da, dm, dc_R, dc_D, dICU_R, dICU_D, dICUrec, dhospital, Nc_work, seasonality, E_susc, E_inf, E_hosp,# SEIRD parameters
                  area, p, # spatially stratified parameters. 
                  s, a, h, c, m_C, m_ICU, # age-stratified parameters
                  place, Nc): # stratified parameters that determine stratification dimensions


        ##################
        ## Format input ##
        ##################

        # Remove negative derivatives to ease further computation
        # Note: this model will only use the fraction itself, not its derivative
        f_VOC[:,1][f_VOC[:,1] < 0] = 0
        # Split derivatives and fraction
        d_VOC = f_VOC[1,:]
        f_VOC = f_VOC[0,:]
        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)
        
        ################################
        ## calculate total population ##
        ################################

        T = S + E + I + A + M + C + C_icurec + ICU + R

        ################################
        ## Compute infection pressure ##
        ################################

        # For total population and for the relevant compartments I and A
        G = place.shape[0] # spatial stratification
        N = Nc.shape[1] # age stratification

        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*place + np.identity(G)*(place @ (1-np.outer(p,p)))
        
        # Expand beta to size G based on local population density
        beta = stratify_beta(beta_R, beta_U, beta_M, area, T.sum(axis=1))
        
        ### RESCALING INFECTIVITY ###
        # Rescale all local beta according to the nationally-aggregated prevalence of the VOCs
        beta *= np.sum(f_VOC*K_inf)
        
        # Rescale beta according to seasonality (nationally aggregated)
        beta *= seasonality
        
        ### RESCALING HOSPITALISATION PROPENSITY ###
        # rescale h according to the prevalence of the VOCs
        h = np.sum(np.outer(h, f_VOC*K_hosp),axis=1)
        # Rescale h according to vaccination status per region and age
        h_bar = np.expand_dims(h, axis=0) * E_hosp
        
        ### RESCALING LATENT PERIOD ###
        # rescale sigma according to the prevalence of the VOCs
        sigma = np.sum(f_VOC*sigma)
        
        ### Define effective local populations (T, I and A) and local average infectivity
        # MAKE SURE THIS IS CORRECT!
        T_eff = np.transpose(place_eff) @ T # total
        I_eff = np.transpose(place_eff) @ I # presymptomatic I_presy
        A_eff = np.transpose(place_eff) @ A # asymptomatic I_asy
        I_dens = (I_eff+A_eff) / T_eff
        E_inf_eff = place_eff @ E_inf / np.expand_dims(np.sum(place_eff,axis=0),axis=1)
        
        # Rescale beta according to vaccination status per region and age. Result is beta[g,h,i,j] of dimension (G,G,N,N)
        # only second index of beta is iterated over. Add empty age indices.
        beta_bar = np.expand_dims(np.expand_dims(np.expand_dims(beta, axis=0),axis=2),axis=2) # shape = (1, 11, 1, 1)
        # First and third index of E_susc is iterated over
        E_susc = np.expand_dims(np.expand_dims(E_susc, axis=2),axis=1) # shape = (11, 1, 10, 1)
        E_inf_eff = np.expand_dims(np.expand_dims(E_inf_eff, axis=0),axis=2) # shape = (1, 11, 1, 10)
        beta_bar = beta_bar * E_susc * E_inf_eff
        
        # Calculate Nc^gh_ij. It would be more efficient to take this out of the class altogether
        kroneckerG = np.diag(np.ones(G))
        kroneckerG = np.expand_dims(np.expand_dims(kroneckerG, axis=2), axis=2)
        Nc_g_total = np.expand_dims(Nc, axis=1) # shape (11, 1, 10, 10)
        Nc_g_work = np.expand_dims(Nc_work, axis=0) # shape (1, 11, 10, 10)
        Nc_bar = kroneckerG * Nc_g_total + (1-kroneckerG) * Nc_g_work
        
        # Use all input in the jit-defined loop function
        dS_inf = jit_main_function_spatial(place_eff, S, beta_bar, Nc_bar, I_dens)
        
        ####################################################
        ## Add spatial dimension to age-stratified params ##
        ####################################################

        # ... such that the dimensions are correct in the set of ODEs.
        a = np.expand_dims(a, axis=0)
        # h = np.expand_dims(h, axis=0) # already done above
        c = np.expand_dims(c, axis=0)
        m_C = np.expand_dims(m_C, axis=0)
        m_ICU = np.expand_dims(m_ICU, axis=0)
        dc_R = np.expand_dims(dc_R, axis=0)
        dc_D = np.expand_dims(dc_D, axis=0)
        dICU_R = np.expand_dims(dICU_R, axis=0)
        dICU_D = np.expand_dims(dICU_D, axis=0)
        dICUrec = np.expand_dims(dICUrec, axis=0)

        ############################
        ## Compute system of ODEs ##
        ############################

        dS  = - dS_inf + zeta*R
        dE  = dS_inf - E/sigma 
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - A/da
        dM = ((1-a)/omega)*I - M*((1-h_bar)/dm) - M*h_bar/dhospital
        dC = M*(h_bar/dhospital)*c - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D))
        dICUstar = M*(h_bar/dhospital)*(1-c) - (1-m_ICU)*ICU/(dICU_R) - m_ICU*ICU/(dICU_D)
        dC_icurec = (1-m_ICU)*ICU/(dICU_R) - C_icurec*(1/dICUrec)
        dR  = A/da + ((1-h_bar)/dm)*M + (1-m_C)*C*(1/(dc_R)) + C_icurec*(1/dICUrec) - zeta*R
        dD  = (m_ICU/(dICU_D))*ICU + (m_C/(dc_D))*C 
        dH_in = M*(h_bar/dhospital) - H_in
        dH_out =  (1-m_C)*C*(1/(dc_R)) +  m_C*C*(1/(dc_D)) + m_ICU/(dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = M*(h_bar/dhospital) - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D)) - m_ICU*ICU/(dICU_D)- C_icurec*(1/dICUrec)       

        return (dS, dE, dI, dA, dM, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot)

class Economic_Model(BaseModel):

    # ...state variables and parameters
    state_names = ['x', 'c', 'c_desired','f', 'd', 'l','O', 'S']
    parameter_names = ['x_0', 'c_0', 'f_0', 'l_0', 'IO', 'O_j', 'n', 'on_site', 'C', 'S_0','b','rho','delta_S','zeta','tau','gamma_F','gamma_H']
    parameters_stratified_names = [['epsilon_S','epsilon_D','epsilon_F']]
    stratification = ['A']
    coordinates = [read_economic_labels('NACE64')]

    # Bookkeeping of 2D stock matrix
    state_2d = ["S"]

     # ..transitions/equations
    @staticmethod

    def integrate(t, x, c, c_desired, f, d, l, O, S, x_0, c_0, f_0, l_0, IO, O_j, n, on_site, C, S_0, b, rho, delta_S, zeta, tau, gamma_F, gamma_H, epsilon_S, epsilon_D, epsilon_F, A):
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
