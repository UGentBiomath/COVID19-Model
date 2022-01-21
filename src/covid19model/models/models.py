# Copyright (c) 2021 by T.W. Alleman, D. Van Hauwermeiren, BIOMATH, Ghent University. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
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

@jit(fastmath=True, nopython=True)
def jit_matmul_2D_1D(A, b):
    """A simple jitted implementation of a 2Dx1D matrix multiplication
    """
    n = A.shape[0]
    f = A.shape[1]
    out = np.zeros(n, np.float64)
    for i in range(n):
            for k in range(f):
                out[i] += A[i, k] * b[k]
    return out

@jit(fastmath=True, nopython=True)
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

@jit(fastmath=True, nopython=True)
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

@jit(fastmath=True, nopython=True)
def jit_outer(a, b):
    """A jitted implementation of np.outer"""
    out = np.zeros((len(a),len(b)), np.float64)
    for i in range(len(a)):
        for j in range(len(b)):
            out[i,j] = a[i]*b[j]
    return out

@jit(fastmath=True, nopython=True)
def negative_values_replacement_2D(A, B):
    for i in range(A.shape[0]):
        A[i,:][B[i,:]<0] = 0
    return A

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

        l = 1.0 # length of discrete timestep (by default one day)
        n = 20 # number of draws to average in one timestep (slows down calculations but converges to a deterministic result when > 20)

        # calculate total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        T = S + I + R

        # Make a dictionary containing the transitions and their propensities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        keys = ['StoI','ItoR']
        probabilities = [1 - np.exp( - beta*np.matmul(Nc,I/T) ),
                        (1 - np.exp(- l * gamma ))*np.ones(S.size),
                        ]
        states = [S,I]
        propensity={}
        for i in range(len(keys)):
            prop=[]
            for j in range(S.size):
                if states[i][j]<=0:
                    prop.append(0)
                else:
                    draw=np.array([])
                    for l in range(n):
                        draw = np.append(draw,np.random.binomial(states[i][j],probabilities[i][j]))
                    draw = np.rint(np.mean(draw))
                    prop.append( draw )
            propensity.update({keys[i]: np.asarray(prop)})

        # calculate the states at timestep k+1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        S_new  = S - propensity['StoI']
        I_new =  I + propensity['StoI'] - propensity['ItoR']
        R_new  =  R + propensity['ItoR']

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

class COVID19_SEIQRD(BaseModel):
    """
    Biomath extended SEIQRD model for COVID-19, Deterministic implementation
    Can account for re-infection and co-infection with a new COVID-19 variant.

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states
        e.g. {'S': N, 'E': np.ones(n_stratification)} with N being the total population and n_stratifications the number of stratified layers
        initialising zeros is thus not required

        S : susceptible
        E : exposed
        I : infected
        A : asymptomatic
        M : mild
        C : cohort
        C_icurec : cohort after recovery from ICU
        ICU : intensive care
        R : recovered
        D : deceased
        H_in : new hospitalizations
        H_out : new hospital discharges
        H_tot : total patients in Belgian hospitals

        parameters : dictionary
        containing the values of all parameters (both stratified and not)
        these can be obtained with the function model_parameters.get_COVID19_SEIRD_parameters()

        Non-stratified parameters
        -------------------------
        beta : probability of infection when encountering an infected person
        sigma : length of the latent period
        omega : length of the pre-symptomatic infectious period
        zeta : effect of re-susceptibility and seasonality
        a : probability of an asymptomatic cases
        m : probability of an initially mild infection (m=1-a)
        da : duration of the infection in case of asymptomatic
        dm : duration of the infection in case of mild
        dc : average length of a hospital stay when not in ICU
        dICU_R : average length of a hospital stay in ICU in case of recovery
        dICU_D: average length of a hospital stay in ICU in case of death
        dhospital : time before a patient reaches the hospital

        Age-stratified parameters
        --------------------
        s: relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU

        Other parameters
        ----------------
        Nc : contact matrix between all age groups in stratification

    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'C_icurec','ICU', 'R', 'D','H_in','H_out','H_tot']
    parameter_names = ['beta', 'sigma', 'omega', 'zeta','da', 'dm', 'dc_R','dc_D','dICU_R', 'dICU_D', 'dICUrec','dhospital']
    parameters_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU']]
    stratification = ['Nc']

    # ..transitions/equations
    @staticmethod
    @jit(nopython=True)
    def integrate(t, S, E, I, A, M, C, C_icurec, ICU, R, D, H_in, H_out, H_tot,
                  beta, sigma, omega, zeta, da, dm, dc_R, dc_D, dICU_R, dICU_D, dICUrec, dhospital,
                  s, a, h, c, m_C, m_ICU,
                  Nc):
        """
        Biomath extended SEIQRD model for COVID-19

        *Deterministic implementation*
        """

        # calculate total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        T = S + E + I + A + M + C + C_icurec + ICU + R

        # Compute infection pressure (IP) of both variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        IP = beta*s*jit_matmul_2D_1D(Nc, (I+A)/T)

        # Compute the  rates of change in every population compartment
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dS  = - IP*S + zeta*R 
        dE  = IP*S - E/sigma 
        dI = (1/sigma)*E - (1/omega)*I 
        dA = (a/omega)*I - A/da      
        dM = ((1-a)/omega)*I - M*((1-h)/dm) - M*h/dhospital

        dC = M*(h/dhospital)*c - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D))
        dICUstar = M*(h/dhospital)*(1-c) - (1-m_ICU)*ICU/(dICU_R) - m_ICU*ICU/(dICU_D)

        dC_icurec = (1-m_ICU)*ICU/(dICU_R) - C_icurec*(1/dICUrec)
        dR  = A/da + ((1-h)/dm)*M + (1-m_C)*C*(1/(dc_R)) + C_icurec*(1/dICUrec) - zeta*R 
        dD  = (m_ICU/(dICU_D))*ICU + (m_C/(dc_D))*C 
        dH_in = M*(h/dhospital) - H_in
        dH_out =  (1-m_C)*C*(1/(dc_R)) +  m_C*C*(1/(dc_D)) + m_ICU/(dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = M*(h/dhospital) - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D)) - m_ICU*ICU/(dICU_D)- C_icurec*(1/dICUrec) 

        return (dS, dE, dI, dA, dM, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot)

class COVID19_SEIQRD_stratified_vacc(BaseModel):
    """
    Biomath extended SEIRD model for COVID-19, Deterministic implementation
    Can account for re-infection and co-infection with a new COVID-19 variant.

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states
        e.g. {'S': N, 'E': np.ones(n_stratification)} with N being the total population and n_stratifications the number of stratified layers
        initialising zeros is thus not required

        S : susceptible
        E : exposed
        I : infected
        A : asymptomatic
        M : mild
        C : cohort
        C_icurec : cohort after recovery from ICU
        ICU : intensive care
        R : recovered
        D : deceased
        H_in : new hospitalizations
        H_out : new hospital discharges
        H_tot : total patients in Belgian hospitals

        parameters : dictionary
        containing the values of all parameters (both stratified and not)
        these can be obtained with the function model_parameters.get_COVID19_SEIRD_parameters()

        Non-stratified parameters
        -------------------------
        beta : probability of infection when encountering an infected person
        f_VOC : (first row) fraction of alternative COVID-19 variant, (second row) derivative of fraction of alternative COVID-19 variant
        f_immune_escape : sequential fraction of immune escape of new variant
        K_inf : infectivity gain of variants (infectivity of new variant = K * infectivity of index variant)
        K_hosp : hospitalization propensity gain of alternative COVID-19 variants (severity of new variant = K * infectivity of reference variant)
        sigma : length of the latent period
        omega : length of the pre-symptomatic infectious period
        zeta : effect of re-susceptibility and seasonality
        a : probability of an asymptomatic cases
        m : probability of an initially mild infection (m=1-a)
        da : duration of the infection in case of asymptomatic
        dm : duration of the infection in case of mild
        dc : average length of a hospital stay when not in ICU
        dICU_R : average length of a hospital stay in ICU in case of recovery
        dICU_D: average length of a hospital stay in ICU in case of death
        dhospital : time before a patient reaches the hospital

        Age-stratified parameters
        --------------------
        s: relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU

        Other parameters
        ----------------
        Nc : contact matrix between all age groups in stratification

    """
    
    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'C_icurec','ICU', 'R', 'D','H_in','H_out','H_tot']
    parameter_names = ['beta', 'f_VOC', 'f_immune_escape', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta','da', 'dm','dICUrec','dhospital','N_vacc', 'd_vacc', 'e_i', 'e_s', 'e_h']
    parameters_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D','dICU_R','dICU_D'],[]]
    stratification = ['Nc','doses']

    # ..transitions/equations
    @staticmethod
    @jit(nopython=True)
    def integrate(t, S, E, I, A, M, C, C_icurec, ICU, R, D, H_in, H_out, H_tot,
                  beta, f_VOC, f_immune_escape, K_inf, K_hosp, sigma, omega, zeta, da, dm,  dICUrec, dhospital, N_vacc, d_vacc, e_i, e_s, e_h,
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D,
                  Nc, doses):
        """
        Biomath extended SEIRD model for COVID-19

        *Deterministic implementation*
        """

        # jit wisdom:
        # the jit-compatible versions of np.outer, np.sum, np.matmul have large speedups as compared to their numpy counterparts however:
        # replacing np.outer, np.sum with jit-compatible variants does not result in a speedup (and even slows the code slightly down)
        # replacing @ by np.matmul results in a speedup for matrices of sufficient size, but slows the code down for smaller systems

        # - negative values check to replace np.where, negative_values_replacement_2D(A, B)

        # Construct vector K_inf
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Prepend a 'one' in front of K_inf and K_hosp (cannot use np.insert with jit compilation)
        K_inf = np.array( ([1,] + list(K_inf)), np.float64)
        K_hosp = np.array( ([1,] + list(K_hosp)), np.float64)
        
        # Modeling immune escape
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Remove negative derivatives to ease further computation (jit compatible in 1D but not in 2D!)
        f_VOC[1,:][f_VOC[1,:] < 0] = 0
        # Split derivatives and fraction
        d_VOC = f_VOC[1,:]
        f_VOC = f_VOC[0,:]

        # calculate total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        T = np.expand_dims(np.sum(S + E + I + A + M + C + C_icurec + ICU + R, axis=1),axis=1)

        # Account for higher hospitalisation propensity and changes in vaccination parameters due to new variant
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        sigma = np.sum(f_VOC*sigma)
        h = np.sum(jit_outer(h, f_VOC*K_hosp),axis=1)
        h[h > 1] = 1
        e_i = jit_matmul_1D_2D(f_VOC, e_i) #jit_matmul_1D_2D(f_VOC, e_i) performs slower than @ (maybe because matrices are quite small)
        e_s = jit_matmul_1D_2D(f_VOC, e_s) 
        e_h = jit_matmul_1D_2D(f_VOC, e_h)

        # Expand dims on first stratification axis (age)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        dR[:,1] =  N_vacc[:,0]*f_R # 0 --> 1 dose
        # Compute transitioning in two shot circuit
        dS[:,2] =  N_vacc[:,2]*f_S # 0 --> 2 doses
        dR[:,2] =  N_vacc[:,2]*f_R # 0 --> 2 doses

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,1] + E[:,1] + I[:,1] + A[:,1] + R[:,1]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,1]/VE
        f_R = R[:,1]/VE
        # Compute transitioning in one short circuit
        dS[:,1] = dS[:,1] - N_vacc[:,1]*f_S
        dR[:,1] = dR[:,1] - N_vacc[:,1]*f_R
        # Compute transitioning in two shot circuit
        dS[:,2] = dS[:,2] + N_vacc[:,1]*f_S
        dR[:,2] = dR[:,2] + N_vacc[:,1]*f_R

        # waned vaccine, 2 --> B
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,2]+ R[:,2] + S[:,3] + R[:,3]
        # 2 dose circuit
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,2]/VE
        f_R = R[:,2]/VE
        # Compute transitioning in two shot circuit
        dS[:,2] = dS[:,2] - N_vacc[:,3]*f_S
        dR[:,2] = dR[:,2] - N_vacc[:,3]*f_R
        # Compute transitioning in booster circuit
        dS[:,4] = dS[:,4] + N_vacc[:,3]*f_S
        dR[:,4] = dR[:,4] + N_vacc[:,3]*f_R
        # waned vaccine circuit
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,3]/VE
        f_R = R[:,3]/VE
        # Compute transitioning in two shot circuit
        dS[:,3] = dS[:,3] - N_vacc[:,3]*f_S
        dR[:,3] = dR[:,3] - N_vacc[:,3]*f_R
        # Compute transitioning in booster circuit
        dS[:,4] = dS[:,4] + N_vacc[:,3]*f_S
        dR[:,4] = dR[:,4] + N_vacc[:,3]*f_R

        # Update the S and R state
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        S_post_vacc = S + dS
        R_post_vacc = R + dR

        # Compute dS that makes S and R equal to zero
        #dS[S_post_vacc < 0] = 0 - S[S_post_vacc < 0]
        #dR[R_post_vacc < 0] = 0 - R[R_post_vacc < 0]
        # Set S and R equal to zero
        #S_post_vacc[S_post_vacc < 0] = 0
        #R_post_vacc[R_post_vacc < 0] = 0

        #################################
        ## Compute system of equations ##
        #################################

        # Compute infection pressure (IP) of all variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        IP = np.expand_dims( np.sum( jit_outer(beta*s*jit_matmul_2D_1D(Nc, np.sum(((I+A)/T)*(1-e_i), axis=1)), f_VOC*K_inf), axis=1), axis=1)

        # Compute the  rates of change in every population compartment
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        h_acc = (1-e_h)*h

        dS  = dS - IP*S_post_vacc*(1-e_s)
        dE  = IP*S_post_vacc*(1-e_s) - E/sigma 
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

        # Waning of vaccines
        # ~~~~~~~~~~~~~~~~~~

        # Waning of second dose
        r_waning_vacc = 1/((6/12)*365)
        dS[:,2] = dS[:,2] - r_waning_vacc*S_post_vacc[:,2]
        dR[:,2] = dR[:,2] - r_waning_vacc*R_post_vacc[:,2]
        dS[:,3] = dS[:,3] + r_waning_vacc*S_post_vacc[:,2]
        dR[:,3] = dR[:,3] + r_waning_vacc*R_post_vacc[:,2]
        
        # Waning of booster dose
        # No waning of booster dose

        # Waning of natural immunity
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        dS[:,0] = dS[:,0] + zeta*R_post_vacc[:,0] 
        dR[:,0] = dR[:,0] - zeta*R_post_vacc[:,0]

        # Immune escape
        # ~~~~~~~~~~~~~

        dS = dS + np.sum(f_immune_escape*d_VOC)*R
        dR = dR - np.sum(f_immune_escape*d_VOC)*R     

        return (dS, dE, dI, dA, dM, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot)

class COVID19_SEIQRD_spatial(BaseModel):
    """
    BIOMATH extended SEIRD model for COVID-19, spatially explicit. Based on COVID_SEIRD and Arenas (2020).
    Can account for re-infection and co-infection with a new COVID-19 variants.

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states
        e.g. {'S': N, 'E': np.ones(n_stratification)} with N being the total population and n_stratifications the number of stratified layers
        initialising zeros is thus not required
        
        S : susceptible
        E : exposed
        I : infected
        A : asymptomatic
        M : mild
        C : cohort
        C_icurec : cohort after recovery from ICU
        ICU : intensive care
        R : recovered
        D : deceased

        H_in : new hospitalizations
        H_out : new hospital discharges
        H_tot : total patients in Belgian hospitals
        
    parameters : dictionary
        containing the values of all parameters (both stratified and not)
        these can be obtained with the function parameters.get_COVID19_SEIRD_parameters()

        Non-stratified parameters
        -------------------------
        beta : probability of infection when encountering an infected person
        sigma : length of the latent period
        omega : length of the pre-symptomatic infectious period
        zeta : effect of re-susceptibility and seasonality
        a : probability of an asymptomatic cases
        m : probability of an initially mild infection (m=1-a)
        da : duration of the infection in case of asymptomatic
        dm : duration of the infection in case of mild
        der : duration of stay in emergency room/buffer ward
        dc : average length of a hospital stay when not in ICU
        dICU_R : average length of a hospital stay in ICU in case of recovery
        dICU_D: average length of a hospital stay in ICU in case of death
        dhospital : time before a patient reaches the hospital

        Age-stratified parameters
        -------------------------
        s: relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU

        Spatially-stratified parameters
        -------------------------------
        place : normalised mobility data. place[g][h] denotes the fraction of the population in patch g that goes to patch h
        p : mobility parameter (1 by default = no measures)

        Other parameters
        ----------------
        Nc : N-by-N contact matrix between all age groups in stratification

    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'C_icurec', 'ICU', 'R', 'D', 'H_in', 'H_out', 'H_tot']
    parameter_names = ['beta_R', 'beta_U', 'beta_M', 'sigma', 'omega', 'zeta', 'da', 'dm', 'dc_R', 'dc_D', 'dICU_R', 'dICU_D', 'dICUrec', 'dhospital', 'Nc_work']
    parameters_stratified_names = [['area', 'p'], ['s','a','h', 'c', 'm_C','m_ICU']]
    stratification = ['place','Nc'] # mobility and social interaction: name of the dimension (better names: ['nis', 'age'])
    coordinates = ['place'] # 'place' is interpreted as a list of NIS-codes appropriate to the geography
    coordinates.append(None) # age dimension has no coordinates (just integers, which is fine)

    # ..transitions/equations
    @staticmethod

    def integrate(t, S, E, I, A, M, C, C_icurec, ICU, R, D, H_in, H_out, H_tot, # time + SEIRD classes
                  beta_R, beta_U, beta_M, sigma, omega, zeta, da, dm, dc_R, dc_D, dICU_R, dICU_D, dICUrec, dhospital, Nc_work,# SEIRD parameters
                  area, p,  # spatially stratified parameters. 
                  s, a, h, c, m_C, m_ICU, # age-stratified parameters
                  place, Nc): # stratified parameters that determine stratification dimensions

        """
        BIOMATH extended SEIRD model for COVID-19
        """

        ################################
        ## calculate total population ##
        ################################

        T = S + E + I + A + M + C + C_icurec + ICU + R

        ####################################
        ## Compute the infection pressure ##
        ####################################

        # For total population and for the relevant compartments I and A
        G = place.shape[0] # spatial stratification
        N = Nc.shape[1] # age stratification
        
        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*place + np.identity(G)*np.matmul(place, (1-np.outer(p,p)))
        # infer aggregation (prov, arr or mun)
        agg = None
        if G == 11:
            agg = 'prov'
        elif G == 43:
            agg = 'arr'
        elif G == 581:
            agg = 'mun'
        else:
            raise Exception(f"Space is {G}-fold stratified. This is not recognized as being stratification at Belgian province, arrondissement, or municipality level.")
        # Expand beta to size G
        beta = stratify_beta(beta_R, beta_U, beta_M, agg, area, T.sum(axis=1))
        # Compute populations after application of 'place' to obtain the S, I and A populations
        T_work = np.matmul(np.transpose(place_eff), T)
        S_work = np.matmul(np.transpose(place_eff), S)
        I_work = np.matmul(np.transpose(place_eff), I)
        A_work = np.matmul(np.transpose(place_eff), A)
        # Apply work contacts to place modified populations
        multip_work = np.squeeze( np.matmul(((I_work + A_work)/T_work)[:,np.newaxis,:], Nc_work))
        multip_work *= beta[:,np.newaxis]
        # Apply all other contacts to non-place modified populations
        multip_rest = np.squeeze( np.matmul(((I + A)/T)[:,np.newaxis,:], Nc-Nc_work))
        multip_rest *= beta[:,np.newaxis]
        # Compute rates of change
        dS_inf = S_work * multip_work + S * multip_rest

        ############################
        ## Compute system of ODEs ##
        ############################
        
        ### non-vaccinated population
        dS  = - dS_inf + zeta*R
        dE  = dS_inf - E/sigma
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - A/da
        dM = ((1-a)/omega)*I - M*((1-h)/dm) - M*h/dhospital
        dC = M*(h/dhospital)*c - (1-m_C)*C*(1/(dc_R)) - m_C*C*(1/(dc_D))
        dC_icurec = (1-m_ICU)*ICU/(dICU_R) - C_icurec*(1/dICUrec)
        dICUstar = M*(h/dhospital)*(1-c) - (1-m_ICU)*ICU/(dICU_R) - m_ICU*ICU/(dICU_D)
        dR  = A/da + ((1-h)/dm)*M + (1-m_C)*C*(1/dc_R) + C_icurec*(1/dICUrec) - zeta*R
        dD  = (m_ICU/dICU_D)*ICU + (m_C/dc_D)*C  

        ## Hospital rates of changes
        dH_in = M*(h/dhospital) - H_in
        dH_out =  (1-m_C)*C*(1/dc_R) +  m_C*C*(1/dc_D) + (m_ICU/dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = M*(h/dhospital) - (1-m_C)*C*(1/dc_R) -  m_C*C*(1/dc_D) - (m_ICU/dICU_D)*ICU - C_icurec*(1/dICUrec)

        return (dS, dE, dI, dA, dM, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot)


class COVID19_SEIQRD_spatial_stratified_vacc(BaseModel):
    """
    insert uitleg
    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'C_icurec', 'ICU', 'R', 'D', 'H_in', 'H_out', 'H_tot']
    parameter_names = ['beta_R', 'beta_U', 'beta_M', 'f_VOC', 'f_immune_escape', 'K_inf', 'K_hosp', 'sigma', 'omega', 'zeta', 'da', 'dm', 'dc_R', 'dc_D', 'dICU_R', 'dICU_D', 'dICUrec', 'dhospital', 'N_vacc', 'e_i', 'e_s', 'e_h', 'd_vacc', 'Nc_work']
    parameters_stratified_names = [['area', 'p'], ['s','a','h', 'c', 'm_C','m_ICU'],[]]
    stratification = ['place','Nc','doses'] # mobility and social interaction: name of the dimension (better names: ['nis', 'age'])
    coordinates = ['place', None, None] # 'place' is interpreted as a list of NIS-codes appropriate to the geography

    # ..transitions/equations
    @staticmethod

    def integrate(t, S, E, I, A, M, C, C_icurec, ICU, R, D, H_in, H_out, H_tot, # time + SEIRD classes
                  beta_R, beta_U, beta_M, f_VOC, f_immune_escape, K_inf, K_hosp, sigma, omega, zeta, da, dm, dc_R, dc_D, dICU_R, dICU_D, dICUrec, dhospital, N_vacc, e_i, e_s, e_h, d_vacc, Nc_work,# SEIRD parameters
                  area, p,  # spatially stratified parameters. 
                  s, a, h, c, m_C, m_ICU, # age-stratified parameters
                  place, Nc, doses): # stratified parameters that determine stratification dimensions

        ############################
        ## Modeling immune escape ##
        ############################

        # Remove negative derivatives to ease further computation
        f_VOC[1,:][np.where(f_VOC[1,:] < 0)] = 0

        # Split derivatives and fraction
        d_VOC = f_VOC[1,:]
        f_VOC = f_VOC[0,:]

        #################################################
        ## Compute variant weighted-average properties ##
        #################################################

        # Prepend a 'one' in front of K_inf and K_hosp
        K_inf = np.insert(K_inf, 0, 1)
        K_hosp = np.insert(K_hosp, 0, 1)

        if sum(f_VOC) != 1:
            raise ValueError(
                "The sum of the fractions of the VOCs is not equal to one, please check your time dependant VOC function"
            )
        sigma = np.sum(f_VOC*sigma)
        h = np.sum(np.outer(h, f_VOC*K_hosp),axis=1)
        e_i = np.matmul(f_VOC, e_i)
        e_s = np.matmul(f_VOC, e_s)
        e_h = np.matmul(f_VOC, e_h)

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

        dS = np.zeros(S.shape)
        dR = np.zeros(R.shape)

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
        dR[:,:,1] =  N_vacc[:,:,0]*f_R # 0 --> 1 dose
        # Compute transitioning in two shot circuit
        dS[:,:,2] =  N_vacc[:,:,2]*f_S # 0 --> 2 doses
        dR[:,:,2] =  N_vacc[:,:,2]*f_R # 0 --> 2 doses

        # 1 --> 2 
        # ~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,1] + E[:,:,1] + I[:,:,1] + A[:,:,1] + R[:,:,1]
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,1]/VE
        f_R = R[:,:,1]/VE
        # Compute transitioning in one short circuit
        dS[:,:,1] = dS[:,:,1] - N_vacc[:,:,1]*f_S
        dR[:,:,1] = dR[:,:,1] - N_vacc[:,:,1]*f_R
        # Compute transitioning in two shot circuit
        dS[:,:,2] = dS[:,:,2] + N_vacc[:,:,1]*f_S
        dR[:,:,2] = dR[:,:,2] + N_vacc[:,:,1]*f_R

        # waned vaccine, 2 --> B
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Compute vaccine eligible population
        VE = S[:,:,2]+ R[:,:,2] + S[:,:,3] + R[:,:,3]
        # 2 dose circuit
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,2]/VE
        f_R = R[:,:,2]/VE
        # Compute transitioning in two shot circuit
        dS[:,:,2] = dS[:,:,2] - N_vacc[:,:,3]*f_S
        dR[:,:,2] = dR[:,:,2] - N_vacc[:,:,3]*f_R
        # Compute transitioning in booster circuit
        dS[:,:,4] = dS[:,:,4] + N_vacc[:,:,3]*f_S
        dR[:,:,4] = dR[:,:,4] + N_vacc[:,:,3]*f_R
        # waned vaccine circuit
        # Compute fraction of VE to distribute vaccins
        f_S = S[:,:,3]/VE
        f_R = R[:,:,3]/VE
        # Compute transitioning in two shot circuit
        dS[:,:,3] = dS[:,:,3] - N_vacc[:,:,3]*f_S
        dR[:,:,3] = dR[:,:,3] - N_vacc[:,:,3]*f_R
        # Compute transitioning in booster circuit
        dS[:,:,4] = dS[:,:,4] + N_vacc[:,:,3]*f_S
        dR[:,:,4] = dR[:,:,4] + N_vacc[:,:,3]*f_R

        # Update the S and R state
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        S_post_vacc = S + dS
        R_post_vacc = R + dR

        # Compute dS that makes S and R equal to zero
        dS[np.where(S_post_vacc < 0)] = 0 - S[np.where(S_post_vacc < 0)]
        dR[np.where(R_post_vacc < 0)] = 0 - R[np.where(R_post_vacc < 0)]
        # Set S and R equal to zero
        S_post_vacc[np.where(S_post_vacc < 0)] = 0
        R_post_vacc[np.where(R_post_vacc < 0)] = 0

        ################################
        ## calculate total population ##
        ################################

        T = np.sum(S + E + I + A + M + C + C_icurec + ICU + R, axis=2) # Sum over doses

        ################################
        ## Compute infection pressure ##
        ################################

        # For total population and for the relevant compartments I and A
        G = place.shape[0] # spatial stratification
        N = Nc.shape[1] # age stratification
        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*place + np.identity(G)*np.matmul(place, (1-np.outer(p,p)))
        # infer aggregation (prov, arr or mun)
        agg = None
        if G == 11:
            agg = 'prov'
        elif G == 43:
            agg = 'arr'
        elif G == 581:
            agg = 'mun'
        else:
            raise Exception(f"Space is {G}-fold stratified. This is not recognized as being stratification at Belgian province, arrondissement, or municipality level.")
        # Expand beta to size G
        beta = stratify_beta(beta_R, beta_U, beta_M, agg, area, T.sum(axis=1))*sum(f_VOC*K_inf)
        # Compute populations after application of 'place' to obtain the S, I and A populations
        T_work = np.matmul(np.transpose(place_eff), T)
        T_work = np.expand_dims(T_work, axis=2)
        # I have verified on a dummy example that the following line of code:
        S_work = np.transpose(np.matmul(np.transpose(S_post_vacc), place_eff))
        # Is equivalent to the following for loop:
        # S_work = np.zeros(S.shape)
        #for idx in range(S.shape[2]):
        #    S_work[:,:,idx] = np.matmul(np.transpose(place_eff), S[:,:,idx]) 
        I_work = np.transpose(np.matmul(np.transpose(I), place_eff))
        A_work = np.transpose(np.matmul(np.transpose(A), place_eff))
        infpop = np.sum( (I_work + A_work)/T_work*(1-e_i), axis=2)
        # (11, 10, 10) x (11, 10, 5)
        multip_work = np.matmul(Nc_work, infpop[:,:,np.newaxis])
        multip_work *= beta[:,np.newaxis, np.newaxis]
        # Apply all other contacts to non-place modified populations
        infpop = np.sum( (I + A)/np.expand_dims(T, axis=2)*(1-e_i), axis=2)
        multip_rest = np.matmul(Nc-Nc_work, infpop[:,:,np.newaxis])
        multip_rest *= beta[:,np.newaxis,np.newaxis]
        # Compute rates of change
        dS_inf = (S_work * multip_work + S_post_vacc * multip_rest)*(1-e_s)

        ############################
        ## Compute system of ODEs ##
        ############################

        h_acc = (1-e_h)*h

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

        # Waning of second dose
        r_waning_vacc = 1/((6/12)*365)
        dS[:,:,2] = dS[:,:,2] - r_waning_vacc*S_post_vacc[:,:,2]
        dR[:,:,2] = dR[:,:,2] - r_waning_vacc*R_post_vacc[:,:,2]
        dS[:,:,3] = dS[:,:,3] + r_waning_vacc*S_post_vacc[:,:,2]
        dR[:,:,3] = dR[:,:,3] + r_waning_vacc*R_post_vacc[:,:,2]
        
        # Waning of booster dose
        # No waning of booster dose

        # Waning of natural immunity
        dS[:,:,0] = dS[:,:,0] + zeta*R_post_vacc[:,:,0] 
        dR[:,:,0] = dR[:,:,0] - zeta*R_post_vacc[:,:,0]       

        # Immune escape
        # ~~~~~~~~~~~~~
        dS = dS + sum(f_immune_escape*d_VOC)*R
        dR = dR - sum(f_immune_escape*d_VOC)*R   

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
