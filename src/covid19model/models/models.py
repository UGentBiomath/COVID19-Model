# Original implementation by Ryan S. Mcgee can be found using the following link: https://github.com/ryansmcgee/seirsplus
# Copyright (c) 2020 by T.W. Alleman, D. Van Hauwermeiren, BIOMATH, Ghent University. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as numpy
import numpy as np
import scipy as scipy
import scipy.integrate
import pandas as pd
import random
from random import choices
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import copy
import multiprocessing
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from .utils import read_coordinates_nis, dens_dep
from ..optimization import pso

# set color schemes
#From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
orange = "#E69F00"
light_blue = "#56B4E9"
green = "#009E73"
yellow = "#F0E442"
blue = "#0072B2"
red = "#D55E00"
pink = "#CC79A7"
black = "#000000"
Okabe_Ito = (orange, light_blue, green, yellow, blue, red, pink, black)
plt.rcParams["axes.prop_cycle"] = matplotlib.cycler('color', Okabe_Ito)

# increase font sizes
# the code below is not wrong, but kinda annoying if you continuously import
# this model in a notebook using the load_ext magic
#multiplier = 1.5
#keys = ("font.size", )
#for key in keys:
#    plt.rcParams[key] *= multiplier
plt.rcParams["font.size"] = 15
plt.rcParams["lines.linewidth"] = 3


from .base import BaseModel

class COVID19_SEIRD(BaseModel):
    """
    Biomath extended SEIRD model for COVID-19

    Deterministic implementation

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
        ER: emergency room, buffer ward (hospitalized state)
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
        --------------------
        s: relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU

    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'ER', 'C', 'C_icurec','ICU', 'R', 'D','H_in','H_out','H_tot']
    parameter_names = ['beta', 'sigma', 'omega', 'zeta','da', 'dm', 'der', 'dc_R','dc_D','dICU_R', 'dICU_D', 'dICUrec','dhospital']
    parameters_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU']]
    stratification = ['Nc']
    apply_compliance_to = 'Nc'

    # ..transitions/equations
    @staticmethod
    def integrate(t, S, E, I, A, M, ER, C, C_icurec, ICU, R, D, H_in, H_out, H_tot,
                  beta, sigma, omega, zeta, da, dm, der, dc_R, dc_D, dICU_R, dICU_D, dICUrec,
                  dhospital, s, a, h, c, m_C, m_ICU, Nc):
        """
        Biomath extended SEIRD model for COVID-19

        *Deterministic implementation*
        """

        # calculate total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        T = S + E + I + A + M + ER + C + C_icurec + ICU + R

        # Compute the  rates of change in every population compartment
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dS  = - beta*s*np.matmul(Nc,((I+A)/T)*S) + zeta*R
        dE  = beta*s*np.matmul(Nc,((I+A)/T)*S) - E/sigma
        dI = (1/sigma)*E - (1/omega)*I
        dA = (a/omega)*I - A/da
        dM = ((1-a)/omega)*I - M*((1-h)/dm) - M*h/dhospital
        dER = M*(h/dhospital) - (1/der)*ER
        dC = c*(1/der)*ER - (1-m_C)*C*(1/dc_R) - m_C*C*(1/dc_D)
        dC_icurec = ((1-m_ICU)/dICU_R)*ICU - C_icurec*(1/dICUrec)
        dICUstar = (1-c)*(1/der)*ER - (1-m_ICU)*ICU/dICU_R - m_ICU*ICU/dICU_D
        dR  = A/da + ((1-h)/dm)*M + (1-m_C)*C*(1/dc_R) + C_icurec*(1/dICUrec) - zeta*R
        dD  = (m_ICU/dICU_D)*ICU + (m_C/dc_D)*C
        dH_in = M*(h/dhospital) - H_in
        dH_out =  (1-m_C)*C*(1/dc_R) +  m_C*C*(1/dc_D) + (m_ICU/dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = M*(h/dhospital) - (1-m_C)*C*(1/dc_R) -  m_C*C*(1/dc_D) - (m_ICU/dICU_D)*ICU - C_icurec*(1/dICUrec)

        return (dS, dE, dI, dA, dM, dER, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot)

class COVID19_SEIRD_sto(BaseModel):
    """
    Biomath extended SEIRD model for COVID-19

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states
        e.g. {'S': N, 'E': np.ones(n_stratification)} with N being the total population and n_stratifications the number of stratified layers
        initialising zeros is thus not required
    parameters : dictionary
        containing the values of all parameters (both stratified and not)
        these can be obtained with the function parameters.get_COVID19_SEIRD_parameters()

    """

    # ...state variables and parameters

    state_names = ['S', 'E', 'I', 'A', 'M', 'ER', 'C', 'C_icurec','ICU', 'R', 'D', 'H_in', 'H_out', 'H_tot']
    parameter_names = ['beta', 'sigma', 'omega', 'zeta', 'da', 'dm', 'der', 'dc_R', 'dc_D', 'dICU_R', 'dICU_D', 'dICUrec', 'dhospital']
    parameters_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU']]
    stratification = ['Nc']
    apply_compliance_to = 'Nc'

    # ..transitions/equations
    @staticmethod

    def integrate(t, S, E, I, A, M, ER, C, C_icurec, ICU, R, D, H_in, H_out, H_tot,
                  beta, sigma, omega, zeta, da, dm, der, dc_R, dc_D, dICU_R, dICU_D, dICUrec,
                  dhospital, s, a, h, c, m_C, m_ICU, Nc):

        """
        BIOMATH extended SEIRD model for COVID-19

        *Antwerp University stochastic implementation*
        """

        # Define solver parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        l = 1.0 # length of discrete timestep
        n = 1 # number of draws to average in one timestep (slows down calculations but converges to deterministic results when > 20)
        T = S + E + I + A + M + ER + C + C_icurec + ICU + R # calculate total population per age bin using 2D array

        # Make a dictionary containing the transitions and their propensities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        keys = ['StoE','EtoI','ItoA','ItoM','AtoR','MtoR','MtoER','ERtoC','ERtoICU','CtoR','ICUtoCicurec','CicurectoR','CtoD','ICUtoD','RtoS']
        probabilities = [1 - np.exp( - l*s*beta*np.matmul(Nc,((I+A)/T)) ),
                        (1 - np.exp(- l * (1/sigma) ))*np.ones(S.size),
                        1 - np.exp(- l * a * (1/omega) ),
                        1 - np.exp(- l * (1-a)* (1/omega) ),
                        (1 - np.exp(- l * (1/da) ))*np.ones(S.size),
                        (1 - np.exp(- l * (1-h)* (1/dm) ))*np.ones(S.size),
                        1 - np.exp(- l * h * (1/dhospital) ),
                        1 - np.exp(- l * c * (1/der) ),
                        1 - np.exp(- l * (1-c) * (1/der) ),
                        (1 - np.exp(- l * (1-m_C) * (1/dc_R) ))*np.ones(S.size),
                        (1 - np.exp(- l * (1-m_ICU) * (1/dICU_R) ))*np.ones(S.size),
                        (1 - np.exp(- l * (1/dICUrec) ))*np.ones(S.size),
                        (1 - np.exp(- l * m_C * (1/dc_D) ))*np.ones(S.size),
                        (1 - np.exp(- l * m_ICU * (1/dICU_D) ))*np.ones(S.size),
                        (1 - np.exp(- l * zeta ))*np.ones(S.size),
                        ]
        states = [S,E,I,I,A,M,M,ER,ER,C,ICU,C_icurec,C,ICU,R]
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
        S_new  = S - propensity['StoE'] + propensity['RtoS']
        E_new  =  E + propensity['StoE'] - propensity['EtoI']
        I_new =  I + propensity['EtoI'] - propensity['ItoA'] - propensity['ItoM']
        A_new =  A + propensity['ItoA'] - propensity['AtoR']
        M_new =  M + propensity['ItoM'] - propensity['MtoR'] - propensity['MtoER']
        ER_new = ER + propensity['MtoER'] - propensity['ERtoC'] - propensity['ERtoICU']
        C_new =  C + propensity['ERtoC'] - propensity['CtoR'] - propensity['CtoD']
        C_icurec_new =  C_icurec + propensity['ICUtoCicurec'] - propensity['CicurectoR']
        ICU_new =  ICU +  propensity['ERtoICU'] - propensity['ICUtoCicurec'] - propensity['ICUtoD']
        R_new  =  R + propensity['AtoR'] + propensity['MtoR'] + propensity['CtoR'] + propensity['CicurectoR'] - propensity['RtoS']
        D_new  = D +  propensity['ICUtoD'] +  propensity['CtoD']
        H_in_new = propensity['ERtoC'] + propensity['ERtoICU']
        H_out_new = propensity['CtoR'] + propensity['CicurectoR']
        H_tot_new = H_tot + H_in_new - H_out_new - propensity['ICUtoD'] -  propensity['CtoD']

        # Add protection against states < 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        output = (S_new, E_new, I_new, A_new, M_new, ER_new, C_new, C_icurec_new,ICU_new, R_new, D_new,H_in_new,H_out_new,H_tot_new)
        for i in range(len(output)):
            output[i][output[i]<0] = 0

        return output

class COVID19_SEIRD_sto_spatial(BaseModel):
    """
    BIOMATH stochastic extended SEIRD model for COVID-19, spatially explicit

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states
        e.g. {'S': N, 'E': np.ones(n_stratification)} with N being the total population and n_stratifications the number of stratified layers
        initialising zeros is thus not required
    parameters : dictionary
        containing the values of all parameters (both stratified and not)
        these can be obtained with the function parameters.get_COVID19_SEIRD_parameters()

    """

    # ...state variables and parameters

    state_names = ['S', 'E', 'I', 'A', 'M', 'ER', 'C', 'C_icurec','ICU', 'R', 'D','H_in','H_out','H_tot']
    parameter_names = ['beta', 'sigma', 'omega', 'zeta','da', 'dm', 'der','dhospital']
    parameters_stratified_names = [['area', 'sg'], ['s','a','h', 'c', 'm_C','m_ICU', 'dc_R', 'dc_D', 'dICU_R', 'dICU_D', 'dICUrec', 'pi']]
    stratification = ['place','Nc']
    coordinates = [read_coordinates_nis()]
    coordinates.append(None)
    apply_compliance_to = 'Nc'

    # ..transitions/equations
    @staticmethod

    def integrate(t, S, E, I, A, M, ER, C, C_icurec, ICU, R, D, H_in, H_out, H_tot, # time + SEIRD classes
                  beta, sigma, omega, zeta, da, dm, der, dhospital, # SEIRD parameters
                  area, sg,  # spatially stratified parameters
                  s, a, h, c, m_C, m_ICU, dc_R, dc_D, dICU_R, dICU_D, dICUrec, pi, # age-stratified parameters
                  place, Nc): # not really sure?
#     def integrate(t, S, E, I, A, M, ER, C, C_icurec, ICU, R, D, H_in, H_out, H_tot,
#                   beta, sigma, omega, zeta, da, dm, der, dc_R, dc_D, dICU_R, dICU_D, dICUrec,
#                   dhospital, s, a, h, c, m_C, m_ICU, place, Nc):

        """
        BIOMATH extended SEIRD model for COVID-19

        *Antwerp University stochastic implementation*
        """

        # Define solver parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~

        l = 1.0 # length of discrete timestep
        n = 1 # number of draws to average in one timestep (slows down calculations but converges to deterministic results when > 20)
        T = S + E + I + A + M + ER + C + C_icurec + ICU + R # calculate total population per age bin using 2D array

        
        # Define all the parameters needed to calculate the probability for an agent to get infected (following Arenas 2020)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # Effective population per age class per patch: T[patch][age] due to mobility pi[age]
        # For total population and for the relevant compartments I and A
        G = P.shape[0] # spatial stratification
        N = Nc.shape[0] # age stratification
        T_eff = np.zeros(initN.shape) # initialise
        A_eff = np.zeros(initN.shape)
        I_eff = np.zeros(initN.shape)
        for g in range(G):
            for h in range(G):
                # total population
                sum1 = (1 - pi) * np.identity(G)[h][g]
                sum2 = pi * P[h][g]
                T_eff[g] += ( sum1 + sum2 ) * initN[h]
                A_eff[g] += ( sum1 + sum2 ) * A[h]
                I_eff[g] += ( sum1 + sum2 ) * A[h]
                
        # Density dependence parameter per patch: f[patch]
        def dens_dep(rho, xi):
            f = 1 + (1 - np.exp(-xi*rho))
            return f
        # xi is hard-coded for now
        xi = 0.01 # km^-2
        T_eff_total = T_eff.sum(axis=1)
        rho = T_eff_total / area
        f = 1 + (1 - np.exp(-xi * rho))
        
        # Normalisation factor per age class: zi[age]
        Ti = T.sum(axis=0)
        f = dens_dep(T_eff.sum(axis=1) / area, xi)
        denom = np.zeros(N)
        for h in range(G):
            value = f[h] * T_eff[h]
            denom += value
        zi = Ti / denom
        
        # The probability to get infected in the 'home patch' when in a particular age class: P[patch][age]
        # initialisation for the summation over all ages below
        T_eff_temp = np.zeros(Nc.shape)
        A_eff_temp = np.zeros(Nc.shape)
        I_eff_temp = np.zeros(Nc.shape)
        s_temp = np.array([s]*43)
        Nc_temp = np.array([Nc]*43)
        for h in range(G):
            for j in range(N):
                T_eff_temp[h,j] = T_eff[h]
                A_eff_temp[h,j] = A_eff[h]
                I_eff_temp[h,j] = I_eff[h]

        argument = np.zeros([G,N])
        for j in range(N):
            argument += - beta * s * zi * f * Nc[:,:,j] * ( I_eff_temp[:,:,j] + A_eff_temp[:,:,j] ) / T_eff_temp[:,:,j]

        P = 1 - np.exp(argument)
        
        # the probability to get infected in any patch when in a particular age class: Pbis[patch][age]
        Pbis = np.zeros(P.shape) # initialise
        for j in range(N):
            Pbis[:,j] = np.matmul(np.transpose(place), P[:,j])
            
        # the combined probability, depending on the overall mobility
        bigP = np.zeros(P.shape) # initialise
        for h in range(G):
            bigP[h] = (1 - pi) * P[h] + pi * Pbis[h]
            
        # To be added: effect of average family size (sigma^g or sg)
        
        
        # Make a dictionary containing the propensities of the system
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        keys = ['StoE','EtoI','ItoA','ItoM','AtoR','MtoR','MtoER','ERtoC','ERtoICU','CtoR','ICUtoCicurec','CicurectoR','CtoD','ICUtoD','RtoS']
 
        
        matrix_1 = np.zeros([place.shape[0],Nc.shape[0]])
        for i in range(place.shape[0]):
            # matrix_1[i,:] =
            matrix_1[i,:] = -l*s*beta*np.matmul(Nc,((I[i,:]+A[i,:])/T[i,:]))
        matrix_2 = np.matmul(place,matrix_1)

        probabilities = [1 - np.exp(matrix_2),
                        (1 - np.exp(- l * (1/sigma) ))*np.ones([place.shape[0],Nc.shape[0]]),
                        1 - np.exp(- l * a * (1/omega) )*np.ones([place.shape[0],Nc.shape[0]]),
                        1 - np.exp(- l * (1-a)* (1/omega) )*np.ones([place.shape[0],Nc.shape[0]]),
                        (1 - np.exp(- l * (1/da) ))*np.ones([place.shape[0],Nc.shape[0]]),
                        (1 - np.exp(- l * (1-h)* (1/dm) ))*np.ones([place.shape[0],Nc.shape[0]]),
                        1 - np.exp(- l * h * (1/dhospital) )*np.ones([place.shape[0],Nc.shape[0]]),
                        1 - np.exp(- l * c * (1/der) )*np.ones([place.shape[0],Nc.shape[0]]),
                        1 - np.exp(- l * (1-c) * (1/der) )*np.ones([place.shape[0],Nc.shape[0]]),
                        (1 - np.exp(- l * (1-m_C) * (1/dc_R) ))*np.ones([place.shape[0],Nc.shape[0]]),
                        (1 - np.exp(- l * (1-m_ICU) * (1/dICU_R) ))*np.ones([place.shape[0],Nc.shape[0]]),
                        (1 - np.exp(- l * (1/dICUrec) ))*np.ones([place.shape[0],Nc.shape[0]]),
                        (1 - np.exp(- l * m_C * (1/dc_D) ))*np.ones([place.shape[0],Nc.shape[0]]),
                        (1 - np.exp(- l * m_ICU * (1/dICU_D) ))*np.ones([place.shape[0],Nc.shape[0]]),
                        (1 - np.exp(- l * zeta ))*np.ones([place.shape[0],Nc.shape[0]]),
                        ]

        states = [S,E,I,I,A,M,M,ER,ER,C,ICU,C_icurec,C,ICU,R]
        propensity={}
        for i in range(len(keys)):
            prop=np.zeros([place.shape[0],Nc.shape[0]])
            for j in range(place.shape[0]):
                for k in range(Nc.shape[0]):
                    if states[i][j][k]<=0:
                        prop[j,k]=0
                else:
                    draw=np.array([])
                    for l in range(n):
                        draw = np.append(draw,np.random.binomial(states[i][j][k],probabilities[i][j][k]))
                    draw = np.rint(np.mean(draw))
                    prop[j,k] = draw
            propensity.update({keys[i]: np.asarray(prop)})

        # calculate the states at timestep k+1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        S_new  = S - propensity['StoE'] + propensity['RtoS']
        E_new  =  E + propensity['StoE'] - propensity['EtoI']
        I_new =  I + propensity['EtoI'] - propensity['ItoA'] - propensity['ItoM']
        A_new =  A + propensity['ItoA'] - propensity['AtoR']
        M_new =  M + propensity['ItoM'] - propensity['MtoR'] - propensity['MtoER']
        ER_new = ER + propensity['MtoER'] - propensity['ERtoC'] - propensity['ERtoICU']
        C_new =  C + propensity['ERtoC'] - propensity['CtoR'] - propensity['CtoD']
        C_icurec_new =  C_icurec + propensity['ICUtoCicurec'] - propensity['CicurectoR']
        ICU_new =  ICU +  propensity['ERtoICU'] - propensity['ICUtoCicurec'] - propensity['ICUtoD']
        R_new  =  R + propensity['AtoR'] + propensity['MtoR'] + propensity['CtoR'] + propensity['CicurectoR'] - propensity['RtoS']
        D_new  = D +  propensity['ICUtoD'] +  propensity['CtoD']
        H_in_new = propensity['ERtoC'] + propensity['ERtoICU']
        H_out_new = propensity['CtoR'] + propensity['CicurectoR']
        H_tot_new = H_tot + H_in_new - H_out_new - propensity['ICUtoD'] -  propensity['CtoD']

        # Add protection against states < 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        output = (S_new, E_new, I_new, A_new, M_new, ER_new, C_new, C_icurec_new,ICU_new, R_new, D_new,H_in_new,H_out_new,H_tot_new)
        for i in range(len(output)):
            output[i][output[i]<0] = 0

        return output


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
