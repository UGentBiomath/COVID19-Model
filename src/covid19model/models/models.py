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

from .utils import stratify_beta # read_coordinates_nis, dens_dep
from ..optimization import pso
from .QALY import create_life_table 

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
    Biomath extended SEIRD model for COVID-19, Deterministic implementation
    Can account for re-infection, vaccination and co-infection with a new COVID-19 variant.
    
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
        ICU_tot : intensive care
        R : recovered
        D : deceased
        H_in : new hospitalizations
        H_out : new hospital discharges
        H_tot : total patients in Belgian hospitals
        VE : vaccination eligible states (S + R + E + I + A), needed in time-dependent vaccination function
        V : vaccinated (people that have become immune + part that is not immune but not (yet) infected)
        V_new : newly vaccinated each day
        alpha : fraction of alternative COVID-19 variant

    parameters : dictionary
        containing the values of all parameters (both stratified and not)
        these can be obtained with the function model_parameters.get_COVID19_SEIRD_parameters()

        Non-stratified parameters
        -------------------------
        beta : probability of infection when encountering an infected person
        K : infectivity gain of alternative COVID-19 variants (infectivity of new variant = K * infectivity of old variant)
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
        injection_day : number of days after start of simulation when new strain is injected
        injection_ratio : ratio of new strain vs total amount of virus on injection_day

        Age-stratified parameters
        --------------------
        s: relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU

            Hypothetical vaccination study
            ------------------------------
        v : daily vaccination rate (percentage of population to be vaccinated); used for hypothetical vaccination study
        e : vaccine effectivity
        N_vacc : daily number of people vaccinated in each age group; used for real vaccination study
        leakiness : leakiness of the vaccine (proportion of vaccinated people that contribute to infections)

        Other parameters
        ----------------
        Nc : contact matrix between all age groups in stratification

    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'ER', 'C', 'C_icurec','ICU', 'R', 'D','H_in','H_out','H_tot', 
                    'VE', 'V', 'V_new','alpha']
    parameter_names = ['beta', 'K', 'sigma', 'omega', 'zeta','da', 'dm', 'der', 'dc_R','dc_D','dICU_R', 
                        'dICU_D', 'dICUrec','dhospital', 'injection_day', 'injection_ratio']
    parameters_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU', 'v', 'e','N_vacc', 'leakiness']]
    stratification = ['Nc']

    # ..transitions/equations
    @staticmethod
    def integrate(t, S, E, I, A, M, ER, C, C_icurec, ICU, R, D, H_in, H_out, H_tot, VE, V, V_new, alpha,
                  beta, K, sigma, omega, zeta, da, dm, der, dc_R, dc_D, dICU_R, dICU_D, dICUrec, dhospital, injection_day, injection_ratio, 
                  s, a, h, c, m_C, m_ICU, v, e, N_vacc, leakiness,
                  Nc):
        """
        Biomath extended SEIRD model for COVID-19

        *Deterministic implementation*
        """

        # calculate total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        T = S + E + I + A + M + ER + C + C_icurec + ICU + R + V
        # vaccination eligible states
        VE = S + R + E + I + A

        # Compute infection pressure (IP) of both variants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if Nc is None:
            print(t)
        IP_old = (1-alpha)*beta*s*np.matmul(Nc,((I+A+leakiness*V)/T)) # leakiness
        IP_new = alpha*K*beta*s*np.matmul(Nc,((I+A+leakiness*V)/T))

        # Compute the  rates of change in every population compartment
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dS  = - (IP_old + IP_new)*S + zeta*R - v*e*S - N_vacc/VE*S
        dE  = (IP_old + IP_new)*S - E/sigma - v*e*E - N_vacc/VE*E + (IP_old + IP_new)*(1-e)*V 
        dI = (1/sigma)*E - (1/omega)*I - N_vacc/VE*I
        dA = (a/omega)*I - A/da - N_vacc/VE*A        
        dM = ((1-a)/omega)*I - M*((1-h)/dm) - M*h/dhospital
        dER = M*(h/dhospital) - (1/der)*ER
        dC = c*(1/der)*ER - (1-m_C)*C*(1/dc_R) - m_C*C*(1/dc_D)
        dC_icurec = ((1-m_ICU)/dICU_R)*ICU - C_icurec*(1/dICUrec)
        dICUstar = (1-c)*(1/der)*ER - (1-m_ICU)*ICU/dICU_R - m_ICU*ICU/dICU_D
        dR  = A/da + ((1-h)/dm)*M + (1-m_C)*C*(1/dc_R) + C_icurec*(1/dICUrec) - zeta*R +  v*e*S + v*e*E - N_vacc/VE*R
        dD  = (m_ICU/dICU_D)*ICU + (m_C/dc_D)*C
        dH_in = M*(h/dhospital) - H_in
        dH_out =  (1-m_C)*C*(1/dc_R) +  m_C*C*(1/dc_D) + (m_ICU/dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = M*(h/dhospital) - (1-m_C)*C*(1/dc_R) -  m_C*C*(1/dc_D) - (m_ICU/dICU_D)*ICU - C_icurec*(1/dICUrec)
        dV_new = N_vacc/VE*S + N_vacc/VE*R + N_vacc/VE*E + N_vacc/VE*I + N_vacc/VE*A - V_new
        dV = N_vacc/VE*S + N_vacc/VE*R + N_vacc/VE*E + N_vacc/VE*I + N_vacc/VE*A - (IP_old + IP_new)*(1-e)*V
        dVE = dS + dR + dE + dI + dA
        # Update fraction of new COVID-19 variant
        dalpha = IP_new/(IP_old+IP_new) - alpha
         # If A and I are both zero, a division error occurs
        dalpha[np.isnan(dalpha)] = 0

        # On injection_day, inject injection_ratio new strain to alpha (but only if alpha is still zero)
        if (t >= injection_day) & (alpha.sum().sum()==0):
            dalpha += injection_ratio

        return (dS, dE, dI, dA, dM, dER, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot, dVE, dV, dV_new, dalpha)

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

    state_names = ['S', 'E', 'I', 'A', 'M', 'ER', 'C', 'C_icurec','ICU', 
                   'R', 'D', 'H_in', 'H_out', 'H_tot']
    parameter_names = ['beta', 'sigma', 'omega', 'zeta', 'da', 'dm', 'der', 
                       'dc_R', 'dc_D', 'dICU_R', 'dICU_D', 'dICUrec', 'dhospital']
    parameters_stratified_names = [['s','a','h', 'c', 'm_C','m_ICU']]
    stratification = ['Nc']

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

    
class COVID19_SEIRD_spatial(BaseModel):
    """
    BIOMATH extended SEIRD model for COVID-19, spatially explicit. Based on COVID_SEIRD and Arenas (2020).

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

        Non-stratified parameters
        -------------------------
        beta_R : probability of infection when encountering an infected person in rural environment
        beta_U : probability of infection when encountering an infected person in urban environment
        beta_M : probability of infection when encountering an infected person in metropolitan environment
        K : infectivity gain of alternative COVID-19 variants (infectivity of new variant = K * infectivity of old variant)
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
        xi : factor controlling the contact dependence on density f
        injection_day : number of days after start of simulation when new strain is injected
        injection_ratio : ratio of new strain vs total amount of virus on injection_day

        Age-stratified parameters
        -------------------------
        s: relative susceptibility to infection
        a : probability of a subclinical infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in Cohort (non-ICU)
        m_C : mortality in Cohort
        m_ICU : mortality in ICU
        pi : mobility parameter (1 by default = no measures)
        N_vacc : daily number of people vaccinated in each age group
        e : vaccine effectivity
        leakiness : leakiness of the vaccine (proportion of vaccinated people that contribute to infections)

        Spatially-stratified parameters
        -------------------------------
        place : normalised mobility data. place[g][h] denotes the fraction of the population in patch g that goes to patch h
        area : area[g] is the area of patch g in square kilometers. Used for the density dependence factor f.
        sg : average size of a household per patch. Not used as of yet.


        Other parameters
        ----------------
        Nc : contact matrix between all age groups in stratification

    """

    # ...state variables and parameters

    state_names = ['S', 'E', 'I', 'A', 'M', 'ER', 'C', 'C_icurec','ICU', 'R', 'D','H_in','H_out','H_tot', 'VE', 'V', 'V_new','alpha']
    parameter_names = ['beta_R', 'beta_U', 'beta_M', 'K', 'sigma', 'omega', 'zeta','da', 'dm', 'der','dhospital', 
                        'dc_R', 'dc_D', 'dICU_R', 'dICU_D', 'dICUrec', 'xi', 'injection_day', 'injection_ratio']
    parameters_stratified_names = [['area', 'sg', 'p'], ['s','a','h', 'c', 'm_C','m_ICU', 'v', 'e', 'N_vacc', 'leakiness']]
    stratification = ['place','Nc'] # mobility and social interaction: name of the dimension (better names: ['nis', 'age'])
    coordinates = ['place'] # 'place' is interpreted as a list of NIS-codes appropriate to the geography
    coordinates.append(None) # age dimension has no coordinates (just integers, which is fine)

    # ..transitions/equations
    @staticmethod

    def integrate(t, S, E, I, A, M, ER, C, C_icurec, ICU, R, D, H_in, H_out, H_tot, VE, V, V_new, alpha, # time + SEIRD classes
                  beta_R, beta_U, beta_M, K, sigma, omega, zeta, da, dm, der, dhospital, dc_R, dc_D, 
                        dICU_R, dICU_D, dICUrec, xi, injection_day,  injection_ratio,# SEIRD parameters
                  area, sg, p,  # spatially stratified parameters. Might delete sg later.
                  s, a, h, c, m_C, m_ICU, v, e, N_vacc, leakiness, # age-stratified parameters
                  place, Nc): # stratified parameters that determine stratification dimensions

        """
        BIOMATH extended SEIRD model for COVID-19
        """

        # calculate total population
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        T = S + E + I + A + M + ER + C + C_icurec + ICU + R + V # calculate total population per age bin using 2D array
        VE = S + R + E + I + A

        # Define all the parameters needed to determine the rates of change
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # For total population and for the relevant compartments I and A
        G = place.shape[0] # spatial stratification
        N = Nc.shape[0] # age stratification
        
        # Define effective mobility matrix place_eff from user-defined parameter p[patch]
        place_eff = np.outer(p, p)*place + np.identity(G)*np.matmul(place, (1-np.outer(p,p)))
        # TO DO: add age stratification for p
        
        # Effective population per age class per patch: T[patch][age] due to mobility expressed in place and/or regulated by p[patch]
        T_eff = np.matmul(np.transpose(place_eff), T)
        A_eff = np.matmul(np.transpose(place_eff), A)
        I_eff = np.matmul(np.transpose(place_eff), I)
        V_eff = np.matmul(np.transpose(place_eff), V)
        alpha_eff = np.matmul(np.transpose(place_eff), alpha)
                
        # The number of susceptibles in age class i from patch g that work in patch h. Susc[patch,patch,age]
        Susc = place_eff[:,:,np.newaxis]*S[:,np.newaxis,:]
        V_Susc = place_eff[:,:,np.newaxis]*V[:,np.newaxis,:]
        
        # Density dependence per patch: f[patch]
        T_eff_total = T_eff.sum(axis=1)
        rho = T_eff_total / area
        f = 1 + (1 - np.exp(-xi * rho))

        # Normalisation factor per age class: zi[age]
        # Population per age class
        Ti = T.sum(axis=0)
        zi = Ti / np.matmul(np.transpose(T_eff),f)
        
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
        
        # Define spatially stratified infectivity beta with three degrees of freedom beta_R, beta_U, beta_M, based on stratification
        # Default values for RU_threshold and UM_threshold are taken. beta[patch]
        beta = stratify_beta(beta_R, beta_U, beta_M, agg)
        
        # Define actual beta due to VOCs, which is in general age-dependent. beta_weighted_av[patch,age], 
        beta_weighted_av = (1-alpha_eff)*beta[:,np.newaxis] + alpha_eff*K*beta[:,np.newaxis]
        
        # Define the number of contacts multiplier per patch and age, multip[patch,age]
        multip = np.matmul( (I_eff + A_eff + leakiness*V)/T_eff , np.transpose(Nc) )
        
        # Multiply with correctional term for density f[patch], normalisation per age zi[age], and age-dependent susceptibility s[age]
        multip *= np.outer(f, s*zi)
        
        # Multiply with beta_weighted_av[patch,age]
        # if infectivity depends on VISITED region (beta^h), beta_localised = True
        # if infectivity depends on region of origin (beta^g), beta_localised = False
        # Change this in hard-code depending on preference
        beta_localised = True
        if beta_localised:
            multip *= beta_weighted_av
        else:
            Susc *= beta_weighted_av[:,np.newaxis,:]
            V_Susc *= beta_weighted_av[:,np.newaxis,:]
        
        # So far we have all the interaction happening in the *visited* patch h. We want to know how this affects the people *from* g.
        # We need sum over a patch index h, which is the second index (axis=1). Result is dS_inf[patch,age] and dV_inf[patch,age].
        dS_inf = (Susc * multip[np.newaxis,:,:]).sum(axis=1)
        dV_inf = (V_Susc * multip[np.newaxis,:,:]).sum(axis=1)

        dS  = -dS_inf + zeta*R - N_vacc/VE*S
        dE  = dS_inf - E/sigma - N_vacc/VE*E + (1-e)*dV_inf # Unsuccesful vaccinations are added to Exposed population
        dI = (1/sigma)*E - (1/omega)*I - N_vacc/VE*I
        dA = (a/omega)*I - A/da - N_vacc/VE*A
        dM = ((1-a)/omega)*I - M*((1-h)/dm) - M*h/dhospital
        dER = M*(h/dhospital) - (1/der)*ER
        dC = c*(1/der)*ER - (1-m_C)*C*(1/dc_R) - m_C*C*(1/dc_D)
        dC_icurec = ((1-m_ICU)/dICU_R)*ICU - C_icurec*(1/dICUrec)
        dICUstar = (1-c)*(1/der)*ER - (1-m_ICU)*ICU/dICU_R - m_ICU*ICU/dICU_D
        dR  = A/da + ((1-h)/dm)*M + (1-m_C)*C*(1/dc_R) + C_icurec*(1/dICUrec) - zeta*R - N_vacc/VE*R
        dD  = (m_ICU/dICU_D)*ICU + (m_C/dc_D)*C
        dH_in = M*(h/dhospital) - H_in
        dH_out =  (1-m_C)*C*(1/dc_R) +  m_C*C*(1/dc_D) + (m_ICU/dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = M*(h/dhospital) - (1-m_C)*C*(1/dc_R) -  m_C*C*(1/dc_D) - (m_ICU/dICU_D)*ICU - C_icurec*(1/dICUrec)
        dV_new = N_vacc/VE*S + N_vacc/VE*R + N_vacc/VE*E + N_vacc/VE*I + N_vacc/VE*A - V_new
        dV = N_vacc/VE*S + N_vacc/VE*R + N_vacc/VE*E + N_vacc/VE*I + N_vacc/VE*A - (1-e)*dV_inf
        dVE = dS + dR + dE + dI + dA
        dalpha = alpha*K/(1-alpha+alpha*K) - alpha
        # If A and I are both zero, a division error occurs
        dalpha[np.isnan(dalpha)] = 0
        



        # On injection_day, inject injection_ratio new strain to alpha (but only if alpha is still zero)
        if (t >= injection_day) & (alpha.sum().sum()==0):
            dalpha += injection_ratio
        

        return (dS, dE, dI, dA, dM, dER, dC, dC_icurec, dICUstar, dR, dD, dH_in, dH_out, dH_tot, dVE, dV_new, dV, dalpha)
    
    
class COVID19_SEIRD_sto_spatial(BaseModel):
    """
    TO DO: this needs to be entirely updated!
    
    BIOMATH stochastic extended SEIRD model for COVID-19, spatially explicit. Note: enable discrete=True in model simulation.

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
    parameter_names = ['beta', 'sigma', 'omega', 'zeta','da', 'dm', 'der','dhospital', 'dc_R', 'dc_D', 'dICU_R', 'dICU_D', 'dICUrec', 'xi']
    parameters_stratified_names = [['area', 'sg'], ['s','a','h', 'c', 'm_C','m_ICU', 'pi']]
    stratification = ['place','Nc'] # mobility and social interaction: name of the dimension (better names: ['nis', 'age'])
    coordinates = ['place'] # 'place' is interpreted as a list of NIS-codes appropriate to the geography
    coordinates.append(None) # age dimension has no coordinates (just integers, which is fine)

    # ..transitions/equations
    @staticmethod

    def integrate(t, S, E, I, A, M, ER, C, C_icurec, ICU, R, D, H_in, H_out, H_tot, # time + SEIRD classes
                  beta, sigma, omega, zeta, da, dm, der, dhospital, dc_R, dc_D, dICU_R, dICU_D, dICUrec, xi, # SEIRD parameters
                  area, sg,  # spatially stratified parameters. Might delete sg later.
                  s, a, h, c, m_C, m_ICU, pi, # age-stratified parameters
                  place, Nc): # stratified parameters that determine stratification dimensions

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
        G = place.shape[0] # spatial stratification
        N = Nc.shape[0] # age stratification
        T_eff = np.zeros([G,N]) # initialise
        A_eff = np.zeros([G,N])
        I_eff = np.zeros([G,N])
        for g in range(G):
            for i in range(N):
                sumT = 0
                sumA = 0
                sumI = 0
                # h is taken, so iterator for a sum is gg
                for gg in range(G):
                    term1 = (1 - pi[i]) * np.identity(G)[gg][g]
                    term2 = pi[i] * place[gg][g]
                    sumT += (term1 + term2) * T[gg][i]
                    sumA += (term1 + term2) * A[gg][i]
                    sumI += (term1 + term2) * I[gg][i]
                T_eff[g][i] = sumT
                A_eff[g][i] = sumA
                I_eff[g][i] = sumI

        # Density dependence per patch: f[patch]
        T_eff_total = T_eff.sum(axis=1)
        rho = T_eff_total / area
        f = 1 + (1 - np.exp(-xi * rho))

        # Normalisation factor per age class: zi[age]
        # Population per age class
        Ti = T.sum(axis=0)
        denom = np.zeros(N)
        for gg in range(G):
            value = f[gg] * T_eff[gg]
            denom += value
        zi = Ti / denom

        # The probability to get infected in the 'home patch' when in a particular age class: P[patch][age]
        # initialisation for the summation over all ages below
        argument = np.zeros([G,N])
        for i in range(N):
            for g in range(G):
                summ = 0
                for j in range(N):
                    term = - beta * s[i] * zi[i] * f[g] * Nc[i,j] * (I_eff[g,j] + A_eff[g,j]) / T_eff[g,j] # this used to be T_eff[g,i]
                    summ += term
                argument[g,i] = summ
        P = 1 - np.exp(l*argument) # multiplied by length of timestep
        
        # The probability to get infected in any patch when in a particular age class: Pbis[patch][age]
        Pbis = np.zeros([G,N]) # initialise
        # THIS NEEDS TO BE CHANGED if PLACE BECOMES AGE-STRATIFIED
        for i in range(N):
            for g in range(G):
                summ = 0
                for gg in range(G):
                    term = place[g,gg] * P[gg,i]
                    summ += term
                Pbis[g,i] = summ

        # The total probability bigP[patch][age], depending on mobility parameter pi[age]
        bigP = np.zeros([G,N])
        for i in range(N):
            for g in range(G):
                bigP[g,i] = (1 - pi[i]) * P[g,i] + pi[i] * Pbis[g,i]


        # To be added: effect of average family size (sigma^g or sg)


        # Make a dictionary containing the propensities of the system
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        keys = ['StoE','EtoI','ItoA','ItoM','AtoR','MtoR','MtoER','ERtoC','ERtoICU','CtoR','ICUtoCicurec','CicurectoR','CtoD','ICUtoD','RtoS']


        # Probabilities for a single agent to migrate between SEIR compartments in one unit of the timestep (typically days)
        probabilities = [bigP,
                        (1 - np.exp(- l * (1/sigma) ))*np.ones([G,N]),
                        1 - np.exp(- l * a * (1/omega) )*np.ones([G,N]),
                        1 - np.exp(- l * (1-a)* (1/omega) )*np.ones([G,N]),
                        (1 - np.exp(- l * (1/da) ))*np.ones([G,N]),
                        (1 - np.exp(- l * (1-h)* (1/dm) ))*np.ones([G,N]),
                        1 - np.exp(- l * h * (1/dhospital) )*np.ones([G,N]),
                        1 - np.exp(- l * c * (1/der) )*np.ones([G,N]),
                        1 - np.exp(- l * (1-c) * (1/der) )*np.ones([G,N]),
                        (1 - np.exp(- l * (1-m_C) * (1/dc_R) ))*np.ones([G,N]),
                        (1 - np.exp(- l * (1-m_ICU) * (1/dICU_R) ))*np.ones([G,N]),
                        (1 - np.exp(- l * (1/dICUrec) ))*np.ones([G,N]),
                        (1 - np.exp(- l * m_C * (1/dc_D) ))*np.ones([G,N]),
                        (1 - np.exp(- l * m_ICU * (1/dICU_D) ))*np.ones([G,N]),
                        (1 - np.exp(- l * zeta ))*np.ones([G,N]),
                        ]
        
        states = [S, E, I, I, A, M, M, ER, ER, C, ICU, C_icurec, C, ICU, R]
        propensity={}
        # Calculate propensity for each migration (listed in keys)
        for k in range(len(keys)):
            prop=np.zeros([G,N])
            for g in range(G):
                for i in range(N):
                    # If state is empty, no one can migrate out of it
                    if states[k][g][i]<=0:
                        prop[g,i]=0
                    else:
                        draw=np.array([])
                        # Loop over number of draws. Calculate binomial random number per draw and pick average
                        for l in range(n):
                            draw = np.append(draw,np.random.binomial(states[k][g][i],probabilities[k][g][i]))
                        draw = np.rint(np.mean(draw)) # round to nearest integer
                        prop[g,i] = draw
            # Define migration flow
            propensity.update({keys[k]: np.asarray(prop)})

        # calculate the states at timestep k+1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Add and subtract the ins and outs calculated binomially above
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
        output = (S_new, E_new, I_new, A_new, M_new, ER_new, C_new, C_icurec_new, ICU_new, R_new, D_new, H_in_new, H_out_new, H_tot_new)
        # Any SEIR class with a negative population is brought to zero
        for i in range(len(output)):
            output[i][output[i]<0] = 0

        return output


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
