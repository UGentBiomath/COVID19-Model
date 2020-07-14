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

from .utils import sample_beta_binomial
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
        C : cohort (normal care hospital section)
        Cicurec : cohort after recovery from ICU
        ICU : intensive care
        R : recovered
        D : dead
        ...Q : state in quarantine

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
        dc : average length of a hospital stay when not in ICU
        dICU : average length of a hospital stay in ICU
        dhospital : time before a patient reaches the hospital
        totalTests: number of tests
        psi_FP : probability of a false positive
        psi_PP : probability of a correct test
        dq : days in quarantine

        Stratified parameters
        --------------------
        s: relative susceptibility to infection
        h : probability of hospitalisation for a mild infection
        c : probability of hospitalisation in cohort (non-ICU) (c=1-icu)
        m0 : mortality in ICU
        icu : probability of hospitalisation in ICU

    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'ER', 'C', 'C_icurec',
                   'ICU', 'R', 'D', 'SQ', 'EQ', 'IQ', 'AQ', 'MQ', 'RQ','H_in','H_out','H_tot']
    parameter_names = ['beta', 'sigma', 'omega', 'zeta','da', 'dm', 'der', 'dc_R','dc_D','dICU_R', 'dICU_D', 'dICUrec',
                       'dhospital', 'totalTests', 'psi_FP', 'psi_PP', 'dq']
    parameters_stratified_names = ['s','a','h', 'c', 'm0_C','m0_ICU']
    stratification = 'Nc'
    apply_compliance_to = 'Nc'

    # ..transitions/equations
    @staticmethod
    def integrate(t, S, E, I, A, M, ER, C, C_icurec, ICU, R, D, SQ, EQ, IQ, AQ, MQ, RQ,H_in,H_out,H_tot,
                  beta, sigma, omega, zeta, da, dm, der, dc_R, dc_D, dICU_R, dICU_D, dICUrec,
                  dhospital, totalTests, psi_FP, psi_PP, dq, s, a, h, c, m0_C,m0_ICU, Nc):
        """
        Biomath extended SEIRD model for COVID-19

        *Deterministic implementation*
        """

        # Model equations
        Ctot = C + C_icurec
        # calculate total population per age bin using 2D array
        N = S + E + I + A + M + ER + Ctot + ICU + R + SQ + EQ + IQ + AQ + MQ + RQ
        # calculate the test rates for each pool using the total number of available tests
        nT = S + E + I + A + M + R
        theta_S = totalTests/nT
        theta_S[theta_S > 1] = 1
        theta_E = totalTests/nT
        theta_E[theta_E > 1] = 1
        theta_I = totalTests/nT
        theta_I[theta_I > 1] = 1
        theta_A = totalTests/nT
        theta_A[theta_A > 1] = 1
        theta_M = totalTests/nT
        theta_M[theta_M > 1] = 1
        theta_R = totalTests/nT
        theta_R[theta_R > 1] = 1
        # calculate rates of change using the 2D arrays
        dS  = - beta*s*np.matmul(Nc,((I+A)/N)*S) - theta_S*psi_FP*S + SQ/dq + zeta*R
        dE  = beta*s*np.matmul(Nc,((I+A)/N)*S) - E/sigma - theta_E*psi_PP*E
        dI = (1/sigma)*E - (1/omega)*I - theta_I*psi_PP*I
        dA = (a/omega)*I - A/da - theta_A*psi_PP*A
        dM = ((1-a)/omega)*I - M*((1-h)/dm) - M*h/dhospital - theta_M*psi_PP*M
        dER = (M+MQ)*(h/dhospital) - (1/der)*ER
        dC = c*(1/der)*ER - (1-m0_C)*C*(1/dc_R) - m0_C*C*(1/dc_D)
        dC_icurec = ((1-m0_ICU)/dICU_R)*ICU - C_icurec*(1/dICUrec)
        dICUstar = (1-c)*(1/der)*ER - (1-m0_ICU)*ICU/dICU_R - m0_ICU*ICU/dICU_D
        dR  = A/da + ((1-h)/dm)*M + (1-m0_C)*C*(1/dc_R) + C_icurec*(1/dICUrec) + AQ/dq + MQ*((1-h)/dm) + RQ/dq - zeta*R
        dD  = (m0_ICU/dICU_D)*ICU + (m0_C/dc_D)*C
        dSQ = theta_S*psi_FP*S - SQ/dq
        dEQ = theta_E*psi_PP*E - EQ/sigma
        dIQ = theta_I*psi_PP*I + (1/sigma)*EQ - (1/omega)*IQ
        dAQ = theta_A*psi_PP*A + (a/omega)*IQ - AQ/dq
        dMQ = theta_M*psi_PP*M + ((1-a)/omega)*IQ - ((1-h)/dm)*MQ - (h/dhospital)*MQ
        dRQ = theta_R*psi_FP*R - RQ/dq
        dH_in = (M+MQ)*(h/dhospital) - H_in
        dH_out =  (1-m0_C)*C*(1/dc_R) +  m0_C*C*(1/dc_D) + (m0_ICU/dICU_D)*ICU + C_icurec*(1/dICUrec) - H_out
        dH_tot = (M+MQ)*(h/dhospital) - (1-m0_C)*C*(1/dc_R) -  m0_C*C*(1/dc_D) - (m0_ICU/dICU_D)*ICU - C_icurec*(1/dICUrec)
        return (dS, dE, dI, dA, dM, dER, dC, dC_icurec,
                dICUstar, dR, dD, dSQ, dEQ, dIQ, dAQ, dMQ, dRQ,dH_in,dH_out,dH_tot)

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
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'C_icurec','ICU', 'R', 'D','H_in','H_out']
    parameter_names = ['beta', 'd', 'sigma', 'omega', 'zeta', 'a', 'm', 'da', 'dm', 'dc', 'dICU', 'dICUrec','dhospital']
    parameters_stratified_names = ['s','h', 'c', 'm0', 'icu']
    stratification = 'Nc'
    apply_compliance_to = 'Nc'

    # ..transitions/equations
    @staticmethod
    def integrate(t, S, E, I, A, M, C, C_icurec, ICU, R, D, H_in, H_out,
                  beta, d, sigma, omega, zeta, a, m, da, dm, dc, dICU, dICUrec,
                  dhospital, s, h, c, m0, icu, Nc):
        """
        BIOMATH extended SEIRD model for COVID-19

        *Antwerp University stochastic implementation*
        """
        # length of discrete timestep
        l = 1.0
        # number of draws to average
        n = 12
        # calculate total population per age bin using 2D array
        N = S + E + I + A + M + C + C_icurec + ICU + R

        # Make a dictionary containing the propensities of the system
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        keys = ['StoE','EtoI','ItoA','ItoM','AtoR','MtoR','MtoC','MtoICU','CtoR','ICUtoCicurec','CicurectoR','CtoD','ICUtoD','RtoS']
        probabilities = [1 - np.exp( - l*s*beta*np.matmul(Nc,((I+A)/N)) ),
                        (1 - np.exp(- l * (1/sigma) ))*np.ones(S.size),
                        1 - np.exp(- l * a * (1/omega) ),
                        1 - np.exp(- l * m * (1/omega) ),
                        (1 - np.exp(- l * (1/da) ))*np.ones(S.size),
                        (1 - np.exp(- l * (1/dm) ))*np.ones(S.size),
                        1 - np.exp(- l * h * c * (1/dhospital) ),
                        1 - np.exp(- l * h * (1-c) * (1/dhospital) ),
                        (1 - np.exp(- l * (1-m0) * (1/dc) ))*np.ones(S.size),
                        (1 - np.exp(- l * (1-m0) * (1/dICU) ))*np.ones(S.size),
                        (1 - np.exp(- l * (1/dICUrec) ))*np.ones(S.size),
                        (1 - np.exp(- l * m0 * (1/dc) ))*np.ones(S.size),
                        (1 - np.exp(- l * m0 * (1/dICU) ))*np.ones(S.size),
                        (1 - np.exp(- l * zeta ))*np.ones(S.size),
                        ]
        states = [S,E,I,I,A,M,M,M,C,ICU,C_icurec,C,ICU,R]
        propensity={}
        for i in range(len(keys)):
            prop=[]
            for j in range(S.size):
                if states[i][j]<=0:
                    prop.append(0)
                else:
                    draw=np.array([])
                    for l in range(n):
                        if i == 1:
                            draw = np.append(draw,sample_beta_binomial(states[i][j],probabilities[i][j],d))
                        else:
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
        M_new =  M + propensity['ItoM'] - propensity['MtoR'] - propensity['MtoC'] - propensity['MtoICU']
        C_new =  C + propensity['MtoC'] - propensity['CtoR'] - propensity['CtoD']
        C_icurec_new =  C_icurec + propensity['ICUtoCicurec'] - propensity['CicurectoR']
        ICU_new =  ICU +  propensity['MtoICU'] - propensity['ICUtoCicurec'] - propensity['ICUtoD']
        R_new  =  R + propensity['AtoR'] + propensity['MtoR'] + propensity['CtoR'] + propensity['CicurectoR'] - propensity['RtoS']
        D_new  = D +  propensity['ICUtoD'] +  propensity['CtoD']
        # derived variables
        H_in_new = propensity['MtoC'] + propensity['MtoICU']
        H_out_new = propensity['CtoR'] + propensity['CicurectoR']

        # protection against states < 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        output = (S_new, E_new, I_new, A_new, M_new, C_new, C_icurec_new,ICU_new, R_new, D_new,H_in_new,H_out_new)
        for i in range(len(output)):
            output[i][output[i]<0] = 0

        return output


class SEIRSAgeModel():
    """
    A class to simulate the Deterministic extended SEIRS Model with optionl age-structuring
    =======================================================================================
    Params:
    """

    def __init__(self, initN, beta, sigma, omega, Nc=0, zeta=0,a=0,m=0,h=0,c=0,da=0,dm=0,dc=0,dICU=0,dICUrec=0,dhospital=0,m0=0,totalTests=0,
                psi_FP=0,psi_PP=0,dq=14,initE=0,initI=0,initA=0,initM=0,initC=0,initCicurec=0,initICU=0,initR=0,
                initD=0,initSQ=0,initEQ=0,initIQ=0,initAQ=0,initMQ=0,initRQ=0,monteCarlo=False,n_samples=1):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clinical parameters
        self.beta   = beta
        self.sigma  = sigma
        self.omega = omega
        self.Nc     = Nc
        self.zeta     = zeta
        self.a     = a
        self.m     = m
        self.h     = h
        self.c     = c
        self.da     = da
        self.dm     = dm
        self.dc      = dc
        self.dICU   = dICU
        self.dICUrec = dICUrec
        self.dhospital     = dhospital
        self.m0     = m0

        # Testing-related parameters:
        self.totalTests = totalTests
        self.psi_FP    = psi_FP
        self.psi_PP  = psi_PP
        self.dq     = dq

        # monte-carlo sampling is an attribute of the model
        self.monteCarlo = monteCarlo
        self.n_samples = n_samples

        # added for the compliance function
        self.compliance=False
        self.old_Nc=0
        self.final_Nc=0
        self.t0 = 7
        self.k = 1

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape inital condition in Nc.shape[0] x 1 2D arrays:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initial condition must be an attribute of class: WAS NOT ADDED ORIGINALLY
        self.initN = numpy.reshape(initN,[Nc.shape[0],1])
        self.initE = numpy.reshape(initE,[Nc.shape[0],1])
        self.initI = numpy.reshape(initI,[Nc.shape[0],1])
        self.initA = numpy.reshape(initA,[Nc.shape[0],1])
        self.initM = numpy.reshape(initM,[Nc.shape[0],1])
        initCtot = initC + initCicurec
        self.initC = numpy.reshape(initC,[Nc.shape[0],1])
        self.initCicurec = numpy.reshape(initCicurec,[Nc.shape[0],1])
        self.initCtot = numpy.reshape(initCtot,[Nc.shape[0],1])
        self.initICU = numpy.reshape(initICU,[Nc.shape[0],1])
        self.initR = numpy.reshape(initR,[Nc.shape[0],1])
        self.initD = numpy.reshape(initD,[Nc.shape[0],1])
        self.initSQ = numpy.reshape(initSQ,[Nc.shape[0],1])
        self.initEQ = numpy.reshape(initEQ,[Nc.shape[0],1])
        self.initIQ = numpy.reshape(initIQ,[Nc.shape[0],1])
        self.initAQ = numpy.reshape(initAQ,[Nc.shape[0],1])
        self.initMQ = numpy.reshape(initMQ,[Nc.shape[0],1])
        self.initRQ = numpy.reshape(initRQ,[Nc.shape[0],1])
        self.initH_in = numpy.reshape(numpy.zeros(9),[Nc.shape[0],1])
        self.initH_out = numpy.reshape(numpy.zeros(9),[Nc.shape[0],1])

        #~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~
        self.t       = 0
        self.tmax    = 0 # will be set when run() is called
        self.tseries = numpy.array([0])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # per age category:
        self.N          = self.initN.astype(int)
        self.numE       = self.initE.astype(int)
        self.numI       = self.initI.astype(int)
        self.numA      = self.initA.astype(int)
        self.numM       = self.initM.astype(int)
        self.numCtot       = self.initCtot.astype(int)
        self.numC       = self.initC.astype(int)
        self.numCicurec       = self.initCicurec.astype(int)
        self.numICU       = self.initICU.astype(int)
        self.numR       = self.initR.astype(int)
        self.numD       = self.initD.astype(int)
        self.numSQ      = self.initSQ.astype(int)
        self.numEQ      = self.initEQ.astype(int)
        self.numIQ      = self.initIQ.astype(int)
        self.numAQ     = self.initAQ.astype(int)
        self.numMQ      = self.initMQ.astype(int)
        self.numRQ      = self.initRQ.astype(int)
        self.numS = numpy.reshape(self.N[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numE[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numI[:,-1],[Nc.shape[0],1])- numpy.reshape(self.numA[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numM[:,-1],[Nc.shape[0],1])
        - numpy.reshape(self.numCtot[:,-1],[Nc.shape[0],1]) -  numpy.reshape(self.numICU[:,-1],[Nc.shape[0],1])
        - numpy.reshape(self.numR[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numD[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numSQ[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numEQ[:,-1],[Nc.shape[0],1])
        - numpy.reshape(self.numIQ[:,-1],[Nc.shape[0],1])- numpy.reshape(self.numAQ[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numMQ[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numRQ[:,-1],[Nc.shape[0],1])
        self.numH_in = self.initH_in.astype(int)
        self.numH_out = self.initH_out.astype(int)

    def reset(self):
        Nc = self.Nc

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape inital condition in Nc.shape[0] x 1 2D arrays:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initial condition must be an attribute of class: WAS NOT ADDED ORIGINALLY
        self.initN = numpy.reshape(self.initN,[Nc.shape[0],1])
        self.initE = numpy.reshape(self.initE,[Nc.shape[0],1])
        self.initI = numpy.reshape(self.initI,[Nc.shape[0],1])
        self.initA = numpy.reshape(self.initA,[Nc.shape[0],1])
        self.initM = numpy.reshape(self.initM,[Nc.shape[0],1])
        self.initC = numpy.reshape(self.initC,[Nc.shape[0],1])
        self.initCicurec = numpy.reshape(self.initCicurec,[Nc.shape[0],1])
        initCtot = self.initC + self.initCicurec
        self.initCtot = numpy.reshape(initCtot,[Nc.shape[0],1])
        self.initICU = numpy.reshape(self.initICU,[Nc.shape[0],1])
        self.initR = numpy.reshape(self.initR,[Nc.shape[0],1])
        self.initD = numpy.reshape(self.initD,[Nc.shape[0],1])
        self.initSQ = numpy.reshape(self.initSQ,[Nc.shape[0],1])
        self.initEQ = numpy.reshape(self.initEQ,[Nc.shape[0],1])
        self.initIQ = numpy.reshape(self.initIQ,[Nc.shape[0],1])
        self.initAQ = numpy.reshape(self.initAQ,[Nc.shape[0],1])
        self.initMQ = numpy.reshape(self.initMQ,[Nc.shape[0],1])
        self.initRQ = numpy.reshape(self.initRQ,[Nc.shape[0],1])
        self.initH_in = numpy.reshape(numpy.zeros(9),[Nc.shape[0],1])
        self.initH_out = numpy.reshape(numpy.zeros(9),[Nc.shape[0],1])

        #~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~
        self.t       = 0
        self.tmax    = 0 # will be set when run() is called
        self.tseries = numpy.array([0])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # per age category:
        self.N          = self.initN.astype(int)
        self.numE       = self.initE.astype(int)
        self.numI       = self.initI.astype(int)
        self.numA      = self.initA.astype(int)
        self.numM       = self.initM.astype(int)
        self.numCtot       = self.initCtot.astype(int)
        self.numC       = self.initC.astype(int)
        self.numCicurec       = self.initCicurec.astype(int)
        self.numICU       = self.initICU.astype(int)
        self.numR       = self.initR.astype(int)
        self.numD       = self.initD.astype(int)
        self.numSQ      = self.initSQ.astype(int)
        self.numEQ      = self.initEQ.astype(int)
        self.numIQ      = self.initIQ.astype(int)
        self.numAQ     = self.initAQ.astype(int)
        self.numMQ      = self.initMQ.astype(int)
        self.numRQ      = self.initRQ.astype(int)
        self.numS = numpy.reshape(self.N[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numE[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numI[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numA[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numM[:,-1],[Nc.shape[0],1])
        - numpy.reshape(self.numCtot[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numICU[:,-1],[Nc.shape[0],1])
        - numpy.reshape(self.numR[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numD[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numSQ[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numEQ[:,-1],[Nc.shape[0],1])
        - numpy.reshape(self.numIQ[:,-1],[Nc.shape[0],1])- numpy.reshape(self.numAQ[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numMQ[:,-1],[Nc.shape[0],1]) - numpy.reshape(self.numRQ[:,-1],[Nc.shape[0],1])
        self.numH_in = self.initH_in.astype(int)
        self.numH_out = self.initH_out.astype(int)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    @staticmethod
    def system_dfes(t, variables, beta, sigma,omega, Nc, zeta, a, m, h, c, da, dm, dc, dICU, dICUrec, dhospital, m0, ICU, totalTests, psi_FP, psi_PP, dq):

        # input is a 1D-array
        # first extract seperate variables in 1D-array
        S,E,I,A,M,C,Cicurec,ICU,R,D,SQ,EQ,IQ,AQ,MQ,RQ = variables.reshape(16,Nc.shape[0])
        # reshape all age dependent parameters to a Nc.shape[0]x1 2D-array
        a = numpy.reshape(a,[Nc.shape[0],1])
        m = numpy.reshape(m,[Nc.shape[0],1])
        h = numpy.reshape(h,[Nc.shape[0],1])
        c = numpy.reshape(c,[Nc.shape[0],1])
        m0 = numpy.reshape(m0,[Nc.shape[0],1])
        # reshape all variables to a Nc.shape[0]x1 2D-array
        S = numpy.reshape(S,[Nc.shape[0],1])
        E = numpy.reshape(E,[Nc.shape[0],1])
        I = numpy.reshape(I,[Nc.shape[0],1])
        A = numpy.reshape(A,[Nc.shape[0],1])
        M = numpy.reshape(M,[Nc.shape[0],1])
        C = numpy.reshape(C,[Nc.shape[0],1])
        Cicurec = numpy.reshape(Cicurec,[Nc.shape[0],1])
        ICU = numpy.reshape(ICU,[Nc.shape[0],1])
        R = numpy.reshape(R,[Nc.shape[0],1])
        D = numpy.reshape(D,[Nc.shape[0],1])
        SQ = numpy.reshape(SQ,[Nc.shape[0],1])
        EQ = numpy.reshape(EQ,[Nc.shape[0],1])
        IQ = numpy.reshape(IQ,[Nc.shape[0],1])
        AQ = numpy.reshape(AQ,[Nc.shape[0],1])
        MQ = numpy.reshape(MQ,[Nc.shape[0],1])
        RQ = numpy.reshape(RQ,[Nc.shape[0],1])
        Ctot = C + Cicurec
        # calculate total population per age bin using 2D array
        N   = S + E + I + A + M + Ctot + ICU + R + SQ + EQ + IQ + AQ + MQ + RQ
        # calculate the test rates for each pool using the total number of available tests
        nT = S + E + I + A + M + R
        theta_S = totalTests/nT
        theta_S[theta_S > 1] = 1
        theta_E = totalTests/nT
        theta_E[theta_E > 1] = 1
        theta_I = totalTests/nT
        theta_I[theta_I > 1] = 1
        theta_A = totalTests/nT
        theta_A[theta_A > 1] = 1
        theta_M = totalTests/nT
        theta_M[theta_M > 1] = 1
        theta_R = totalTests/nT
        theta_R[theta_R > 1] = 1
        # calculate rates of change using the 2D arrays
        dS  = - beta*numpy.matmul(Nc,((I+A)/N)*S) - theta_S*psi_FP*S + SQ/dq + zeta*R
        dE  = beta*numpy.matmul(Nc,((I+A)/N)*S) - E/sigma - theta_E*psi_PP*E
        dI = (1/sigma)*E - (1/omega)*I - theta_I*psi_PP*I
        dA = (a/omega)*I - A/da - theta_A*psi_PP*A
        dM = (m/omega)*I - M*((1-h)/dm) - M*h/dhospital - theta_M*psi_PP*M
        dC = c*(M+MQ)*(h/dhospital) - C*(1/dc)
        dICUstar = (1-c)*(M+MQ)*(h/(dhospital)) - ICU/dICU
        dCicurec = ((1-m0)/dICU)*ICU - Cicurec*(1/dICUrec)
        dR  = A/da + ((1-h)/dm)*M + C*(1/dc) + Cicurec*(1/dICUrec) + AQ/dq + MQ*((1-h)/dm) + RQ/dq - zeta*R
        dD  = (m0/dICU)*ICU
        dSQ = theta_S*psi_FP*S - SQ/dq
        dEQ = theta_E*psi_PP*E - EQ/sigma
        dIQ = theta_I*psi_PP*I + (1/sigma)*EQ - (1/omega)*IQ
        dAQ = theta_A*psi_PP*A + (a/omega)*IQ - AQ/dq
        dMQ = theta_M*psi_PP*M + (m/omega)*IQ - ((1-h)/dm)*MQ - (h/dhospital)*MQ
        dRQ = theta_R*psi_FP*R - RQ/dq
        # reshape output back into a 1D array of similar dimension as input
        out = numpy.array([dS,dE,dI,dA,dM,dC,dCicurec,dICUstar,dR,dD,dSQ,dEQ,dIQ,dAQ,dMQ,dRQ])
        out = numpy.reshape(out,16*Nc.shape[0])
        return out

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def logistic(self,t,final,k,t0):
        f = final/(1+numpy.exp(-k*(t-t0)))
        return f

    def compliance_fcn(self,time,old_Nc,final_Nc,k,t0):
        new_Nc = numpy.zeros([old_Nc.shape[0],old_Nc.shape[1]])
        for i in range(old_Nc.shape[0]):
            for j in range(old_Nc.shape[1]):
                new_Nc[i,j] = old_Nc[i,j] + self.logistic(time,1,k,t0)*(final_Nc[i,j]-old_Nc[i,j])
        return new_Nc

    def run_epoch(self, runtime, dt=1):
        if self.compliance == True:
            timer = 0
            while timer < runtime:
                timer = timer + 1
                self.Nc = self.compliance_fcn(timer,self.old_Nc,self.final_Nc,self.k,self.t0)
                # Repeat normal run_epoch function but using 1 day as the timestep
                t_eval    = numpy.arange(start=self.t+1, stop=self.t+2, step=dt)
                t_span          = (self.t, self.t+1)
                init_cond = numpy.array([self.numS[:,-1], self.numE[:,-1], self.numI[:,-1], self.numA[:,-1], self.numM[:,-1], self.numC[:,-1],self.numCicurec[:,-1], self.numICU[:,-1], self.numR[:,-1], self.numD[:,-1], self.numSQ[:,-1], self.numEQ[:,-1],self.numIQ[:,-1], self.numAQ[:,-1], self.numMQ[:,-1], self.numRQ[:,-1]])
                init_cond = numpy.reshape(init_cond,16*self.Nc.shape[0])
                solution        = scipy.integrate.solve_ivp(lambda t, X: SEIRSAgeModel.system_dfes(t, X, self.beta, self.sigma, self.omega, self.Nc, self.zeta, self.a, self.m, self.h, self.c, self.da,
                                    self.dm, self.dc,self.dICU,self.dICUrec,self.dhospital,self.m0,self.ICU,self.totalTests,self.psi_FP,self.psi_PP,self.dq), t_span=[self.t, self.tmax], y0=init_cond, t_eval=t_eval)
                S,E,I,A,M,C,Cicurec,ICU,R,F,SQ,EQ,IQ,AQ,MQ,RQ = numpy.split(numpy.transpose(solution['y']),16,axis=1)
                Ctot = C + Cicurec
                # calculate hospital in and hospital out
                h = self.h
                c = self.c
                m0 = self.m0
                dhospital = self.dhospital
                dc = self.dc
                dICU = self.dICU
                dICUrec = self.dICUrec
                H_in=numpy.zeros([len(M),self.Nc.shape[0]])
                H_out=numpy.zeros([len(M),self.Nc.shape[0]])
                for i in range(len(H_in)):
                    H_in[i,:]=(M[i,:]+MQ[i,:])*(h/dhospital)
                    H_out[i,:] =  C[i,:]*(1/dc) + (m0/dICU)*ICU[i,:] + Cicurec[i,:]*(1/dICUrec)
                # append output
                self.tseries    = numpy.append(self.tseries, solution['t'])
                self.numS       = numpy.append(self.numS, numpy.transpose(S),axis=1)
                self.numE       = numpy.append(self.numE, numpy.transpose(E),axis=1)
                self.numI       = numpy.append(self.numI, numpy.transpose(I),axis=1)
                self.numA       = numpy.append(self.numA, numpy.transpose(A),axis=1)
                self.numM       = numpy.append(self.numM, numpy.transpose(M),axis=1)
                self.numCtot    = numpy.append(self.numCtot, numpy.transpose(Ctot),axis=1)
                self.numC       = numpy.append(self.numC, numpy.transpose(C),axis=1)
                self.numCicurec = numpy.append(self.numCicurec, numpy.transpose(Cicurec),axis=1)
                self.numICU     = numpy.append(self.numICU, numpy.transpose(ICU),axis=1)
                self.numR       = numpy.append(self.numR, numpy.transpose(R),axis=1)
                self.numD       = numpy.append(self.numD, numpy.transpose(F),axis=1)
                self.numSQ      = numpy.append(self.numSQ, numpy.transpose(SQ),axis=1)
                self.numEQ      = numpy.append(self.numEQ, numpy.transpose(EQ),axis=1)
                self.numAQ      = numpy.append(self.numAQ, numpy.transpose(AQ),axis=1)
                self.numMQ      = numpy.append(self.numMQ, numpy.transpose(MQ),axis=1)
                self.numRQ      = numpy.append(self.numRQ, numpy.transpose(RQ),axis=1)
                self.t = self.tseries[-1]
                self.numH_in      = numpy.append(self.numH_in, numpy.transpose(H_in),axis=1)
                self.numH_out      = numpy.append(self.numH_out, numpy.transpose(H_out),axis=1)
                #print(self.numH_out.shape,self.t)
            else:
                self.compliance = False
        else:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Create a list of times at which the ODE solver should output system values.
            # Append this list of times as the model's timeseries
            t_eval    = numpy.arange(start=self.t+1, stop=self.t+runtime, step=dt)
            #print(t_eval)
            # Define the range of time values for the integration:
            t_span          = (self.t, self.t+runtime)

            # Define the initial conditions as the system's current state:
            # (which will be the t=0 condition if this is the first run of this model,
            # else where the last sim left off)
            init_cond = numpy.array([self.numS[:,-1], self.numE[:,-1], self.numI[:,-1], self.numA[:,-1], self.numM[:,-1], self.numC[:,-1],self.numCicurec[:,-1], self.numICU[:,-1], self.numR[:,-1], self.numD[:,-1], self.numSQ[:,-1], self.numEQ[:,-1],self.numIQ[:,-1], self.numAQ[:,-1], self.numMQ[:,-1], self.numRQ[:,-1]])
            init_cond = numpy.reshape(init_cond,16*self.Nc.shape[0])

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Solve the system of differential eqns:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            solution        = scipy.integrate.solve_ivp(lambda t, X: SEIRSAgeModel.system_dfes(t, X, self.beta, self.sigma, self.omega, self.Nc, self.zeta, self.a, self.m, self.h, self.c, self.da,
            self.dm, self.dc,self.dICU,self.dICUrec,self.dhospital,self.m0,self.ICU,self.totalTests,self.psi_FP,self.psi_PP,self.dq), t_span=[self.t, self.tmax], y0=init_cond, t_eval=t_eval)

            # output of size (nTimesteps * Nc.shape[0])
            S,E,I,A,M,C,Cicurec,ICU,R,F,SQ,EQ,IQ,AQ,MQ,RQ = numpy.split(numpy.transpose(solution['y']),16,axis=1)
            Ctot = C + Cicurec

            # calculate hospital in and hospital out
            h = self.h
            c = self.c
            m0 = self.m0
            dhospital = self.dhospital
            dc = self.dc
            dICU = self.dICU
            dICUrec = self.dICUrec
            H_in=numpy.zeros([len(M),self.Nc.shape[0]])
            H_out=numpy.zeros([len(M),self.Nc.shape[0]])
            for i in range(len(H_in)):
                H_in[i,:]=(M[i,:]+MQ[i,:])*(h/dhospital)
                H_out[i,:] =  C[i,:]*(1/dc) + (m0/dICU)*ICU[i,:] + Cicurec[i,:]*(1/dICUrec)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store the solution output as the model's time series and data series:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # transpose before appending
            # append per category:
            self.tseries    = numpy.append(self.tseries, solution['t'])
            self.numS       = numpy.append(self.numS, numpy.transpose(S),axis=1)
            self.numE       = numpy.append(self.numE, numpy.transpose(E),axis=1)
            self.numI       = numpy.append(self.numI, numpy.transpose(I),axis=1)
            self.numA       = numpy.append(self.numA, numpy.transpose(A),axis=1)
            self.numM       = numpy.append(self.numM, numpy.transpose(M),axis=1)
            self.numCtot    = numpy.append(self.numCtot, numpy.transpose(Ctot),axis=1)
            self.numC       = numpy.append(self.numC, numpy.transpose(C),axis=1)
            self.numCicurec = numpy.append(self.numCicurec, numpy.transpose(Cicurec),axis=1)
            self.numICU     = numpy.append(self.numICU, numpy.transpose(ICU),axis=1)
            self.numR       = numpy.append(self.numR, numpy.transpose(R),axis=1)
            self.numD       = numpy.append(self.numD, numpy.transpose(F),axis=1)
            self.numSQ      = numpy.append(self.numSQ, numpy.transpose(SQ),axis=1)
            self.numEQ      = numpy.append(self.numEQ, numpy.transpose(EQ),axis=1)
            self.numAQ      = numpy.append(self.numAQ, numpy.transpose(AQ),axis=1)
            self.numMQ      = numpy.append(self.numMQ, numpy.transpose(MQ),axis=1)
            self.numRQ      = numpy.append(self.numRQ, numpy.transpose(RQ),axis=1)
            self.t = self.tseries[-1]
            self.numH_in      = numpy.append(self.numH_in, numpy.transpose(H_in),axis=1)
            self.numH_out      = numpy.append(self.numH_out, numpy.transpose(H_out),axis=1)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def run(self, T, checkpoints, dt=1, verbose=False):
    #def run(self, T, dt=1, checkpoints=None, verbose=False):

        if(T>0):
            self.tmax += T + 1
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            numCheckpoints = len(checkpoints['t'])
            paramNames = ['beta', 'sigma', 'Nc', 'zeta', 'a', 'm', 'h', 'c','da','dm','dc','dICU','dICUrec','dhospital','m0','totalTests',
                          'psi_FP','psi_PP','dq']
            for param in paramNames:
                # For params that don't have given checkpoint values (or bad value given),
                # set their checkpoint values to the value they have now for all checkpoints.
                if(param not in list(checkpoints.keys())
                    or not isinstance(checkpoints[param], (list, numpy.ndarray))
                    or len(checkpoints[param])!=numCheckpoints):
                    checkpoints[param] = [getattr(self, param)]*numCheckpoints
            # Before using checkpoints, save variables to be changed by method
            beforeChk=[]
            for key in checkpoints.keys():
                if key != 't':
                    beforeChk.append(getattr(self,key))

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(not checkpoints):
            self.run_epoch(runtime=self.tmax, dt=dt)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #print("t = %.2f" % self.t)
            if(verbose):
                print("\t S   = " + str(self.numS[:,-1]))
                print("\t E   = " + str(self.numE[:,-1]))
                print("\t I   = " + str(self.numI[:,-1]))
                print("\t A   = " + str(self.numA[:,-1]))
                print("\t M   = " + str(self.numM[:,-1]))
                print("\t C   = " + str(self.numC[:,-1]))
                print("\t ICU   = " + str(self.numICU[:,-1]))
                print("\t R   = " + str(self.numR[:,-1]))
                print("\t D   = " + str(self.numD[:,-1]))
                print("\t SQ   = " + str(self.numSQ[:,-1]))
                print("\t EQ   = " + str(self.numEQ[:,-1]))
                print("\t IQ   = " + str(self.numIQ[:,-1]))
                print("\t AQ   = " + str(self.numAQ[:,-1]))
                print("\t MQ   = " + str(self.numMQ[:,-1]))
                print("\t RQ   = " + str(self.numRQ[:,-1]))


        else: # checkpoints provided
            for checkpointIdx, checkpointTime in enumerate(checkpoints['t']):
                # Run the sim until the next checkpoint time:
                #print(checkpointTime-self.t)
                self.run_epoch(runtime=checkpointTime-self.t, dt=dt)
                # Having reached the checkpoint, update applicable parameters:
                #print("[Checkpoint: Updating parameters]")
                for param in paramNames:
                    if param == 'Nc':
                        # initialise flag and timer
                        self.compliance = True
                        self.old_Nc = self.Nc
                        self.final_Nc = checkpoints[param][checkpointIdx]
                        #setattr(self, param, checkpoints[param][checkpointIdx])
                    else:
                        setattr(self, param, checkpoints[param][checkpointIdx])

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                if(verbose):
                    print("\t S   = " + str(self.numS[:,-1]))
                    print("\t E   = " + str(self.numE[:,-1]))
                    print("\t I   = " + str(self.numI[:,-1]))
                    print("\t A   = " + str(self.numA[:,-1]))
                    print("\t M   = " + str(self.numM[:,-1]))
                    print("\t C   = " + str(self.numC[:,-1]))
                    print("\t ICU   = " + str(self.numICU[:,-1]))
                    print("\t R   = " + str(self.numR[:,-1]))
                    print("\t D   = " + str(self.numD[:,-1]))
                    print("\t SQ   = " + str(self.numSQ[:,-1]))
                    print("\t EQ   = " + str(self.numEQ[:,-1]))
                    print("\t IQ   = " + str(self.numIQ[:,-1]))
                    print("\t AQ   = " + str(self.numAQ[:,-1]))
                    print("\t MQ   = " + str(self.numMQ[:,-1]))
                    print("\t RQ   = " + str(self.numRQ[:,-1]))

            if(self.t < self.tmax):
                #print(self.t)
                self.run_epoch(runtime=self.tmax-self.t-1, dt=dt)
                # Reset all parameter values that were changed back to their original value
                i = 0
                for key in checkpoints.keys():
                    if key != 't':
                        setattr(self,key,beforeChk[i])
                        i = i+1
        return self

    def sim(self, T, dt=1, checkpoints=None,trace=None):
        tN = int(T) +1
        if trace is not None:
            #Perform input check on trace dictionary
            #Check that all parNames are actual model parameters
            possibleNames = ['beta', 'sigma', 'omega','Nc', 'zeta', 'a', 'm', 'h', 'c','da','dm','dc','dICU','dICUrec','dhospital','m0','totalTests',
                             'psi_FP','psi_PP','dq']
            for key in trace.keys():
                if key not in trace:
                    raise Exception('The parametername provided by user in position {} of trace dictionary is not an actual model parameter. Please check its spelling.'.format(i))
            self.n_samples = len(trace[key])
        else:
            self.n_samples = 1

        # pre-allocate a 3D matrix for the raw results
        self.S = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.E = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.I = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.A = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.M = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.C = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.Cicurec = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.Ctot = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.ICU = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.R = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.D = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.SQ = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.EQ = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.IQ = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.AQ = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.MQ = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.RQ = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.H_in = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        self.H_out = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        # total hospitalised
        self.H = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        # total infected
        self.InfTot = numpy.zeros([self.Nc.shape[0],tN,self.n_samples])
        # pre-allocate a 2D matrix for the results summed over all age bins
        self.sumS = numpy.zeros([tN,self.n_samples])
        self.sumE = numpy.zeros([tN,self.n_samples])
        self.sumI = numpy.zeros([tN,self.n_samples])
        self.sumA = numpy.zeros([tN,self.n_samples])
        self.sumM = numpy.zeros([tN,self.n_samples])
        self.sumC = numpy.zeros([tN,self.n_samples])
        self.sumCicurec = numpy.zeros([tN,self.n_samples])
        self.sumCtot = numpy.zeros([tN,self.n_samples])
        self.sumICU = numpy.zeros([tN,self.n_samples])
        self.sumR = numpy.zeros([tN,self.n_samples])
        self.sumD = numpy.zeros([tN,self.n_samples])
        self.sumSQ = numpy.zeros([tN,self.n_samples])
        self.sumEQ = numpy.zeros([tN,self.n_samples])
        self.sumIQ = numpy.zeros([tN,self.n_samples])
        self.sumAQ = numpy.zeros([tN,self.n_samples])
        self.sumMQ = numpy.zeros([tN,self.n_samples])
        self.sumRQ = numpy.zeros([tN,self.n_samples])
        self.sumH_in = numpy.zeros([tN,self.n_samples])
        self.sumH_out = numpy.zeros([tN,self.n_samples])
        # total hospitalised
        self.sumH = numpy.zeros([tN,self.n_samples])
        # total infected
        self.sumInfTot = numpy.zeros([tN,self.n_samples])
        # simulation loop
        for i in range(self.n_samples):
            if trace is not None:
                for key in trace.keys():
                    lst = list(trace[key])
                    param=random.sample(lst, 1)
                    setattr(self,key,random.sample(lst, 1)[0])

            # reset self to initial conditioin
            self.reset()
            # perform simulation
            self.run(int(T),checkpoints)
            # append raw results to 3D matrix
            self.S[:,:,i] = self.numS
            self.E[:,:,i] = self.numE
            self.I[:,:,i] = self.numI
            self.A[:,:,i] = self.numA
            self.M[:,:,i] = self.numM
            self.C[:,:,i] = self.numC
            self.Cicurec[:,:,i] = self.numCicurec
            self.Ctot[:,:,i] = self.numCtot
            self.ICU[:,:,i] = self.numICU
            self.R[:,:,i] = self.numR
            self.D[:,:,i] = self.numD
            self.SQ[:,:,i] = self.numSQ
            self.EQ[:,:,i] = self.numEQ
            self.IQ[:,:,i] = self.numIQ
            self.AQ[:,:,i] = self.numAQ
            self.MQ[:,:,i] = self.numMQ
            self.RQ[:,:,i] = self.numRQ
            self.H_in[:,:,i] = self.numH_in
            self.H_out[:,:,i] = self.numH_out
            # total hospitalised
            self.H[:,:,i] = self.numCtot + self.numICU
            # total infected
            self.InfTot[:,:,i] = self.numCtot +  self.numICU + self.numI + self.numA + self.numM
            # convert raw results to sums of all age categories
            self.sumS[:,i] = self.numS.sum(axis=0)
            self.sumE[:,i] = self.numE.sum(axis=0)
            self.sumI[:,i] = self.numI.sum(axis=0)
            self.sumA[:,i] = self.numA.sum(axis=0)
            self.sumM[:,i] = self.numM.sum(axis=0)
            self.sumC[:,i] = self.numC.sum(axis=0)
            self.sumCicurec[:,i] = self.numCicurec.sum(axis=0)
            self.sumCtot[:,i] = self.numCtot.sum(axis=0)
            self.sumICU[:,i] = self.numICU.sum(axis=0)
            self.sumR[:,i] = self.numR.sum(axis=0)
            self.sumD[:,i] = self.numD.sum(axis=0)
            self.sumSQ[:,i] = self.numSQ.sum(axis=0)
            self.sumEQ[:,i] = self.numEQ.sum(axis=0)
            self.sumIQ[:,i] = self.numIQ.sum(axis=0)
            self.sumAQ[:,i] = self.numAQ.sum(axis=0)
            self.sumMQ[:,i] = self.numMQ.sum(axis=0)
            self.sumRQ[:,i] = self.numRQ.sum(axis=0)
            self.sumH_in[:,i] = self.numH_in.sum(axis=0)
            self.sumH_out[:,i] = self.numH_out.sum(axis=0)
            # total hospitalised
            self.sumH[:,i] = self.numCtot.sum(axis=0) + self.numICU.sum(axis=0)
            # total infected
            self.sumInfTot[:,i] = self.numCtot.sum(axis=0) + self.numICU.sum(axis=0)+ self.numI.sum(axis=0) + self.numA.sum(axis=0) + self.numM.sum(axis=0)
        return self

    def sampleFromDistribution(self,filename,k):
        df = pd.read_csv(filename)
        x = df.iloc[:,0]
        y = df.iloc[:,1]
        return(numpy.asarray(choices(x, y, k = k)))

    def plotPopulationStatus(self,filename=None,getfig=False):
        # extend with plotting data and using dates (extra argument startDate)
        fig, ax = plt.subplots()
        ax.plot(self.tseries,numpy.mean(self.sumS,axis=1),color=black)
        ax.fill_between(self.tseries, numpy.percentile(self.sumS,90,axis=1), numpy.percentile(self.sumS,10,axis=1),color=black,alpha=0.2)
        ax.plot(self.tseries,numpy.mean(self.sumE,axis=1),color=orange)
        ax.fill_between(self.tseries, numpy.percentile(self.sumE,90,axis=1), numpy.percentile(self.sumE,10,axis=1),color=orange,alpha=0.2)
        #I = self.sumA + self.sumM + self.sumCtot + self.sumMi + self.sumICU
        ax.plot(self.tseries,numpy.mean(self.sumInfTot,axis=1),color=red)
        ax.fill_between(self.tseries, numpy.percentile(self.sumInfTot,90,axis=1), numpy.percentile(self.sumInfTot,10,axis=1),color=red,alpha=0.2)
        ax.plot(self.tseries,numpy.mean(self.sumR,axis=1),color=green)
        ax.fill_between(self.tseries, numpy.percentile(self.sumR,90,axis=1), numpy.percentile(self.sumR,10,axis=1),color=green,alpha=0.2)
        ax.legend(('susceptible','exposed','total infected','immune'), loc="upper left", bbox_to_anchor=(1,1))
        ax.set_xlabel('days')
        ax.set_ylabel('number of patients')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # enable the grid
        plt.grid(True)
        # To specify the number of ticks on both or any single axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if filename is not None:
            plt.savefig(filename,dpi=600,bbox_inches='tight')
        if getfig:
            return fig, ax
        else:
            plt.show()

    def plotInfected(self,asymptomatic=False,mild=False,filename=None,getfig=False):
        # extend with plotting data and using dates (extra argument startDate)
        fig, ax = plt.subplots()
        if asymptomatic is not False:
            ax.plot(self.tseries,numpy.mean(self.sumA,axis=1),color=blue)
            ax.fill_between(self.tseries, numpy.percentile(self.sumA,90,axis=1), numpy.percentile(self.sumA,10,axis=1),color=blue,alpha=0.2)
        if mild is not False:
            ax.plot(self.tseries,numpy.mean(self.sumM,axis=1),color=green)
            ax.fill_between(self.tseries, numpy.percentile(self.sumM,90,axis=1), numpy.percentile(self.sumM,10,axis=1),color=green,alpha=0.2)
        #H = self.sumCtot + self.sumMi + self.sumICU
        ax.plot(self.tseries,numpy.mean(self.sumH,axis=1),color=orange)
        ax.fill_between(self.tseries, numpy.percentile(self.sumH,90,axis=1), numpy.percentile(self.sumH,10,axis=1),color=orange,alpha=0.2)
        ax.plot(self.tseries,numpy.mean(self.sumICU,axis=1),color=red)
        ax.fill_between(self.tseries, numpy.percentile(self.sumICU,90,axis=1), numpy.percentile(self.sumICU,10,axis=1),color=red,alpha=0.2)
        ax.plot(self.tseries,numpy.mean(self.sumD,axis=1),color=black)
        ax.fill_between(self.tseries, numpy.percentile(self.sumD,90,axis=1), numpy.percentile(self.sumD,10,axis=1),color=black,alpha=0.2)
        if mild is not False and asymptomatic is not False:
            legend_labels = ('asymptomatic','mild','hospitalised','ICU','dead')
        elif mild is not False and asymptomatic is False:
            legend_labels = ('mild','hospitalised','ICU','dead')
        elif mild is False and asymptomatic is not False:
            legend_labels = ('asymptomatic','hospitalised','ICU','dead')
        elif mild is False and asymptomatic is False:
            legend_labels = ('hospitalised','ICU','dead')
        ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1,1))
        ax.set_xlabel('days')
        ax.set_ylabel('number of patients')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # enable the grid
        plt.grid(True)
        # To specify the number of ticks on both or any single axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if filename is not None:
            plt.savefig(filename,dpi=600,bbox_inches='tight')
        if getfig:
            return fig, ax
        else:
            plt.show()

    def LSQ(self,thetas,data,parNames,positions,weights,checkpoints):
        # ------------------
        # Prepare simulation
        # ------------------
        # reset all numX
        self.reset()
        # assign estimates to correct variable
        i = 0
        for param in parNames:
            if param == 'extraTime':
                setattr(self,param,int(round(thetas[i])))
            else:
                setattr(self,param,thetas[i])
            i = i + 1
        # Compute length of data
        n = len(data)
        # Compute simulation time --> build in some redundancy here, datasizes don't have to be equal to eachother.
        T = data[0].size+self.extraTime-1
        #print(n,T,thetas,data)
        # ------------------
        # Perform simulation
        # ------------------
        #print(self.dc,self.dICU)
        #print(checkpoints)
        self.sim(T,checkpoints=checkpoints)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumI,self.sumA,self.sumM,self.sumCtot,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ,self.sumH_in,self.sumH_out)

        # ---------------
        # extract results
        # ---------------
        ymodel=[]
        SSE = 0
        for i in range(n):
            som = 0
            for j in positions[i]:
                som = som + numpy.mean(out[j],axis=1).reshape(numpy.mean(out[j],axis=1).size,1)
            ymodel.append(som[self.extraTime:,0].reshape(som[self.extraTime:,0].size,1))
            # calculate quadratic error
            SSE = SSE + weights[i]*sum((ymodel[i]-data[i])**2)
        SSE = SSE[0]
        return(SSE)

    def fit(self,data,parNames,positions,bounds,weights,checkpoints=None,setvar=False,disp=True,polish=True,maxiter=30,popsize=10):
        # -------------------------------
        # Run a series of checks on input
        # -------------------------------
        # Check if data, parNames and positions are lists
        if type(data) is not list or type(parNames) is not list or type(positions) is not list:
            raise Exception('Datatype of arguments data, parNames and positions must be lists. Lists are made by wrapping whatever datatype in square brackets [].')
        # Check that length of positions is equal to the length of data
        if len(data) is not len(positions):
            raise Exception('The number of positions must match the number of dataseries given to function fit.')
        # Check that length of parNames is equal to length of bounds
        if (len(parNames)) is not len(bounds):
            raise Exception('The number of bounds must match the number of parameter names given to function fit.')
        # Check that all parNames are actual model parameters
        possibleNames = ['k','t0','extraTime','beta', 'sigma', 'Nc', 'zeta', 'a', 'm', 'h', 'c','mi','da','dm','dc','dmi','dICU','dICUrec','dmirec','dhospital','m0','maxICU','totalTests',
                        'psi_FP','psi_PP','dq']
        i = 0
        for param in parNames:
            # For params that don't have given checkpoint values (or bad value given),
            # set their checkpoint values to the value they have now for all checkpoints.
            if param not in possibleNames:
                raise Exception('The parametername provided by user in position {} of argument parNames is not an actual model parameter. Please check its spelling.'.format(i))
            i = i + 1

        # ---------------------
        # Run genetic algorithm
        # ---------------------
        #optim_out = scipy.optimize.differential_evolution(self.LSQ, bounds, args=(data,parNames,positions,weights),disp=disp,polish=polish,workers=-1,maxiter=maxiter, popsize=popsize,tol=1e-18)
        #theta_hat = optim_out.x
        p_hat, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = pso.pso(self.LSQ, bounds, args=(data,parNames,positions,weights,checkpoints), swarmsize=popsize, maxiter=maxiter,
                                                                                   processes=multiprocessing.cpu_count(),minfunc=1e-9, minstep=1e-9,debug=True, particle_output=True)
        theta_hat = p_hat

        # ---------------------------------------------------
        # If setattr is True: assign estimated thetas to self
        # ---------------------------------------------------
        if setvar is True:
            #self.extraTime = int(round(theta_hat[0]))
            i = 0
            for param in parNames:
                if param == 'extraTime':
                    setattr(self,param,int(round(theta_hat[i])))
                else:
                    setattr(self,param,theta_hat[i])
                i  = i + 1

        return self,theta_hat

    def plotFit(self,index,data,positions,checkpoints=None,trace=None,dataMkr=['o','v','s','*','^'],modelClr=['green','orange','red','black','blue'],legendText=None,titleText=None,filename=None,getfig=False):
        # ------------------
        # Prepare simulation
        # ------------------
        # reset all numX
        self.reset()
        # Compute number of dataseries
        n = len(data)
        # Compute simulation time
        T = data[0].size+self.extraTime-1

        # ------------------
        # Perform simulation
        # ------------------
        self.sim(T,checkpoints=checkpoints,trace=trace)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumI,self.sumA,self.sumM,self.sumCtot,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ,self.sumH_in,self.sumH_out)

        # -----------
        # Plot result
        # -----------
        # Create shifted index vector using self.extraTime
        timeObj = index[0]
        timestampStr = timeObj.strftime("%Y-%m-%d")
        index_acc = pd.date_range(timestampStr,freq='D',periods=data[0].size + self.extraTime) - datetime.timedelta(days=self.extraTime)
        # Plot figure
        fig, ax = plt.subplots()
        # Plot data
        for i in range(n):
            ax.scatter(index,data[i],color="black",marker=dataMkr[i])
        # Plot model prediction
        for i in range(n):
            ymodel = 0
            for j in positions[i]:
                ymodel = ymodel + out[j]
            ax.plot(index_acc,numpy.mean(ymodel,axis=1),'--',color=modelClr[i])
            ax.fill_between(index_acc,numpy.percentile(ymodel,95,axis=1),
                 numpy.percentile(ymodel,5,axis=1),color=modelClr[i],alpha=0.2)
        # Attributes
        if legendText is not None:
            ax.legend(legendText, loc="upper left", bbox_to_anchor=(1,1))
        if titleText is not None:
            ax.set_title(titleText,{'fontsize':18})
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%Y'))
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
            'rotation', 90)
        ax.set_xlim( index_acc[self.extraTime-3], pd.to_datetime(index_acc[-1]+ datetime.timedelta(days=1)))
        ax.set_ylabel('number of patients')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # enable the grid
        plt.grid(True)
        # To specify the number of ticks on both or any single axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if filename is not None:
            plt.savefig(filename,dpi=600,bbox_inches='tight')
        if getfig:
            return fig, ax
        else:
            plt.show()

    def passInitial(self):
        self.initE = numpy.reshape(numpy.mean(self.E[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initI = numpy.reshape(numpy.mean(self.I[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initA = numpy.reshape(numpy.mean(self.A[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initM = numpy.reshape(numpy.mean(self.M[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initC = numpy.reshape(numpy.mean(self.C[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initCicurec = numpy.reshape(numpy.mean(self.Cicurec[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initICU = numpy.reshape(numpy.mean(self.ICU[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initR = numpy.reshape(numpy.mean(self.R[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initD = numpy.reshape(numpy.mean(self.D[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initSQ = numpy.reshape(numpy.mean(self.SQ[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initEQ = numpy.reshape(numpy.mean(self.EQ[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initIQ = numpy.reshape(numpy.mean(self.IQ[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initAQ = numpy.reshape(numpy.mean(self.AQ[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initMQ = numpy.reshape(numpy.mean(self.MQ[:,-1,:],axis=1),[self.Nc.shape[0],1])
        self.initRQ = numpy.reshape(numpy.mean(self.RQ[:,-1,:],axis=1),[self.Nc.shape[0],1])
        return self

    def constructHorizon(self,thetas,parNames,policy_period):
        # from length of theta list and number of parameters, length of horizon can be calculated
        N = int(len(thetas)/len(parNames))
        # Time
        t = []
        for i in range(N-1):
            t.append(policy_period*(i+1))
        checkpoints = {'t': t}
        # Initialise empty list for every control handle
        for i in range(len(parNames)):
            checkpoints.update({parNames[i] : []})
        # Append to list
        for i in range(len(parNames)):
            if parNames[i] == 'Nc':
                for j in range(0,N):
                    if j == 0:
                        setattr(self, parNames[i],numpy.array([thetas[i*N+j]]))
                    else:
                        checkpoints[parNames[i]].append(numpy.array([thetas[i*N + j]]))
            else:
                for j in range(0,N):
                    if j == 0:
                        setattr(self, parNames[i],numpy.array([thetas[i*N+j]]))
                    else:
                        checkpoints[parNames[i]].append(numpy.array([thetas[i*N + j]]))
        return(checkpoints)

    def constructHorizonRealTimeMPC(self,thetas,parNames,policy_period):
        # from length of theta list and number of parameters, length of horizon can be calculated
        N = int(len(thetas)/len(parNames))
        # Time
        t = []
        for i in range(N):
            t.append(policy_period*i)
        checkpoints = {'t': t}
        # Initialise empty list for every control handle
        for i in range(len(parNames)):
            checkpoints.update({parNames[i] : []})
        # Append to list
        for i in range(len(parNames)):
            if parNames[i] == 'Nc':
                # There is a bug here, parNames[i] is 'Nc' but somehow the if doesn't end up here
                for j in range(0,N):
                    checkpoints[parNames[i]].append(numpy.array([thetas[i*N + j]]))
            else:
                for j in range(0,N):
                    checkpoints[parNames[i]].append(numpy.array([thetas[i*N + j]]))
        return(checkpoints)

    def calcMPCsse(self,thetas,parNames,setpoints,positions,weights,policy_period,P):
        # ------------------------------------------------------
        # Building the prediction horizon checkpoints dictionary
        # ------------------------------------------------------

        # from length of theta list and number of parameters, length of horizon can be calculated
        N = int(len(thetas)/len(parNames))

        # Build prediction horizon
        thetas_lst=[]
        for i in range(len(parNames)):
            for j in range(0,N):
                thetas_lst.append(thetas[i*N + j])
            for k in range(P-N):
                thetas_lst.append(thetas[i*N + j])
        chk = self.constructHorizon(thetas_lst,parNames,policy_period)

        # ------------------
        # Perform simulation
        # ------------------

        # Set correct simtime
        T = chk['t'][-1] + policy_period
        # run simulation
        self.reset()
        self.sim(T,checkpoints=chk)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumI,self.sumA,self.sumM,self.sumCtot,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ)

        # ---------------
        # Calculate error
        # ---------------
        error = 0
        ymodel =[]
        for i in range(len(setpoints)):
            som = 0
            for j in positions[i]:
                som = som + numpy.mean(out[j],axis=1).reshape(numpy.mean(out[j],axis=1).size,1)
            ymodel.append(som.reshape(som.size,1))
            # calculate error
        for i in range(len(ymodel)):
            error = error + weights[i]*(ymodel[i]-setpoints[i])**2
        SSE = sum(error)[0]
        return(SSE)

    def optimizePolicy(self,parNames,bounds,setpoints,positions,weights,policy_period=7,N=6,P=12,disp=True,polish=True,maxiter=100,popsize=20):
        # -------------------------------
        # Run a series of checks on input
        # -------------------------------
        # Check if parNames, bounds, setpoints and positions are lists
        if type(parNames) is not list or type(bounds) is not list or type(setpoints) is not list or type(positions) is not list:
            raise Exception('Datatype of arguments parNames, bounds, setpoints and positions must be lists. Lists are made by wrapping whatever datatype in square brackets [].')
        # Check that length of parNames is equal to the length of bounds
        if len(parNames) is not len(bounds):
            raise Exception('The number of controlled parameters must match the number of bounds given to function MPCoptimize.')
        # Check that length of setpoints is equal to length of positions
        if len(setpoints) is not len(positions):
            raise Exception('The number of output positions must match the number of setpoints names given to function MPCoptimize.')
        # Check that all parNames are actual model parameters
        possibleNames = ['beta', 'sigma', 'Nc', 'zeta', 'a', 'm', 'h', 'c','mi','da','dm','dc','dmi','dICU','dICUrec','dmirec','dhospital','m0','maxICU','totalTests',
                        'psi_FP','psi_PP','dq']
        i = 0
        for param in parNames:
            # For params that don't have given checkpoint values (or bad value given),
            # set their checkpoint values to the value they have now for all checkpoints.
            if param not in possibleNames:
                raise Exception('The parametername provided by user in position {} of argument parNames is not an actual model parameter. Please check its spelling.'.format(i))
            i = i + 1

        # ----------------------------------------------------------------------------------------
        # Convert bounds vector to an appropriate format for scipy.optimize.differential_evolution
        # ----------------------------------------------------------------------------------------
        scipy_bounds=[]
        for i in range(len(parNames)):
            for j in range(N):
                scipy_bounds.append((bounds[i][0],bounds[i][1]))

        # ---------------------
        # Run genetic algorithm
        # ---------------------
        #optim_out = scipy.optimize.differential_evolution(self.calcMPCsse, scipy_bounds, args=(parNames,setpoints,positions,weights,policy_period,P),disp=disp,polish=polish,workers=-1,maxiter=maxiter, popsize=popsize,tol=1e-18)
        #theta_hat = optim_out.x
        p_hat, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = pso.pso(self.calcMPCsse, scipy_bounds, args=(parNames,setpoints,positions,weights,policy_period,P), swarmsize=popsize, maxiter=maxiter,
                                                                                    processes=multiprocessing.cpu_count(),minfunc=1e-9, minstep=1e-9,debug=True, particle_output=True)
        theta_hat = p_hat
        # ---------------------------------------------
        # Assign optimal policy to SEIRSAgeModel object
        # ---------------------------------------------
        self.optimalPolicy = theta_hat
        return(theta_hat)

    def plotOptimalPolicy(self,parNames,setpoints,policy_period,asymptomatic=False,mild=False,filename=None,getfig=False):
        # Construct checkpoints dictionary using the optimalPolicy list
        # Mind that constructHorizon also sets self.Parameters to the first optimal value of every control handle
        # This is done because the first checkpoint cannot be at time 0.
        checkpoints=self.constructHorizon(self.optimalPolicy,parNames,policy_period)
        # First run the simulation
        self.reset()
        self.sim(T=len(checkpoints['t'])*checkpoints['t'][0],checkpoints=checkpoints)

        # Then perform plot
        fig, ax = plt.subplots()
        if asymptomatic is not False:
            ax.plot(self.tseries,numpy.mean(self.sumA,axis=1),color=blue)
            ax.fill_between(self.tseries, numpy.percentile(self.sumA,90,axis=1), numpy.percentile(self.sumA,10,axis=1),color=blue,alpha=0.2)
        if mild is not False:
            ax.plot(self.tseries,numpy.mean(self.sumM,axis=1),color=green)
            ax.fill_between(self.tseries, numpy.percentile(self.sumM,90,axis=1), numpy.percentile(self.sumM,10,axis=1),color=green,alpha=0.2)
        H = self.sumCtot + self.sumICU
        ax.plot(self.tseries,numpy.mean(H,axis=1),color=orange)
        ax.fill_between(self.tseries, numpy.percentile(H,90,axis=1), numpy.percentile(H,10,axis=1),color=orange,alpha=0.2)
        icu = self.sumMi + self.sumICU
        ax.plot(self.tseries,numpy.mean(icu,axis=1),color=red)
        ax.fill_between(self.tseries, numpy.percentile(icu,90,axis=1), numpy.percentile(icu,10,axis=1),color=red,alpha=0.2)
        ax.plot(self.tseries,numpy.mean(self.sumD,axis=1),color=black)
        ax.fill_between(self.tseries, numpy.percentile(self.sumD,90,axis=1), numpy.percentile(self.sumD,10,axis=1),color=black,alpha=0.2)
        if mild is not False and asymptomatic is not False:
            legend_labels = ('asymptomatic','mild','hospitalised','ICU','dead')
        elif mild is not False and asymptomatic is False:
            legend_labels = ('mild','hospitalised','ICU','dead')
        elif mild is False and asymptomatic is not False:
            legend_labels = ('asymptomatic','hospitalised','ICU','dead')
        elif mild is False and asymptomatic is False:
            legend_labels = ('hospitalised','ICU','dead')
        ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1,1))
        ax.set_xlabel('days')
        ax.set_ylabel('number of patients')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # enable the grid
        plt.grid(True)
        # To specify the number of ticks on both or any single axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if filename is not None:
            plt.savefig(filename,dpi=600,bbox_inches='tight')
        if getfig:
            return fig, ax
        else:
            plt.show()

    def mergeDict(self,T,dict1, dict2):
        # length of dict1 is needed later on
        orig_len = len(dict1['t'])
        merged = {}
        # add latest simulation time to dict2
        end = T
        #end = dict1['t'][-1]
        for i in range(len(dict2['t'])):
            dict2['t'][i] = dict2['t'][i]+end
        # merge dictionaries by updating
        temp = {**dict2, **dict1}
        # loop over all key-value pairs
        for key,value in temp.items():
            if key in dict1 and key in dict2:
                for i in range(len(dict2[key])):
                    value.append(dict2[key][i])
                merged[key] = value
            elif key in dict1 and not key in dict2:
                if key != 'Nc':
                    for i in range(len(dict2['t'])):
                        dict1[key].append(getattr(self,key))
                    merged[key] = dict1[key]
                else:
                    for i in range(len(dict2['t'])):
                        dict1[key].append(getattr(self,key))
                    merged[key] = dict1[key]
            elif key in dict2 and not key in dict1:
                if key != 'Nc':
                    for i in range(orig_len):
                        dict2[key].insert(0,getattr(self,key))
                    merged[key] = dict2[key]
                else:
                    for i in range(orig_len):
                        dict2[key].insert(0,getattr(self,key))
                    merged[key] = dict2[key]
        return(merged)

    def realTimeScenario(self,startDate,data,positions,pastPolicy,futurePolicy=None,trace=None,T_extra=14,dataMkr=['o','v','s','*','^'],
                                modelClr=['green','orange','red','black','blue'],legendText=None,titleText=None,filename=None,getfig=False):

        # Initialize a vector of dates starting on the user provided startDate and of length data
        # Calculate length of data to obtain an initial simulation time
        t_data = pd.date_range(startDate, freq='D', periods=data[0].size)
        T = len(t_data) + self.extraTime - 1 + int(T_extra) # number of datapoints

        # make a deepcopy --> if you modify a python dictionary in a function it will be modified globally
        dict1_orig = copy.deepcopy(pastPolicy)
        dict2_orig = copy.deepcopy(futurePolicy)

        # add estimated extraTime to past policy vector
        for i in range(len(dict1_orig['t'])):
            dict1_orig['t'][i] = dict1_orig['t'][i] + self.extraTime

        # Create a merged dictionary accounting for estimated 'extraTime'
        if futurePolicy is not None:
            chk = self.mergeDict((T-int(T_extra)-1),dict1_orig,dict2_orig)
            T = chk['t'][-1]+int(T_extra)
        else:
            chk = pastPolicy
        # ------------------
        # Prepare simulation
        # ------------------
        # reset all numX
        self.reset()

        # ------------------
        # Perform simulation
        # ------------------
        self.sim(T,checkpoints=chk,trace=trace)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumI,self.sumA,self.sumM,self.sumCtot,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ,self.sumH_in,self.sumH_out)

        # -----------
        # Plot result
        # -----------
        # Create shifted index vector using self.extraTime
        t_acc = pd.date_range(startDate,freq='D',periods=T+1)-datetime.timedelta(days=self.extraTime+1)
        # Plot figure
        fig, ax = plt.subplots()
        # Plot data
        for i in range(len(data)):
            ax.scatter(t_data,data[i],color="black",marker=dataMkr[i])
        # Plot model prediction
        for i in range(len(data)):
            ymodel = 0
            for j in positions[i]:
                ymodel = ymodel + out[j]
            ax.plot(t_acc,numpy.mean(ymodel,axis=1),'--',color=modelClr[i])
            ax.fill_between(t_acc,numpy.percentile(ymodel,95,axis=1),
                 numpy.percentile(ymodel,5,axis=1),color=modelClr[i],alpha=0.3)
        # Attributes
        if legendText is not None:
            ax.legend(legendText, loc="upper left", bbox_to_anchor=(1,1))
        if titleText is not None:
            ax.set_title(titleText,{'fontsize':18})
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%Y'))
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
            'rotation', 90)
        ax.set_xlim( t_acc[self.extraTime-3], pd.to_datetime(t_acc[-1]))
        ax.set_ylabel('number of patients')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # enable the grid
        plt.grid(True)
        # To specify the number of ticks on both or any single axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if filename is not None:
            plt.savefig(filename,dpi=600,bbox_inches='tight')
        if getfig:
            return fig, ax
        else:
            plt.show()

    def realTimeMPC(self,startDate,data,positions,pastPolicy,parNames,bounds,setpoints,weights,
                        policy_period=7,N=6,P=12,disp=True,polish=True,maxiter=100,popsize=20,
                        dataMkr=['o','v','s','*','^'],modelClr=['green','orange','red','black','blue'],legendText=None,titleText=None,filename=None,getfig=False):

        # -------------------------------------------------------
        # Step 1: Run simulation untill the end of the dataseries
        # -------------------------------------------------------
        # Initialize a vector of dates starting on the user provided startDate and of length data
        t_data = pd.date_range(startDate, freq='D', periods=data[0].size)
        # Calculate length of data to obtain an initial simulation time
        T = len(t_data) + self.extraTime - 1 # number of datapoints
        # make a deepcopy of pastPolicy
        dict1_orig = copy.deepcopy(pastPolicy)
        # add estimated extraTime to past policy vector
        for i in range(len(dict1_orig['t'])):
            dict1_orig['t'][i] = dict1_orig['t'][i] + self.extraTime - 1
        # reset all numX
        self.reset()
        # run simulation
        self.sim(T,checkpoints=dict1_orig)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumI,self.sumA,self.sumM,self.sumCtot,self.sumMi,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ)

        # ----------------------------------------------------------------------
        # Step 2: Pass population pools to MPC optimiser, save initial condition
        # ----------------------------------------------------------------------
        # Assign self.initX to local variable initX
        initE = self.initE
        initA = self.initA
        initM = self.initM
        initC = self.initC
        initCicurec = self.initCicurec
        initICU = self.initICU
        initR = self.initR
        initD = self.initD
        initSQ = self.initSQ
        initEQ = self.initEQ
        initAQ = self.initAQ
        initMQ = self.initMQ
        initRQ = self.initRQ
        self.passInitial()

        # ---------------------------
        # Step 3: Optimize controller
        # ---------------------------
        self.optimizePolicy(parNames,bounds,setpoints,positions,weights,policy_period,N,P,disp,polish,maxiter,popsize)
        # Write a different constructHorizon function because this does not work very well
        dict2_orig=self.constructHorizonRealTimeMPC(self.optimalPolicy,parNames,policy_period)

        # ---------------------------
        # Step 4: Merge dictionaries
        # ---------------------------
        chk = self.mergeDict((T-1),dict1_orig,dict2_orig)

        # -------------------------------
        # Step 5: Reset initial condition
        # -------------------------------
        # Assign local variable initX back to self.initX
        self.initE = initE
        self.initA = initA
        self.initM = initM
        self.initC = initC
        self.initCicurec = initCicurec
        self.initCmirec = initCmirec
        self.initMi = initMi
        self.initICU = initICU
        self.initR = initR
        self.initD = initD
        self.initSQ = initSQ
        self.initEQ = initEQ
        self.initAQ = initAQ
        self.initMQ = initMQ
        self.initRQ = initRQ

        # ----------------------
        # Step 6: Simulate model
        # ----------------------
        self.reset()
        T = chk['t'][-1]+int(policy_period)
        self.sim(T,checkpoints=chk)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumA,self.sumM,self.sumCtot,self.sumMi,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ)

        # -------------------
        # Step 7: Plot result
        # -------------------
        # Create shifted index vector using self.extraTime
        t_acc = pd.date_range(startDate,freq='D',periods=T+1)-datetime.timedelta(days=self.extraTime-1)
        # Plot figure
        fig, ax = plt.subplots()
        # Plot data
        for i in range(len(data)):
            ax.scatter(t_data,data[i],color="black",marker=dataMkr[i])
        # Plot model prediction
        for i in range(len(data)):
            ymodel = 0
            for j in positions[i]:
                ymodel = ymodel + out[j]
            ax.plot(t_acc,numpy.mean(ymodel,axis=1),'--',color=modelClr[i])
            ax.fill_between(t_acc,numpy.percentile(ymodel,95,axis=1),
                 numpy.percentile(ymodel,5,axis=1),color=modelClr[i],alpha=0.2)
        # Attributes
        if legendText is not None:
            ax.legend(legendText, loc="upper left", bbox_to_anchor=(1,1))
        if titleText is not None:
            ax.set_title(titleText,{'fontsize':18})
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%Y'))
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
            'rotation', 90)
        ax.set_xlim( t_acc[self.extraTime-3], pd.to_datetime(t_acc[-1]))
        ax.set_ylabel('number of patients')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # enable the grid
        plt.grid(True)
        # To specify the number of ticks on both or any single axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if filename is not None:
            plt.savefig(filename,dpi=600,bbox_inches='tight')
        if getfig:
            return fig, ax
        else:
            plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
