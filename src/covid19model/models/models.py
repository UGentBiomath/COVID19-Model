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
import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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



class SEIRSAge(BaseModel):
    """
    Biomath SEIRS model

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        e.g. {'S': N, 'E': np.ones(n_age_classes)} with N total population
        initialising zeros is not required
    parameters : dictionary
        containing the values of all parameters (both stratified and not)
        these can be obtained with the function parameters.get_agemodel_parameters()

    """

    # ...state variables and parameters
    state_names = ['S', 'E', 'I', 'A', 'M', 'C', 'Cicurec',
                   'ICU', 'R', 'D', 'SQ', 'EQ', 'IQ', 'AQ', 'MQ', 'RQ']
    parameter_names = ['beta', 'sigma', 'omega', 'zeta', 'a', 'm', 'da', 'dm', 'dc', 'dICU', 'dICUrec',
                       'dhospital', 'totalTests', 'psi_FP', 'psi_PP', 'dq']
    parameters_stratified_names = ['h', 'c', 'm0', 'icu']
    stratification = 'Nc'

    # ..transitions/equations
    @staticmethod
    def integrate(t, S, E, I, A, M, C, Cicurec, ICU, R, D, SQ, EQ, IQ, AQ, MQ, RQ,
                  beta, sigma, omega, zeta, a, m, da, dm, dc, dICU, dICUrec,
                  dhospital, totalTests, psi_FP, psi_PP, dq, h, c, m0, icu, Nc):
        """Basic SIR model"""

        # Model equations
        Ctot = C + Cicurec
        # calculate total population per age bin using 2D array
        N = S + E + I + A + M + Ctot + ICU + R + SQ + EQ + IQ + AQ + MQ + RQ
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
        dS  = - beta*np.matmul(Nc,((I+A)/N)*S) - theta_S*psi_FP*S + SQ/dq + zeta*R
        dE  = beta*np.matmul(Nc,((I+A)/N)*S) - E/sigma - theta_E*psi_PP*E
        dI = (1/sigma)*E - (1/omega)*I - theta_I*psi_PP*I
        dA = (a/omega)*I - A/da - theta_A*psi_PP*A
        dM = (m/omega)*I - M*((1-h)/dm) - M*h/dhospital - theta_M*psi_PP*M
        dC = c*(M+MQ)*(h/dhospital) - C*(1/dc)
        dCicurec = ((1-m0)/dICU)*ICU - Cicurec*(1/dICUrec)
        dICUstar = (1-c)*(M+MQ)*(h/dhospital) - ICU/dICU
        dR  = A/da + ((1-h)/dm)*M + C*(1/dc) + Cicurec*(1/dICUrec) + AQ/dq + MQ*((1-h)/dm) + RQ/dq - zeta*R
        dD  = (m0/dICU)*ICU
        dSQ = theta_S*psi_FP*S - SQ/dq
        dEQ = theta_E*psi_PP*E - EQ/sigma
        dIQ = theta_I*psi_PP*I + (1/sigma)*EQ - (1/omega)*IQ
        dAQ = theta_A*psi_PP*A + (a/omega)*IQ - AQ/dq
        dMQ = theta_M*psi_PP*M + (m/omega)*IQ - ((1-h)/dm)*MQ - (h/dhospital)*MQ
        dRQ = theta_R*psi_FP*R - RQ/dq

        return (dS, dE, dI, dA, dM, dC, dCicurec,
                dICUstar, dR, dD, dSQ, dEQ, dIQ, dAQ, dMQ, dRQ)


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

    def run_epoch(self, runtime, dt=1):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create a list of times at which the ODE solver should output system values.
        # Append this list of times as the model's timeseries
        t_eval    = numpy.arange(start=self.t+1, stop=self.t+runtime, step=dt)

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
                self.run_epoch(runtime=checkpointTime-self.t, dt=dt)
                # Having reached the checkpoint, update applicable parameters:
                #print("[Checkpoint: Updating parameters]")
                for param in paramNames:
                    setattr(self, param, checkpoints[param][checkpointIdx])

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

            if(self.t < self.tmax):
                self.run_epoch(runtime=self.tmax-self.t, dt=dt)
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
        possibleNames = ['extraTime','beta', 'sigma', 'Nc', 'zeta', 'a', 'm', 'h', 'c','mi','da','dm','dc','dmi','dICU','dICUrec','dmirec','dhospital','m0','maxICU','totalTests',
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

    def plotFit(self,index,data,positions,checkpoints=None,dataMkr=['o','v','s','*','^'],modelClr=['green','orange','red','black','blue'],legendText=None,titleText=None,filename=None,getfig=False):
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
        self.sim(T,checkpoints=checkpoints)
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

class SEIRSNetworkModel():
    """
    A class to simulate the SEIRS Stochastic Network Model
    =====================================================
    Params: G       Network adjacency matrix (numpy array) or Networkx graph object.
    EXTEND LIST
    """
    def __init__(self, G, beta, sigma, omega, initN,zeta=0, p=0,a=0, m=0, h=0, c=0, mi = 0, da=0, dm=0, dc=0, dmi=0, dICU=0, dICUrec=0, dmirec=0,dhospital=0, m0=0,
                    maxICU=0,theta_S=0, theta_E=0, theta_I=0, theta_A=0, theta_M=0, theta_R=0, phi_S=0, phi_E=0, phi_I=0, phi_A=0, phi_M = 0, phi_R=0,psi_FP=0, psi_PP=0,dq=0,initE=1, initI = 0, initA=0,
                    initM=0, initC=0, initCmirec=0, initCicurec=0, initMi = 0, initICU = 0, initR=0, initD=0,initSQ=0, initEQ=0, initIQ=0, initAQ=0, initMQ=0, initRQ=0,
                    monteCarlo=False,repeats=1,node_groups=None):


        #~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Adjacency matrix:
        #~~~~~~~~~~~~~~~~~~~~~~~~
        self.update_G(G)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initiate Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta   = numpy.array(beta).reshape((self.numNodes, 1))  if isinstance(beta, (list, numpy.ndarray)) else numpy.full(fill_value=beta, shape=(self.numNodes,1))
        self.omega  = numpy.array(sigma).reshape((self.numNodes, 1))  if isinstance(omega, (list, numpy.ndarray)) else numpy.full(fill_value=omega, shape=(self.numNodes,1))
        self.zeta     = numpy.array(zeta).reshape((self.numNodes, 1))    if isinstance(zeta, (list, numpy.ndarray)) else numpy.full(fill_value=zeta, shape=(self.numNodes,1))
        self.p      = numpy.array(p).reshape((self.numNodes, 1))     if isinstance(p, (list, numpy.ndarray)) else numpy.full(fill_value=p, shape=(self.numNodes,1))
        self.a  = numpy.array(a).reshape((self.numNodes, 1)) if isinstance(a, (list, numpy.ndarray)) else numpy.full(fill_value=a, shape=(self.numNodes,1))
        self.m  = numpy.array(m).reshape((self.numNodes, 1)) if isinstance(m, (list, numpy.ndarray)) else numpy.full(fill_value=m, shape=(self.numNodes,1))
        self.h  = numpy.array(h).reshape((self.numNodes, 1)) if isinstance(h, (list, numpy.ndarray)) else numpy.full(fill_value=h, shape=(self.numNodes,1))
        self.c  = numpy.array(c).reshape((self.numNodes, 1)) if isinstance(c, (list, numpy.ndarray)) else numpy.full(fill_value=c, shape=(self.numNodes,1))
        self.mi  = numpy.array(mi).reshape((self.numNodes, 1)) if isinstance(mi, (list, numpy.ndarray)) else numpy.full(fill_value=mi, shape=(self.numNodes,1))
        self.da  = numpy.array(da).reshape((self.numNodes, 1)) if isinstance(da, (list, numpy.ndarray)) else numpy.full(fill_value=da, shape=(self.numNodes,1))
        self.dm  = numpy.array(dm).reshape((self.numNodes, 1)) if isinstance(dm, (list, numpy.ndarray)) else numpy.full(fill_value=dm, shape=(self.numNodes,1))
        self.dc  = numpy.array(dc).reshape((self.numNodes, 1)) if isinstance(dc, (list, numpy.ndarray)) else numpy.full(fill_value=dc, shape=(self.numNodes,1))
        self.dmi  = numpy.array(dmi).reshape((self.numNodes, 1)) if isinstance(dmi, (list, numpy.ndarray)) else numpy.full(fill_value=dmi, shape=(self.numNodes,1))
        self.dICU  = numpy.array(dICU).reshape((self.numNodes, 1)) if isinstance(dICU, (list, numpy.ndarray)) else numpy.full(fill_value=dICU, shape=(self.numNodes,1))
        self.dICUrec  = numpy.array(dICUrec).reshape((self.numNodes, 1)) if isinstance(dICUrec, (list, numpy.ndarray)) else numpy.full(fill_value=dICUrec, shape=(self.numNodes,1))
        self.dmirec  = numpy.array(dmirec).reshape((self.numNodes, 1)) if isinstance(dmirec, (list, numpy.ndarray)) else numpy.full(fill_value=dmirec, shape=(self.numNodes,1))
        self.dhospital  = numpy.array(dhospital).reshape((self.numNodes, 1)) if isinstance(dhospital, (list, numpy.ndarray)) else numpy.full(fill_value=dhospital, shape=(self.numNodes,1))
        self.m0  = numpy.array(m0).reshape((self.numNodes, 1)) if isinstance(m0, (list, numpy.ndarray)) else numpy.full(fill_value=m0, shape=(self.numNodes,1))
        self.maxICU  = numpy.array(maxICU).reshape((self.numNodes, 1)) if isinstance(maxICU, (list, numpy.ndarray)) else numpy.full(fill_value=maxICU, shape=(self.numNodes,1))
        self.theta_S  = numpy.array(theta_S).reshape((self.numNodes, 1)) if isinstance(theta_S, (list, numpy.ndarray)) else numpy.full(fill_value=theta_S, shape=(self.numNodes,1))
        self.theta_E  = numpy.array(theta_E).reshape((self.numNodes, 1)) if isinstance(theta_E, (list, numpy.ndarray)) else numpy.full(fill_value=theta_E, shape=(self.numNodes,1))
        self.theta_I  = numpy.array(theta_I).reshape((self.numNodes, 1)) if isinstance(theta_I, (list, numpy.ndarray)) else numpy.full(fill_value=theta_I, shape=(self.numNodes,1))
        self.theta_A  = numpy.array(theta_A).reshape((self.numNodes, 1)) if isinstance(theta_A, (list, numpy.ndarray)) else numpy.full(fill_value=theta_A, shape=(self.numNodes,1))
        self.theta_M  = numpy.array(theta_M).reshape((self.numNodes, 1)) if isinstance(theta_M, (list, numpy.ndarray)) else numpy.full(fill_value=theta_M, shape=(self.numNodes,1))
        self.theta_R  = numpy.array(theta_R).reshape((self.numNodes, 1)) if isinstance(theta_R, (list, numpy.ndarray)) else numpy.full(fill_value=theta_R, shape=(self.numNodes,1))
        self.phi_S  = numpy.array(phi_S).reshape((self.numNodes, 1)) if isinstance(phi_S, (list, numpy.ndarray)) else numpy.full(fill_value=phi_S, shape=(self.numNodes,1))
        self.phi_E  = numpy.array(phi_E).reshape((self.numNodes, 1)) if isinstance(phi_E, (list, numpy.ndarray)) else numpy.full(fill_value=phi_E, shape=(self.numNodes,1))
        self.phi_I  = numpy.array(phi_I).reshape((self.numNodes, 1)) if isinstance(phi_I, (list, numpy.ndarray)) else numpy.full(fill_value=phi_I, shape=(self.numNodes,1))
        self.phi_A  = numpy.array(phi_A).reshape((self.numNodes, 1)) if isinstance(phi_A, (list, numpy.ndarray)) else numpy.full(fill_value=phi_A, shape=(self.numNodes,1))
        self.phi_M  = numpy.array(phi_M).reshape((self.numNodes, 1)) if isinstance(phi_M, (list, numpy.ndarray)) else numpy.full(fill_value=phi_M, shape=(self.numNodes,1))
        self.phi_R  = numpy.array(phi_R).reshape((self.numNodes, 1)) if isinstance(phi_R, (list, numpy.ndarray)) else numpy.full(fill_value=phi_R, shape=(self.numNodes,1))
        self.psi_FP  = numpy.array(psi_FP).reshape((self.numNodes, 1)) if isinstance(psi_FP, (list, numpy.ndarray)) else numpy.full(fill_value=psi_FP, shape=(self.numNodes,1))
        self.psi_PP  = numpy.array(psi_PP).reshape((self.numNodes, 1)) if isinstance(psi_PP, (list, numpy.ndarray)) else numpy.full(fill_value=psi_PP, shape=(self.numNodes,1))
        self.dq  = numpy.array(dq).reshape((self.numNodes, 1)) if isinstance(dq, (list, numpy.ndarray)) else numpy.full(fill_value=dq, shape=(self.numNodes,1))

        # monte-carlo sampling is an attribute of the model
        self.monteCarlo = monteCarlo
        # node-groups should also ben an attribute of the model
        self.node_groups = node_groups
        # initN is used to extrapolate results to given population
        self.initN = initN
        # number of repeats
        self.repeats = repeats

        if self.monteCarlo is True:
            self.sigma = numpy.full(fill_value=self.sampleFromDistribution('../data/incubation.csv',self.numNodes).reshape((self.numNodes, 1)), shape=(self.numNodes,1))
        else:
            self.sigma  = numpy.array(sigma).reshape((self.numNodes, 1)) if isinstance(sigma, (list, numpy.ndarray)) else numpy.full(fill_value=sigma, shape=(self.numNodes,1))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each node can undergo up to 4 transitions (sans vitality/re-susceptibility returns to S state),
        # so there are ~numNodes*4 events/timesteps expected; initialize numNodes*5 timestep slots to start
        # (will be expanded during run if needed)
        self.tseries = numpy.zeros(5*self.numNodes)
        self.N      = numpy.zeros(5*self.numNodes)
        self.numS   = numpy.zeros(5*self.numNodes)
        self.numE   = numpy.zeros(5*self.numNodes)
        self.numI   = numpy.zeros(5*self.numNodes)
        self.numA   = numpy.zeros(5*self.numNodes)
        self.numM   = numpy.zeros(5*self.numNodes)
        self.numC   = numpy.zeros(5*self.numNodes)
        self.numCmirec   = numpy.zeros(5*self.numNodes)
        self.numCicurec   = numpy.zeros(5*self.numNodes)
        self.numMi   = numpy.zeros(5*self.numNodes)
        self.numICU   = numpy.zeros(5*self.numNodes)
        self.numR   = numpy.zeros(5*self.numNodes)
        self.numD   = numpy.zeros(5*self.numNodes)
        self.numSQ   = numpy.zeros(5*self.numNodes)
        self.numEQ   = numpy.zeros(5*self.numNodes)
        self.numIQ   = numpy.zeros(5*self.numNodes)
        self.numAQ   = numpy.zeros(5*self.numNodes)
        self.numMQ   = numpy.zeros(5*self.numNodes)
        self.numRQ   = numpy.zeros(5*self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t      = 0
        self.tmax   = 0 # will be set when run() is called
        self.tidx   = 0
        self.tseries[0] = 0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initial condition must be an attribute of class: WAS NOT ADDED ORIGINALLY
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.initE = initE
        self.initI = initI
        self.initA = initA
        self.initM = initM
        self.initC = initC
        self.initCmirec = initCmirec
        self.initCicurec = initCicurec
        self.initMi = initMi
        self.initICU = initICU
        self.initR = initR
        self.initD = initD
        self.initSQ = initSQ
        self.initEQ = initEQ
        self.initIQ = initIQ
        self.initAQ = initAQ
        self.initMQ = initMQ
        self.initRQ = initRQ

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.numE[0] = int(initE)
        self.numI[0] = int(initI)
        self.numA[0] = int(initA)
        self.numM[0] = int(initM)
        self.numC[0] = int(initC)
        self.numCmirec[0] = int(initCmirec)
        self.numCicurec[0] = int(initCicurec)
        self.numMi[0] = int(initMi)
        self.numICU[0] = int(initICU)
        self.numR[0] = int(initR)
        self.numD[0] = int(initD)
        self.numSQ[0] = int(initSQ)
        self.numEQ[0] = int(initEQ)
        self.numIQ[0] = int(initIQ)
        self.numAQ[0] = int(initAQ)
        self.numMQ[0] = int(initMQ)
        self.numRQ[0] = int(initRQ)

        self.numS[0] = self.numNodes - self.numE[0] - self.numI[0]  - self.numA[0] - self.numM[0] - self.numC[0] - self.numCmirec[0] - self.numCicurec[0] - self.numMi[0] - self.numICU[0] - self.numSQ[0] - self.numEQ[0] - self.numIQ[0]  - self.numAQ[0] - self.numMQ[0]- self.numRQ[0]- self.numR[0] - self.numD[0]
        self.N[0]    = self.numS[0] + self.numE[0] + self.numI[0] + self.numA[0] + self.numM[0] + self.numC[0] + self.numCmirec[0] + self.numCicurec[0]  + self.numSQ[0] + self.numEQ[0] + self.numIQ[0] + self.numAQ[0] + self.numMQ[0] + self.numRQ[0] + self.numR[0]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Node states:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.S      = 1
        self.E      = 2
        self.I      = 3
        self.A      = 4
        self.M      = 5
        self.C      = 6
        self.Cmirec = 7
        self.Cicurec= 8
        self.Mi     = 9
        self.ICU    = 10
        self.SQ     = 11
        self.EQ     = 12
        self.IQ     = 13
        self.AQ     = 14
        self.MQ     = 15
        self.RQ     = 16
        self.R      = 17
        self.D      = 18
        self.X = numpy.array([self.S]*int(self.numS[0]) + [self.E]*int(self.numE[0]) + [self.I]*int(self.numI[0]) + [self.A]*int(self.numA[0]) + [self.M]*int(self.numM[0])
        + [self.C]*int(self.numC[0]) + [self.Cmirec]*int(self.numCmirec[0]) + [self.Cicurec]*int(self.numCicurec[0])+ [self.Mi]*int(self.numMi[0])
        + [self.ICU]*int(self.numICU[0]) + [self.SQ]*int(self.numSQ[0]) + [self.EQ]*int(self.numEQ[0]) + [self.IQ]*int(self.numIQ[0])+ [self.AQ]*int(self.numAQ[0])
        + [self.MQ]*int(self.numMQ[0]) + [self.RQ]*int(self.numRQ[0]) + [self.R]*int(self.numR[0]) + [self.D]*int(self.numD[0])).reshape((self.numNodes,1))
        numpy.random.shuffle(self.X)

        self.transitions =  {
                                'StoE': {'currentState':self.S, 'newState':self.E},
                                'EtoI': {'currentState':self.E, 'newState':self.I},
                                'ItoA': {'currentState':self.I, 'newState':self.A},
                                'ItoM': {'currentState':self.I, 'newState':self.M},
                                'MtoC': {'currentState':self.M, 'newState':self.C},
                                'MtoMi': {'currentState':self.M, 'newState':self.Mi},
                                'MtoICU': {'currentState':self.M, 'newState':self.ICU},
                                'AtoR': {'currentState':self.A, 'newState':self.R},
                                'MtoR': {'currentState':self.M, 'newState':self.R},
                                'MitoCmirec': {'currentState':self.Mi, 'newState':self.Cmirec},
                                'ICUtoCicurec': {'currentState':self.ICU, 'newState':self.Cicurec},
                                'ICUtoD': {'currentState':self.ICU, 'newState':self.D},
                                'CtoR': {'currentState':self.C, 'newState':self.R},
                                'CmirectoR': {'currentState':self.Cmirec, 'newState':self.R},
                                'CicurectoR': {'currentState':self.Cicurec, 'newState':self.R},
                                'StoSQ': {'currentState':self.S, 'newState':self.SQ},
                                'ItoIQ': {'currentState':self.I, 'newState':self.IQ},
                                'EtoEQ': {'currentState':self.E, 'newState':self.EQ},
                                'AtoAQ': {'currentState':self.A, 'newState':self.AQ},
                                'MtoMQ': {'currentState':self.M, 'newState':self.MQ},
                                'RtoRQ': {'currentState':self.R, 'newState':self.RQ},
                                'SQtoS': {'currentState':self.SQ, 'newState':self.S},
                                'EQtoIQ': {'currentState':self.EQ, 'newState':self.IQ},
                                'IQtoAQ': {'currentState':self.IQ, 'newState':self.AQ},
                                'IQtoMQ': {'currentState':self.IQ, 'newState':self.MQ},
                                'MQtoC': {'currentState':self.MQ, 'newState':self.C},
                                'MQtoMi': {'currentState':self.MQ, 'newState':self.Mi},
                                'MQtoICU': {'currentState':self.MQ, 'newState':self.ICU},
                                'MQtoR': {'currentState':self.MQ, 'newState':self.R},
                                'AQtoR': {'currentState':self.AQ, 'newState':self.R},
                                'RQtoR': {'currentState':self.AQ, 'newState':self.R},
                                'RtoS': {'currentState':self.R, 'newState':self.S},
                                '_toS': {'currentState':True, 'newState':self.S},
                            }

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node subgroup data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nodeGroupData = None
        if(node_groups):
            self.nodeGroupData = {}
            for groupName, nodeList in node_groups.items():
                self.nodeGroupData[groupName] = {'nodes':   numpy.array(nodeList),
                                                 'mask':    numpy.isin(range(self.numNodes), nodeList).reshape((self.numNodes,1))}
                self.nodeGroupData[groupName]['numS']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numE']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numI']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numA']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numM']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numC']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numCmirec']  = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numCicurec'] = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numMi']      = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numICU']     = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numSQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numEQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numIQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numAQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numMQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numRQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numR']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['N']          = numpy.zeros(5*self.numNodes)

                self.nodeGroupData[groupName]['numS'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numA'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.A)
                self.nodeGroupData[groupName]['numM'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.M)
                self.nodeGroupData[groupName]['numC'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.C)
                self.nodeGroupData[groupName]['numCmirec'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Cmirec)
                self.nodeGroupData[groupName]['numCicurec'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Cicurec)
                self.nodeGroupData[groupName]['numMi'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Mi)
                self.nodeGroupData[groupName]['numICU'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.ICU)
                self.nodeGroupData[groupName]['numSQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.SQ)
                self.nodeGroupData[groupName]['numEQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.EQ)
                self.nodeGroupData[groupName]['numIQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.IQ)
                self.nodeGroupData[groupName]['numAQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.AQ)
                self.nodeGroupData[groupName]['numMQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.MQ)
                self.nodeGroupData[groupName]['numRQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.RQ)
                self.nodeGroupData[groupName]['numR'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numD'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D)
                self.nodeGroupData[groupName]['N'][0]       = self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0]
                + self.nodeGroupData[groupName]['numA'][0] + self.nodeGroupData[groupName]['numM'][0] + self.nodeGroupData[groupName]['numC'][0] + self.nodeGroupData[groupName]['numCmirec'][0] + self.nodeGroupData[groupName]['numCicurec'][0] + self.nodeGroupData[groupName]['numMi'][0]
                + self.nodeGroupData[groupName]['numICU'][0] + self.nodeGroupData[groupName]['numSQ'][0] + self.nodeGroupData[groupName]['numEQ'][0] + self.nodeGroupData[groupName]['numIQ'][0]
                + self.nodeGroupData[groupName]['numAQ'][0] + self.nodeGroupData[groupName]['numMQ'][0] +  self.nodeGroupData[groupName]['numRQ'][0] + self.nodeGroupData[groupName]['numR'][0]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def reset(self):
        node_groups = self.node_groups

        # A function which re-initialises the network with the initial conditions
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each node can undergo up to 4 transitions (sans vitality/re-susceptibility returns to S state),
        # so there are ~numNodes*4 events/timesteps expected; initialize numNodes*5 timestep slots to start
        # (will be expanded during run if needed)
        self.tseries = numpy.zeros(5*self.numNodes)
        self.N      = numpy.zeros(5*self.numNodes)
        self.numS   = numpy.zeros(5*self.numNodes)
        self.numE   = numpy.zeros(5*self.numNodes)
        self.numI   = numpy.zeros(5*self.numNodes)
        self.numA   = numpy.zeros(5*self.numNodes)
        self.numM   = numpy.zeros(5*self.numNodes)
        self.numC   = numpy.zeros(5*self.numNodes)
        self.numCmirec   = numpy.zeros(5*self.numNodes)
        self.numCicurec   = numpy.zeros(5*self.numNodes)
        self.numMi   = numpy.zeros(5*self.numNodes)
        self.numICU   = numpy.zeros(5*self.numNodes)
        self.numR   = numpy.zeros(5*self.numNodes)
        self.numD   = numpy.zeros(5*self.numNodes)
        self.numSQ   = numpy.zeros(5*self.numNodes)
        self.numEQ   = numpy.zeros(5*self.numNodes)
        self.numIQ   = numpy.zeros(5*self.numNodes)
        self.numAQ   = numpy.zeros(5*self.numNodes)
        self.numMQ   = numpy.zeros(5*self.numNodes)
        self.numRQ   = numpy.zeros(5*self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t      = 0
        self.tmax   = 0 # will be set when run() is called
        self.tidx   = 0
        self.tseries[0] = 0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.numE[0] = int(self.initE)
        self.numI[0] = int(self.initI)
        self.numA[0] = int(self.initA)
        self.numM[0] = int(self.initM)
        self.numC[0] = int(self.initC)
        self.numCmirec[0] = int(self.initCmirec)
        self.numCicurec[0] = int(self.initCicurec)
        self.numMi[0] = int(self.initMi)
        self.numICU[0] = int(self.initICU)
        self.numR[0] = int(self.initR)
        self.numD[0] = int(self.initD)
        self.numSQ[0] = int(self.initSQ)
        self.numEQ[0] = int(self.initEQ)
        self.numIQ[0] = int(self.initIQ)
        self.numAQ[0] = int(self.initAQ)
        self.numMQ[0] = int(self.initMQ)
        self.numRQ[0] = int(self.initRQ)
        self.numS[0] = self.numNodes - self.numE[0] - self.numI[0] - self.numA[0] - self.numM[0] - self.numC[0] - self.numCmirec[0] - self.numCicurec[0] - self.numMi[0] - self.numICU[0]- self.numSQ[0] - self.numEQ[0] - self.numIQ[0] - self.numAQ[0] - self.numMQ[0]- self.numRQ[0]- self.numR[0] - self.numD[0]
        self.N[0]    = self.numS[0] + self.numE[0] + self.numI[0] + self.numA[0] + self.numM[0] + self.numC[0] + self.numCmirec[0] + self.numCicurec[0]  + self.numSQ[0] + self.numEQ[0] + self.numIQ[0] + self.numAQ[0] + self.numMQ[0] + self.numRQ[0] + self.numR[0]


        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Node states:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.S      = 1
        self.E      = 2
        self.I      = 3
        self.A      = 4
        self.M      = 5
        self.C      = 6
        self.Cmirec = 7
        self.Cicurec= 8
        self.Mi     = 9
        self.ICU    = 10
        self.SQ     = 11
        self.EQ     = 12
        self.IQ     = 13
        self.AQ     = 14
        self.MQ     = 15
        self.RQ     = 16
        self.R      = 17
        self.D      = 18
        self.X = numpy.array([self.S]*int(self.numS[0]) + [self.E]*int(self.numE[0]) + [self.I]*int(self.numI[0]) + [self.A]*int(self.numA[0]) + [self.M]*int(self.numM[0])
        + [self.C]*int(self.numC[0]) + [self.Cmirec]*int(self.numCmirec[0]) + [self.Cicurec]*int(self.numCicurec[0])+ [self.Mi]*int(self.numMi[0])
        + [self.ICU]*int(self.numICU[0]) + [self.SQ]*int(self.numSQ[0]) + [self.EQ]*int(self.numEQ[0]) + [self.IQ]*int(self.numIQ[0])+ [self.AQ]*int(self.numAQ[0])
        + [self.MQ]*int(self.numMQ[0]) + [self.RQ]*int(self.numRQ[0]) + [self.R]*int(self.numR[0]) + [self.D]*int(self.numD[0])).reshape((self.numNodes,1))
        numpy.random.shuffle(self.X)

        self.transitions =  {
                                'StoE': {'currentState':self.S, 'newState':self.E},
                                'EtoI': {'currentState':self.E, 'newState':self.I},
                                'ItoA': {'currentState':self.I, 'newState':self.A},
                                'ItoM': {'currentState':self.I, 'newState':self.M},
                                'MtoC': {'currentState':self.M, 'newState':self.C},
                                'MtoMi': {'currentState':self.M, 'newState':self.Mi},
                                'MtoICU': {'currentState':self.M, 'newState':self.ICU},
                                'AtoR': {'currentState':self.A, 'newState':self.R},
                                'MtoR': {'currentState':self.M, 'newState':self.R},
                                'MitoCmirec': {'currentState':self.Mi, 'newState':self.Cmirec},
                                'ICUtoCicurec': {'currentState':self.ICU, 'newState':self.Cicurec},
                                'ICUtoD': {'currentState':self.ICU, 'newState':self.D},
                                'CtoR': {'currentState':self.C, 'newState':self.R},
                                'CmirectoR': {'currentState':self.Cmirec, 'newState':self.R},
                                'CicurectoR': {'currentState':self.Cicurec, 'newState':self.R},
                                'StoSQ': {'currentState':self.S, 'newState':self.SQ},
                                'ItoIQ': {'currentState':self.I, 'newState':self.IQ},
                                'EtoEQ': {'currentState':self.E, 'newState':self.EQ},
                                'AtoAQ': {'currentState':self.A, 'newState':self.AQ},
                                'MtoMQ': {'currentState':self.M, 'newState':self.MQ},
                                'RtoRQ': {'currentState':self.R, 'newState':self.RQ},
                                'SQtoS': {'currentState':self.SQ, 'newState':self.S},
                                'EQtoIQ': {'currentState':self.EQ, 'newState':self.IQ},
                                'IQtoAQ': {'currentState':self.IQ, 'newState':self.AQ},
                                'IQtoMQ': {'currentState':self.IQ, 'newState':self.MQ},
                                'MQtoC': {'currentState':self.MQ, 'newState':self.C},
                                'MQtoMi': {'currentState':self.MQ, 'newState':self.Mi},
                                'MQtoICU': {'currentState':self.MQ, 'newState':self.ICU},
                                'MQtoR': {'currentState':self.MQ, 'newState':self.R},
                                'AQtoR': {'currentState':self.AQ, 'newState':self.R},
                                'RQtoR': {'currentState':self.AQ, 'newState':self.R},
                                'RtoS': {'currentState':self.R, 'newState':self.S},
                                '_toS': {'currentState':True, 'newState':self.S},
                            }

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node subgroup data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nodeGroupData = None
        if(node_groups):
            self.nodeGroupData = {}
            for groupName, nodeList in node_groups.items():
                self.nodeGroupData[groupName] = {'nodes':   numpy.array(nodeList),
                                                 'mask':    numpy.isin(range(self.numNodes), nodeList).reshape((self.numNodes,1))}
                self.nodeGroupData[groupName]['numS']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numE']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numI']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numA']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numM']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numC']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numCmirec']  = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numCicurec'] = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numMi']      = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numICU']     = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numSQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numEQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numIQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numAQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numMQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numRQ']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numR']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['numD']       = numpy.zeros(5*self.numNodes)
                self.nodeGroupData[groupName]['N']          = numpy.zeros(5*self.numNodes)

                self.nodeGroupData[groupName]['numS'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numA'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.A)
                self.nodeGroupData[groupName]['numM'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.M)
                self.nodeGroupData[groupName]['numC'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.C)
                self.nodeGroupData[groupName]['numCmirec'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Cmirec)
                self.nodeGroupData[groupName]['numCicurec'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Cicurec)
                self.nodeGroupData[groupName]['numMi'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Mi)
                self.nodeGroupData[groupName]['numICU'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.ICU)
                self.nodeGroupData[groupName]['numSQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.SQ)
                self.nodeGroupData[groupName]['numEQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.EQ)
                self.nodeGroupData[groupName]['numIQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.IQ)
                self.nodeGroupData[groupName]['numAQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.AQ)
                self.nodeGroupData[groupName]['numMQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.MQ)
                self.nodeGroupData[groupName]['numRQ'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.RQ)
                self.nodeGroupData[groupName]['numR'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numD'][0]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D)
                self.nodeGroupData[groupName]['N'][0]       = self.nodeGroupData[groupName]['numS'][0] + self.nodeGroupData[groupName]['numE'][0] + self.nodeGroupData[groupName]['numI'][0]
                + self.nodeGroupData[groupName]['numA'][0] + self.nodeGroupData[groupName]['numM'][0] + self.nodeGroupData[groupName]['numC'][0] + self.nodeGroupData[groupName]['numCmirec'][0] + self.nodeGroupData[groupName]['numCicurec'][0] + self.nodeGroupData[groupName]['numMi'][0]
                + self.nodeGroupData[groupName]['numICU'][0] + self.nodeGroupData[groupName]['numSQ'][0] + self.nodeGroupData[groupName]['numEQ'][0] + self.nodeGroupData[groupName]['numIQ'][0]
                + self.nodeGroupData[groupName]['numAQ'][0] + self.nodeGroupData[groupName]['numMQ'][0] +  self.nodeGroupData[groupName]['numRQ'][0] + self.nodeGroupData[groupName]['numR'][0]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def node_degrees(self, Amat):
        return Amat.sum(axis=0).reshape(self.numNodes,1)   # sums of adj matrix cols

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def update_G(self, new_G):
        self.G = new_G
        # Adjacency matrix:
        if type(new_G)==numpy.ndarray:
            self.Adj = scipy.sparse.csr_matrix(new_G)
        elif type(new_G)==networkx.classes.graph.Graph:
            self.Adj = networkx.adj_matrix(new_G) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes   = int(self.Adj.shape[1])
        self.degree     = numpy.asarray(self.node_degrees(self.Adj)).astype(float)

        return

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # De eigenlijke vergelijkingen!
    def calc_propensities(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication

        numContacts_A = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numA[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_A = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.A) )

        numContacts_E = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numE[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_E = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.E) )

        numContacts_I = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numI[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_I = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.I) )

        numContacts_SQ = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numSQ[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_SQ = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.SQ) )

        numContacts_EQ = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numEQ[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_EQ = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.EQ) )

        numContacts_IQ = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numIQ[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_IQ = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.IQ) )

        numContacts_AQ = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numAQ[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_AQ = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.AQ) )

        numContacts_MQ = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numMQ[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_MQ = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.MQ) )

        numContacts_RQ = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numRQ[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_RQ = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.RQ) )

        numContacts_C = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numC[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_C = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.C) )

        numContacts_Cmirec = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numCmirec[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_Cmirec = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.Cmirec) )

        numContacts_Cicurec = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numCicurec[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_Cicurec = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.Cicurec) )

        numContacts_ICU = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numICU[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_ICU = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.ICU) )

        numContacts_Mi = numpy.zeros(shape=(self.numNodes,1))
        if(numpy.any(self.numMi[self.tidx])
            and numpy.any(self.beta!=0)):
            numContacts_Mi = numpy.asarray( scipy.sparse.csr_matrix.dot(self.Adj, self.X==self.Mi) )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities_StoE   = ( self.p*((self.beta*(self.numI[self.tidx]+self.numA[self.tidx]) )/self.N[self.tidx])
                                + (1-self.p)*numpy.divide((self.beta*(numContacts_I + numContacts_A)), self.degree, out=numpy.zeros_like(self.degree), where=self.degree!=0)
                              )*(self.X==self.S)
        propensities_EtoI   = (1/self.sigma)*(self.X==self.E)
        propensities_ItoA   = (self.a/self.omega)*(self.X==self.I)
        propensities_ItoM   = (self.m/self.omega)*(self.X==self.I)
        propensities_MtoC   = (self.h*self.c/self.dhospital)*(self.X==self.M)
        propensities_MtoMi  = (self.h*self.mi/self.dhospital)*(self.X==self.M)
        propensities_MtoICU = self.h*((1-self.c-self.mi)/self.dhospital)*(self.X==self.M)
        propensities_AtoR   = (1/self.da)*(self.X==self.A)
        propensities_MtoR   = ((1-self.h)/self.dm)*(self.X==self.M)
        propensities_MitoCmirec   = (1/self.dmi)*(self.X==self.Mi)
        propensities_ICUtoCicurec   = ((1-self.m0)/self.dICU)*(self.X==self.ICU)
        propensities_ICUtoD   = (self.m0/self.dICU)*(self.X==self.ICU)
        propensities_CtoR   = (1/self.dc)*(self.X==self.C)
        propensities_CmirectoR   = (1/self.dmirec)*(self.X==self.Cmirec)
        propensities_CicurectoR   = (1/self.dICUrec)*(self.X==self.Cicurec)
        propensities_StoSQ = (self.theta_S + self.phi_S*(numContacts_SQ + numContacts_EQ + numContacts_IQ + numContacts_AQ + numContacts_MQ + numContacts_RQ + numContacts_C + numContacts_Mi + numContacts_ICU + numContacts_Cmirec + numContacts_Cicurec))*self.psi_FP*(self.X==self.S)
        propensities_EtoEQ = (self.theta_E + self.phi_E*(numContacts_SQ + numContacts_EQ + numContacts_IQ + numContacts_AQ + numContacts_MQ + numContacts_RQ + numContacts_C + numContacts_Mi + numContacts_ICU + numContacts_Cmirec + numContacts_Cicurec))*self.psi_PP*(self.X==self.E)
        propensities_ItoIQ = (self.theta_I + self.phi_I*(numContacts_SQ + numContacts_EQ + numContacts_IQ + numContacts_AQ + numContacts_MQ + numContacts_RQ + numContacts_C + numContacts_Mi + numContacts_ICU + numContacts_Cmirec + numContacts_Cicurec))*self.psi_FP*(self.X==self.I)
        propensities_AtoAQ = (self.theta_A + self.phi_A*(numContacts_SQ + numContacts_EQ + numContacts_IQ + numContacts_AQ + numContacts_MQ + numContacts_RQ + numContacts_C + numContacts_Mi + numContacts_ICU + numContacts_Cmirec + numContacts_Cicurec))*self.psi_PP*(self.X==self.A)
        propensities_MtoMQ = (self.theta_M + self.phi_M*(numContacts_SQ + numContacts_EQ + numContacts_IQ + numContacts_AQ + numContacts_MQ + numContacts_RQ + numContacts_C + numContacts_Mi + numContacts_ICU + numContacts_Cmirec + numContacts_Cicurec))*self.psi_PP*(self.X==self.M)
        propensities_RtoRQ = (self.theta_R + self.phi_R*(numContacts_SQ + numContacts_EQ + numContacts_IQ + numContacts_AQ + numContacts_MQ + numContacts_RQ + numContacts_C + numContacts_Mi + numContacts_ICU + numContacts_Cmirec + numContacts_Cicurec))*self.psi_FP*(self.X==self.R)
        propensities_SQtoS = (1/self.dq)*(self.X==self.SQ)
        propensities_EQtoIQ = (1/self.sigma)*(self.X==self.EQ)
        propensities_IQtoAQ = (self.a/self.omega)*(self.X==self.IQ)
        propensities_IQtoMQ = (self.m/self.omega)*(self.X==self.IQ)
        propensities_MQtoC = (self.h*self.c/self.dhospital)*(self.X==self.MQ)
        propensities_MQtoMi = (self.h*self.mi/self.dhospital)*(self.X==self.MQ)
        propensities_MQtoICU = (self.h*(1-self.c-self.mi)/self.dhospital)*(self.X==self.MQ)
        propensities_MQtoR = ((1-self.h)/self.dm)*(self.X==self.MQ)
        propensities_AQtoR = (1/self.dq)*(self.X==self.AQ)
        propensities_RQtoR = (1/self.dq)*(self.X==self.RQ)
        propensities_RtoS   = self.zeta*(self.X==self.R)
        #propensities__toS   = self.nu*(self.X!=self.F)

        propensities = numpy.hstack([propensities_StoE, propensities_EtoI,
                                     propensities_ItoA,
                                     propensities_ItoM, propensities_MtoC,
                                     propensities_MtoMi, propensities_MtoICU,
                                     propensities_AtoR, propensities_MtoR,
                                     propensities_MitoCmirec, propensities_MitoCmirec,
                                     propensities_ICUtoCicurec,
                                     propensities_ICUtoD, propensities_CtoR,
                                     propensities_CmirectoR, propensities_CicurectoR,
                                     propensities_StoSQ, propensities_EtoEQ,
                                     propensities_ItoIQ,
                                     propensities_AtoAQ, propensities_MtoMQ,
                                     propensities_RtoRQ, propensities_SQtoS,
                                     propensities_EQtoIQ, propensities_IQtoAQ,
                                     propensities_IQtoMQ,
                                     propensities_MQtoC, propensities_MQtoMi,
                                     propensities_MQtoICU, propensities_MQtoR,
                                     propensities_AQtoR, propensities_MQtoR,
                                     propensities_RQtoR, propensities_RtoS]) #propensities__toS

        columns = ['StoE', 'EtoI', 'ItoA', 'ItoM', 'MtoC', 'MtoMi', 'MtoICU','AtoR','MtoR','MitoCmirec','MitoCmirec','ICUtoCicurec','ICUtoD','CtoR','CmirectoR','CicurectoR',
                    'StoSQ','EtoEQ', 'ItoIQ','AtoAQ','MtoMQ','RtoRQ','SQtoS','EQtoIQ','IQtoAQ','IQtoMQ','MQtoC','MQtoMi','MQtoICU','MQtoR','AQtoR','MQtoR','RQtoR','RtoS'] #'_toS'
        return propensities, columns
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def increase_data_series_length(self):
        self.tseries = numpy.pad(self.tseries, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numS = numpy.pad(self.numS, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numE = numpy.pad(self.numE, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numI = numpy.pad(self.numI, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numA = numpy.pad(self.numA, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numM = numpy.pad(self.numM, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numC = numpy.pad(self.numC, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numCmirec = numpy.pad(self.numCmirec, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numCicurec = numpy.pad(self.numCicurec, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numMi = numpy.pad(self.numMi, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numICU = numpy.pad(self.numICU, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numSQ = numpy.pad(self.numSQ, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numEQ = numpy.pad(self.numEQ, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numIQ = numpy.pad(self.numIQ, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numAQ = numpy.pad(self.numAQ, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numMQ = numpy.pad(self.numMQ, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numRQ = numpy.pad(self.numRQ, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numR = numpy.pad(self.numR, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD = numpy.pad(self.numD, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.N = numpy.pad(self.N, [(0, 5*self.numNodes)], mode='constant', constant_values=0)

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']     = numpy.pad(self.nodeGroupData[groupName]['numS'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numE']     = numpy.pad(self.nodeGroupData[groupName]['numE'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numI']     = numpy.pad(self.nodeGroupData[groupName]['numI'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numA']     = numpy.pad(self.nodeGroupData[groupName]['numA'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numM']     = numpy.pad(self.nodeGroupData[groupName]['numM'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numC']     = numpy.pad(self.nodeGroupData[groupName]['numC'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numCmirec']     = numpy.pad(self.nodeGroupData[groupName]['numCmirec'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numCicurec']     = numpy.pad(self.nodeGroupData[groupName]['numCicurec'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numMi']     = numpy.pad(self.nodeGroupData[groupName]['numMi'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numICU']     = numpy.pad(self.nodeGroupData[groupName]['numICU'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numSQ']     = numpy.pad(self.nodeGroupData[groupName]['numSQ'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numEQ']     = numpy.pad(self.nodeGroupData[groupName]['numEQ'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numIQ']     = numpy.pad(self.nodeGroupData[groupName]['numIQ'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numAQ']     = numpy.pad(self.nodeGroupData[groupName]['numAQ'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numMQ']     = numpy.pad(self.nodeGroupData[groupName]['numMQ'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numRQ']     = numpy.pad(self.nodeGroupData[groupName]['numRQ'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numR']     = numpy.pad(self.nodeGroupData[groupName]['numR'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['numD']     = numpy.pad(self.nodeGroupData[groupName]['numD'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
                self.nodeGroupData[groupName]['N']        = numpy.pad(self.nodeGroupData[groupName]['N'], [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        return None
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def finalize_data_series(self):
        self.tseries = numpy.array(self.tseries, dtype=float)[:self.tidx+1]
        self.numS = numpy.array(self.numS, dtype=float)[:self.tidx+1]
        self.numE = numpy.array(self.numE, dtype=float)[:self.tidx+1]
        self.numI = numpy.array(self.numI, dtype=float)[:self.tidx+1]
        self.numA = numpy.array(self.numA, dtype=float)[:self.tidx+1]
        self.numM = numpy.array(self.numM, dtype=float)[:self.tidx+1]
        self.numC = numpy.array(self.numC, dtype=float)[:self.tidx+1]
        self.numCmirec = numpy.array(self.numCmirec, dtype=float)[:self.tidx+1]
        self.numCicurec = numpy.array(self.numCicurec, dtype=float)[:self.tidx+1]
        self.numMi = numpy.array(self.numMi, dtype=float)[:self.tidx+1]
        self.numICU = numpy.array(self.numICU, dtype=float)[:self.tidx+1]
        self.numSQ = numpy.array(self.numSQ, dtype=float)[:self.tidx+1]
        self.numEQ = numpy.array(self.numEQ, dtype=float)[:self.tidx+1]
        self.numIQ = numpy.array(self.numIQ, dtype=float)[:self.tidx+1]
        self.numAQ = numpy.array(self.numAQ, dtype=float)[:self.tidx+1]
        self.numMQ = numpy.array(self.numMQ, dtype=float)[:self.tidx+1]
        self.numRQ = numpy.array(self.numRQ, dtype=float)[:self.tidx+1]
        self.numR = numpy.array(self.numR, dtype=float)[:self.tidx+1]
        self.numD = numpy.array(self.numD, dtype=float)[:self.tidx+1]
        self.N = numpy.array(self.N, dtype=float)[:self.tidx+1]

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS']    = numpy.array(self.nodeGroupData[groupName]['numS'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numE']    = numpy.array(self.nodeGroupData[groupName]['numE'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numI']    = numpy.array(self.nodeGroupData[groupName]['numI'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numA']    = numpy.array(self.nodeGroupData[groupName]['numA'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numM']    = numpy.array(self.nodeGroupData[groupName]['numM'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numC']    = numpy.array(self.nodeGroupData[groupName]['numC'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numCmirec']    = numpy.array(self.nodeGroupData[groupName]['numCmirec'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numCicurec']    = numpy.array(self.nodeGroupData[groupName]['numCicurec'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numMi']    = numpy.array(self.nodeGroupData[groupName]['numMi'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numICU']    = numpy.array(self.nodeGroupData[groupName]['numICU'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numSQ']    = numpy.array(self.nodeGroupData[groupName]['numSQ'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numEQ']    = numpy.array(self.nodeGroupData[groupName]['numEQ'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numIQ']    = numpy.array(self.nodeGroupData[groupName]['numIQ'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numAQ']    = numpy.array(self.nodeGroupData[groupName]['numAQ'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numMQ']    = numpy.array(self.nodeGroupData[groupName]['numMQ'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numRQ']    = numpy.array(self.nodeGroupData[groupName]['numRQ'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numR']    = numpy.array(self.nodeGroupData[groupName]['numR'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['numD']    = numpy.array(self.nodeGroupData[groupName]['numD'], dtype=float)[:self.tidx+1]
                self.nodeGroupData[groupName]['N']       = numpy.array(self.nodeGroupData[groupName]['N'], dtype=float)[:self.tidx+1]
        return None

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run_iteration(self):

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = numpy.random.rand()
        r2 = numpy.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        # Terminate when probability of all events is 0:
        if(propensities.sum() <= 0.0):
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate alpha
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities_flat   = propensities.ravel(order='F')
        cumsum              = propensities_flat.cumsum()
        alpha               = propensities_flat.sum()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Compute the time until the next event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tau = (1/alpha)*numpy.log(float(1/r1))
        self.t += tau

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. Compute which event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        transitionIdx   = numpy.searchsorted(cumsum,r2*alpha)
        transitionNode  = transitionIdx % self.numNodes
        transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6. Update node states and data series
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.D), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
        self.X[transitionNode] = self.transitions[transitionType]['newState']

        self.tidx += 1

        self.tseries[self.tidx]  = self.t
        self.numS[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.I), a_min=0, a_max=self.numNodes)
        self.numA[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.A), a_min=0, a_max=self.numNodes)
        self.numM[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.M), a_min=0, a_max=self.numNodes)
        self.numC[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.C), a_min=0, a_max=self.numNodes)
        self.numCmirec[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.Cmirec), a_min=0, a_max=self.numNodes)
        self.numCicurec[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.Cicurec), a_min=0, a_max=self.numNodes)
        self.numMi[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.Mi), a_min=0, a_max=self.numNodes)
        self.numICU[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.ICU), a_min=0, a_max=self.numNodes)
        self.numSQ[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.SQ), a_min=0, a_max=self.numNodes)
        self.numEQ[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.EQ), a_min=0, a_max=self.numNodes)
        self.numIQ[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.IQ), a_min=0, a_max=self.numNodes)
        self.numAQ[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.AQ), a_min=0, a_max=self.numNodes)
        self.numMQ[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.MQ), a_min=0, a_max=self.numNodes)
        self.numRQ[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.RQ), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.numD[self.tidx]     = numpy.clip(numpy.count_nonzero(self.X==self.D), a_min=0, a_max=self.numNodes)
        self.N[self.tidx]        = numpy.clip((self.numS[self.tidx] + self.numE[self.tidx] + self.numI[self.tidx] + self.numA[self.tidx] + self.numM[self.tidx] + self.numC[self.tidx]
        + self.numCmirec[self.tidx] + self.numCicurec[self.tidx] + self.numMi[self.tidx] + self.numICU[self.tidx] + self.numSQ[self.tidx] + self.numEQ[self.tidx] + self.numIQ[self.tidx]
        + self.numAQ[self.tidx] + self.numMQ[self.tidx] + self.numRQ[self.tidx] + self.numR[self.tidx]), a_min=0, a_max=self.numNodes)

        if(self.nodeGroupData):
            for groupName in self.nodeGroupData:
                self.nodeGroupData[groupName]['numS'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.S)
                self.nodeGroupData[groupName]['numE'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.E)
                self.nodeGroupData[groupName]['numI'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.I)
                self.nodeGroupData[groupName]['numA'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.A)
                self.nodeGroupData[groupName]['numM'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.M)
                self.nodeGroupData[groupName]['numC'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.C)
                self.nodeGroupData[groupName]['numCmirec'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Cmirec)
                self.nodeGroupData[groupName]['numCicurec'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Cicurec)
                self.nodeGroupData[groupName]['numMi'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.Mi)
                self.nodeGroupData[groupName]['numICU'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.ICU)
                self.nodeGroupData[groupName]['numSQ'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.SQ)
                self.nodeGroupData[groupName]['numEQ'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.EQ)
                self.nodeGroupData[groupName]['numIQ'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.IQ)
                self.nodeGroupData[groupName]['numAQ'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.AQ)
                self.nodeGroupData[groupName]['numMQ'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.MQ)
                self.nodeGroupData[groupName]['numRQ'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.RQ)
                self.nodeGroupData[groupName]['numR'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.R)
                self.nodeGroupData[groupName]['numD'][self.tidx]    = numpy.count_nonzero(self.nodeGroupData[groupName]['mask']*self.X==self.D)
                self.nodeGroupData[groupName]['N'][self.tidx]       = numpy.clip((self.nodeGroupData[groupName]['numS'][0]
                + self.nodeGroupData[groupName]['numE'][0]  + self.nodeGroupData[groupName]['numI'][0] + self.nodeGroupData[groupName]['numA'][0] + self.nodeGroupData[groupName]['numM'][0]
                + self.nodeGroupData[groupName]['numC'][0]  + self.nodeGroupData[groupName]['numCmirec'][0] + self.nodeGroupData[groupName]['numCicurec'][0]
                + self.nodeGroupData[groupName]['numMi'][0] + self.nodeGroupData[groupName]['numICU'][0] + self.nodeGroupData[groupName]['numSQ'][0]
                + self.nodeGroupData[groupName]['numEQ'][0] + self.nodeGroupData[groupName]['numIQ'][0] + self.nodeGroupData[groupName]['numAQ'][0]
                + self.nodeGroupData[groupName]['numMQ'][0] + self.nodeGroupData[groupName]['numRQ'][0] + self.nodeGroupData[groupName]['numR'][0]), a_min=0, a_max=self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Terminate if tmax reached or num infectious and num exposed is 0:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # if(self.t >= self.tmax or (self.numSM[self.tidx]<1 and self.numM[self.tidx]<1 and self.numH[self.tidx]<1 and self.numC[self.tidx]<1 and self.numHH[self.tidx]<1 and self.numCH[self.tidx]<1 and self.numE[self.tidx]<1 )):
        #     self.finalize_data_series()
        #     return False

        if(self.t > self.tmax):
            self.finalize_data_series()
            return False
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return True
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, checkpoints=None, print_interval=10, verbose=False):
        if(T>0):
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            # Before using checkpoints, save variables to be changed by method
            beforeChk=[]
            for key in checkpoints.keys():
                if key != 't':
                    beforeChk.append(getattr(self,key))
            numCheckpoints = len(checkpoints['t'])
            paramNames = ['G', 'beta', 'sigma', 'omega', 'initN','zeta', 'p','a','m', 'h', 'c', 'mi', 'da', 'dm', 'dc', 'dmi', 'dICU', 'dICUrec', 'dmirec','dhospital', 'm0',
            'maxICU','theta_S', 'theta_E', 'theta_I', 'theta_A', 'theta_M', 'theta_R','phi_S','phi_E','phi_I','phi_A','phi_R','psi_FP','psi_PP']

            for chkpt_param, chkpt_values in checkpoints.items():
                assert(isinstance(chkpt_values, (list, numpy.ndarray)) and len(chkpt_values)==numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times ("+str(numCheckpoints)+") for each checkpoint parameter."
            checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
            if(checkpointIdx >= numCheckpoints):
                # We are out of checkpoints, stop checking them:
                checkpoints = None
            else:
                checkpointTime = checkpoints['t'][checkpointIdx]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print_reset = True
        running     = True
        while running:

            running = self.run_iteration()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle checkpoints if applicable:
            if(checkpoints):
                if(self.t >= checkpointTime):
                    print("[Checkpoint: Updating parameters]")
                    # A checkpoint has been reached, update param values:
                    for param in paramNames:
                        if(param in list(checkpoints.keys())):
                            if(param=='G'):
                                self.update_G(checkpoints[param][checkpointIdx])
                            else:
                                setattr(self, param, checkpoints[param][checkpointIdx] if isinstance(checkpoints[param][checkpointIdx], (list, numpy.ndarray)) else numpy.full(fill_value=checkpoints[param][checkpointIdx], shape=(self.numNodes,1)))
                    # Update scenario flags to represent new param values:
                    #self.update_scenario_flags()
                    # Update the next checkpoint time:
                    checkpointIdx  = numpy.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
                    if(checkpointIdx >= numCheckpoints):
                        # Reset all parameter values that were changed back to their original value
                        i = 0
                        for key in checkpoints.keys():
                            if key != 't':
                                setattr(self,key,beforeChk[i])
                                i = i+1
                        # We are out of checkpoints, stop checking them:
                        checkpoints = None
                    else:
                        checkpointTime = checkpoints['t'][checkpointIdx]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if(print_interval):
                if(print_reset and (int(self.t) % print_interval == 0)):
                    print("t = %.2f" % self.t)
                    if(verbose):
                        print("\t S   = " + str(self.numS[self.tidx]))
                        print("\t E   = " + str(self.numE[self.tidx]))
                        print("\t I   = " + str(self.numI[self.tidx]))
                        print("\t A   = " + str(self.numA[self.tidx]))
                        print("\t M   = " + str(self.numM[self.tidx]))
                        print("\t C   = " + str(self.numC[self.tidx]))
                        print("\t Cmirec   = " + str(self.numCmirec[self.tidx]))
                        print("\t Cicurec   = " + str(self.numCicurec[self.tidx]))
                        print("\t Mi   = " + str(self.numMi[self.tidx]))
                        print("\t ICU   = " + str(self.numICU[self.tidx]))
                        print("\t SQ   = " + str(self.numSQ[self.tidx]))
                        print("\t EQ   = " + str(self.numEQ[self.tidx]))
                        print("\t IQ   = " + str(self.numIQ[self.tidx]))
                        print("\t AQ   = " + str(self.numAQ[self.tidx]))
                        print("\t MQ   = " + str(self.numMQ[self.tidx]))
                        print("\t RQ   = " + str(self.numRQ[self.tidx]))
                        print("\t R   = " + str(self.numR[self.tidx]))
                        print("\t D   = " + str(self.numD[self.tidx]))
                    print_reset = False
                elif(not print_reset and (int(self.t) % 10 != 0)):
                    print_reset = True
        return self

    def format_numX(self,T):
        # output of stochastic model is not returned in daily intervals,
        # to use the same attribute functions as the deterministic model
        # the results must be interpolated
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r = self.initN/self.numNodes #ratio of number of nodes vs. total population, used to make an extrapolation
        x = self.tseries
        t = numpy.linspace(0,T,T+1)
        # sometimes simulator stops before reaching time T because change was too small
        # if this is the case append one timestep
        if x[-1] < T:
            x=numpy.append(x,T+1)
            self.numS=numpy.append(self.numS,self.numS[-1])
            self.numE=numpy.append(self.numE,self.numE[-1])
            self.numI=numpy.append(self.numI,self.numE[-1])
            self.numA=numpy.append(self.numA,self.numA[-1])
            self.numM=numpy.append(self.numM,self.numM[-1])
            self.numC=numpy.append(self.numC,self.numC[-1])
            self.numCmirec=numpy.append(self.numCmirec,self.numCmirec[-1])
            self.numCicurec=numpy.append(self.numCicurec,self.numCicurec[-1])
            self.numMi=numpy.append(self.numMi,self.numMi[-1])
            self.numICU=numpy.append(self.numICU,self.numICU[-1])
            self.numR=numpy.append(self.numR,self.numR[-1])
            self.numD=numpy.append(self.numD,self.numD[-1])
            self.numSQ=numpy.append(self.numSQ,self.numSQ[-1])
            self.numEQ=numpy.append(self.numEQ,self.numEQ[-1])
            self.numIQ=numpy.append(self.numIQ,self.numEQ[-1])
            self.numAQ=numpy.append(self.numAQ,self.numAQ[-1])
            self.numMQ=numpy.append(self.numMQ,self.numMQ[-1])
            self.numRQ=numpy.append(self.numRQ,self.numRQ[-1])
        # Use interpolate function to match self.numS with timevector t
        inte = inter.interp1d(x,self.numS)
        self.numS = inte(t)*r
        inte = inter.interp1d(x,self.numE)
        self.numE = inte(t)*r
        inte = inter.interp1d(x,self.numI)
        self.numI = inte(t)*r
        inte = inter.interp1d(x,self.numA)
        self.numA = inte(t)*r
        inte = inter.interp1d(x,self.numM)
        self.numM = inte(t)*r
        inte = inter.interp1d(x,self.numC)
        self.numC = inte(t)*r
        inte = inter.interp1d(x,self.numCmirec)
        self.numCmirec = inte(t)*r
        inte = inter.interp1d(x,self.numCicurec)
        self.numCicurec = inte(t)*r
        inte = inter.interp1d(x,self.numMi)
        self.numMi = inte(t)*r
        inte = inter.interp1d(x,self.numICU)
        self.numICU = inte(t)*r
        inte = inter.interp1d(x,self.numR)
        self.numR = inte(t)*r
        inte = inter.interp1d(x,self.numD)
        self.numD = inte(t)*r
        inte = inter.interp1d(x,self.numSQ)
        self.numSQ = inte(t)*r
        inte = inter.interp1d(x,self.numEQ)
        self.numEQ = inte(t)*r
        inte = inter.interp1d(x,self.numIQ)
        self.numIQ = inte(t)*r
        inte = inter.interp1d(x,self.numAQ)
        self.numAQ = inte(t)*r
        inte = inter.interp1d(x,self.numMQ)
        self.numMQ = inte(t)*r
        inte = inter.interp1d(x,self.numRQ)
        self.numRQ = inte(t)*r
        # replace self.tseries with a np.linspace(0,T,T+1)
        self.tseries=t
        return self

    def sampleFromDistribution(self,filename,k):
        df = pd.read_csv(filename)
        x = df.iloc[:,0]
        y = df.iloc[:,1]
        return(numpy.asarray(choices(x, y, k = k)))

    def sim(self, T, dt=1, checkpoints=None, verbose=False):
        tN = int(T) + 1
        # pre-allocate a 2D matrix for the results summed over all age bins
        self.sumS = numpy.zeros([tN,self.repeats])
        self.sumE = numpy.zeros([tN,self.repeats])
        self.sumI = numpy.zeros([tN,self.repeats])
        self.sumA = numpy.zeros([tN,self.repeats])
        self.sumM = numpy.zeros([tN,self.repeats])
        self.sumC = numpy.zeros([tN,self.repeats])
        self.sumCmirec = numpy.zeros([tN,self.repeats])
        self.sumCicurec = numpy.zeros([tN,self.repeats])
        self.sumCtot = numpy.zeros([tN,self.repeats])
        self.sumMi = numpy.zeros([tN,self.repeats])
        self.sumICU = numpy.zeros([tN,self.repeats])
        self.sumR = numpy.zeros([tN,self.repeats])
        self.sumD = numpy.zeros([tN,self.repeats])
        self.sumSQ = numpy.zeros([tN,self.repeats])
        self.sumEQ = numpy.zeros([tN,self.repeats])
        self.sumIQ = numpy.zeros([tN,self.repeats])
        self.sumAQ = numpy.zeros([tN,self.repeats])
        self.sumMQ = numpy.zeros([tN,self.repeats])
        self.sumRQ = numpy.zeros([tN,self.repeats])
        # simulation loop
        i=0
        while i <= self.repeats-1:
            # reset self to initial condition
            self.reset()
            # perform simulation
            self.run(int(T),checkpoints)
            # format all vectors numX
            self.format_numX(int(T))

            self.sumS[:,i] = self.numS
            self.sumE[:,i] = self.numE
            self.sumI[:,i] = self.numI
            self.sumA[:,i] = self.numA
            self.sumM[:,i] = self.numM
            self.sumC[:,i] = self.numC
            self.sumCmirec[:,i] = self.numCmirec
            self.sumCicurec[:,i] = self.numCicurec
            self.sumCtot[:,i] = self.numC + self.numCmirec + self.numCicurec
            self.sumMi[:,i] = self.numMi
            self.sumICU[:,i] = self.numICU
            self.sumR[:,i] = self.numR
            self.sumD[:,i] = self.numD
            self.sumSQ[:,i] = self.numSQ
            self.sumEQ[:,i] = self.numEQ
            self.sumIQ[:,i] = self.numIQ
            self.sumAQ[:,i] = self.numAQ
            self.sumMQ[:,i] = self.numMQ
            self.sumRQ[:,i] = self.numRQ
            i = i + 1
        return self

    def plotPopulationStatus(self,filename=None,getfig=False):
        # extend with plotting data and using dates (extra argument startDate)
        fig, ax = plt.subplots()
        ax.plot(self.tseries,numpy.mean(self.sumS,axis=1),color=black)
        ax.fill_between(self.tseries, numpy.percentile(self.sumS,90,axis=1), numpy.percentile(self.sumS,10,axis=1),color=black,alpha=0.2)
        ax.plot(self.tseries,numpy.mean(self.sumE,axis=1),color=orange)
        ax.fill_between(self.tseries, numpy.percentile(self.sumE,90,axis=1), numpy.percentile(self.sumE,10,axis=1),color=orange,alpha=0.2)
        I = self.sumA + self.sumM + self.sumCtot + self.sumMi + self.sumICU
        ax.plot(self.tseries,numpy.mean(I,axis=1),color=red)
        ax.fill_between(self.tseries, numpy.percentile(I,90,axis=1), numpy.percentile(I,10,axis=1),color=red,alpha=0.2)
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
        H = self.sumCtot + self.sumMi + self.sumICU
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

    def LSQ(self,thetas,data,parNames,positions,weights):
        # ------------------
        # Prepare simulation
        # ------------------
        # reset all numX
        self.reset()
        # assign estimates to correct variable
        extraTime = int(thetas[0])
        i = 0
        for param in parNames:
            setattr(self,param,thetas[i+1])
            i = i + 1

        # Compute length of data
        n = len(data)
        # Compute simulation time --> build in some redundancy here, datasizes don't have to be equal to eachother.
        T = data[0].size+extraTime-1
        # Set initial condition
        # ...

        # ------------------
        # Perform simulation
        # ------------------
        self.sim(T)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumA,self.sumM,self.sumCtot,self.sumMi,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ)

        # ---------------
        # extract results
        # ---------------
        ymodel=[]
        SSE = 0
        for i in range(n):
            som = 0
            for j in positions[i]:
                som = som + numpy.mean(out[j],axis=1).reshape(numpy.mean(out[j],axis=1).size,1)
            ymodel.append(som[extraTime:,0].reshape(som[extraTime:,0].size,1))
            # calculate quadratic error
            SSE = SSE + weights[i]*sum((ymodel[i]-data[i])**2)
        SSE = SSE[0]
        print(SSE)
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
        if (len(parNames)+1) is not len(bounds):
            raise Exception('The number of bounds must match the number of parameter names given to function fit.')
        # Check that all parNames are actual model parameters
        possibleNames = ['G', 'beta', 'sigma', 'initN','zeta', 'p','a','m', 'h', 'c', 'mi', 'da', 'dm', 'dc', 'dmi', 'dICU', 'dICUrec', 'dmirec','dhospital', 'm0',
            'maxICU','theta_S', 'theta_E', 'theta_A', 'theta_M', 'theta_R','phi_S','phi_E','phi_A','phi_R','psi_FP','psi_PP']
        i = 0
        for param in parNames:
            # For params that don't have given checkpoint values (or bad value given),
            # set their checkpoint values to the value they have now for all checkpoints.
            if param not in possibleNames:
                raise Exception('The parametername provided by user in position {} of argument parNames is not an actual model parameter. Please check its spelling.'.format(i))
            else:
                if param == 'G':
                    raise Exception('Cannot fit parameter G because this is a network object')
            i = i + 1

        # ---------------------
        # Run genetic algorithm
        # ---------------------
        #optim_out = scipy.optimize.differential_evolution(self.LSQ, bounds, args=(data,parNames,positions,weights),disp=disp,polish=polish,workers=-1,maxiter=maxiter, popsize=popsize,tol=1e-18)
        #theta_hat = optim_out.x
        p_hat, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = pso.pso(self.LSQ, bounds, args=(data,parNames,positions,weights), swarmsize=popsize, maxiter=maxiter,
                                                                                    processes=multiprocessing.cpu_count(),minfunc=1e-9, minstep=1e-9,debug=True, particle_output=True)
        theta_hat = p_hat
        print(theta_hat)

        # ---------------------------------------------------
        # If setattr is True: assign estimated thetas to self
        # ---------------------------------------------------
        if setvar is True:
            self.extraTime = int(theta_hat[0])
            i = 0
            for param in parNames:
                setattr(self,param,theta_hat[i+1])
                i  = i + 1

        return self,theta_hat

    def plotFit(self,index,data,positions,dataMkr=['o','v','s','*','^'],modelClr=['green','orange','red','black','blue'],legendText=None,titleText=None,filename=None,getfig=False):
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
        self.sim(T)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumA,self.sumM,self.sumCtot,self.sumMi,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ)

        # -----------
        # Plot result
        # -----------
        # Create shifted index vector using self.extraTime
        timeObj = index[0]
        timestampStr = timeObj.strftime("%Y-%m-%d")
        index_acc = pd.date_range(timestampStr,freq='D',periods=data[0].size + self.extraTime) - datetime.timedelta(days=self.extraTime-1)
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
            ax.fill_between(index_acc,numpy.percentile(ymodel,99,axis=1),
                 numpy.percentile(ymodel,1,axis=1),color=modelClr[i],alpha=0.3)
        # Attributes
        if legendText is not None:
            ax.legend(legendText, loc="upper left", bbox_to_anchor=(1,1))
        if titleText is not None:
            ax.set_title(titleText,{'fontsize':18})
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%Y'))
        plt.setp(plt.gca().xaxis.get_majorticklabels(),
            'rotation', 90)
        ax.set_xlim( index_acc[self.extraTime-3], pd.to_datetime(index_acc[-1]))
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
                for i in range(len(dict2['t'])):
                    dict1[key].append(getattr(self,key))
                merged[key] = dict1[key]
            elif key in dict2 and not key in dict1:
                for i in range(orig_len):
                    dict2[key].insert(0,getattr(self,key))
                merged[key] = dict2[key]
        return merged

    def realTimeScenario(self,startDate,data,positions,pastPolicy,futurePolicy=None,T_extra=14,dataMkr=['o','v','s','*','^'],
                                modelClr=['green','orange','red','black','blue'],legendText=None,titleText=None,filename=None,getfig=False):
        # This function will by default plot the user provided country data + model prediction starting on the first day of the dates vectors
        # To match the current data with the model prediction, a checkpoints dictionary pastPolicy must be given to the function.
        # If none of the additional arguments are provided, the model will simply plot the the prediction up until the end date provided by the user.
        # Optionally, one out of two things can be done. 1) Provide an additional checkpoints dictionary, this can be used to perform scenario analysis.
        # 2) Run an MPC optimisation starting on the end date provided by the user.
        # The correct working of this function requires a model that was calibrated to the data.

        # Initialize a vector of dates starting on the user provided startDate and of length data
        # Calculate length of data to obtain an initial simulation time
        t_data = pd.date_range(startDate, freq='D', periods=data[0].size)
        T = len(t_data) + self.extraTime - 1 + int(T_extra) # number of datapoints

        # add estimated extraTime to past policy vector
        for i in range(len(pastPolicy['t'])):
            pastPolicy['t'][i] = pastPolicy['t'][i] + self.extraTime - 1
        chk = pastPolicy

        # Create a merged dictionary accounting for estimated 'extraTime'
        if futurePolicy is not None:
            chk = self.mergeDict((T-int(T_extra)-1),pastPolicy,futurePolicy)
            T = chk['t'][-1]+int(T_extra)

        # ------------------
        # Prepare simulation
        # ------------------
        # reset all numX
        self.reset()

        # ------------------
        # Perform simulation
        # ------------------
        self.sim(T,checkpoints=chk)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumA,self.sumM,self.sumCtot,self.sumMi,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ)

        # -----------
        # Plot result
        # -----------
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a custom method for generating
# power-law-like graphs with exponential tails
# both above and below the degree mean and
# where the mean degree be easily down-shifted
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    # Generate a random preferential attachment power law graph as a starting point.
    # By the way this graph is constructed, it is expected to have 1 connected component.
    # Every node is added along with m=8 edges, so the min degree is m=8.
    if(base_graph):
        graph = base_graph.copy()
    else:
        assert(n is not None), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = networkx.barabasi_albert_graph(n=n, m=m)

    # To get a graph with power-law-esque properties but without the fixed minimum degree,
    # We modify the graph by probabilistically dropping some edges from each node.
    for node in graph:
        neighbors = list(graph[node].keys())
        quarantineEdgeNum = int( max(min(numpy.random.exponential(scale=scale, size=1), len(neighbors)), min_num_edges) )
        quarantineKeepNeighbors = numpy.random.choice(neighbors, size=quarantineEdgeNum, replace=False)
        for neighbor in neighbors:
            if(neighbor not in quarantineKeepNeighbors):
                graph.remove_edge(node, neighbor)

    return graph

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def plot_degree_distn(graph, max_degree=None, show=True, use_seaborn=True):
    import matplotlib.pyplot as pyplot
    if(use_seaborn):
        import seaborn
        seaborn.set_style('ticks')
        seaborn.despine()
    # Get a list of the node degrees:
    if type(graph)==numpy.ndarray:
        nodeDegrees = graph.sum(axis=0).reshape((graph.shape[0],1))   # sums of adj matrix cols
    elif type(graph)==networkx.classes.graph.Graph:
        nodeDegrees = [d[1] for d in graph.degree()]
    else:
        raise BaseException("Input an adjacency matrix or networkx object only.")
    # Calculate the mean degree:
    meanDegree = numpy.mean(nodeDegrees)
    # Generate a histogram of the node degrees:
    pyplot.hist(nodeDegrees, bins=range(max(nodeDegrees)), alpha=0.5, color='tab:blue', label=('mean degree = %.1f' % meanDegree))
    pyplot.xlim(0, max(nodeDegrees) if not max_degree else max_degree)
    pyplot.xlabel('degree')
    pyplot.ylabel('num nodes')
    pyplot.legend(loc='upper right')
    if(show):
        pyplot.show()
