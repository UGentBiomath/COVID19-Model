#!/usr/bin/env python
# coding: utf-8

# Load required packages
#~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import scipy
from scipy.integrate import odeint
import matplotlib.dates as mdates
import matplotlib
import scipy.stats as st
import networkx
import timeit

# Import in-house code from /src directory
# First Python must be told where to find models.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import models

# Construct the network G
# ~~~~~~~~~~~~~~~~~~~~~~~
numNodes = 120000
baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=7)
# Baseline normal interactions:
G_norm     = models.custom_exponential_graph(baseGraph, scale=200)
models.plot_degree_distn(G_norm, max_degree=40)

# Construct the network G under social distancing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
numNodes = 120000
baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=2)
# Baseline normal interactions:
G_dist     = models.custom_exponential_graph(baseGraph, scale=20000)
models.plot_degree_distn(G_dist, max_degree=40)

# Define model
# ~~~~~~~~~~~~
model = models.SEIRSNetworkModel(
                                 # network connectivty
                                 G = G_norm,
                                 p       = 0.6,
                                 # clinical parameters
                                 beta    = 0.3, 
                                 sigma   = 5.2,
                                 zeta    = 0,
                                 sm      = 0.50,
                                 m       = (1-0.50)*0.81,
                                 h       = (1-0.50)*0.15,
                                 c       = (1-0.50)*0.04,
                                 dsm     = 14,
                                 dm      = 14,
                                 dhospital = 1,
                                 dh      = 21,
                                 dcf     = 18.5,
                                 dcr     = 22.0,
                                 mc0     = 0.49,
                                 ICU     = 2000,
                                 # testing
                                 theta_S = 0,
                                 theta_E = 0,
                                 theta_SM= 0,
                                 theta_M = 0,
                                 theta_R = 0,
                                 psi_FP = 0,
                                 psi_PP = 1,
                                 dq     = 14,                                 
                                 # back-tracking
                                 phi_S   = 0,
                                 phi_E   = 0,
                                 phi_SM  = 0,
                                 phi_R   = 0,
                                 # initial condition
                                 initN = 11.43e6, #results are extrapolated to entire population
                                 initE = 100,
                                 initSM = 0, 
                                 initM = 0,
                                 initH = 0,
                                 initC = 0,
                                 initHH = 0,
                                 initCH = 0,
                                 initR = 0,
                                 initF = 0,
                                 initSQ = 0,
                                 initEQ = 0,
                                 initSMQ = 0,
                                 initMQ = 0,
                                 initRQ = 0
                            )

# Perform optimisation and time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
start = timeit.default_timer()
# hardcoded data vectors
ICUvect= np.array([[5,24,33,53,79,100,130,164,238,290,322,381,474,605,690,789,867,927,1021,1088,1144,1205,1245,1261,1257,1260,1276,1285,1278,1262,1232,1234,1223]])
hospital = np.array([[58,97,163,264,368,496,648,841,1096,1380,1643,1881,2137,2715,3068,3640,4077,4468,4884,4975,5206,5358,5492,5509,5600,5738,5692,5590,5610,5635,5409,5393,5536]])
# vector with dates
index=pd.date_range('2020-03-13', freq='D', periods=ICUvect.size)
# data series used to calibrate model must be given to function 'plotFit' as a list
idx = -26
index = index[0:idx]
data=[np.transpose(ICUvect[:,0:idx]),np.transpose(hospital[:,0:idx])]
# set optimisation settings
parNames = ['beta','p'] # must be a list!
positions = [np.array([7]),np.array([6,7])] # must be a list!
bounds=[(1,100),(0.1,0.5),(0.5,0.9)] # must be a list!
weights = np.array([0,1])
# run optimisation
theta = model.fit(data,parNames,positions,bounds,weights,setvar=True,maxiter=10,popsize=5)
stop = timeit.default_timer()
print('Required time: ', stop - start,' seconds')

# Make a graphical representation of results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model.plotFit(index,data,positions,modelClr=['red','orange'],legendText=('ICU (model)','ICU (data)','Hospital (model)','Hospital (data)'),titleText='Belgium')
