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
numNodes = 90000
baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=3)
# Baseline normal interactions:
G_norm     = models.custom_exponential_graph(baseGraph, scale=500)
models.plot_degree_distn(G_norm, max_degree=40)

# Construct the network G under social distancing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
numNodes = 90000
baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=1)
# Baseline normal interactions:
G_dist     = models.custom_exponential_graph(baseGraph, scale=200000)
models.plot_degree_distn(G_dist, max_degree=40)

# Define model
# ~~~~~~~~~~~~
model = models.SEIRSNetworkModel(
                                 # network connectivty
                                 G = G_norm,
                                 p       = 0.51,
                                 # clinical parameters
                                 beta    = 0.03, 
                                 sigma   = 4.0,
                                 omega   = 1.5,
                                 zeta    = 0,
                                 a = 0.43, # probability of an asymptotic (supermild) infection
                                 m = 1-0.43, # probability of a mild infection
                                 h = 0.20, # probability of hospitalisation for a mild infection
                                 c = 2/3, # probability of hospitalisation in cohort
                                 mi = 1/6, # probability of hospitalisation in midcare
                                 da = 6.5, # days of infection when asymptomatic (supermild)
                                 dm = 6.5, # days of infection when mild
                                 dc = 7,
                                 dmi = 14,
                                 dICU = 14,
                                 dICUrec = 6,
                                 dmirec = 6,
                                 dhospital = 5, # days before reaching the hospital when heavy or critical
                                 m0 = 0.49, # mortality in ICU
                                 maxICU = 2000,
                                 # testing
                                 theta_S = 0,
                                 theta_E = 0,
                                 theta_A = 0,
                                 theta_M = 0,
                                 theta_R = 0,
                                 psi_FP = 0,
                                 psi_PP = 1,
                                 dq     = 14,                                 
                                 # back-tracking
                                 phi_S   = 0,
                                 phi_E   = 0,
                                 phi_I   = 0,
                                 phi_A   = 0,
                                 phi_R   = 0,
                                 # initial condition
                                 initN = 11.43e6, #results are extrapolated to entire population
                                 initE = 0,
                                 initI = 1,
                                 initA = 0, 
                                 initM = 0,
                                 initC = 0,
                                 initCmirec=0,
                                 initCicurec=0,
                                 initR = 0,
                                 initD = 0,
                                 initSQ = 0,
                                 initEQ = 0,
                                 initIQ = 0,
                                 initAQ = 0,
                                 initMQ = 0,
                                 initRQ = 0,
                                 # monte-carlo sampling
                                 monteCarlo = False,
                                 repeats = 3
                            )

# Create checkpoints dictionary
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
chk = {'t':       [40,80], 
       'G':       [G_dist,G_norm],
       'p':       [0.01,0.6],
      }

# Run simulation and time
# ~~~~~~~~~~~~~~~~~~~~~~~
start = timeit.default_timer()
model.sim(T=100,checkpoints=chk)
stop = timeit.default_timer()
print('Required time: ', stop - start,' seconds')

# Make a graphical representation of results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model.plotPopulationStatus(filename='populationStatus.svg')
model.plotInfected(filename='infected.svg')
