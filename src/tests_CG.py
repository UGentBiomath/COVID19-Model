# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:23:58 2020

@author: CGarneau
"""


import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from ipywidgets import interact,fixed,FloatSlider,IntSlider,ToggleButtons
import pandas as pd
import datetime
import scipy
from scipy.integrate import odeint
import matplotlib.dates as mdates
import matplotlib
import scipy.stats as st
import networkx
import economic_model
import models


### Construct the infection model

if 0:
    # Construct the network G
    numNodes = 6000
    baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=7)
    # Baseline normal interactions:
    G_norm     = models.custom_exponential_graph(baseGraph, scale=200)
    models.plot_degree_distn(G_norm, max_degree=40)
    
    # Construct the network G under social distancing
    numNodes = 6000
    baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=2)
    # Baseline normal interactions:
    G_dist     = models.custom_exponential_graph(baseGraph, scale=20000)
    models.plot_degree_distn(G_dist, max_degree=40)




EcoMod = economic_model.EconomicModel
print(EcoMod)
EcoMod.LoadInputs(EcoMod)

adapt = EcoMod.Adaptation
inputs = EcoMod.Inputs

if 0:
    baseScen = inputs["Employment"]
    ConfineScen = baseScen.copy()
    WaH = adapt["Work at home"]
    mix = adapt["Mix home - office"]
    WaW = adapt["Work at work"]
    for i in range(len(baseScen)):
        ConfineScen[i]=baseScen[i] * (WaH[i] + mix[i] + WaW[i]) / 100
    
    print( ConfineScen.sum() / baseScen.sum())
    
    
    

    
    w_work = np.array([adapt["Work at home"], 
              adapt["Mix home - office"], 
              adapt["Work at work"], 
              adapt["Temporary Unemployed"],
              adapt["Absent"]])



test = EcoMod.AssignOccupation(EcoMod,2000, True)
#print(test)
#test= EcoMod.ChangeConfinementPolicy(EcoMod,test,31, 1)
#print(test)

 #   def ChangeConfinementPolicy(self,StatusMatrix, SectorClass, NewPolicy):
 

print(EcoMod.ComputeOccupation(EcoMod,test))
#    def ComputeOccupation(StatusMatrix):









