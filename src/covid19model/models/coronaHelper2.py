#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:01:22 2020

@author: twallema
Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved.
"""

import numpy as np
import pandas as pd
from random import choices
import scipy
from scipy.integrate import odeint
import math
import networkx
from scipy import interpolate as inter
from gekko import GEKKO

from . import models


def sampleFromDistribution(filename,k):
    df = pd.read_csv(filename)
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    return(np.asarray(choices(x, y, k = k)))

def runSimulation(initN,beta,sigmavect,Nc,zeta,smvect,mvect,hvect,cvect,dsm,dm,dhospitalvect,dh,dcfvect,dcrvect,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE,
 initSM, initM, initH, initC,initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,method,modelType,checkpoints,**stoArgs):
    tN = simtime + 1
    if monteCarlo == True: 
        n_samples = dcfvect.size
        S = np.zeros([tN,n_samples])
        E = np.zeros([tN,n_samples])
        SM = np.zeros([tN,n_samples])
        M = np.zeros([tN,n_samples])
        H = np.zeros([tN,n_samples])
        C = np.zeros([tN,n_samples])
        HH = np.zeros([tN,n_samples])
        CH = np.zeros([tN,n_samples])
        R = np.zeros([tN,n_samples])
        F = np.zeros([tN,n_samples])
        SQ = np.zeros([tN,n_samples])
        EQ = np.zeros([tN,n_samples])
        SMQ = np.zeros([tN,n_samples])
        MQ = np.zeros([tN,n_samples])
        RQ = np.zeros([tN,n_samples])
        i=0
        t = np.linspace(0,simtime,tN)
        for sigma in sigmavect:
            dcf = dcfvect[i]
            dcr = dcrvect[i]
            dhospital = dhospitalvect[i]
            sm = smvect[i]
            m = (1-sm)*0.81 
            h = (1-sm)*0.14 
            c = (1-sm)*0.05 
            # perform simulation
            if modelType == 'deterministic':
                if method == 'findInfected' or method == 'findTime' or method == 'none':
                    model = models.SEIRSAgeModel(initN=initN,beta=beta,sigma=sigma,Nc=Nc,zeta=zeta,sm=sm,m=m,h=h,c=c,dsm=dsm,dm=dm,dhospital=dhospital,dh=dh,dcf=dcf,dcr=dcr,mc0=mc0,ICU=ICU,
                        totalTests=totalTests,psi_FP=psi_FP,psi_PP=psi_PP,dq=dq,
                        initE=initE,initSM=initSM,initM=initM,initH=initH,initC=initC,initHH=initHH,initCH=initCH,initR=initR,initF=initF,initSQ=initSQ,initEQ=initEQ,initSMQ=initSMQ,initMQ=initMQ,
                        initRQ=initRQ)
                    y = model.run(T=simtime,checkpoints=checkpoints)
                elif method == 'findGovernmentResponse':
                    extraTime = stoArgs['extraTime']
                    measureTime = stoArgs['measureTime']
                    initE = 1
                    Nc0 = 11.2
                    checkpoints = {
                        't':        [measureTime+extraTime],
                        'Nc':       [Nc]
                    }
                    model = models.SEIRSAgeModel(initN=initN,beta=beta,sigma=sigma,Nc=Nc0,zeta=zeta,sm=sm,m=m,h=h,c=c,dsm=dsm,dm=dm,dhospital=dhospital,dh=dh,dcf=dcf,dcr=dcr,mc0=mc0,ICU=ICU,
                        totalTests=totalTests,psi_FP=psi_FP,psi_PP=psi_PP,dq=dq,
                        initE=initE,initSM=initSM,initM=initM,initH=initH,initC=initC,initHH=initHH,initCH=initCH,initR=initR,initF=initF,initSQ=initSQ,initEQ=initEQ,initSMQ=initSMQ,initMQ=initMQ,
                        initRQ=initRQ)
                    y = model.run(T=simtime,checkpoints=checkpoints)
                else:
                    raise Exception('Suitable methods to run model are: none, findTime, findInfected, findGovernmentResponse. The provided method was: {}'.format(method))
            elif modelType == 'stochastic':
                if method == 'findInfected' or method == 'findTime' or method == 'none':
                    model = models.SEIRSNetworkModel(G=stoArgs['G'],beta=beta,sigma=sigma,zeta=zeta,p=stoArgs['p'],sm=sm,m=m,h=h,c=c,dsm=dsm,dm=dm,dhospital=dhospital,dh=dh,dcf=dcf,dcr=dcr,mc0=mc0,ICU=ICU,theta_S=theta_S,
                            theta_E=theta_E,theta_SM=theta_SM,theta_M=theta_M,theta_R=theta_R,phi_S=phi_S,phi_E=phi_E,phi_SM=phi_SM,phi_R=phi_R,psi_FP=psi_FP,psi_PP=psi_PP,
                            dq=dq,initE=initE,initSM=initSM,initM=initM,initH=initH,initC=initC,initHH=initHH,initCH=initCH,initR=initR,initF=initF,initSQ=initSQ,initEQ=initEQ,initSMQ=initSMQ,
                            initMQ=initMQ,initRQ=initRQ)
                    y = model.run(T=simtime,checkpoints=checkpoints)
                    # output is not returned every single day, so the results must be interpolated
                    x = y.tseries
                    if x[-1] < simtime:
                        x=np.append(x,simtime+1)
                        y.numS=np.append(y.numS,y.numS[-1])
                        y.numE=np.append(y.numE,y.numE[-1])
                        y.numSM=np.append(y.numSM,y.numSM[-1])
                        y.numM=np.append(y.numM,y.numM[-1])
                        y.numH=np.append(y.numH,y.numH[-1])
                        y.numC=np.append(y.numC,y.numC[-1])
                        y.numHH=np.append(y.numHH,y.numHH[-1])
                        y.numCH=np.append(y.numCH,y.numCH[-1])
                        y.numR=np.append(y.numR,y.numR[-1])
                        y.numF=np.append(y.numF,y.numF[-1])
                        y.numSQ=np.append(y.numSQ,y.numSQ[-1])
                        y.numEQ=np.append(y.numEQ,y.numEQ[-1])
                        y.numSMQ=np.append(y.numSMQ,y.numSMQ[-1])
                        y.numMQ=np.append(y.numMQ,y.numMQ[-1])
                        y.numRQ=np.append(y.numRQ,y.numRQ[-1])

                    # first variable
                    inte = inter.interp1d(x,y.numS)
                    y.numS = inte(t)

                    inte = inter.interp1d(x,y.numE)
                    y.numE = inte(t)

                    inte = inter.interp1d(x,y.numSM)
                    y.numSM = inte(t)

                    inte = inter.interp1d(x,y.numM)
                    y.numM = inte(t)

                    inte = inter.interp1d(x,y.numH)
                    y.numH = inte(t)

                    inte = inter.interp1d(x,y.numC)
                    y.numC = inte(t)

                    inte = inter.interp1d(x,y.numHH)
                    y.numHH = inte(t)

                    inte = inter.interp1d(x,y.numCH)
                    y.numCH = inte(t)

                    inte = inter.interp1d(x,y.numR)
                    y.numR = inte(t)

                    inte = inter.interp1d(x,y.numF)
                    y.numF = inte(t)

                    inte = inter.interp1d(x,y.numSQ)
                    y.numSQ = inte(t)

                    inte = inter.interp1d(x,y.numEQ)
                    y.numEQ = inte(t)

                    inte = inter.interp1d(x,y.numSMQ)
                    y.numSMQ = inte(t)

                    inte = inter.interp1d(x,y.numMQ)
                    y.numMQ = inte(t)

                    inte = inter.interp1d(x,y.numRQ)
                    y.numRQ = inte(t)

                elif method == 'findGovernmentResponse':
                    extraTime = stoArgs['extraTime']
                    measureTime = stoArgs['measureTime']
                    initE = 1
                    beta0 = 0.244
                    checkpoints = {
                        't':        [measureTime+extraTime],
                        'beta':     [beta]
                    }
                    model = models.SEIRSNetworkModel(G=stoArgs['G'],beta=beta,sigma=sigma,zeta=zeta,p=stoArgs['p'],sm=sm,m=m,h=h,c=c,dsm=dsm,dm=dm,dhospital=dhospital,dh=dh,dcf=dcf,dcr=dcr,mc0=mc0,ICU=ICU,theta_S=theta_S,
                            theta_E=theta_E,theta_SM=theta_SM,theta_M=theta_M,theta_R=theta_R,phi_S=phi_S,phi_E=phi_E,phi_SM=phi_SM,phi_R=phi_R,psi_FP=psi_FP,psi_PP=psi_PP,
                            dq=dq,initE=initE,initSM=initSM,initM=initM,initH=initH,initC=initC,initHH=initHH,initCH=initCH,initR=initR,initF=initF,initSQ=initSQ,initEQ=initEQ,initSMQ=initSMQ,
                            initMQ=initMQ,initRQ=initRQ)
                    y = model.run(T=simtime,checkpoints=checkpoints)
                    # output is not returned every single day, so the results must be interpolated
                    x = y.tseries
                    if x[-1] < simtime:
                        x=np.append(x,simtime+1)
                        y.numS=np.append(y.numS,y.numS[-1])
                        y.numE=np.append(y.numE,y.numE[-1])
                        y.numSM=np.append(y.numSM,y.numSM[-1])
                        y.numM=np.append(y.numM,y.numM[-1])
                        y.numH=np.append(y.numH,y.numH[-1])
                        y.numC=np.append(y.numC,y.numC[-1])
                        y.numHH=np.append(y.numHH,y.numHH[-1])
                        y.numCH=np.append(y.numCH,y.numCH[-1])
                        y.numR=np.append(y.numR,y.numR[-1])
                        y.numF=np.append(y.numF,y.numF[-1])
                        y.numSQ=np.append(y.numSQ,y.numSQ[-1])
                        y.numEQ=np.append(y.numEQ,y.numEQ[-1])
                        y.numSMQ=np.append(y.numSMQ,y.numSMQ[-1])
                        y.numMQ=np.append(y.numMQ,y.numMQ[-1])
                        y.numRQ=np.append(y.numRQ,y.numRQ[-1])

                    # first variable
                    inte = inter.interp1d(x,y.numS)
                    y.numS = inte(t)

                    inte = inter.interp1d(x,y.numE)
                    y.numE = inte(t)

                    inte = inter.interp1d(x,y.numSM)
                    y.numSM = inte(t)

                    inte = inter.interp1d(x,y.numM)
                    y.numM = inte(t)

                    inte = inter.interp1d(x,y.numH)
                    y.numH = inte(t)

                    inte = inter.interp1d(x,y.numC)
                    y.numC = inte(t)

                    inte = inter.interp1d(x,y.numHH)
                    y.numHH = inte(t)

                    inte = inter.interp1d(x,y.numCH)
                    y.numCH = inte(t)

                    inte = inter.interp1d(x,y.numR)
                    y.numR = inte(t)

                    inte = inter.interp1d(x,y.numF)
                    y.numF = inte(t)

                    inte = inter.interp1d(x,y.numSQ)
                    y.numSQ = inte(t)

                    inte = inter.interp1d(x,y.numEQ)
                    y.numEQ = inte(t)

                    inte = inter.interp1d(x,y.numSMQ)
                    y.numSMQ = inte(t)

                    inte = inter.interp1d(x,y.numMQ)
                    y.numMQ = inte(t)

                    inte = inter.interp1d(x,y.numRQ)
                    y.numRQ = inte(t)                    
                else:
                    raise Exception('Suitable methods to run function dxdt are: none, findTime, findInfected, findGovernmentResponse. The provided method was: {}'.format(method))
            else:
                raise Exception('Modeltype is either deterministic or stochastic. The provided modeltype was: {}'.format(modelType))
            
            # extract results
            if modelType == "deterministic":
                S[:,i] = y.sumS.reshape(y.sumS.size,1)[:,0] 
                E[:,i] = y.sumE.reshape(y.sumE.size,1)[:,0] 
                SM[:,i] = y.sumSM.reshape(y.sumSM.size,1)[:,0] 
                M[:,i] = y.sumM.reshape(y.sumM.size,1)[:,0] 
                H[:,i] = y.sumH.reshape(y.sumH.size,1)[:,0] 
                C[:,i] = y.sumC.reshape(y.sumC.size,1)[:,0] 
                HH[:,i] = y.sumHH.reshape(y.sumHH.size,1)[:,0] 
                CH[:,i] = y.sumCH.reshape(y.sumCH.size,1)[:,0] 
                R[:,i] = y.sumR.reshape(y.sumR.size,1)[:,0] 
                F[:,i] = y.sumF.reshape(y.sumF.size,1)[:,0] 
                SQ[:,i] = y.sumSQ.reshape(y.sumSQ.size,1)[:,0] 
                EQ[:,i] = y.sumEQ.reshape(y.sumEQ.size,1)[:,0] 
                SMQ[:,i] = y.sumSMQ.reshape(y.sumSMQ.size,1)[:,0] 
                MQ[:,i] = y.sumMQ.reshape(y.sumMQ.size,1)[:,0] 
                RQ[:,i] = y.sumRQ.reshape(y.sumRQ.size,1)[:,0]

            elif modelType == "stochastic":
                S[:,i] = y.numS.reshape(y.numS.size,1)[:,0] 
                E[:,i] = y.numE.reshape(y.numE.size,1)[:,0] 
                SM[:,i] = y.numSM.reshape(y.numSM.size,1)[:,0] 
                M[:,i] = y.numM.reshape(y.numM.size,1)[:,0] 
                H[:,i] = y.numH.reshape(y.numH.size,1)[:,0] 
                C[:,i] = y.numC.reshape(y.numC.size,1)[:,0] 
                HH[:,i] = y.numHH.reshape(y.numHH.size,1)[:,0] 
                CH[:,i] = y.numCH.reshape(y.numCH.size,1)[:,0] 
                R[:,i] = y.numR.reshape(y.numR.size,1)[:,0] 
                F[:,i] = y.numF.reshape(y.numF.size,1)[:,0] 
                SQ[:,i] = y.numSQ.reshape(y.numSQ.size,1)[:,0] 
                EQ[:,i] = y.numEQ.reshape(y.numEQ.size,1)[:,0] 
                SMQ[:,i] = y.numSMQ.reshape(y.numSMQ.size,1)[:,0] 
                MQ[:,i] = y.numMQ.reshape(y.numMQ.size,1)[:,0] 
                RQ[:,i] = y.numRQ.reshape(y.numRQ.size,1)[:,0] 
            i = i + 1
    else:
        S = np.zeros([tN,1])
        E = np.zeros([tN,1])
        SM = np.zeros([tN,1])
        M = np.zeros([tN,1])
        H = np.zeros([tN,1])
        C = np.zeros([tN,1])
        HH = np.zeros([tN,1])
        CH = np.zeros([tN,1])
        R = np.zeros([tN,1])
        F = np.zeros([tN,1])
        SQ = np.zeros([tN,1])
        EQ = np.zeros([tN,1])
        SMQ = np.zeros([tN,1])
        MQ = np.zeros([tN,1])
        RQ = np.zeros([tN,1])
        t = np.linspace(0,simtime,tN)
        dcf = dcfvect
        dcr = dcrvect
        dhospital = dhospitalvect
        sm = smvect
        m = (1-sm)*0.81 
        h = (1-sm)*0.14 
        c = (1-sm)*0.05
        sigma = sigmavect 
        # perform simulation
        if modelType == 'deterministic':
            if method == 'findInfected' or method == 'findTime' or method == 'none':
                model = models.SEIRSAgeModel(initN=initN,beta=beta,sigma=sigma,Nc=Nc,zeta=zeta,sm=sm,m=m,h=h,c=c,dsm=dsm,dm=dm,dhospital=dhospital,dh=dh,dcf=dcf,dcr=dcr,mc0=mc0,ICU=ICU,
                    totalTests=totalTests,psi_FP=psi_FP,psi_PP=psi_PP,dq=dq,
                    initE=initE,initSM=initSM,initM=initM,initH=initH,initC=initC,initHH=initHH,initCH=initCH,initR=initR,initF=initF,initSQ=initSQ,initEQ=initEQ,initSMQ=initSMQ,initMQ=initMQ,
                    initRQ=initRQ)
                y = model.run(T=simtime,checkpoints=checkpoints)
            elif method == 'findGovernmentResponse':
                extraTime = stoArgs['extraTime']
                measureTime = stoArgs['measureTime']
                initE = 1
                Nc0 = 11.2
                checkpoints = {
                    't':        [measureTime+extraTime],
                    'Nc':       [Nc]
                }
                model = models.SEIRSAgeModel(initN=initN,beta=beta,sigma=sigma,Nc=Nc,zeta=zeta,sm=sm,m=m,h=h,c=c,dsm=dsm,dm=dm,dhospital=dhospital,dh=dh,dcf=dcf,dcr=dcr,mc0=mc0,ICU=ICU,
                    totalTests=totalTests,psi_FP=psi_FP,psi_PP=psi_PP,dq=dq,
                    initE=initE,initSM=initSM,initM=initM,initH=initH,initC=initC,initHH=initHH,initCH=initCH,initR=initR,initF=initF,initSQ=initSQ,initEQ=initEQ,initSMQ=initSMQ,initMQ=initMQ,
                    initRQ=initRQ)
                y = model.run(T=simtime,checkpoints=checkpoints)
            else:
                raise Exception('Suitable methods to run the model are: none, findTime, findInfected, findGovernmentResponse. The provided method was: {}'.format(method))
        
        elif modelType == 'stochastic':
            if method == 'findInfected' or method == 'findTime' or method == 'none':
                model = models.SEIRSNetworkModel(G=stoArgs['G'],beta=beta,sigma=sigma,zeta=zeta,p=stoArgs['p'],sm=sm,m=m,h=h,c=c,dsm=dsm,dm=dm,dhospital=dhospital,dh=dh,dcf=dcf,dcr=dcr,mc0=mc0,ICU=ICU,theta_S=theta_S,
                            theta_E=theta_E,theta_SM=theta_SM,theta_M=theta_M,theta_R=theta_R,phi_S=phi_S,phi_E=phi_E,phi_SM=phi_SM,phi_R=phi_R,psi_FP=psi_FP,psi_PP=psi_PP,
                            dq=dq,initE=initE,initSM=initSM,initM=initM,initH=initH,initC=initC,initHH=initHH,initCH=initCH,initR=initR,initF=initF,initSQ=initSQ,initEQ=initEQ,initSMQ=initSMQ,
                            initMQ=initMQ,initRQ=initRQ)
                print(simtime)
                y = model.run(T=simtime,checkpoints=checkpoints)
                # output is not returned every single day, so the results must be interpolated
                x = y.tseries
                if x[-1] < simtime:
                    x=np.append(x,simtime+1)
                    y.numS=np.append(y.numS,y.numS[-1])
                    y.numE=np.append(y.numE,y.numE[-1])
                    y.numSM=np.append(y.numSM,y.numSM[-1])
                    y.numM=np.append(y.numM,y.numM[-1])
                    y.numH=np.append(y.numH,y.numH[-1])
                    y.numC=np.append(y.numC,y.numC[-1])
                    y.numHH=np.append(y.numHH,y.numHH[-1])
                    y.numCH=np.append(y.numCH,y.numCH[-1])
                    y.numR=np.append(y.numR,y.numR[-1])
                    y.numF=np.append(y.numF,y.numF[-1])
                    y.numSQ=np.append(y.numSQ,y.numSQ[-1])
                    y.numEQ=np.append(y.numEQ,y.numEQ[-1])
                    y.numSMQ=np.append(y.numSMQ,y.numSMQ[-1])
                    y.numMQ=np.append(y.numMQ,y.numMQ[-1])
                    y.numRQ=np.append(y.numRQ,y.numRQ[-1])

                # first variable
                inte = inter.interp1d(x,y.numS)
                y.numS = inte(t)

                inte = inter.interp1d(x,y.numE)
                y.numE = inte(t)

                inte = inter.interp1d(x,y.numSM)
                y.numSM = inte(t)

                inte = inter.interp1d(x,y.numM)
                y.numM = inte(t)

                inte = inter.interp1d(x,y.numH)
                y.numH = inte(t)

                inte = inter.interp1d(x,y.numC)
                y.numC = inte(t)

                inte = inter.interp1d(x,y.numHH)
                y.numHH = inte(t)

                inte = inter.interp1d(x,y.numCH)
                y.numCH = inte(t)

                inte = inter.interp1d(x,y.numR)
                y.numR = inte(t)

                inte = inter.interp1d(x,y.numF)
                y.numF = inte(t)

                inte = inter.interp1d(x,y.numSQ)
                y.numSQ = inte(t)

                inte = inter.interp1d(x,y.numEQ)
                y.numEQ = inte(t)

                inte = inter.interp1d(x,y.numSMQ)
                y.numSMQ = inte(t)

                inte = inter.interp1d(x,y.numMQ)
                y.numMQ = inte(t)

                inte = inter.interp1d(x,y.numRQ)
                y.numRQ = inte(t)

            elif method == 'findGovernmentResponse':
                extraTime = stoArgs['extraTime']
                measureTime = stoArgs['measureTime']
                initE = 1
                beta0 = 0.290
                checkpoints = {
                    't':        [measureTime+extraTime],
                    'beta':     [beta]
                }
                model = models.SEIRSNetworkModel(G=stoArgs['G'],beta=beta,sigma=sigma,zeta=zeta,p=stoArgs['p'],sm=sm,m=m,h=h,c=c,dsm=dsm,dm=dm,dhospital=dhospital,dh=dh,dcf=dcf,dcr=dcr,mc0=mc0,ICU=ICU,theta_S=theta_S,
                            theta_E=theta_E,theta_SM=theta_SM,theta_M=theta_M,theta_R=theta_R,phi_S=phi_S,phi_E=phi_E,phi_SM=phi_SM,phi_R=phi_R,psi_FP=psi_FP,psi_PP=psi_PP,
                            dq=dq,initE=initE,initSM=initSM,initM=initM,initH=initH,initC=initC,initHH=initHH,initCH=initCH,initR=initR,initF=initF,initSQ=initSQ,initEQ=initEQ,initSMQ=initSMQ,
                            initMQ=initMQ,initRQ=initRQ)
                y = model.run(T=simtime,checkpoints=checkpoints)
                # output is not returned every single day, so the results must be interpolated
                x = y.tseries
                if x[-1] < simtime:
                    x=np.append(x,simtime+1)
                    y.numS=np.append(y.numS,y.numS[-1])
                    y.numE=np.append(y.numE,y.numE[-1])
                    y.numSM=np.append(y.numSM,y.numSM[-1])
                    y.numM=np.append(y.numM,y.numM[-1])
                    y.numH=np.append(y.numH,y.numH[-1])
                    y.numC=np.append(y.numC,y.numC[-1])
                    y.numHH=np.append(y.numHH,y.numHH[-1])
                    y.numCH=np.append(y.numCH,y.numCH[-1])
                    y.numR=np.append(y.numR,y.numR[-1])
                    y.numF=np.append(y.numF,y.numF[-1])
                    y.numSQ=np.append(y.numSQ,y.numSQ[-1])
                    y.numEQ=np.append(y.numEQ,y.numEQ[-1])
                    y.numSMQ=np.append(y.numSMQ,y.numSMQ[-1])
                    y.numMQ=np.append(y.numMQ,y.numMQ[-1])
                    y.numRQ=np.append(y.numRQ,y.numRQ[-1])

                # first variable
                inte = inter.interp1d(x,y.numS)
                y.numS = inte(t)

                inte = inter.interp1d(x,y.numE)
                y.numE = inte(t)

                inte = inter.interp1d(x,y.numSM)
                y.numSM = inte(t)

                inte = inter.interp1d(x,y.numM)
                y.numM = inte(t)

                inte = inter.interp1d(x,y.numH)
                y.numH = inte(t)

                inte = inter.interp1d(x,y.numC)
                y.numC = inte(t)

                inte = inter.interp1d(x,y.numHH)
                y.numHH = inte(t)

                inte = inter.interp1d(x,y.numCH)
                y.numCH = inte(t)

                inte = inter.interp1d(x,y.numR)
                y.numR = inte(t)

                inte = inter.interp1d(x,y.numF)
                y.numF = inte(t)

                inte = inter.interp1d(x,y.numSQ)
                y.numSQ = inte(t)

                inte = inter.interp1d(x,y.numEQ)
                y.numEQ = inte(t)

                inte = inter.interp1d(x,y.numSMQ)
                y.numSMQ = inte(t)

                inte = inter.interp1d(x,y.numMQ)
                y.numMQ = inte(t)

                inte = inter.interp1d(x,y.numRQ)
                y.numRQ = inte(t)                    
            else:
                raise Exception('Suitable methods to run model are: none, findTime, findInfected, findGovernmentResponse. The provided method was: {}'.format(method))
        
        else:
            raise Exception('Modeltype is either deterministic or stochastic. The provided modeltype was: {}'.format(modelType))
         
        # extract results
        if modelType == "deterministic":
            S[:,0] = y.sumS.reshape(y.sumS.size,1)[:,0] 
            E[:,0] = y.sumE.reshape(y.sumE.size,1)[:,0] 
            SM[:,0] = y.sumSM.reshape(y.sumSM.size,1)[:,0] 
            M[:,0] = y.sumM.reshape(y.sumM.size,1)[:,0] 
            H[:,0] = y.sumH.reshape(y.sumH.size,1)[:,0] 
            C[:,0] = y.sumC.reshape(y.sumC.size,1)[:,0] 
            HH[:,0] = y.sumHH.reshape(y.sumHH.size,1)[:,0] 
            CH[:,0] = y.sumCH.reshape(y.sumCH.size,1)[:,0] 
            R[:,0] = y.sumR.reshape(y.sumR.size,1)[:,0] 
            F[:,0] = y.sumF.reshape(y.sumF.size,1)[:,0] 
            SQ[:,0] = y.sumSQ.reshape(y.sumSQ.size,1)[:,0] 
            EQ[:,0] = y.sumEQ.reshape(y.sumEQ.size,1)[:,0] 
            SMQ[:,0] = y.sumSMQ.reshape(y.sumSMQ.size,1)[:,0] 
            MQ[:,0] = y.sumMQ.reshape(y.sumMQ.size,1)[:,0] 
            RQ[:,0] = y.sumRQ.reshape(y.sumRQ.size,1)[:,0]            
        elif modelType == "stochastic":
            S[:,0] = y.numS.reshape(y.numS.size,1)[:,0] 
            E[:,0] = y.numE.reshape(y.numE.size,1)[:,0] 
            SM[:,0] = y.numSM.reshape(y.numSM.size,1)[:,0] 
            M[:,0] = y.numM.reshape(y.numM.size,1)[:,0] 
            H[:,0] = y.numH.reshape(y.numH.size,1)[:,0] 
            C[:,0] = y.numC.reshape(y.numC.size,1)[:,0] 
            HH[:,0] = y.numHH.reshape(y.numHH.size,1)[:,0] 
            CH[:,0] = y.numCH.reshape(y.numCH.size,1)[:,0] 
            R[:,0] = y.numR.reshape(y.numR.size,1)[:,0] 
            F[:,0] = y.numF.reshape(y.numF.size,1)[:,0] 
            SQ[:,0] = y.numSQ.reshape(y.numSQ.size,1)[:,0] 
            EQ[:,0] = y.numEQ.reshape(y.numEQ.size,1)[:,0] 
            SMQ[:,0] = y.numSMQ.reshape(y.numSMQ.size,1)[:,0] 
            MQ[:,0] = y.numMQ.reshape(y.numMQ.size,1)[:,0] 
            RQ[:,0] = y.numRQ.reshape(y.numRQ.size,1)[:,0] 

    if modelType == 'deterministic':
        return(t,S,E,SM,M,H,C,HH,CH,R,F,SQ,EQ,SMQ,MQ,RQ)
    elif modelType == 'stochastic':
        return(t,S,E,SM,M,H,C,HH,CH,R,F,SQ,EQ,SMQ,MQ,RQ,y.numNodes)


def LSQ(thetas,data,fitTo,
            initN,sigmavect,Nc,zeta,smvect,mvect,hvect,cvect,dsm,dm,dhospital,dh,dcfvect,dcrvect,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,
            phi_S,phi_E,phi_SM,phi_R,monteCarlo,method,modelType,checkpoints,stoArgs):

    if method == 'findInfected':
        # check if number of provided bounds is two
        if len(thetas) != 2:
            raise Exception('Number of bounds for method findInfected is 2. The number of provided bounds was: {}'.format(len(thetas)))
        # define length of simulation from provided data
        simtime = data.size-1
        # assign estimates to correct varaiable
        beta = thetas[0]
        B0=thetas[1]
        # calculate initial condition
        if modelType == 'stochastic':
            raise Exception('A stochastic model should be calibrated using the method findTime. The provided calibration method was: {}'.format(method))
        initN = initN
        initE = np.ones(Nc.shape[0])*B0
        initSM = np.zeros(Nc.shape[0])
        initM = np.zeros(Nc.shape[0])
        initH = np.zeros(Nc.shape[0])
        initC = np.zeros(Nc.shape[0])
        initHH = np.zeros(Nc.shape[0])
        initCH = np.zeros(Nc.shape[0])
        initR = np.zeros(Nc.shape[0])
        initF = np.zeros(Nc.shape[0])
        initSQ = np.zeros(Nc.shape[0])
        initEQ = np.zeros(Nc.shape[0])
        initSMQ = np.zeros(Nc.shape[0])
        initMQ = np.zeros(Nc.shape[0])
        initRQ = np.zeros(Nc.shape[0]) 
        # run simulation
        y = runSimulation(initN,beta,sigmavect,Nc,zeta,smvect,mvect,hvect,cvect,dsm,dm,dhospital,dh,dcfvect,dcrvect,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,
        phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC,initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,method,modelType,checkpoints,**stoArgs)
        # extract results
        ymodel=0
        for i in fitTo:
            ymodel = ymodel + np.mean(y[i],axis=1).reshape(np.mean(y[i],axis=1).size,1)
        # calculate quadratic error
        SSE = sum((ymodel-data)**2)

    elif method == 'findTime':
        # check if number of provided bounds is two or three for deterministic/stochastic respectively
        # assign the estimates to the correct variables
        if modelType == 'deterministic':
            if len(thetas) != 2:
                raise Exception('Number of bounds for deterministic model and method findTime is 2. The number of provided bounds was: {}'.format(len(thetas)))
            beta = thetas[0]
            extraTime = int(thetas[1])
            stoArgs.update({'extraTime': int(thetas[1])})
        elif modelType == 'stochastic':
            if len(thetas) != 3:
                raise Exception('Number of bounds for stochastic model and method findTime is 3. The number of provided bounds was: {}'.format(len(thetas)))
            beta = thetas[0]
            extraTime = int(thetas[1])
            stoArgs.update({'extraTime': int(thetas[1])})
            p = thetas[2]
            stoArgs.update({'p': thetas[2]})
        else:
            raise Exception('Invalid modelType. The provided modelType was: {}'.format(modelType))
        # define length of simulation from provided data
        simtime = data.size+extraTime-1
        # calculate initial condition
        initN = initN
        initE = np.ones(Nc.shape[0])
        initSM = np.zeros(Nc.shape[0])
        initM = np.zeros(Nc.shape[0])
        initH = np.zeros(Nc.shape[0])
        initC = np.zeros(Nc.shape[0])
        initHH = np.zeros(Nc.shape[0])
        initCH = np.zeros(Nc.shape[0])
        initR = np.zeros(Nc.shape[0])
        initF = np.zeros(Nc.shape[0])
        initSQ = np.zeros(Nc.shape[0])
        initEQ = np.zeros(Nc.shape[0])
        initSMQ = np.zeros(Nc.shape[0])
        initMQ = np.zeros(Nc.shape[0])
        initRQ = np.zeros(Nc.shape[0])  
        # run simulation
        y = runSimulation(initN,beta,sigmavect,Nc,zeta,smvect,mvect,hvect,cvect,dsm,dm,dhospital,dh,dcfvect,dcrvect,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,
        phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC,initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,method,modelType,checkpoints,**stoArgs)
        if modelType == 'deterministic':
            # extract results
            ymodel=0
            for i in fitTo:
                ymodel = ymodel + (np.mean(y[i],axis=1).reshape(np.mean(y[i],axis=1).size,1))
            ymodel = ymodel[extraTime-1:-1,0].reshape(ymodel[extraTime-1:-1,0].size,1)
            # calculate quadratic error
            SSE = sum((ymodel-data)**2)          
        elif modelType == 'stochastic':
            r = initN/y[-1] # ratio between stochastic population and total population
            # extract results
            ymodel=0
            for i in fitTo:
                ymodel = ymodel + (np.mean(y[i],axis=1).reshape(np.mean(y[i],axis=1).size,1))*r # extrapolate to whole population
            ymodel = ymodel[extraTime-1:-1,0].reshape(ymodel[extraTime-1:-1,0].size,1)
            # calculate quadratic error
            SSE = sum((ymodel-data)**2)
        
    elif method == 'findGovernmentResponse':
        # check if number of provided bounds is three
        if len(thetas) != 3:
            raise Exception('Number of bounds for method findGovernmentResponse is 3. The number of provided bounds was: {}'.format(len(thetas)))
        # assign beta and normal Nc
        beta = 0.0314
        Nc = np.array([11.2])
        # assign estimates to correct variable
        Nc_star = np.array([thetas[0]])  
        extraTime = int(thetas[1])
        stoArgs.update({'extraTime': int(thetas[1])})
        measureTime = int(thetas[2])
        stoArgs.update({'measureTime': int(thetas[2])})
        checkpoints={
            't':  [extraTime+measureTime],
            'Nc': [Nc_star]
        }
        # define length of simulation from provided data
        simtime = data.size+int(extraTime)-1
        # calculate initial condition
        initN = initN
        initE = np.ones(Nc.shape[0])
        initSM = np.zeros(Nc.shape[0])
        initM = np.zeros(Nc.shape[0])
        initH = np.zeros(Nc.shape[0])
        initC = np.zeros(Nc.shape[0])
        initHH = np.zeros(Nc.shape[0])
        initCH = np.zeros(Nc.shape[0])
        initR = np.zeros(Nc.shape[0])
        initF = np.zeros(Nc.shape[0])
        initSQ = np.zeros(Nc.shape[0])
        initEQ = np.zeros(Nc.shape[0])
        initSMQ = np.zeros(Nc.shape[0])
        initMQ = np.zeros(Nc.shape[0])
        initRQ = np.zeros(Nc.shape[0])
        method='none'  
        # run simulation
        y = runSimulation(initN,beta,sigmavect,Nc,zeta,smvect,mvect,hvect,cvect,dsm,dm,dhospital,dh,dcfvect,dcrvect,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,
        phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC,initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,method,modelType,checkpoints,**stoArgs)
        # extract results
        ymodel=0
        for i in fitTo:
            ymodel = ymodel + np.mean(y[i],axis=1).reshape(np.mean(y[i],axis=1).size,1)
        ymodel = ymodel[extraTime-1:-1,0].reshape(ymodel[extraTime-1:-1,0].size,1)
        # calculate quadratic error
        SSE = sum(abs(ymodel-data))    
    elif method == 'socialInteraction':
        # source: https://github.com/kieshaprem/covid19-agestructureSEIR-wuhan-social-distancing/tree/master/data
        Nc_home = np.loadtxt("Belgium/BELhome.txt", dtype='f', delimiter='\t')
        Nc_work = np.loadtxt("Belgium/BELwork.txt", dtype='f', delimiter='\t')
        Nc_schools = np.loadtxt("Belgium/BELschools.txt", dtype='f', delimiter='\t')
        Nc_others = np.loadtxt("Belgium/BELothers.txt", dtype='f', delimiter='\t')
        Nc_all = np.loadtxt("Belgium/BELall.txt", dtype='f', delimiter='\t')
        Nc = Nc_all
        checkpoints={
            't':  [26,29,29+5,29+10,29+15],
            'Nc': [Nc_all-Nc_schools,
                    Nc_home + thetas[0]*(1-0.20)*Nc_work +thetas[0]*(1-0.70)*Nc_others,
                    Nc_home + thetas[1]*(1-0.40)*Nc_work + thetas[1]*(1-0.70)*Nc_others,
                    Nc_home + thetas[2]*(1-0.52)*Nc_work + thetas[2]*(1-0.70)*Nc_others,
                    Nc_home + thetas[3]*(1-0.52)*Nc_work + thetas[3]*(1-0.70)*Nc_others]
        }
        # define length of simulation from provided data
        extraTime = 27
        simtime = data.size+27-1
        beta = 0.032155
        # calculate initial condition
        initN = initN
        initE = np.ones(Nc.shape[0])
        initSM = np.zeros(Nc.shape[0])
        initM = np.zeros(Nc.shape[0])
        initH = np.zeros(Nc.shape[0])
        initC = np.zeros(Nc.shape[0])
        initHH = np.zeros(Nc.shape[0])
        initCH = np.zeros(Nc.shape[0])
        initR = np.zeros(Nc.shape[0])
        initF = np.zeros(Nc.shape[0])
        initSQ = np.zeros(Nc.shape[0])
        initEQ = np.zeros(Nc.shape[0])
        initSMQ = np.zeros(Nc.shape[0])
        initMQ = np.zeros(Nc.shape[0])
        initRQ = np.zeros(Nc.shape[0])
        # run simulation
        method='findTime'
        y = runSimulation(initN,beta,sigmavect,Nc,zeta,smvect,mvect,hvect,cvect,dsm,dm,dhospital,dh,dcfvect,dcrvect,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,
        phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC,initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,method,modelType,checkpoints)
        if modelType == 'deterministic':
            # extract results
            ymodel=0
            for i in fitTo:
                ymodel = ymodel + (np.mean(y[i],axis=1).reshape(np.mean(y[i],axis=1).size,1))
            ymodel = ymodel[extraTime-1:-1,0].reshape(ymodel[extraTime-1:-1,0].size,1)
            # calculate quadratic error
            SSE = sum((ymodel-data)**2)   
    else:
            raise Exception('Method not suited for least-squares fit: choose either findTime, findInfected or findGovernmentResponse. The provided method was: {}'.format(method))
    return(SSE)

def modelFit(bounds,data,fitTo,initN,Nc,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,monteCarlo,n_samples,method,modelType,checkpoints,
disp,polish,maxiter,popsize,**stoArgs):
    # Monte Carlo sampling of parameters gamma, dHD, dHI, sm, and call to genetic optimisation algorithm is performed here
    if monteCarlo == True:
        sigmavect = sampleFromDistribution('corona_incubatie.csv',n_samples)
        dcfvect = np.random.normal(18.5, 5.2, n_samples)
        dcrvect = np.random.normal(22.0, 5.2, n_samples)
        smvect = np.random.normal(0.86, 0.04/1.96, n_samples)
        mvect = (1-smvect)*0.81
        hvect = (1-smvect)*0.14
        cvect = (1-smvect)*0.05
        dhospitalvect = np.random.normal(9.10, 0.50/1.96, n_samples)
        thetas = scipy.optimize.differential_evolution(LSQ, bounds, args=(data,fitTo,initN,sigmavect,Nc,zeta,smvect,mvect,hvect,cvect,dsm,dm,dhospitalvect,dh,dcfvect,dcrvect,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,monteCarlo,method,modelType,checkpoints,stoArgs),disp=disp,polish=polish,workers=5,maxiter=maxiter, popsize=popsize,tol=1e-18)
    else:
        sigma = 5.2
        dcf = 18.5 
        dcr = 22.0
        sm = 0.86
        m = (1-sm)*0.81 
        h = (1-sm)*0.14 
        c = (1-sm)*0.05
        dhospital = 9.1
        thetas = scipy.optimize.differential_evolution(LSQ, bounds, args=(data,fitTo,initN,sigma,Nc,zeta,sm,m,h,c,dsm,dm,dhospital,dh,dcf,dcr,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,monteCarlo,method,modelType,checkpoints,stoArgs),disp=disp,polish=polish,workers=5,maxiter=maxiter, popsize=popsize,tol=1e-18)
    fit = thetas.x
    return(fit)

def simModel(initN,beta,Nc,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                          initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,checkpoints,**stoArgs):
    # This function is a wrapper for 'runSimulation' to include monte carlo sampling and extract the results in a dictionary
    # Monte Carlo sampling of parameters gamma, dHD, dHI, sm, and call to genetic optimisation algorithm is performed here
    if monteCarlo == True:
        sigmavect = sampleFromDistribution('corona_incubatie.csv',n_samples)
        dcfvect = np.random.normal(18.5, 5.2, n_samples)
        dcrvect = np.random.normal(22.0, 5.2, n_samples)
        smvect = np.random.normal(0.86, 0.04/1.96, n_samples)
        mvect = (1-smvect)*0.81
        hvect = (1-smvect)*0.14
        cvect = (1-smvect)*0.05
        dhospitalvect = np.random.normal(9.10, 0.50/1.96, n_samples)
        simout = runSimulation(initN,beta,sigmavect,Nc,zeta,smvect,mvect,hvect,cvect,dsm,dm,dhospitalvect,dh,dcfvect,dcrvect,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,
        phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC,initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,method,modelType,checkpoints,**stoArgs)
    
    else:
        sigma = 5.2
        dcf = 18.5 
        dcr = 22.0
        sm = 0.86
        m = (1-sm)*0.81 
        h = (1-sm)*0.14 
        c = (1-sm)*0.05
        dhospital = 9.1 
        simout = runSimulation(initN,beta,sigma,Nc,zeta,sm,m,h,c,dsm,dm,dhospital,dh,dcf,dcr,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,
        initE, initSM, initM, initH, initC,initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,method,modelType,checkpoints,**stoArgs)
    
    # -----------------------------------------------------------------------------
    # extract results, rescale to population size initN in case of stochastic model
    # -----------------------------------------------------------------------------
    if modelType == 'deterministic':
        simout = {
            't':    simout[0],
            'S':    simout[1],
            'E':    simout[2],
            'SM':    simout[3],
            'M':    simout[4],
            'H':    simout[5],
            'C':    simout[6],
            'HH':    simout[7],
            'CH':    simout[8],
            'R':    simout[9],
            'F':    simout[10],
            'SQ':    simout[11],
            'EQ':    simout[12],
            'SMQ':    simout[13],
            'MQ':    simout[14],
            'RQ':    simout[15],
        }    
    elif modelType == 'stochastic':
        r = initN/simout[-1]
        simout = {
            't':    simout[0],
            'S':    simout[1]*r,
            'E':    simout[2]*r,
            'SM':    simout[3]*r,
            'M':    simout[4]*r,
            'H':    simout[5]*r,
            'C':    simout[6]*r,
            'HH':    simout[7]*r,
            'CH':    simout[8]*r,
            'R':    simout[9]*r,
            'F':    simout[10]*r,
            'SQ':    simout[11]*r,
            'SMQ':    simout[12]*r,
            'MQ':    simout[13]*r,
            'RQ':    simout[14]*r,
        }          

    return(simout)

def constructHorizon(theta,period):    
    n = len(theta)
    t = np.zeros([n-1])
    for i in range(n-1):
        t[i] = period*(i+1) 
    checkpoints = {'t': t,
                   'Nc': theta[1:]}
    return(checkpoints)

def constructHorizonPlot(theta,period):
    if type(theta) is np.ndarray:
        n = theta.size
        Nc = np.ones([period*n+1])
        for i in range(n):
            Nc[period*i:(period*i+period)]=theta[i]
    elif type(theta) is float:
        n = 1
        Nc = np.ones([period*n])
        for i in range(n):
            Nc[period*i:(period*i+period)]=theta
    else:
        raise Exception('Theta must be a vector or float. The provided datatype was: {}'.format(type(theta)))
    return(Nc)
    
def constructHorizonTesting(theta1,theta2,period):    
    n = len(theta1)
    t = np.zeros([n-1])
    for i in range(n-1):
        t[i] = period*(i+1) 
    checkpoints = {'t': t,
                   'Nc': theta1[1:],
                   'totalTests': theta2[1:]}
    return(checkpoints)

def constructHorizonTestingPlot(theta1,theta2,period):
    if type(theta1) is np.ndarray:
        n = theta1.size
        Nc = np.ones([period*n+1])
        theta_M = np.ones([period*n+1])
        for i in range(n):
            if i == 0:
                Nc[period*i:(period*i+period)+1]=theta1[i]
                theta_M[period*i:(period*i+period)+1]=theta2[i]
            else:
                Nc[period*i:(period*i+period)]=theta1[i]
                theta_M[period*i:(period*i+period)]=theta2[i]                
    elif type(theta1) is float:
        n = 1
        Nc = np.ones([period*n])
        theta_M = np.ones([period*n])
        for i in range(n):
            if i == 0:
                Nc[period*i:(period*i+period)+1]=theta1
                theta_M[period*i:(period*i+period)+1]=theta2
            else:
                Nc[period*i:(period*i+period)]=theta1[i]
                theta_M[period*i:(period*i+period)]=theta2[i]     
    else:
        raise Exception('Theta must be a vector or float. The provided datatype was: {}'.format(type(theta1)))
    return(Nc,theta_M)

def MPCcalcWeights(thetas,initN,beta,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                          initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,discrete,roundOff,period,P,stoArgs):
    controlDoF = 1
    if controlDoF == 1:
        thetas[thetas<5.6] = 1.8
        thetas[(thetas>=5.6)&(thetas<8)] = 6
        thetas[thetas>=8] = 11.2
        # Add thetas to a list
        Ncs=[]
        for i in range(thetas.size):
            Ncs.append(np.array([thetas[i]]))
        # Build prediction horizon
        for i in range(P-thetas.size):
            Ncs.append(Ncs[-1])
        checkpoints = constructHorizon(Ncs,period)
        # Set correct simtime
        simtime = checkpoints['t'].size*period
        # run simulation
        method == 'none' # nothing special 
        Nc = np.array([thetas[0]]) # first checkpoint cannot be at time 0
        simout = simModel(initN,beta,Nc,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                            initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,checkpoints,**stoArgs)
        if monteCarlo == True:
            CH = np.mean(simout['CH'],axis=1)
            CH = np.mean(simout['CH'],axis=1).reshape(CH.size,1)    
        else:
            CH = simout['CH']
        # regeling van de kritiek zieken
        y_sp = ICU # maximum aantal bedden op ICU
        ymodel = CH # voorspelde aantal kritiek zieken bij ingang beta
        error = y_sp - ymodel # vector met fouten in de tijd
        SSE = sum(error**2)

    elif controlDoF == 2:
        # Split list of thetas in half
        length = thetas.size
        middle_index = length//2
        thetas1 = thetas[:middle_index]
        # Discretise thetas1 (=Nc)
        thetas1[thetas1<5.6] = 1.8
        thetas1[(thetas1>=5.6)&(thetas1<8)] = 6
        thetas1[thetas1>=8] = 11.2
        thetas2 = thetas[middle_index:]
        # Add thetas to list
        Ncs1=[]
        for i in range(thetas1.size):
            Ncs1.append(np.array([thetas1[i]]))
        Ncs2=[]
        for i in range(thetas2.size):
            Ncs2.append(np.array([thetas2[i]]))
        # Build prediction horizons
        for i in range(P-thetas.size//2):
            Ncs1.append(Ncs1[-1])
            Ncs2.append(Ncs2[-1])
        # Construct checkpoints dictionary
        checkpoints = constructHorizonTesting(Ncs1,Ncs2,period)
        #print(checkpoints)
        # Define simtime
        simtime = checkpoints['t'].size*period     
        # run simulation
        method == 'none' # nothing special 
        Nc = np.array([thetas[0]]) # first checkpoint cannot be at time 0
        totalTests = np.array([thetas[middle_index]])
        simout = simModel(initN,beta,Nc,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                            initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,checkpoints,**stoArgs)
        if monteCarlo == True:
            CH = np.mean(simout['CH'],axis=1)
            CH = np.mean(simout['CH'],axis=1).reshape(CH.size,1)    
        else:
            CH = simout['CH']
        # regeling van de kritiek zieken
        y_sp = ICU # maximum aantal bedden op ICU
        ymodel = CH # voorspelde aantal kritiek zieken bij ingang beta
        error = y_sp - ymodel # vector met fouten in de tijd
        SSE = sum(error**2)
    return(SSE)

def MPCcalcWeightsAge(thetas,initN,beta,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                          initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,discrete,period,P):

    # source: https://github.com/kieshaprem/covid19-agestructureSEIR-wuhan-social-distancing/tree/master/data
    Nc_home = np.loadtxt("Belgium/BELhome.txt", dtype='f', delimiter='\t')
    Nc_work = np.loadtxt("Belgium/BELwork.txt", dtype='f', delimiter='\t')
    Nc_schools = np.loadtxt("Belgium/BELschools.txt", dtype='f', delimiter='\t')
    Nc_others = np.loadtxt("Belgium/BELothers.txt", dtype='f', delimiter='\t')
    Nc_all = np.loadtxt("Belgium/BELall.txt", dtype='f', delimiter='\t')
    # Use values of thetas to build a list object Ncs containing discrete scenarios
    Ncs=[]
    for i in range(thetas.size):
        if thetas[i]<=1 and thetas[i]>=0:
            Ncs.append(Nc_all)
        elif thetas[i]<=2 and thetas[i]> 1:
            Ncs.append(Nc_home + Nc_schools + 0.01*(1-0.52)*Nc_work + 0.01*(1-0.70)*Nc_others)
        elif thetas[i]<=3 and thetas[i]> 2:
            Ncs.append(Nc_home + 0.01*(1-0.52)*Nc_work + 0.01*(1-0.70)*Nc_others)

    # build prediction horizon
    for i in range(P-thetas.size):
        Ncs.append(Ncs[-1])
    checkpoints = constructHorizon(Ncs,period)
    simtime = checkpoints['t'].size*period
    # run simulation
    method == 'none' # nothing special
    Nc = Ncs[0] # first checkpoint cannot be at time 0
    simout = simModel(initN,beta,Nc,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                          initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,checkpoints)
    if monteCarlo == True:
        CH = np.mean(simout['CH'],axis=1)
        CH = np.mean(simout['CH'],axis=1).reshape(CH.size,1)    
    else:
        CH = simout['CH']
    # regeling van de kritiek zieken
    y_sp = ICU # maximum aantal bedden op ICU
    ymodel = CH # voorspelde aantal kritiek zieken bij ingang beta
    error = y_sp - ymodel # vector met fouten in de tijd
    SSE = sum(error**2)
    return(SSE)

def MPCoptimize(initN,beta,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                          initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,discrete,roundOff,period,P,N,
                          disp,polish,maxiter,popsize,**stoArgs):
    controlDoF = 1
    if controlDoF == 1:
        # Geef bounds op
        bounds=[]
        for i in range(N):
            bounds.append((0,11.2))
        # Perform optimisation                                                                           
        fit = scipy.optimize.differential_evolution(MPCcalcWeights, bounds, args=(initN,beta,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                          initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,discrete,roundOff,period,P,stoArgs),disp=disp,polish=polish,workers=-1,maxiter=maxiter, popsize=popsize,tol=1e-30)
        thetas=fit.x

    elif controlDoF == 2:
        # Geef bounds op
        bounds=[]
        # First variable is Nc
        for i in range(N):
            bounds.append((0,11.2))
        # Second variable is theta_M
        for i in range(N):
            bounds.append((0,1e6))        
        # Perform optimisation   
        fit = scipy.optimize.differential_evolution(MPCcalcWeights, bounds, args=(initN,beta,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                          initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,discrete,roundOff,period,P,stoArgs),disp=disp,polish=polish,workers=-1,maxiter=maxiter, popsize=popsize,tol=1e-30)
        thetas=fit.x      
    print(thetas)
    return(thetas)
 
def MPCoptimizeAge(initN,beta,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,totalTests,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                          initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,discrete,period,P,N,
                          disp,polish,maxiter,popsize):
    # source: https://github.com/kieshaprem/covid19-agestructureSEIR-wuhan-social-distancing/tree/master/data
    Nc_home = np.loadtxt("Belgium/BELhome.txt", dtype='f', delimiter='\t')
    Nc_work = np.loadtxt("Belgium/BELwork.txt", dtype='f', delimiter='\t')
    Nc_schools = np.loadtxt("Belgium/BELschools.txt", dtype='f', delimiter='\t')
    Nc_others = np.loadtxt("Belgium/BELothers.txt", dtype='f', delimiter='\t')
    Nc_all = np.loadtxt("Belgium/BELall.txt", dtype='f', delimiter='\t')
    
    # Geef bounds op
    bounds=[]
    for i in range(N):
        bounds.append((0,3))
    # Prepare solver
    # Perform optimisation (CONTINUOUS)                                                                        
    fit = scipy.optimize.differential_evolution(MPCcalcWeightsAge, bounds, args=(initN,beta,zeta,dsm,dm,dhospital,dh,mc0,ICU,theta_S,theta_E,theta_SM,theta_M,theta_R,psi_FP,psi_PP,dq,phi_S,phi_E,phi_SM,phi_R,initE, initSM, initM, initH, initC, 
                          initHH,initCH,initR,initF,initSQ,initEQ,initSMQ,initMQ,initRQ,simtime,monteCarlo,n_samples,method,modelType,discrete,period,P),disp=disp,polish=polish,workers=-1,maxiter=maxiter, popsize=popsize,tol=1e-18,mutation=(1.9, 1.99), recombination=1)
    thetas = fit.x

    # discretise thetas if needed
    thetas=fit.x
    Ncs=[]
    for i in range(thetas.size):
        if thetas[i]<=1 and thetas[i]>=0:
            Ncs.append(Nc_all)
        elif thetas[i]<=2 and thetas[i]> 1:
            Ncs.append(Nc_home + Nc_schools + 0.01*(1-0.52)*Nc_work + 0.01*(1-0.70)*Nc_others)
        elif thetas[i]<=3 and thetas[i]> 2:
            Ncs.append(Nc_home + 0.01*(1-0.52)*Nc_work + 0.01*(1-0.70)*Nc_others)
    return(Ncs,thetas)

# You cannot keep extending the control horizon because the number of parameters will get so big
# that optimisation becomes a problem. To simulate the full course of the outbreak, it is better
# to optimise one policy interval, advance the simulation to the next policy interval and repeat
def MPClongTerm(y0,nat,mort,dSM,dM,dZ,m,z,h,mh,ICU,monteCarlo,n_samples,period,maxiter,popsize,polish,disp,P,N,discrete,roundOff,Kh,Kd,Ki,nPeriods):
    betaVect=[]
    for i in range(nPeriods):
        # optimise control horizon over prediction horizon
        beta = MPCoptimize(y0,nat,mort,dSM,dM,dZ,m,z,h,mh,ICU,monteCarlo,n_samples,period,maxiter,popsize,polish,disp,P,N,discrete,roundOff,Kh,Kd,Ki)
        betaVect.append(beta[0])
        # advance the simulation one policy interval
        simtime = period # - 2
        tN = simtime + 1 
        t = np.linspace(0,simtime,tN)
        u = np.ones([tN])
        u = u*beta[0]
        simout = simModel(y0,nat,mort,u,dSM,dM,dZ,m,z,h,mh,ICU,tN,simtime,monteCarlo,n_samples,'variableBeta')
        O = simout[1]
        B = simout[2]
        SM = simout[3]
        M = simout[4]
        Z = simout[5]
        H = simout[6]
        I = simout[7]
        D = simout[8]
        T = simout[9]
        O = np.mean(O,axis=1)
        B = np.mean(B,axis=1)
        SM = np.mean(SM,axis=1)
        M = np.mean(M,axis=1)
        Z = np.mean(Z,axis=1)
        H = np.mean(H,axis=1)
        I = np.mean(I,axis=1)
        D = np.mean(D,axis=1)
        T = np.mean(T,axis=1)
        y0 = np.array([O[-1],B[-1],SM[-1],M[-1],Z[-1],H[-1],I[-1],D[-1],T[-1]])
    return(betaVect)  
    
