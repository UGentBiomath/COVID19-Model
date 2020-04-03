#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:01:22 2020

@author: twallema
"""

import numpy as np
import pandas as pd
from random import choices
import scipy
from scipy.integrate import odeint
import math
from simple_pid import PID

def sampleFromDistribution(filename,k):
    df = pd.read_csv(filename)
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    return(np.asarray(choices(x, y, k = k)))
    
def dxdt(y,t,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,simtime,method,*findTime):
    # Initialise solution
    O = y[0]
    B = y[1]
    SM = y[2]
    M = y[3]
    Z = y[4]
    H = y[5]
    I = y[6]
    D = y[7]
    T = y[8]
    # Account for increased mortality if critical patients receive no care
    if H > ICU:
        mh=0.49*(ICU/H)+1*((H-ICU)/H)
    # Depending on the required simulation of the model do:
    if method == 'none' or method == 'findTime' or method == 'findInfected':
        beta = beta #basically do nothing
    elif method == 'findGovernmentResponse':
        # unpack extraTime and measureTime
        extraTime = findTime[0][0]
        measureTime = findTime[0][1]
        if t < extraTime+measureTime:
            beta = 0.266
    elif method == 'variableBeta':
        tstar=math.floor(t)
        if tstar >= simtime:
            tstar = simtime
            beta = beta[tstar]
        else:
            beta = beta[tstar]
    # Implementation of the model equations
    dOdt = (nat-mort)*O - beta*(B/T)*O - beta*(SM/T)*O - beta*(M/T)*O
    dBdt = beta*O*(B/T + SM/T + M/T) - B/gamma - mort*B
    dSMdt = sm*B/gamma - SM/dSM
    dMdt = m*B/gamma - M/dM
    dZdt = z*B/gamma - Z/dZ
    dHdt = h*B/gamma - mh*H/dHD - (1-mh)*H/dHI
    dIdt = SM/dSM + M/dM + Z/dZ - mort*I + (1-mh)*H/dHI
    dDdt = mh*H/dHD
    dTdt = (nat-mort)*O - mort*B - mort*I - H/dHD
    return [dOdt,dBdt,dSMdt,dMdt,dZdt,dHdt,dIdt,dDdt,dTdt]

def runSimulation(y0,nat,mort,beta,gammaVect,dSM,dM,dZ,dHIvect,dHDvect,smvect,m,z,h,mh,ICU,tN,simtime,monteCarlo,method,*findTime):
    if monteCarlo == True: 
        n_samples = dHDvect.size
        O = np.zeros([tN,n_samples])
        B = np.zeros([tN,n_samples])
        SM = np.zeros([tN,n_samples])
        M = np.zeros([tN,n_samples])
        Z = np.zeros([tN,n_samples])
        H = np.zeros([tN,n_samples])
        I = np.zeros([tN,n_samples])
        D = np.zeros([tN,n_samples])
        T = np.zeros([tN,n_samples])
        i=0
        t = np.linspace(0,simtime,tN)
        for gamma in gammaVect:
            dHD = dHDvect[i]
            dHI = dHIvect[i]
            sm = smvect[i]
            # perform simulation
            if method == 'findInfected' or method == 'findTime' or method == 'variableBeta' or method == 'none':
                y = odeint(dxdt,y0,t,args=(nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,simtime,method))
            elif method == 'findGovernmentResponse':
                extraTime = findTime[0][0]
                measureTime = findTime[0][1]
                y = odeint(dxdt,y0,t,args=(nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,simtime,method,(extraTime,measureTime)))
            else:
                raise Exception('Suitable methods to run function dxdt are: none, findTime, findInfected, findGovernmentResponse or variableBeta. The provided method was: {}'.format(method))
            # extract results
            O[:,i] = y[:,0]
            B[:,i] = y[:,1]
            SM[:,i] = y[:,2]
            M[:,i] = y[:,3]
            Z[:,i] = y[:,4]
            H[:,i] = y[:,5]
            I[:,i] = y[:,6]
            D[:,i] = y[:,7]
            T[:,i] = y[:,8]
            i = i + 1
    else:
        O = np.zeros([tN,1])
        B = np.zeros([tN,1])
        SM = np.zeros([tN,1])
        M = np.zeros([tN,1])
        Z = np.zeros([tN,1])
        H = np.zeros([tN,1])
        I = np.zeros([tN,1])
        D = np.zeros([tN,1])
        T = np.zeros([tN,1])
        t = np.linspace(0,simtime,tN)
        gamma = gammaVect
        dHD = dHDvect
        dHI = dHIvect
        sm = smvect
        # perform simulation
        if method == 'findInfected' or method == 'findTime' or method == 'variableBeta' or method == 'none':
            y = odeint(dxdt,y0,t,args=(nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,simtime,method))
        elif method == 'findGovernmentResponse':
            extraTime = findTime[0][0]
            measureTime = findTime[0][1]
            y = odeint(dxdt,y0,t,args=(nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,simtime,method,(extraTime,measureTime)))
        else:
            raise Exception('Suitable methods to run function dxdt are: none, findTime, findInfected, findGovernmentResponse or variableBeta. The provided method was: {}'.format(method))
        # extract results
        O[:,0] = y[:,0]
        B[:,0] = y[:,1]
        SM[:,0] = y[:,2]
        M[:,0] = y[:,3]
        Z[:,0] = y[:,4]
        H[:,0] = y[:,5]
        I[:,0] = y[:,6]
        D[:,0] = y[:,7]
        T[:,0] = y[:,8]
    return(t,O,B,SM,M,Z,H,I,D,T)

def LSQ(thetas,data,fitTo,nat,mort,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,T0,monteCarlo,method,*findTime):
    if method == 'findInfected':
        # check if number of provided bounds is two
        if len(thetas) != 2:
            raise Exception('Number of bounds for method findInfected is 2. The number of provided bounds was: {}'.format(len(thetas)))
        # define length of simulation from provided data
        simtime = data.size-1
        tN = simtime+1
        # assign estimates to correct varaiable
        beta = thetas[0]
        B0=thetas[1]
        # construct initial condition
        y0 = np.array([T0-data[0,0]-B0,B0,0,data[0,0],0,0,0,0,T0])
        # run simulation
        y=runSimulation(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,tN,simtime,monteCarlo,method)
        # extract results
        ymodel=0
        for i in fitTo:
            ymodel = ymodel + np.mean(y[i],axis=1).reshape(np.mean(y[i],axis=1).size,1)
        # calculate quadratic error
        SSE = sum((ymodel-data)**2)

    elif method == 'findTime':
        # check if number of provided bounds is two
        if len(thetas) != 2:
            raise Exception('Number of bounds for method findTime is 2. The number of provided bounds was: {}'.format(len(thetas)))
        # assign estimates to correct varaiable
        beta = thetas[0]        
        extraTime = int(thetas[1])
        # define length of simulation from provided data
        simtime = data.size+extraTime-1
        tN = simtime+1
        # construct initial condition
        y0 = np.array([T0-1,1,0,0,0,0,0,0,T0])
        # run simulation
        y=runSimulation(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,tN,simtime,monteCarlo,method,(extraTime))
        # extract results
        # extract results
        ymodel=0
        for i in fitTo:
            ymodel = ymodel + np.mean(y[i],axis=1).reshape(np.mean(y[i],axis=1).size,1)
        ymodel = ymodel[extraTime-1:-1,0].reshape(ymodel[extraTime-1:-1,0].size,1)
        # calculate quadratic error
        SSE = sum((ymodel-data)**2)    
        
    elif method == 'findGovernmentResponse':
        # check if number of provided bounds is three
        if len(thetas) != 3:
            raise Exception('Number of bounds for method findGovernmentResponse is 3. The number of provided bounds was: {}'.format(len(thetas)))
        # assign estimates to correct varaiable
        beta = thetas[0]        
        extraTime = int(thetas[1])
        measureTime = int(thetas[2])
        # define length of simulation from provided data
        simtime = data.size+extraTime-1
        tN = simtime+1
        # construct initial condition
        y0 = np.array([T0-1,1,0,0,0,0,0,0,T0])
        # run simulation
        y=runSimulation(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,tN,simtime,monteCarlo,method,(extraTime,measureTime))
        # extract results
        # extract results
        ymodel=0
        for i in fitTo:
            ymodel = ymodel + np.mean(y[i],axis=1).reshape(np.mean(y[i],axis=1).size,1)
        ymodel = ymodel[extraTime-1:-1,0].reshape(ymodel[extraTime-1:-1,0].size,1)
        # calculate quadratic error
        SSE = sum((ymodel-data)**2)    
    else:
            raise Exception('Method not suited for least-squares fit: choose either findTime, findInfected or findGovernmentResponse. The provided method was: {}'.format(method))
    return(SSE)

def modelFit(bounds,data,fitTo,nat,mort,dSM,dM,dZ,m,z,h,mh,ICU,T0,monteCarlo,n_samples,maxiter,popsize,polish,disp,method):
    # Monte Carlo sampling of parameters gamma, dHD, dHI, sm, and call to genetic optimisation algorithm is performed here
    if monteCarlo == True:
        gammaVect = sampleFromDistribution('corona_incubatie.csv',n_samples)
        dHDvect = np.random.normal(18.5, 5.2, n_samples)
        dHIvect = np.random.normal(22.0, 5.2, n_samples)
        smvect = np.random.normal(0.86, 0.04/1.96, n_samples)
        thetas = scipy.optimize.differential_evolution(LSQ, bounds, args=(data,fitTo,nat,mort,gammaVect,dSM,dM,dZ,dHIvect,dHDvect,smvect,m,z,h,mh,ICU,T0,monteCarlo,method),disp=disp,polish=polish,workers=5,maxiter=maxiter, popsize=popsize,tol=1e-18)
    else:
        gamma = 5.2
        dHD = 18.5 
        dHI = 22.0
        sm = 0.86
        thetas = scipy.optimize.differential_evolution(LSQ, bounds, args=(data,fitTo,nat,mort,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,T0,monteCarlo,method),disp=disp,polish=polish,workers=5,maxiter=maxiter, popsize=popsize,tol=1e-18)
    fit = thetas.x
    return(fit)

def simModel(y0,nat,mort,beta,dSM,dM,dZ,m,z,h,mh,ICU,tN,simtime,monteCarlo,n_samples,method,*findTime):
    # This function is a wrapper for 'runSimulation'
    # Monte Carlo sampling of parameters gamma, dHD, dHI, sm, and call to genetic optimisation algorithm is performed here
    if monteCarlo == True:
        gammaVect = sampleFromDistribution('corona_incubatie.csv',n_samples)
        dHDvect = np.random.normal(18.5, 5.2, n_samples)
        dHIvect = np.random.normal(22.0, 5.2, n_samples)
        smvect = np.random.normal(0.86, 0.04/1.96, n_samples)
        simout = runSimulation(y0,nat,mort,beta,gammaVect,dSM,dM,dZ,dHIvect,dHDvect,smvect,m,z,h,mh,ICU,tN,simtime,monteCarlo,method,*findTime)
    else:
        gamma = 5.2
        dHD = 18.5 
        dHI = 22.0
        sm = 0.86
        simout = runSimulation(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,tN,simtime,monteCarlo,method,*findTime)
    return(simout)
 
def constructHorizon(theta,period):
    if type(theta) is np.ndarray:
        n = theta.size
        beta = np.ones([period*n])
        for i in range(n):
            beta[period*i:(period*i+period)]=theta[i]
    elif type(theta) is float:
        n = 1
        beta = np.ones([period*n])
        for i in range(n):
            beta[period*i:(period*i+period)]=theta
    else:
        raise Exception('Theta must be a vector or float. The provided datatype was: {}'.format(type(theta)))
    return(beta)
    
def MPCcalcWeights(thetas,y0,nat,mort,dSM,dM,dZ,m,z,h,mh,ICU,monteCarlo,n_samples,period,P,discrete,roundOff,Kh,Kd,Ki):
    # discretise thetas if wanted
    if discrete == True:
        thetas[(thetas<=0.3)&(thetas>roundOff[0])] = 0.244
        thetas[(thetas<=roundOff[0])&(thetas>roundOff[1])] = 0.20
        thetas[(thetas<=roundOff[1])&(thetas>roundOff[2])] = 0.10
        thetas[thetas<=roundOff[2]] = 0.03

    # build prediction horizon
    Pappend = np.ones([P-thetas.size])*thetas[-1]
    thetas=np.append(thetas,Pappend)
    beta = constructHorizon(thetas,period)
    simtime = beta.size-1
    tN = simtime + 1
    if monteCarlo == True:
        y = simModel(y0,nat,mort,beta,dSM,dM,dZ,m,z,h,mh,ICU,tN,simtime,monteCarlo,n_samples,'variableBeta')
        #y = runSimulation(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,tN,simtime,monteCarlo,'variableBeta')
        #y = runSimulationVariableBetaTripleMonteCarlo(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,tN,simtime,ICU,n_samples,g)
        H = np.mean(y[6],axis=1)
        H = np.mean(y[6],axis=1).reshape(H.size,1)    
    else:
        y = simModel(y0,nat,mort,beta,dSM,dM,dZ,m,z,h,mh,ICU,tN,simtime,monteCarlo,n_samples,'variableBeta')
        #y = runSimulation(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,ICU,tN,simtime,monteCarlo,'variableBeta')
        #y=runSimulationVariableBetaNoMonteCarlo(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,tN,simtime,ICU)
        H = y[6]
    #y = runSimulationVariableBetaTripleMonteCarlo(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,tN,simtime,ICU,n_samples,g)

    #D = np.mean(y[7],axis=1)
    #D = np.mean(y[7],axis=1).reshape(H.size,1)
    #I = np.mean(y[8],axis=1)
    #I = np.mean(y[8],axis=1).reshape(H.size,1)
    # regeling van de kritiek zieken
    y_sp = ICU # maximum aantal bedden op ICU
    ymodel = H # voorspelde aantal kritiek zieken bij ingang beta
    error = Kh*(y_sp - ymodel) # vector met fouten in de tijd
    SSE = sum(error**2)
    # regeling van het aantal doden op het tijdsinterval
    #y_sp = 0 # deaths during the current timeslot
    #y_model = D
    #error = Kd*(y_sp - y_model)
    #SSE = SSE + sum(error**2)
    # regeling van het aantal immunen
    #y_sp = 11430000
    #y_model = I
    #error = Ki*(y_sp - y_model)
    #SSE = SSE + sum(error**2)
    return(SSE)
                 
def MPCoptimize(y0,nat,mort,dSM,dM,dZ,m,z,h,mh,ICU,monteCarlo,n_samples,period,maxiter,popsize,polish,disp,P,N,discrete,roundOff,Kh,Kd,Ki):
    # Geef bounds op
    bounds=[]
    for i in range(N):
        bounds.append((0.03,0.244))
    # Perform optimisation                                                                           
    fit = scipy.optimize.differential_evolution(MPCcalcWeights, bounds, args=(y0,nat,mort,dSM,dM,dZ,m,z,h,mh,ICU,monteCarlo,n_samples,period,P,discrete,roundOff,Kh,Kd,Ki),disp=disp,polish=polish,workers=5,maxiter=maxiter, popsize=popsize,tol=1e-18)
    # discretise thetas if needed
    if discrete == True:
        thetas=fit.x
        thetas[(thetas<=0.3)&(thetas>roundOff[0])] = 0.244
        thetas[(thetas<=roundOff[0])&(thetas>roundOff[1])] = 0.20
        thetas[(thetas<=roundOff[1])&(thetas>roundOff[2])] = 0.10
        thetas[thetas<=roundOff[2]] = 0.03
    else:
        thetas=fit.x
    print(thetas)
    return(thetas)
 
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
    
# Optionally a PID controller can be used
def PIDcorona(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,T0,tN,simtime,ICU,n_samples,g,K,Ki,Kd):
    pid = PID(K, Ki, Kd, setpoint=1900)
    # define controller
    pid.sample_time = 1 # update every day
    pid.output_limits = (0.02, 0.244)
    # calculate new beta
    beta = pid(y0[5])    
    # run model for one timestep
    simtime = 1
    tN = 2             
    y=runSimulation(y0,nat,mort,beta,gamma,dSM,dM,dZ,dHI,dHD,sm,m,z,h,mh,T0,tN,simtime,ICU,n_samples,g)
    #print(y0[5],y[5])
    return(beta,y)
    
