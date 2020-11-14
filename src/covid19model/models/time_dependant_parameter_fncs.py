import os
import random
import numpy as np
import pandas as pd

def ramp_fun(Nc_old, Nc_new, t, tau_days, l, t_start):
    """
    t : timestamp
        current date
    tau : int
        number of days before measures start having an effect
    l : int
        number of additional days after the time delay until full compliance is reached
    """
    return Nc_old + (Nc_new-Nc_old)/l * (t-t_start-tau_days)/pd.Timedelta('1D')

def lockdown_func(t,param,policy0,policy1,l,tau,prevention,start_date):
    """
    Lockdown function handling t as datetime
    
    t : timestamp
        current date
    policy0 : matrix
        policy before lockdown (= no policy)
    policy1 : matrix
        policy during lockdown
    tau : int
        number of days before measures start having an effect
    l : int
        number of additional days after the time delay until full compliance is reached
    start_date : timestamp
        start date of the data
    """
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')
    if t <= start_date + tau_days:
        return policy0
    elif start_date + tau_days < t <= start_date + tau_days + l_days:
        return ramp_fun(policy0, prevention*policy1, t, tau_days, l, start_date)
    else:
        return prevention*policy1
    
def policies_until_september(t,param,start_date,policy0,policy1,policy2,policy3,policy4,policy5,
                               policy6,policy7,policy8,policy9,l,tau,prevention):
    """
    t : timestamp
        current date
    policy0 : matrix
        policy before lockdown (= no policy)
    policy1 : matrix
        policy during lockdown
    policy 2: matrix
        reopening industry
    policy 3: matrix
        merging of two bubbels
    policy 4: matrix
        reopening of businesses
    policy 5: matrix
        partial reopening schools
    policy 6: matrix
        reopening schools, bars, restaurants
    policy 7: matrix
        school holidays, gatherings 15 people, cultural event
    policy 8: matrix
       "second" wave 
    policy 9: matrix
        opening schools
    tau : int
        number of days before measures start having an effect
    l : int
        number of additional days after the time delay until full compliance is reached
    start_date : timestamp
        start date of the data
    

    """

    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')
    t1 = pd.Timestamp('2020-05-04') # reopening industry
    t2 = pd.Timestamp('2020-05-06') # merging of two bubbels
    t3 = pd.Timestamp('2020-05-11') # reopening of businesses
    t4 = pd.Timestamp('2020-05-18') # partial reopening schools
    t5 = pd.Timestamp('2020-06-04') # reopening schools, bars, restaurants
    t6 = pd.Timestamp('2020-07-01') # school holidays, gatherings 15 people, cultural event
    t7 = pd.Timestamp('2020-07-31') # "second" wave
    t8 = pd.Timestamp('2020-09-01') # opening schools
    
    if t <= start_date + tau_days:
        return policy0
    elif start_date + tau_days < t <= start_date + tau_days + l_days:
        return ramp_fun(policy0, prevention*policy1, t, tau_days, l, start_date)
    elif start_date + tau_days + l_days < t <= t1: 
        return prevention*policy1 # lockdown
    elif t1 < t <= t2:
        return prevention*policy2 # re-opening industry
    elif t2 < t <= t3:
        return prevention*policy3
    elif t3 < t <= t4:
        return prevention*policy4
    elif t4 < t <= t5:
        return prevention*policy5
    elif t5 < t <= t6:
        return prevention*policy6
    elif t6 < t <= t7:
        return prevention*policy7
    elif t7 < t <= t8:
        return prevention*policy8
    elif t8 < t:
        return prevention*policy9

def google_lockdown(t,param,df_google, Nc_all, Nc_15min, Nc_1hr, l , tau, prevention):
    
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer: COVID-urgency very low
    t4 = pd.Timestamp('2020-08-01')
    t5 = pd.Timestamp('2020-09-01') # september: lockdown relaxation narrative in newspapers reduces sense of urgency
    t6 = pd.Timestamp('2020-10-19') # lockdown
    t7 = pd.Timestamp('2020-11-16') # schools re-open
    t8 = pd.Timestamp('2020-12-18') # schools close
    t9 = pd.Timestamp('2021-01-04') # schools re-open

    # get mobility reductions
    if t < t1:
        return Nc_all['total']
    elif t1 < t <= df_google.index[-1]:
        row = -df_google[df_google.index == pd.Timestamp(t.date())]/100
    elif t > df_google.index[-1]:
        row=-df_google[df_google.index == df_google.index[-1]]/100
    
    work=(1-row['work'].values)[0]
    transport=(1-row['transport'].values)[0]
    leisure=(1-row['retail_recreation'].values)[0]
    others=(1-row['grocery'].values)[0]

    # define policies
    if t1 < t <= t1 + tau_days:
        school = 0
        return (1/2.3)*Nc_all['home'] + work*Nc_all['work'] + school*Nc_all['schools'] + transport*Nc_all['transport'] + leisure*Nc_all['leisure'] + others*Nc_all['others']
    elif t1 + tau_days < t <= t1 + tau_days + l_days:
        school = 0
        policy_old = (1/2.3)*Nc_all['home'] + work*Nc_all['work'] + school*Nc_all['schools'] + transport*Nc_all['transport'] + leisure*Nc_all['leisure'] + others*Nc_all['others']
        policy_new = (1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others']
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t1)
    elif t1 + tau_days + l_days < t <= t2:
        school = 0
        return (1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others']  
    elif t2 < t <= t3:
        school = 0
        return (1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others']
    elif t3 < t <= t4:
        school = 0
        return (1/2.3)*Nc_15min['home'] + work*Nc_15min['work'] + school*Nc_15min['schools'] + transport*Nc_15min['transport'] + leisure*Nc_15min['leisure'] + others*Nc_15min['others'] 
    elif t4 < t <= t5:
        school = 0
        return (1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others']     
    elif t5 < t <= t6 + tau_days:
        school = 1
        return (1/2.3)*Nc_15min['home'] + work*Nc_15min['work'] + school*Nc_15min['schools'] + transport*Nc_15min['transport'] + leisure*Nc_15min['leisure'] + others*Nc_15min['others']
    elif t6 + tau_days < t <= t6 + tau_days + l_days:
        school = 1
        policy_old = (1/2.3)*Nc_15min['home'] + work*Nc_15min['work'] + school*Nc_15min['schools'] + transport*Nc_15min['transport'] + leisure*Nc_15min['leisure'] + others*Nc_15min['others']
        policy_new = prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + 0*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t6)
    elif t6 + tau_days + l_days < t <= t7:
        school = 0
        return prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])
    elif t7 < t <= t8:
        school = 1
        return prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])
    elif t8 < t <= t9:
        school = 0
        return prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])
    else:
        school = 1
        return prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])

    
def google_lockdown_no_prev(t,param,df_google, Nc_all, Nc_15min, Nc_1hr, l , tau):
    
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')
    prevention = 1

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer: COVID-urgency very low
    t4 = pd.Timestamp('2020-08-01')
    t5 = pd.Timestamp('2020-09-01') # september: lockdown relaxation narrative in newspapers reduces sense of urgency
    t6 = pd.Timestamp('2020-10-19') # lockdown
    t7 = pd.Timestamp('2020-11-16') # schools re-open
    t8 = pd.Timestamp('2020-12-18') # schools close
    t9 = pd.Timestamp('2021-01-04') # schools re-open

    # get mobility reductions
    if t < t1:
        return Nc_all['total']
    elif t1 < t <= df_google.index[-1]:
        row = -df_google[df_google.index == pd.Timestamp(t.date())]/100
    elif t > df_google.index[-1]:
        row=-df_google[df_google.index == df_google.index[-1]]/100
    
    work=(1-row['work'].values)[0]
    transport=(1-row['transport'].values)[0]
    leisure=(1-row['retail_recreation'].values)[0]
    others=(1-row['grocery'].values)[0]

    # define policies
    if t1 < t <= t1 + tau_days:
        school = 0
        return (1/2.3)*Nc_all['home'] + work*Nc_all['work'] + school*Nc_all['schools'] + transport*Nc_all['transport'] + leisure*Nc_all['leisure'] + others*Nc_all['others']
    elif t1 + tau_days < t <= t1 + tau_days + l_days:
        school = 0
        policy_old = (1/2.3)*Nc_all['home'] + work*Nc_all['work'] + school*Nc_all['schools'] + transport*Nc_all['transport'] + leisure*Nc_all['leisure'] + others*Nc_all['others']
        policy_new = (1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others']
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t1)
    elif t1 + tau_days + l_days < t <= t2:
        school = 0
        return (1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others']  
    elif t2 < t <= t3:
        school = 0
        return (1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others']
    elif t3 < t <= t4:
        school = 0
        return (1/2.3)*Nc_15min['home'] + work*Nc_15min['work'] + school*Nc_15min['schools'] + transport*Nc_15min['transport'] + leisure*Nc_15min['leisure'] + others*Nc_15min['others'] 
    elif t4 < t <= t5:
        school = 0
        return (1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others']     
    elif t5 < t <= t6 + tau_days:
        school = 1
        return (1/2.3)*Nc_15min['home'] + work*Nc_15min['work'] + school*Nc_15min['schools'] + transport*Nc_15min['transport'] + leisure*Nc_15min['leisure'] + others*Nc_15min['others']
    elif t6 + tau_days < t <= t6 + tau_days + l_days:
        school = 1
        policy_old = (1/2.3)*Nc_15min['home'] + work*Nc_15min['work'] + school*Nc_15min['schools'] + transport*Nc_15min['transport'] + leisure*Nc_15min['leisure'] + others*Nc_15min['others']
        policy_new = prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + 0*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t6)
    elif t6 + tau_days + l_days < t <= t7:
        school = 0
        return prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])
    elif t7 < t <= t8:
        school = 1
        return prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])
    elif t8 < t <= t9:
        school = 0
        return prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])
    else:
        school = 1
        return prevention*((1/2.3)*Nc_1hr['home'] + work*Nc_1hr['work'] + school*Nc_1hr['schools'] + transport*Nc_1hr['transport'] + leisure*Nc_1hr['leisure'] + others*Nc_1hr['others'])


def social_policy_func(t,param,policy_time,policy1,policy2,tau,l):
    """
    Delayed ramp social policy function to implement a gradual change between policy1 and policy2.
    
    Parameters
    ----------
    t : int
        Time parameter. Runs simultaneously with simulation time
    param : 
        Currently obsolete parameter that may be used in a future stage
    policy_time : int
        Time in the simulation at which a new policy is imposed
    policy1 : float or int or list or matrix
        Value corresponding to the policy before t = policy_time (e.g. full mobility)
    policy2 : float or int or list or matrix (same dimensions as policy1)
        Value corresponding to the policy after t = policy_time (e.g. 50% mobility)
    tau : int
        Delayed ramp parameter: number of days before the new policy has any effect
    l : int
        Delayed ramp parameter: number of days after t = policy_time + tau the new policy reaches full effect (policy2)
        
    Return
    ------
    state : float or int or list or matrix
        Either policy1, policy2 or an intermediate state.
        
    """
    # Nothing changes before policy_time
    if t < policy_time:
        state = policy1
    # From t = policy time onward, the delayed ramp takes effect toward policy2
    else:
        # Time starting at policy_time
        tt = t-policy_time
        if tt <= tau:
            state = policy1
        if (tt > tau) & (tt <= tau + l):
            intermediate = (policy2 - policy1) / l * (tt - tau) + policy1
            state = intermediate
        if tt > tau + l:
            state = policy2
    return state