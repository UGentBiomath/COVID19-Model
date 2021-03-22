import os
import random
import numpy as np
import pandas as pd
from functools import lru_cache

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

def contact_matrix(t, df_google, Nc_all, prev_home=1, prev_schools=1, prev_work=1, prev_transport=1, prev_leisure=1, prev_others=1, school=None, work=None, transport=None, leisure=None, others=None):
    """
    t : timestamp
        current date
    Nc_all : dictionnary
        contact matrices for home, schools, work, transport, leisure and others
    prev_... : float [0,1]
        prevention parameter to estimate
    school, work, transport, leisure, others : float [0,1]
        level of opening of these sectors
        if None, it is calculated from google mobility data
        only school cannot be None!
    """
    
    if t < pd.Timestamp('2020-03-15'):
        CM = Nc_all['total']
    else:
        
        if school is None:
            raise ValueError(
            "Please indicate to which extend schools are open")
        
        if pd.Timestamp('2020-03-15') < t <= df_google.index[-1]:
            #take t.date() because t can be more than a date! (e.g. when tau_days is added)
            row = -df_google[df_google.index == pd.Timestamp(t.date())]/100 
        else:
            row = -df_google.iloc[[-1],:]/100

        if work is None:
            work=(1-row['work'].values)[0]
        if transport is None:
            transport=(1-row['transport'].values)[0]
        if leisure is None:
            leisure=(1-row['retail_recreation'].values)[0]
        if others is None:
            others=(1-row['grocery'].values)[0]

        CM = (prev_home*(1/2.3)*Nc_all['home'] + 
              prev_schools*school*Nc_all['schools'] + 
              prev_work*work*Nc_all['work'] + 
              prev_transport*transport*Nc_all['transport'] + 
              prev_leisure*leisure*Nc_all['leisure'] + 
              prev_others*others*Nc_all['others']) 


    return CM

def mobility_update_func(t,states,param,agg,default_mobility=None):
    """
    Function to update the mobility matrix 'place' in spatially explicit models on a daily basis, from processed Proximus matrices. IMPORTANT: these data are not public, so they are not shared on GitHub. Make sure to copy the fractional-mobility-matrix_staytime_*_*.csv CSVs from the S-drive and locate them in data/interim/mobility/[agg]/staytime.
    
    Input
    -----
    t : timestamp
        current date as datetime object
    states : formal necessity (not used)
    param : formal necessity (not used)
    agg : str
        Denotes the spatial aggregation at hand. Either 'prov', 'arr' or 'mun'
        
    Returns
    -------
    place : matrix
        square matrix with floating points between 0 and 1, dimension depending on agg
    """
    
    # Import date_to_YYYYMMD function
    from ..data.mobility import date_to_YYYYMMDD
    
    # Define absolute location of this file
    abs_dir = os.path.dirname(__file__)
    # Define data location for this particular aggregation level
    data_location = '../../../data/interim/mobility/' + agg + '/staytime/'
    # Define YYYYMMDD date
    YYYYMMDD = date_to_YYYYMMDD(t)
    filename = 'fractional-mobility-matrix_staytime_' + agg + '_' + str(YYYYMMDD) + '.csv'
    try: # if there is data available for this date
        place = pd.read_csv(os.path.join(abs_dir, data_location+filename), \
                        index_col='mllp_postalcode').drop(index='Foreigner', columns='ABROAD').values
    except:
        if default_mobility: # If there is no data available and a user-defined input is given
            place = default_mobility
        else: # No data and no user input: fall back on average mobility
            place = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/' + agg + '/quick-average_staytime_' + agg + \
                                             '.csv'), index_col='mllp_postalcode').drop(index='Foreigner', columns='ABROAD').values
    return place

def lockdown_func(t,states,param,policy0,policy1,l,tau,prevention,start_date):
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
    
def policies_until_september(t,states,param,start_date,policy0,policy1,policy2,policy3,policy4,policy5,
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

def google_lockdown(t,states,param,df_google, Nc_all, Nc_15min, Nc_1hr, l , tau, prevention):
    
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
    elif t1 <= t <= df_google.index[-1]:
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

    
def google_lockdown_no_prev(t,states,param,df_google, Nc_all, Nc_15min, Nc_1hr, l , tau):
    
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
    elif t1 <= t <= df_google.index[-1]:
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

def social_policy_func(t,states,param,policy_time,policy1,policy2,tau,l):
    """
    Delayed ramp social policy function to implement a gradual change between policy1 and policy2. Copied from Michiel and superfluous in the mean time.
    
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Optimized google lockdown function below
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_contact_matrix_function(df_google, Nc_all):
    """
    Nc_all : dictionnary
            contact matrices for home, schools, work, transport, leisure and others
    df_google : dataframe
            google mobility data
    """
    
    df_google_array = df_google.values
    df_google_start = df_google.index[0]
    df_google_end = df_google.index[-1]
    
    @lru_cache()
    def all_contact(t):
        return Nc_all['total']

    @lru_cache()
    def all_contact_no_schools(t):
        return Nc_all['total'] - Nc_all['schools']

    @lru_cache() # once the function is run for a set of parameters, it doesn't need to compile again
    def contact_matrix_4prev(t, prev_home=1, prev_schools=1, prev_work=1, prev_rest = 1,
                       school=None, work=None, transport=None, leisure=None, others=None, home=None, SB=False):
        """
        t : timestamp
            current date
        prev_... : float [0,1]
            prevention parameter to estimate
        school, work, transport, leisure, others : float [0,1]
            level of opening of these sectors
            if None, it is calculated from google mobility data
            only school cannot be None!
        SB : str '2a', '2b' or '2c'
            '2a': september behaviour overall
            '2b': september behaviour, but work = lockdown behaviour
            '2c': september behaviour, but leisure = lockdown behaviour

        """

        if t < pd.Timestamp('2020-03-15'):
            CM = Nc_all['total']
        else:

            if school is None:
                raise ValueError(
                "Please indicate to which extend schools are open")

            if pd.Timestamp('2020-03-15') <= t <= df_google_end:
                #take t.date() because t can be more than a date! (e.g. when tau_days is added)
                idx = int((t - df_google_start) / pd.Timedelta("1 day")) 
                row = -df_google_array[idx]/100
            else:
                row = -df_google[-7:-1].mean()/100 # Extrapolate mean of last week

            if SB == '2a':
                row = -df_google['2020-09-01':'2020-10-01'].mean()/100
            elif SB == '2b':
                row = -df_google['2020-09-01':'2020-10-01'].mean()/100
                row[4] = -df_google['2020-03-15':'2020-04-01'].mean()[4]/100 
            elif SB == '2c':
                row = -df_google['2020-09-01':'2020-10-01'].mean()/100
                row[0] = -df_google['2020-03-15':'2020-04-01'].mean()[0]/100 
                
            # columns: retail_recreation grocery parks transport work residential
            if work is None:
                work= 1-row[4]
            if transport is None:
                transport=1-row[3]
            if leisure is None:
                leisure=1-row[0]
            if others is None:
                others=1-row[1]
            #if home is None:
            #    home = 1-row[5]

            CM = (prev_home*(1/2.3)*Nc_all['home'] + 
                  prev_schools*school*Nc_all['schools'] + 
                  prev_work*work*Nc_all['work'] + 
                  prev_rest*transport*Nc_all['transport'] + 
                  prev_rest*leisure*Nc_all['leisure'] + 
                  prev_rest*others*Nc_all['others']) 


        return CM

    return contact_matrix_4prev, all_contact, all_contact_no_schools