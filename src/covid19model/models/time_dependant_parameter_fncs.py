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

# ~~~~~~~~~~~~~~~~~~~~~
# Vaccination functions
# ~~~~~~~~~~~~~~~~~~~~~

def make_vaccination_function(df_sciensano):
    df_sciensano_start = df_sciensano['V1_tot'].ne(0).idxmax()
    df_sciensano_end = df_sciensano.index[-1]

    @lru_cache()
    def sciensano_first_dose(t):
        # Extrapolate Sciensano n0. vaccinations to the model's native age bins
        N_vacc = np.zeros(9)
        N_vacc[1] = (2/17)*df_sciensano['V1_18_34'][t] # 10-20
        N_vacc[2] = (10/17)*df_sciensano['V1_18_34'][t] # 20-30
        N_vacc[3] = (5/17)*df_sciensano['V1_18_34'][t] + (5/10)*df_sciensano['V1_35_44'][t] # 30-40
        N_vacc[4] = (5/10)*df_sciensano['V1_35_44'][t] + (5/10)*df_sciensano['V1_45_54'][t] # 40-50
        N_vacc[5] = (5/10)*df_sciensano['V1_45_54'][t] + (5/10)*df_sciensano['V1_55_64'][t] # 50-60
        N_vacc[6] = (5/10)*df_sciensano['V1_55_64'][t] + (5/10)*df_sciensano['V1_65_74'][t] # 60-70
        N_vacc[7] = (5/10)*df_sciensano['V1_65_74'][t] + (5/10)*df_sciensano['V1_75_84'][t] # 70-80
        N_vacc[8] = (5/10)*df_sciensano['V1_75_84'][t] + (5/10)*df_sciensano['V1_85+'][t]# 80+
        return N_vacc
    
    return sciensano_first_dose, df_sciensano_start, df_sciensano_end

def vacc_strategy(t, states, param, sciensano_first_dose, df_sciensano_start, df_sciensano_end,
                    daily_dose=30000, delay = 21, vacc_order = [8,7,6,5,4,3,2,1,0], refusal = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]):
    """
    time-dependent function for the Belgian vaccination strategy
    First, all available data from Sciensano are used. Then, the user can specify a custom vaccination strategy of "daily_dose" doses per day,
    given in the order specified by the vector "vacc_order" with a refusal propensity of "refusal" in every age group.
  
    Parameters
    ----------
    t : int
        Simulation time
    states: dict
        Dictionary containing values of model states
    param : dict
        Model parameter dictionary
    sciensano_first_dose : function
        Function returning the number of (first dose) vaccinated individuals at simulation time t, according to the data made public by Sciensano.
    df_sciensano_start : date
        Start date of Sciensano vaccination data frame
    df_sciensano_end : date
        End date of Sciensano vaccination data frame
    daily_dose : int
        Number of doses administered per day. Default is 30000 doses/day.
    delay : int
        Time delay between first dose vaccination and start of immunity. Default is 21 days.
    vacc_order : array
        Vector containing vaccination prioritization preference. Default is old to young. Must be equal in length to the number of age bins in the model.
    refusal: array
        Vector containing the fraction of individuals refusing a vaccine per age group. Default is 30% in every age group. Must be equal in length to the number of age bins in the model.

    Return
    ------
    N_vacc : array
        Number of individuals to be vaccinated at simulation time "t"
        
    """
    
    # Convert time to suitable format
    t = pd.Timestamp(t.date())
    # Convert delay to a timedelta
    delay = pd.Timedelta(str(delay)+'D')
    # Compute the number of vaccine eligible individuals
    VE = states['S'] + states['R']
    
    if t < df_sciensano_start + delay:
        return np.zeros(9)
    elif df_sciensano_start + delay <= t <= df_sciensano_end + delay:
        return sciensano_first_dose(t-delay)
    else:
        N_vacc = np.zeros(9)
        # Vaccines distributed according to vector 'order'
        # With residue 'refusal' remaining in each age group
        idx = 0
        while d > 0:
            if VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]]) > d:
                N_vacc[vacc_order[idx]] = d
                d = 0
            else:
                N_vacc[vacc_order[idx]] = VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                d = d - VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                idx = idx + 1
        return N_vacc


# ~~~~~~~~~~~~~~~~~~~~~~
# Google policy function
# ~~~~~~~~~~~~~~~~~~~~~~

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