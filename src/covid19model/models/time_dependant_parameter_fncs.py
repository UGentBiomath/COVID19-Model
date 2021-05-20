import os
import random
import numpy as np
import pandas as pd
from functools import lru_cache

def delayed_ramp_fun(Nc_old, Nc_new, t, tau_days, l, t_start):
    """
    t : timestamp
        current date
    tau : int
        number of days before measures start having an effect
    l : int
        number of additional days after the time delay until full compliance is reached
    """
    return Nc_old + (Nc_new-Nc_old)/l * (t-t_start-tau_days)/pd.Timedelta('1D')

def ramp_fun(Nc_old, Nc_new, t, t_start, l):
    """
    t : timestamp
        current date
    l : int
        number of additional days after the time delay until full compliance is reached
    """
    return Nc_old + (Nc_new-Nc_old)/l * (t-t_start)/pd.Timedelta('1D')

def load_all_mobility_data(agg, dtype='fractional', beyond_borders=False):
    """
    Function that fetches all available mobility data and adds it to a DataFrame with dates as indices and numpy matrices as values. Make sure to regularly update the mobility data with the notebook notebooks/preprocessing/Quick-update_mobility-matrices.ipynb to get the data for the most recent days. Also returns the average mobility over all available data, which might NOT always be desirable as a back-up mobility.
    
    Input
    -----
    agg : str
        Denotes the spatial aggregation at hand. Either 'prov', 'arr' or 'mun'
    dtype : str
        Choose the type of mobility data to return. Either 'fractional' (default), staytime (all available hours for region g spent in h), or visits (all unique visits from region g to h)
    beyond_borders : boolean
        If true, also include mobility abroad and mobility from foreigners
    
    Returns
    -------
    all_mobility_data : pd.DataFrame
        DataFrame with datetime objects as indices ('DATE') and np.arrays ('place') as value column
    average_mobility_data : np.array
        average mobility matrix over all available dates
    """

    ### Validate input ###
    
    if agg not in ['mun', 'arr', 'prov']:
        raise ValueError(
                    "spatial stratification '{0}' is not legitimate. Possible spatial "
                    "stratifications are 'mun', 'arr', or 'prov'".format(agg)
                )
    if dtype not in ['fractional', 'staytime', 'visits']:
        raise ValueError(
                    "data type '{0}' is not legitimate. Possible mobility matrix "
                    "data types are 'fractional', 'staytime', or 'visits'".format(dtype)
                )
    
    ### Load all available data ###
    
    # Define absolute location of this file
    abs_dir = os.path.dirname(__file__)
    # Define data location for this particular aggregation level
    data_location = f'../../../data/interim/mobility/{agg}/{dtype}'
    
    # Iterate over all available interim mobility data
    all_available_dates=[]
    all_available_places=[]
    directory=os.path.join(abs_dir, f'{data_location}')
    for csv in os.listdir(directory):
        # take YYYYMMDD information from processed CSVs. NOTE: this supposes a particular data name format!
        datum = csv[-12:-4]
        # Create list of datetime objects
        all_available_dates.append(pd.to_datetime(datum, format="%Y%m%d"))
        # Load the CSV as a np.array
        if beyond_borders:
            place = pd.read_csv(f'{directory}/{csv}', index_col='mllp_postalcode').values
        else:
            place = pd.read_csv(f'{directory}/{csv}', index_col='mllp_postalcode').drop(index='Foreigner', columns='ABROAD').values
            if dtype=='fractional':
                # make sure the rows sum up to 1 nicely again after dropping a row and a column
                place = place / place.sum(axis=1)
        # Create list of places
        all_available_places.append(place)
    # Create new empty dataframe with available dates. Load mobility later
    df = pd.DataFrame({'DATE' : all_available_dates, 'place' : all_available_places}).set_index('DATE')
    all_mobility_data = df.copy()
    
    # Take average of all available mobility data
    average_mobility_data = df['place'].values.mean()
    
    return all_mobility_data, average_mobility_data
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Mobility update functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~

class make_mobility_update_function():
    """
    Output the time-dependent mobility function with the data loaded in cache

    Input
    -----
    proximus_mobility_data : DataFrame
        Pandas DataFrame with dates as indices and matrices as values. Output of mobility.get_proximus_mobility_data.
    proximus_mobility_data_avg : np.array
        Average mobility matrix over all matrices
    """
    def __init__(self, proximus_mobility_data, proximus_mobility_data_avg):
        self.proximus_mobility_data = proximus_mobility_data
        self.proximus_mobility_data_avg = proximus_mobility_data_avg

    @lru_cache()
    # Define mobility_update_func
    def __call__(self, t, default_mobility=None):
        """
        time-dependent function which has a mobility matrix of type dtype for every date.
        Note: only works with datetime input (no integer time steps). This

        Input
        -----
        t : timestamp
            current date as datetime object
        states : str
            formal necessity
        param : str
            formal necessity
        default_mobility : np.array or None
            If None (default), returns average mobility over all available dates. Else, return user-defined mobility

        Returns
        -------
        place : np.array
            square matrix with mobility of type dtype (fractional, staytime or visits), dimension depending on agg
        """
        t = pd.Timestamp(t.date())
        try: # if there is data available for this date (if the key exists)
            place = self.proximus_mobility_data['place'][t]
        except:
            if default_mobility: # If there is no data available and a user-defined input is given
                place = self.default_mobility
            else: # No data and no user input: fall back on average mobility
                place = self.proximus_mobility_data_avg
        return place

    def mobility_wrapper_func(self, t, states, param, default_mobility=None):
        t = pd.Timestamp(t.date())
        return self.__call__(t, default_mobility=default_mobility)


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
        return delayed_ramp_fun(policy0, prevention*policy1, t, tau_days, l, start_date)
    else:
        return prevention*policy1

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

# ~~~~~~~~~~~~~
# VOC functions
# ~~~~~~~~~~~~~

def make_VOCB117_function():
    # Load and format VOC data 
    rel_dir = os.path.join(os.path.dirname(__file__), '../../../data/raw/VOCs/sequencing_501YV1_501YV2_501YV3.csv')
    df_VOC = pd.read_csv(rel_dir, parse_dates=True).set_index('collection_date', drop=True).drop(columns=['sampling_week','year', 'week'])
    # Converting the index as date
    df_VOC.index = pd.to_datetime(df_VOC.index)
    # Extrapolate missing dates to obtain a continuous index
    df_VOC = df_VOC.resample('D').interpolate('linear')
    df_VOC['baselinesurv_f_501Y.V1_501Y.V2_501Y.V3'] = (df_VOC['baselinesurv_n_501Y.V1']+df_VOC['baselinesurv_n_501Y.V2']+df_VOC['baselinesurv_n_501Y.V3'])/df_VOC['baselinesurv_total_sequenced']

    @lru_cache()
    def VOCB117_function(t):
        # Function to return fraction of non-wild type SARS variants
        if t < df_VOC.index.min():
            return 0
        elif df_VOC.index.min() <= t <= df_VOC.index.max():
            return df_VOC['baselinesurv_f_501Y.V1_501Y.V2_501Y.V3'][t]
        elif t > df_VOC.index.max():
            return (df_VOC['baselinesurv_n_501Y.V1'][-1]+df_VOC['baselinesurv_n_501Y.V2'][-1]+df_VOC['baselinesurv_n_501Y.V3'][-1])/df_VOC['baselinesurv_total_sequenced'][-1]

    return VOCB117_function

def VOC_wrapper_func(t,states,param, VOC_function):
    t = pd.Timestamp(t.date())
    return VOC_function(t)

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
    delay = pd.Timedelta(str(int(delay))+'D')
    # Compute the number of vaccine eligible individuals
    VE = states['S'] + states['R']
    
    if t <= df_sciensano_start + delay:
        return np.zeros(9)
    elif df_sciensano_start + delay < t <= df_sciensano_end + delay:
        return sciensano_first_dose(t-delay)
    else:
        N_vacc = np.zeros(9)
        # Vaccines distributed according to vector 'order'
        # With residue 'refusal' remaining in each age group
        idx = 0
        while daily_dose > 0:
            if VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]]) > daily_dose:
                N_vacc[vacc_order[idx]] = daily_dose
                daily_dose = 0
            else:
                N_vacc[vacc_order[idx]] = VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                daily_dose = daily_dose - VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                idx = idx + 1
        return N_vacc


# ~~~~~~~~~~~~~~~~~~~~~~
# Google policy function
# ~~~~~~~~~~~~~~~~~~~~~~

class make_contact_matrix_function():
    """
    Class that returns contact matrix based on 4 prevention parameters by default, but has other policies defined as well.

    Input
    -----
    Nc_all : dictionnary
        contact matrices for home, schools, work, transport, leisure and others
    df_google : dataframe
        google mobility data

    Output
    ------

    __class__ : default function
        Default output function, based on contact_matrix_4prev

    """
    def __init__(self, df_google, Nc_all):
        self.df_google = df_google
        self.df_google_array = df_google.values
        self.df_google_start = df_google.index[0]
        self.df_google_end = df_google.index[-1]
        self.Nc_all = Nc_all


    @lru_cache() # once the function is run for a set of parameters, it doesn't need to compile again
    # This is the default output, what was earlier contact_matrix_4prev
    def __call__(self, t, prev_home=1, prev_schools=1, prev_work=1, prev_rest = 1,
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
            CM = self.Nc_all['total']
        else:

            if school is None:
                raise ValueError(
                "Please indicate to which extend schools are open")

            if pd.Timestamp('2020-03-15') <= t <= self.df_google_end:
                #take t.date() because t can be more than a date! (e.g. when tau_days is added)
                idx = int((t - self.df_google_start) / pd.Timedelta("1 day"))
                row = -self.df_google_array[idx]/100
            else:
                row = -self.df_google[-7:-1].mean()/100 # Extrapolate mean of last week

            if SB == '2a':
                row = -self.df_google['2020-09-01':'2020-10-01'].mean()/100
            elif SB == '2b':
                row = -self.df_google['2020-09-01':'2020-10-01'].mean()/100
                row[4] = -self.df_google['2020-03-15':'2020-04-01'].mean()[4]/100
            elif SB == '2c':
                row = -self.df_google['2020-09-01':'2020-10-01'].mean()/100
                row[0] = -self.df_google['2020-03-15':'2020-04-01'].mean()[0]/100

            # columns: retail_recreation grocery parks transport work residential
            if work is None:
                work= 1-row[4]
            if transport is None:
                transport=1-row[3]
            if leisure is None:
                leisure=1-row[0]
            if others is None:
                others=1-row[1]

            CM = (prev_home*self.Nc_all['home'] +
                  prev_schools*school*self.Nc_all['schools'] +
                  prev_work*work*self.Nc_all['work'] +
                  prev_rest*transport*self.Nc_all['transport'] +
                  prev_rest*leisure*self.Nc_all['leisure'] +
                  prev_rest*others*self.Nc_all['others'])

        return CM

    def all_contact(self,t,states,param):
        return self.Nc_all['total']

    def all_contact_no_schools(self,t,states,param):
        return self.Nc_all['total'] - self.Nc_all['schools']

    def policies_wave1_4prev(self, t, states, param, l , tau, prev_schools, prev_work, prev_rest, prev_home):
        '''
        Function that loads the correct prevention parameters and includes compliance for the first wave. Returns contact matrix.
        
        Input
        -----
        t : Timestamp
        states : formality
        param : formality
        l : float
            Compliance parameter for delayed_ramp_fun
        tau : float
            Compliance parameter for delayed_ramp_fun
        prev_{location} : float
            Effectivity of prevention measures at {location}
        
        Returns
        -------
        CM : np.array
            Effective contact matrix (output of __call__ function)
        '''
        all_contact = self.Nc_all['total']
        all_contact_no_schools = self.Nc_all['total'] - self.Nc_all['schools']

        # Convert tau and l to dates
        tau_days = pd.Timedelta(tau, unit='D')
        l_days = pd.Timedelta(l, unit='D')

        # Define additional dates where intensity or school policy changes
        t1 = pd.Timestamp('2020-03-15') # start of lockdown
        t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
        t3 = pd.Timestamp('2020-07-01') # start of summer holidays
        t4 = pd.Timestamp('2020-09-01') # end of summer holidays

        if t <= t1:
            return all_contact
        elif t1 < t < t1 + tau_days:
            return all_contact
        elif t1 + tau_days < t <= t1 + tau_days + l_days:
            t = pd.Timestamp(t.date())
            policy_old = all_contact
            policy_new = self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest,
                                       school=0)
            return self.delayed_ramp_fun(policy_old, policy_new, t, tau_days, l, t1)
        elif t1 + tau_days + l_days < t <= t2:
            t = pd.Timestamp(t.date())
            return self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest, 
                                  school=0)
        elif t2 < t <= t3:
            t = pd.Timestamp(t.date())
            return self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest, 
                                  school=0)
        elif t3 < t <= t4:
            t = pd.Timestamp(t.date())
            return self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest, 
                                  school=0)                     
        else:
            t = pd.Timestamp(t.date())
            return self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest, 
                                  school=0)
        
    def delayed_ramp_fun(self, Nc_old, Nc_new, t, tau_days, l, t_start):
        """
        t : timestamp
            current date
        tau : int
            number of days before measures start having an effect
        l : int
            number of additional days after the time delay until full compliance is reached
        """
        return Nc_old + (Nc_new-Nc_old)/l * (t-t_start-tau_days)/pd.Timedelta('1D')


# Define policy function
def policies_WAVE1(t, states, param, l, prev_schools, prev_work, prev_rest, prev_home):

    # Convert time to timestamp
    t = pd.Timestamp(t.date())

    # Convert l to a date
    l_days = pd.Timedelta(l, unit='D')

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-08-07') # end of 'second wave' in antwerp
    t5 = pd.Timestamp('2020-09-01') # end of summer holidays

    if t <= t1:
        return all_contact(t)
    elif t1 < t <= t1 + l_days:
        policy_old = all_contact(t)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, t1, l)
    elif t1 + l_days < t <= t2:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    ## WARNING: During the summer of 2020, highly localized clusters appeared in Antwerp city, and lockdown measures were taken locally
    ## Do not forget this is a national-level model, you need a spatially explicit model to correctly model localized phenomena.
    ## The following is an ad-hoc tweak to assure a fit on the data during summer in order to be as accurate as possible with the seroprelevance
    elif t3 < t <= t3 + l_days:
        policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, school=0)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, 0.75, school=0)
        return ramp_fun(policy_old, policy_new, t, t3, l)
    elif t3 + l_days < t <= t4:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, 0.75, school=0)
    elif t4 < t <= t5:
        return contact_matrix_4prev(t, prev_home, prev_schools, 0.05, 0.05, 
                              school=0)                                          
    else:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
