import os
import random
import numpy as np
import pandas as pd
import itertools
from functools import lru_cache

##########################
## Compliance functions ##
##########################

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


###############################
## Mobility update functions ##
###############################

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

###################
## VOC functions ##
###################

class make_VOC_function():
    """
    Class that returns a time-dependant parameter function for COVID-19 SEIRD model parameter alpha (variant fraction).
    Current implementation includes the alpha - delta strains.
    If the class is initialized without arguments, a logistic model fitted to prelevance data of the alpha-gamma variant is used. The class can also be initialized with the alpha-gamma prelavence data provided by Prof. Tom Wenseleers.
    A logistic model fitted to prelevance data of the delta variant is always used.

    Input
    -----
    *df_abc: pd.dataFrame (optional)
        Alpha, Beta, Gamma prelevance dataset by Tom Wenseleers, obtained using:
        `from covid19model.data import VOC`
        `df_abc = VOC.get_abc_data()`
        `VOC_function = make_VOC_function(df_abc)`

    Output
    ------

    __class__ : function
        Default variant function

    """
    def __init__(self, *df_abc):
        self.df_abc = df_abc
        self.data_given = False
        if self.df_abc != ():
            self.df_abc = df_abc[0] # First entry in list of optional arguments (dataframe)
            self.data_given = True

    @lru_cache()
    def VOC_abc_data(self,t):
        return self.df_abc.iloc[self.df_abc.index.get_loc(t, method='nearest')]['baselinesurv_f_501Y.V1_501Y.V2_501Y.V3']

    @lru_cache()
    def VOC_abc_logistic(self,t):
        # Parameters obtained by fitting logistic model to weekly prevalence data
        t_sig = pd.Timestamp('2021-02-14')
        k = 0.07
        # Function to return the fraction of the delta-variant
        return 1/(1+np.exp(-k*(t-t_sig)/pd.Timedelta(days=1)))

    @lru_cache()
    def VOC_delta_logistic(self,t):
        # Parameters obtained by fitting logistic model to weekly prevalence data
        t_sig = pd.Timestamp('2021-06-25')
        k = 0.11
        # Function to return the fraction of the delta-variant
        return 1/(1+np.exp(-k*(t-t_sig)/pd.Timedelta(days=1)))

    # Default VOC function includes British and Indian variants
    def __call__(self, t, states, param):
        # Convert time to timestamp
        t = pd.Timestamp(t.date())
        # Introduction Indian variant
        t1 = pd.Timestamp('2021-05-01')
        # Construct alpha
        if t <= t1:
            if self.data_given:
                return np.array([1-self.VOC_abc_data(t), self.VOC_abc_data(t), 0])
            else:
                return np.array([1-self.VOC_abc_logistic(t), self.VOC_abc_logistic(t), 0])
        else:
            return np.array([0, 1-self.VOC_delta_logistic(t), self.VOC_delta_logistic(t)])

###########################
## Vaccination functions ##
###########################

from covid19model.data.model_parameters import construct_initN

class make_vaccination_function():
    """
    Class that returns a two-fold time-dependent parameter function for the vaccination strategy by default. First, first dose data by sciensano are used. In the future, a hypothetical scheme is used. If spatial data is given, the output consists of vaccination data per NIS code.

    Input
    -----
    df : pd.dataFrame
        *either* Sciensano public dataset, obtained using:
        `from covid19model.data import sciensano`
        `df = sciensano.get_sciensano_COVID19_data(update=False)`
        
        *or* public spatial vaccination data, obtained using:
        `from covid19model.data import sciensano`
        `df = sciensano.get_public_spatial_vaccination_data(update=False,agg='arr')`
        
    spatial : Boolean
        True if df is spatially explicit. None by default.

    Output
    ------

    __class__ : function
        Default vaccination function

    """
    def __init__(self, df, age_stratification_size=10):
        # Assign inputs to object
        self.df = df
        self.age_agg = age_stratification_size
        
        # Check if spatial data is provided
        self.spatial = None
        if 'NIS' in self.df.index.names:
            self.spatial = True
            self.space_agg = len(self.df.index.get_level_values('NIS').unique().values)

        # Check if dose data is provided
        self.doses = None
        if 'dose' in self.df.index.names:
            self.doses = True
            self.dose_agg = len(self.df.index.get_level_values('dose').unique().values)

        # Define start- and enddate
        self.df_start = self.df.index.get_level_values('date').min()
        self.df_end = self.df.index.get_level_values('date').max()

        # Define age groups
        if age_stratification_size == 3:
            age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
        elif age_stratification_size == 9:
            age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
        elif age_stratification_size == 10:
            age_classes = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')

        # Perform age conversion
        # Define dataframe with desired format
        iterables=[]
        for index_name in self.df.index.names:
            if index_name != 'age':
                iterables += [self.df.index.get_level_values(index_name).unique()]
            else:
                iterables += [age_classes]
        index = pd.MultiIndex.from_product(iterables, names=self.df.index.names)
        self.new_df = pd.Series(index=index)

        # Four possibilities exist
        if self.spatial:
            if self.doses:
                raise ValueError(
                    "The combination of a spatially explicit, multidose vaccine model is not implemented!"
                )
            else:
                for date in self.df.index.get_level_values('date').unique():
                    for NIS in self.df.index.get_level_values('NIS').unique():
                        data = self.df.loc[(date,NIS)]
                        self.new_df.loc[(date, NIS)] = self.convert_age_stratified_vaccination_data(data, age_classes, self.spatial, NIS).values
        else:
            if self.doses:
                for date in self.df.index.get_level_values('date').unique():
                    for dose in self.df.index.get_level_values('dose').unique():
                        data = self.df.loc[(date, slice(None), dose)]
                        self.new_df.loc[(date, slice(None), dose)] = self.convert_age_stratified_vaccination_data(data, age_classes).values
            else:
                for date in self.df.index.get_level_values('date').unique():
                    data = self.df.loc[(date)]
                    self.new_df.loc[(date)] = self.convert_age_stratified_vaccination_data(data, age_classes).values

        self.df = self.new_df

    def convert_age_stratified_vaccination_data(self, data, age_classes, spatial=None, NIS=None):
        """ 
        A function to convert the sciensano vaccination data to the desired model age groups

        Parameters
        ----------
        data: pd.Series
            A series of age-stratified vaccination incidences. Index must be of type pd.Intervalindex.
        
        age_classes : pd.IntervalIndex
            Desired age groups of the vaccination dataframe.

        spatial: str
            Spatial aggregation: prov, arr or mun
        
        NIS : str
            NIS code of consired spatial element

        Returns
        -------

        out: pd.Series
            Converted data.
        """

        # Pre-allocate new series
        out = pd.Series(index = age_classes, dtype=float)
        # Extract demographics
        if spatial: 
            data_n_individuals = construct_initN(data.index.get_level_values('age'), spatial).loc[NIS,:].values
            demographics = construct_initN(None, spatial).loc[NIS,:].values
        else:
            data_n_individuals = construct_initN(data.index.get_level_values('age'), spatial).values
            demographics = construct_initN(None, spatial).values
        # Loop over desired intervals
        for idx,interval in enumerate(age_classes):
            result = []
            for age in range(interval.left, interval.right):
                try:
                    result.append(demographics[age]/data_n_individuals[data.index.get_level_values('age').contains(age)]*data.iloc[np.where(data.index.get_level_values('age').contains(age))[0][0]])
                except:
                    result.append(0/data_n_individuals[data.index.get_level_values('age').contains(age)]*data.iloc[np.where(data.index.get_level_values('age').contains(age))[0][0]])
            out.iloc[idx] = sum(result)
        return out

    @lru_cache()
    def get_data(self,t):
        if self.spatial:
            if self.doses:
                raise ValueError(
                    "The combination of a spatially explicit, multidose vaccine model is not implemented!"
                )
            else:
                try:
                    return np.array(self.df.loc[t,:,:].values).reshape( (self.space_agg, self.age_agg) )
                except:
                    return np.zeros([self.space_agg, self.age_agg])
        else:
            if self.doses:
                try:
                    return np.array(self.df.loc[t,:,:].values).reshape( (self.age_agg, self.dose_agg) )
                except:
                    return np.zeros([self.age_agg, self.dose_agg])
            else:
                try:
                    return np.array(self.df.loc[t,:].values)
                except:
                    return np.zeros(self.age_agg)

    # Default vaccination strategy = Sciensano data + hypothetical scheme after end of data collection for unidose model only (for now)
    def __call__(self, t, states, param, initN, daily_first_dose=60000, delay_immunity = 21, vacc_order = [8,7,6,5,4,3,2,1,0], stop_idx=9, refusal = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]):
        """
        time-dependent function for the Belgian vaccination strategy
        First, all available first-dose data from Sciensano are used. Then, the user can specify a custom vaccination strategy of "daily_first_dose" first doses per day,
        administered in the order specified by the vector "vacc_order" with a refusal propensity of "refusal" in every age group.
        This vaccination strategy does not distinguish between vaccination doses, individuals are transferred to the vaccination circuit after some time delay after the first dose.
        For use with the model `COVID19_SEIRD` and `COVID19_SEIRD_spatial_vacc` in `~src/models/models.py`

        Parameters
        ----------
        t : int
            Simulation time
        states: dict
            Dictionary containing values of model states
        param : dict
            Model parameter dictionary
        initN : list or np.array
            Demographics according to the epidemiological model age bins
        daily_first_dose : int
            Number of doses administered per day. Default is 30000 doses/day.
        delay_immunity : int
            Time delay between first dose vaccination and start of immunity. Default is 21 days.
        vacc_order : array
            Vector containing vaccination prioritization preference. Default is old to young. Must be equal in length to the number of age bins in the model.
        stop_idx : float
            Index of age group at which the vaccination campaign is halted. An index of 9 corresponds to vaccinating all age groups, an index of 8 corresponds to not vaccinating the age group corresponding with vacc_order[idx].
        refusal: array
            Vector containing the fraction of individuals refusing a vaccine per age group. Default is 30% in every age group. Must be equal in length to the number of age bins in the model.

        Return
        ------
        N_vacc : np.array
            Number of individuals to be vaccinated at simulation time "t" per age, or per [patch,age]

        """

        # Convert time to suitable format
        t = pd.Timestamp(t.date())
        # Convert delay to a timedelta
        delay = pd.Timedelta(str(int(delay_immunity))+'D')
        # Compute the number of vaccine eligible individuals
        if not self.spatial:
            VE = states['S'] + states['R']

        if t <= self.df_end + delay:
            return self.get_data(t-delay)

        # Projection into the future
        else:
            if self.spatial:
                if not self.doses:
                    # No projection implemented
                    return np.zeros([self.space_agg,self.age_agg])
            else:
                if self.doses:
                    return np.zeros([self.age_agg,self.dose_agg])
                else:
                    N_vacc = np.zeros(self.age_agg)
                    idx = 0
                    while daily_first_dose > 0:
                        if idx == stop_idx:
                            daily_first_dose = 0 #End vaccination campaign at age 20
                        elif VE[vacc_order[idx]] - initN[vacc_order[idx]]*refusal[vacc_order[idx]] > daily_first_dose:
                            N_vacc[vacc_order[idx]] = daily_first_dose
                            daily_first_dose = 0
                        else:
                            N_vacc[vacc_order[idx]] = VE[vacc_order[idx]] - initN[vacc_order[idx]]*refusal[vacc_order[idx]]
                            daily_first_dose = daily_first_dose - (VE[vacc_order[idx]] - initN[vacc_order[idx]]*refusal[vacc_order[idx]])
                            idx = idx + 1
                    return N_vacc

############################
## Google policy function ##
############################

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

    def all_contact(self):
        return self.Nc_all['total']

    def all_contact_no_schools(self):
        return self.Nc_all['total'] - self.Nc_all['schools']

    def policies_WAVE1(self, t, states, param, l , prev_schools, prev_work, prev_rest, prev_home):
        '''
        Function that returns the time-dependant social contact matrix Nc for the first 2020 COVID-19 wave. Includes a manual tweaking of the 2020 COVID-19 Antwerp summer wave.

        Input
        -----
        t : Timestamp
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        l : float
            Compliance parameter for ramp_fun
        tau : float
            Compliance parameter for ramp_fun
        prev_{location} : float
            Effectivity of contacts at {location}

        Returns
        -------
        CM : np.array
            Effective contact matrix (output of __call__ function)
        '''
        all_contact = self.Nc_all['total']
        all_contact_no_schools = self.Nc_all['total'] - self.Nc_all['schools']

        # Convert tau and l to dates
        l_days = pd.Timedelta(l, unit='D')

        # Define additional dates where intensity or school policy changes
        t1 = pd.Timestamp('2020-03-15') # start of lockdown
        t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
        t3 = pd.Timestamp('2020-07-01') # start of summer holidays
        t4 = pd.Timestamp('2020-08-07') # peak of 'second wave' in antwerp
        t5 = pd.Timestamp('2020-09-01') # end of summer holidays

        if t <= t1:
            return all_contact
        elif t1 < t <= t1 + l_days:
            t = pd.Timestamp(t.date())
            policy_old = all_contact
            policy_new = self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest,
                                       school=0)
            return self.ramp_fun(policy_old, policy_new, t, t1, l)
        elif t1 + l_days < t <= t2:
            t = pd.Timestamp(t.date())
            return self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest,
                                  school=0)
        elif t2 < t <= t3:
            t = pd.Timestamp(t.date())
            return self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest,
                                  school=0)
        ## WARNING: During the summer of 2020, highly localized clusters appeared in Antwerp city, and lockdown measures were taken locally
        ## Do not forget this is a national-level model, you need a spatially explicit model to correctly model localized phenomena.
        ## The following is an ad-hoc tweak to assure a fit on the data during summer in order to be as accurate as possible with the seroprelevance
        elif t3 < t <= t3 + l_days:
            policy_old = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest, school=0)
            policy_new = self.__call__(t, prev_home, prev_schools, prev_work, 0.75, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t3, l)
        elif t3 + l_days < t <= t4:
            return self.__call__(t, prev_home, prev_schools, prev_work, 0.75, school=0)
        elif t4 < t <= t5:
            return self.__call__(t, prev_home, prev_schools, 0.05, 0.05,
                                school=0)
        else:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=1)

    def policies_WAVE2_full_relaxation(self, t, states, param, l , l_relax, prev_schools, prev_work, prev_rest, prev_home, relaxdate):
        '''
        Function that returns the time-dependant social contact matrix Nc for the second 2020 COVID-19 wave. Includes a full relaxation of measures on relaxdate.

        Input
        -----
        t : Timestamp
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        l : float
            Compliance parameter for ramp_fun
        l_relax : float
            Additional days added after relaxdate
        tau : float
            Compliance parameter for ramp_fun
        prev_{location} : float
            Effectivity of contacts at {location}
        relaxdate : str
            String containing a date (YYYY-MM-DD) on which all measures are relaxed

        Returns
        -------
        CM : np.array (9x9)
            Effective contact matrix (output of __call__ function)
        '''
        t = pd.Timestamp(t.date())

        # Convert compliance tau and l to dates
        l_days = pd.Timedelta(l, unit='D')

        # Convert relaxation l to dates
        l_relax_days = pd.Timedelta(l_relax, unit='D')

        # Define key dates of first wave
        t1 = pd.Timestamp('2020-03-15') # start of lockdown
        t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
        t3 = pd.Timestamp('2020-07-01') # start of summer holidays
        t4 = pd.Timestamp('2020-09-01') # end of summer holidays

        # Define key dates of second wave
        # Note that for days in the future, policies are estimated rather than observed
        t5 = pd.Timestamp('2020-10-19') # lockdown (1)
        t6 = pd.Timestamp('2020-11-02') # lockdown (2)
        t7 = pd.Timestamp('2020-11-16') # schools re-open
        t8 = pd.Timestamp('2020-12-18') # Christmas holiday starts
        t9 = pd.Timestamp('2021-01-04') # Christmas holiday ends
        t10 = pd.Timestamp('2021-02-15') # Spring break starts
        t11 = pd.Timestamp('2021-02-21') # Spring break ends
        t12 = pd.Timestamp('2021-02-28') # Contact increase in children
        t13 = pd.Timestamp('2021-03-26') # Start of Easter holiday
        t14 = pd.Timestamp('2021-04-18') # End of Easter holiday
        t15 = pd.Timestamp(relaxdate) # Relaxation date
        t16 = pd.Timestamp('2021-07-01') # Start of Summer holiday
        t17 = pd.Timestamp('2021-09-01')

        # First wave
        if t <= t4:
            return self.policies_WAVE1(t, states, param, l , prev_schools, prev_work, prev_rest, prev_home)
        
        # Second wave
        elif t4 < t <= t5:
            return self.__call__(t, school=1)
        elif t5  < t <= t5 + l_days:
            policy_old = self.__call__(t, school=1)
            policy_new = self.__call__(t, prev_schools, prev_work, prev_rest,
                                        school=1)
            return self.ramp_fun(policy_old, policy_new, t, t5, l)
        elif t5 + l_days < t <= t6:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=1)
        elif t6 < t <= t7:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=0)
        elif t7 < t <= t8:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=1)
        elif t8 < t <= t9:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=0)
        elif t9 < t <= t10:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=1)
        elif t10 < t <= t11:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=0)
        elif t11 < t <= t12:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=1)
        elif t12 < t <= t13:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=1)
        elif t13 < t <= t14:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                    school=0)
        elif t14 < t <= t15:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=1)
        elif t15 < t <= t15 + l_relax_days:
            policy_old = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                school=1)
            policy_new = self.__call__(t, prev_schools, prev_work, prev_rest,
                                work=1, leisure=1, transport=1, others=1, school=1)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t16 < t <= t17:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                work=0.8, leisure=1, transport=0.90, others=1, school=0)
        else: # full relaxation
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest,
                                work=1, leisure=1, transport=1, others=1, school=1)

    def policies_WAVE2_no_relaxation(self, t, states, param, l , prev_schools, prev_work, prev_rest_lockdown, prev_rest_relaxation, prev_home):
        '''
        Function that returns the time-dependant social contact matrix Nc for the second 2020 COVID-19 wave. Includes a full relaxation of measures on relaxdate.

        Input
        -----
        t : Timestamp
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        l : float
            Compliance parameter for ramp_fun
        tau : float
            Compliance parameter for ramp_fun
        prev_{location} : float
            Effectivity of contacts at {location}

        Returns
        -------
        CM : np.array (9x9)
            Effective contact matrix (output of __call__ function)
        '''
        t = pd.Timestamp(t.date())

        # Convert compliance tau and l to dates
        l_days = pd.Timedelta(l, unit='D')

        # Define key dates of first wave
        t1 = pd.Timestamp('2020-03-15') # start of lockdown
        t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
        t3 = pd.Timestamp('2020-07-01') # start of summer holidays
        t4 = pd.Timestamp('2020-09-01') # end of summer holidays

        # Define key dates of second wave
        t5 = pd.Timestamp('2020-10-19') # lockdown (1)
        t6 = pd.Timestamp('2020-11-02') # lockdown (2)
        t7 = pd.Timestamp('2020-11-16') # schools re-open
        t8 = pd.Timestamp('2020-12-18') # Christmas holiday starts
        t9 = pd.Timestamp('2021-01-04') # Christmas holiday ends
        t10 = pd.Timestamp('2021-02-15') # Spring break starts
        t11 = pd.Timestamp('2021-02-21') # Spring break ends
        t12 = pd.Timestamp('2021-02-28') # Contact increase in children
        t13 = pd.Timestamp('2021-03-26') # Start of Easter holiday
        t14 = pd.Timestamp('2021-04-18') # End of Easter holiday
        t15 = pd.Timestamp('2021-07-01') # Start of Summer holiday
        t16 = pd.Timestamp('2021-08-01') # End of gradual introduction mentality change
        t17 = pd.Timestamp('2021-09-01') # End of Summer holiday
        t18 = pd.Timestamp('2021-11-01') # Start of autumn break
        t19 = pd.Timestamp('2021-11-07') # Start of autumn break
        t20 = pd.Timestamp('2021-12-26') # Start of Christmass break
        t21 = pd.Timestamp('2022-01-06') # End of Christmass break
        t22 = pd.Timestamp('2022-02-28') # Start of Spring Break
        t23 = pd.Timestamp('2022-03-06') # End of Spring Break
        t24 = pd.Timestamp('2022-04-04') # Start of Easter Break
        t25 = pd.Timestamp('2022-04-17') # End of Easter Break

        # First wave
        if t <= t4:
            return self.policies_WAVE1(t, states, param, l , prev_schools, prev_work, prev_rest_lockdown, prev_home)
        
        # Second wave
        if t4 < t <= t5:
            return self.__call__(t, school=1)
        elif t5  < t <= t5 + l_days:
            policy_old = self.__call__(t, school=1)
            policy_new = self.__call__(t, prev_schools, prev_work, prev_rest_lockdown,
                                        school=1)
            return self.ramp_fun(policy_old, policy_new, t, t5, l)
        elif t5 + l_days < t <= t6:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=1)
        elif t6 < t <= t7:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=0)
        elif t7 < t <= t8:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=1)
        elif t8 < t <= t9:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=0)
        elif t9 < t <= t10:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=1)
        elif t10 < t <= t11:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=0)
        elif t11 < t <= t12:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=1)
        elif t12 < t <= t13:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=1)
        elif t13 < t <= t14:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                    school=0)
        elif t14 < t <= t15:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=1)
        elif t15 < t <= t16:
            l = (t16 - t15)/pd.Timedelta(days=1)
            policy_old = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, school=0)
            policy_new = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t15, l)
        elif t16 < t <= t17:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=0)
        elif t17 < t <= t18:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t18 < t <= t19:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                leisure=1.1, work=0.9, transport=1, others=1, school=0)
        elif t19 < t <= t20:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t20 < t <= t21:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                work=0.7, leisure=1.3, transport=1, others=1, school=0)
        elif t21 < t <= t22:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t22 < t <= t23:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                leisure=1.1, work=0.9, transport=1, others=1, school=0)
        elif t23 < t <= t24:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t24 < t <= t25:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                work=0.7, leisure=1.3, transport=1, others=1, school=0)
        else: # full relaxation
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                work=1, leisure=1, transport=1, others=1, school=1)

    def ramp_fun(self, Nc_old, Nc_new, t, t_start, l):
        """
        t : timestamp
            current simulation time
        t_start : timestamp
            start of policy change
        l : int
            number of additional days after the time delay until full compliance is reached
        """
        return Nc_old + (Nc_new-Nc_old)/l * (t-t_start)/pd.Timedelta('1D')

    def delayed_ramp_fun(self, Nc_old, Nc_new, t, tau_days, l, t_start):
        """
        t : timestamp
            current simulation time
        t_start : timestamp
            start of policy  change
        tau : int
            number of days before measures start having an effect
        l : int
            number of additional days after the time delay until full compliance is reached
        """
        return Nc_old + (Nc_new-Nc_old)/l * (t-t_start-tau_days)/pd.Timedelta('1D')
    
    def policies_all(self, t, states, param, l1, l2, prev_schools, prev_work, prev_rest_lockdown, prev_rest_relaxation, prev_home):
        '''
        Function that returns the time-dependant social contact matrix Nc for the second 2020 COVID-19 wave. Includes a full relaxation of measures on relaxdate.
        
        Input
        -----
        t : Timestamp
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        l1 : float
            Compliance parameter for social policies during first lockdown 2020 COVID-19 wave
        l2 : float
            Compliance parameter for social policies during second lockdown 2020 COVID-19 wave        
        prev_{location} : float
            Effectivity of contacts at {location}

        Returns
        -------
        CM : np.array (9x9)
            Effective contact matrix (output of __call__ function)
        '''
        t = pd.Timestamp(t.date())

        # Convert compliance l to dates
        l1_days = pd.Timedelta(l1, unit='D')
        l2_days = pd.Timedelta(l2, unit='D')

        # Define key dates of first wave
        t1 = pd.Timestamp('2020-03-15') # start of lockdown
        t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
        t3 = pd.Timestamp('2020-07-01') # start of summer holidays
        t4 = pd.Timestamp('2020-08-10') # Summer lockdown in Antwerp
        t5 = pd.Timestamp('2020-08-24') # End of summer lockdown in Antwerp
        t6 = pd.Timestamp('2020-09-01') # end of summer holidays
        # Define key dates of second wave
        t7 = pd.Timestamp('2020-10-19') # lockdown (1)
        t8 = pd.Timestamp('2020-11-02') # lockdown (2)
        t9 = pd.Timestamp('2020-11-16') # schools re-open
        t10 = pd.Timestamp('2020-12-18') # Christmas holiday starts
        t11 = pd.Timestamp('2021-01-04') # Christmas holiday ends
        t12 = pd.Timestamp('2021-02-15') # Spring break starts
        t13 = pd.Timestamp('2021-02-21') # Spring break ends
        t14 = pd.Timestamp('2021-02-28') # Contact increase in children
        t15 = pd.Timestamp('2021-03-26') # Start of Easter holiday
        t16 = pd.Timestamp('2021-04-18') # End of Easter holiday
        t17 = pd.Timestamp('2021-07-01') # Start of Summer holiday
        t18 = pd.Timestamp('2021-08-01') # End of gradual introduction mentality change
        t19 = pd.Timestamp('2021-09-01') # End of Summer holiday
        t20 = pd.Timestamp('2021-11-01') # Start of autumn break
        t21 = pd.Timestamp('2021-11-07') # Start of autumn break
        t22 = pd.Timestamp('2021-12-26') # Start of Christmass break
        t23 = pd.Timestamp('2022-01-06') # End of Christmass break
        t24 = pd.Timestamp('2022-02-28') # Start of Spring Break
        t25 = pd.Timestamp('2022-03-06') # End of Spring Break
        t26 = pd.Timestamp('2022-04-04') # Start of Easter Break
        t27 = pd.Timestamp('2022-04-17') # End of Easter Break

        if t <= t1:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=1) 
        elif t1 < t <= t1 + l1_days:
            t = pd.Timestamp(t.date())
            policy_old = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=1) 
            policy_new = self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest_lockdown, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t1, l1)
        elif t1 + l1_days < t <= t2:
            return self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest_lockdown, school=0)
        elif t2 < t <= t3:
            return self.__call__(t, prev_home=prev_home, prev_schools=prev_schools, prev_work=prev_work, prev_rest=prev_rest_lockdown, school=0)                  
        elif t3 < t <= t4:
            l = (t4 - t3)/pd.Timedelta(days=1)
            policy_old = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, school=0)
            policy_new = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t3, l)
        elif t4 < t <= t5:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, school=0)                                          
        elif t5 < t <= t6:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=0)      
        # Second wave
        if t6 < t <= t7:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=1)  
        elif t7  < t <= t7 + l2_days:
            policy_old = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=1)
            policy_new = self.__call__(t, prev_schools, prev_work, prev_rest_lockdown, school=1)
            return self.ramp_fun(policy_old, policy_new, t, t7, l2)
        elif t7 + l2_days < t <= t8:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, 
                                school=1)
        elif t8 < t <= t9:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, 
                                school=0)
        elif t9 < t <= t10:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, 
                                school=1) 
        elif t10 < t <= t11:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, 
                                school=0)
        elif t11 < t <= t12:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, 
                                school=1)
        elif t12 < t <= t13:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, 
                                school=0)    
        elif t13 < t <= t14:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, 
                                school=1)
        elif t14 < t <= t15:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown,
                                school=1)
        elif t15 < t <= t16:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, 
                                    school=0)                           
        elif t16 < t <= t17:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, 
                                school=1)
        elif t17 < t <= t18:
            l = (t18 - t17)/pd.Timedelta(days=1)
            policy_old = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_lockdown, school=0)
            policy_new = self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t17, l)
        elif t18 < t <= t19:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, school=0)
        elif t19 < t <= t20:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, 
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t20 < t <= t21:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation,
                                leisure=1.1, work=0.9, transport=1, others=1, school=0)
        elif t21 < t <= t22:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, 
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t22 < t <= t23:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, 
                                work=0.7, leisure=1.3, transport=1, others=1, school=0) 
        elif t23 < t <= t24:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, 
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t24 < t <= t25:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, 
                                leisure=1.1, work=0.9, transport=1, others=1, school=0)  
        elif t25 < t <= t26:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, 
                                work=1, leisure=1, transport=1, others=1, school=1)           
        elif t26 < t <= t27:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, 
                                work=0.7, leisure=1.3, transport=1, others=1, school=0)                                                                                                   
        else:
            return self.__call__(t, prev_home, prev_schools, prev_work, prev_rest_relaxation, 
                                work=1, leisure=1, transport=1, others=1, school=1)    


##########################
## Seasonality function ##
##########################

class make_seasonality_function():
    """
    Simple class to create a function that controls the season-dependent value of the transmission coefficients. Currently not based on any data, but e.g. weather patterns could be imported if needed.
    """
    def __call__(self, t, states, param, amplitude, peak_shift):
        """
        Default output function. Returns a sinusoid with average value 1.
        
        t : Timestamp
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        amplitude : float
            maximum deviation of output with respect to the average (1)
        peak_shift : float
            phase. Number of days after January 1st after which the maximum value of the seasonality rescaling is reached 
        """
        ref_date = pd.to_datetime('2021-01-01')
        # If peak_shift = 0, the max is on the first of January
        maxdate = pd.Timedelta(days=peak_shift) + ref_date
        # One period is one year long (seasonality)
        t = (t - pd.to_datetime(maxdate))/pd.Timedelta(days=1)/365
        rescaling = 1 + amplitude*np.cos( 2*np.pi*(t))
        return param*rescaling

