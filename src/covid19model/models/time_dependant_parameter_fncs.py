import os
import numpy as np
import pandas as pd
import warnings
from functools import lru_cache
from covid19model.visualization.output import school_vacations_dict

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
    def __init__(self, proximus_mobility_data):
        self.proximus_mobility_data = proximus_mobility_data.sort_index()

    @lru_cache()
    # Define mobility_update_func
    def __call__(self, t):
        """
        time-dependent function which has a mobility matrix of type dtype for every date.
        Note: only works with datetime input (no integer time steps). This

        Input
        -----
        t : timestamp
            current date as datetime object

        Returns
        -------
        place : np.array
            square matrix with mobility of type dtype (fractional, staytime or visits), dimension depending on agg
        """
        t = pd.Timestamp(t.date())
        try: # simplest case: there is data available for this date (the key exists)
            place = self.proximus_mobility_data['place'][t]
        except:
            if t < pd.Timestamp(2020, 2, 10):
                # prepandemic default mobility. Take average of first 20-ish days
                if t.dayofweek < 5:
                    # business days
                    place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek < 5]['place'][:pd.Timestamp(2020, 3, 1)].mean()
                elif t.dayofweek >= 5:
                    # weekend
                    place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek >= 5]['place'][:pd.Timestamp(2020, 3, 1)].mean()
            elif t == pd.Timestamp(2020, 2, 21):
                # first missing date in Proximus data. Just take average
                place = (self.proximus_mobility_data['place'][pd.Timestamp(2020, 2, 20)] + 
                          self.proximus_mobility_data['place'][pd.Timestamp(2020, 2, 22)])/2
            elif t == pd.Timestamp(2020, 12, 18):
                # second missing date in Proximus data. Just take average
                place = (self.proximus_mobility_data['place'][pd.Timestamp(2020, 12, 17)] + 
                          self.proximus_mobility_data['place'][pd.Timestamp(2020, 12, 19)])/2
            elif t > pd.Timestamp(2021, 8, 31):
                # beyond Proximus service. Make a distinction between holiday/non-holiday and weekend/business day
                holiday = False
                for first, duration in school_vacations_dict().items():
                    if (t >= first) and (t < (first + pd.Timedelta(days=duration))):
                        holiday = True
                        # it's a holiday. Take average of summer vacation behaviour
                        if t.dayofweek < 5:
                            # non-weekend holiday
                            place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek < 5]['place'][pd.Timestamp(2021, 7, 1):pd.Timestamp(2021, 8, 31)].mean()
                        elif t.dayofweek >= 5:
                            # weekend holiday
                            place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek >= 5]['place'][pd.Timestamp(2021, 7, 1):pd.Timestamp(2021, 8, 31)].mean()
                if not holiday:
                    # it's not a holiday. Take average of two months before summer vacation
                    if t.dayofweek < 5:
                        # business day
                        place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek < 5]['place'][pd.Timestamp(2021, 6, 1):pd.Timestamp(2021, 6, 30)].mean()
                    elif t.dayofweek >= 5:
                        # regular weekend
                        place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek >= 5]['place'][pd.Timestamp(2021, 6, 1):pd.Timestamp(2021, 6, 30)].mean()
                
        return place

    def mobility_wrapper_func(self, t, states, param):
        t = pd.Timestamp(t.date())
        return self.__call__(t)

###################
## VOC functions ##
###################

class make_VOC_function():
    """
    Class that returns a time-dependant parameter function for COVID-19 SEIQRD model parameter alpha (variant fraction and derivative).
    The model parameter alpha currently consists of 2 rows and 4 columns.
    The first row contains the VOC fractions, the second row contains the derivatives of the fractions. The derivates are needed to model immune escape.
    The columns denote the variants: 0: Wild-Type, 1: Alpha,Beta,Gamma, 2: Delta, 3: Omicron.
    Logistic parameters were manually fitted to the VOC effalence data and are hardcoded in the init function of this module

    Output
    ------

    __class__ : function
        Default variant function.


    """
    def __init__(self, VOC_logistic_growth_parameters):
        self.logistic_parameters=VOC_logistic_growth_parameters
    
    @staticmethod
    def logistic_growth(t,t_sig,k):
        return 1/(1+np.exp(-k*(t-t_sig)/pd.Timedelta(days=1)))

    # Default VOC function includes abc, delta and omicron variants
    def __call__(self, t, states, param):
        # Convert time to timestamp
        t = pd.Timestamp(t.date())
        # Pre-allocate alpha
        alpha = np.zeros([2, len(self.logistic_parameters.index)])
        # Before introduction of first variant, return all zeros
        if t <= min(pd.to_datetime(self.logistic_parameters['t_introduction'].values)):
            return alpha
        else:
            # Retrieve correct index of variant that is currently "growing"
            try:
                idx = [index for index,value in enumerate(self.logistic_parameters['t_introduction'].values) if pd.Timestamp(value) >= t][0]-1
            except:
                idx = len(self.logistic_parameters['t_introduction'].values) - 1
            # Perform computation of currently growing VOC fraction and its derivative
            f = self.logistic_growth(t, pd.Timestamp(self.logistic_parameters.iloc[idx]['t_sigmoid']), self.logistic_parameters.iloc[idx]['k'])
            df = self.logistic_parameters.iloc[idx]['k']*f*(1-f)
            # Decision logic
            if idx == 0:
                alpha[:,idx] = [f,df]
            else:
                alpha[:,idx-1] = [1-f,-df]
                alpha[:,idx] = [f,df]
            return alpha            

###########################
## Vaccination functions ##
###########################

from covid19model.data.utils import construct_initN, convert_age_stratified_property

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
    def __init__(self, df, age_classes=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')):
        age_stratification_size = len(age_classes)
        # Assign inputs to object
        self.df = df
        self.age_agg = age_stratification_size
        
        # Check if spatial data is provided
        self.spatial = None
        if 'NIS' in self.df.index.names:
            self.spatial = True
            self.space_agg = len(self.df.index.get_level_values('NIS').unique().values)
            # infer aggregation (prov, arr or mun)
            if self.space_agg == 11:
                self.agg = 'prov'
            elif self.space_agg == 43:
                self.agg = 'arr'
            elif self.space_agg == 581:
                self.agg = 'mun'
            else:
                raise Exception(f"Space is {G}-fold stratified. This is not recognized as being stratification at Belgian province, arrondissement, or municipality level.")

        # Check if dose data is provided
        self.doses = None
        if 'dose' in self.df.index.names:
            self.doses = True
            self.dose_agg = len(self.df.index.get_level_values('dose').unique().values)

        # Define start- and enddate
        self.df_start = pd.Timestamp(self.df.index.get_level_values('date').min())
        self.df_end = pd.Timestamp(self.df.index.get_level_values('date').max())

        # Perform age conversion
        # Define dataframe with desired format
        iterables=[]
        for index_name in self.df.index.names:
            if index_name != 'age':
                iterables += [self.df.index.get_level_values(index_name).unique()]
            else:
                iterables += [age_classes]
        index = pd.MultiIndex.from_product(iterables, names=self.df.index.names)
        self.new_df = pd.Series(index=index, dtype=float)

        # Four possibilities exist: can this be sped up?
        if self.spatial:
            if self.doses:
                # Shorten?
                for date in self.df.index.get_level_values('date').unique():
                    for NIS in self.df.index.get_level_values('NIS').unique():
                        for dose in self.df.index.get_level_values('dose').unique():
                            data = self.df.loc[(date, NIS, slice(None), dose)]
                            self.new_df.loc[(date, NIS, slice(None), dose)] = self.convert_age_stratified_vaccination_data(data, age_classes, self.agg, NIS).values
            else:
                for date in self.df.index.get_level_values('date').unique():
                    for NIS in self.df.index.get_level_values('NIS').unique():
                        data = self.df.loc[(date,NIS)]
                        self.new_df.loc[(date, NIS)] = self.convert_age_stratified_vaccination_data(data, age_classes, self.agg, NIS).values
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

    def convert_age_stratified_vaccination_data(self, data, age_classes, agg=None, NIS=None):
        """ 
        A function to convert the sciensano vaccination data to the desired model age groups

        Parameters
        ----------
        data: pd.Series
            A series of age-stratified vaccination incidences. Index must be of type pd.Intervalindex.
        
        age_classes : pd.IntervalIndex
            Desired age groups of the vaccination dataframe.

        agg: str
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
        if agg: 
            data_n_individuals = construct_initN(data.index.get_level_values('age'), agg).loc[NIS,:].values
            demographics = construct_initN(None, agg).loc[NIS,:].values
        else:
            data_n_individuals = construct_initN(data.index.get_level_values('age'), agg).values
            demographics = construct_initN(None, agg).values
        # Loop over desired intervals
        for idx,interval in enumerate(age_classes):
            result = []
            for age in range(interval.left, interval.right):
                try:
                    result.append(demographics[age]/data_n_individuals[data.index.get_level_values('age').contains(age)]*data.iloc[np.where(data.index.get_level_values('age').contains(age))[0][0]])
                except:
                    result.append(0)
            out.iloc[idx] = sum(result)
        return out

    @lru_cache()
    def get_data(self,t):
        if self.spatial:
            if self.doses:
                try:
                    return np.array(self.df.loc[t,:,:,:].values).reshape( (self.space_agg, self.age_agg, self.dose_agg) )
                except:
                    return np.zeros([self.space_agg, self.age_agg, self.dose_agg])
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

    def unidose_2021_vaccination_campaign(self, states, initN, daily_doses, delay_immunity, vacc_order, stop_idx, refusal):
        # Compute the number of vaccine eligible individuals
        VE = states['S'] + states['R']
        # Initialize N_vacc
        N_vacc = np.zeros(self.age_agg)
        # Start vaccination loop
        idx = 0
        while daily_doses > 0:
            if idx == stop_idx:
                daily_doses = 0 #End vaccination campaign at age 20
            elif VE[vacc_order[idx]] - initN[vacc_order[idx]]*refusal[vacc_order[idx]] > daily_doses:
                N_vacc[vacc_order[idx]] = daily_doses
                daily_doses = 0
            else:
                N_vacc[vacc_order[idx]] = VE[vacc_order[idx]] - initN[vacc_order[idx]]*refusal[vacc_order[idx]]
                daily_doses = daily_doses - (VE[vacc_order[idx]] - initN[vacc_order[idx]]*refusal[vacc_order[idx]])
                idx = idx + 1
        return N_vacc

    def booster_campaign(self, states, daily_doses, vacc_order, stop_idx, refusal):

        # Compute the number of booster eligible individuals
        VE = states['S'][:,2] + states['E'][:,2] + states['I'][:,2] + states['A'][:,2] + states['R'][:,2] \
                + states['S'][:,3] + states['E'][:,3] + states['I'][:,3] + states['A'][:,3] + states['R'][:,3]
        # Initialize N_vacc
        N_vacc = np.zeros([self.age_agg,self.dose_agg])
        # Booster vaccination strategy without refusal
        idx = 0
        while daily_doses > 0:
            if idx == stop_idx:
                daily_doses= 0 #End vaccination campaign at age 20
            elif VE[vacc_order[idx]] - self.fully_vaccinated_0[vacc_order[idx]]*refusal[vacc_order[idx]] > daily_doses:
                N_vacc[vacc_order[idx],3] = daily_doses
                daily_doses= 0
            else:
                if VE[vacc_order[idx]] - self.fully_vaccinated_0[vacc_order[idx]]*refusal[vacc_order[idx]] >= 0:
                    N_vacc[vacc_order[idx],3] = VE[vacc_order[idx]] - self.fully_vaccinated_0[vacc_order[idx]]*refusal[vacc_order[idx]]
                    daily_doses = daily_doses - (VE[vacc_order[idx]] - self.fully_vaccinated_0[vacc_order[idx]]*refusal[vacc_order[idx]])
                else:
                    N_vacc[vacc_order[idx],3] = 0
                idx = idx + 1
        return N_vacc

    # Default vaccination strategy = Sciensano data + hypothetical scheme after end of data collection for unidose model only (for now)
    def __call__(self, t, states, param, initN, daily_doses=60000, delay_immunity = 21, vacc_order = [8,7,6,5,4,3,2,1,0], stop_idx=9, refusal = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]):
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
        # Compute vaccinated individuals after spring-summer 2021 vaccination campaign
        check_time = pd.Timestamp('2021-10-01')
        # Only for non-spatial multi-vaccindation dose model
        if not self.spatial:
            if self.doses:
                if t == check_time:
                    self.fully_vaccinated_0 = states['S'][:,2] + states['E'][:,2] + states['I'][:,2] + states['A'][:,2] + states['R'][:,2] + \
                                                states['S'][:,3] + states['E'][:,3] + states['I'][:,3] + states['A'][:,3] + states['R'][:,3]
        # Use data
        if t <= self.df_end + delay:
            return self.get_data(t-delay)
        # Projection into the future
        else:
            if self.spatial:
                if self.doses:
                    # No projection implemented
                    return np.zeros([self.space_agg, self.age_agg, self.dose_agg])
                else:
                    # No projection implemented
                    return np.zeros([self.space_agg,self.age_agg])
            else:
                if self.doses:
                    return self.booster_campaign(states, daily_doses, vacc_order, stop_idx, refusal)
                else:
                    return self.unidose_2021_vaccination_campaign(states, initN, daily_doses, delay_immunity, vacc_order, stop_idx, refusal)

                
###########################################
## Rescaling function due to vaccination ##
###########################################

class make_vaccination_rescaling_function():
    """
    Class that returns rescaling parameters time series E_susc, E_inf and E_hosp per province and age (shape = (G,N)), determined by vaccination
    
    Note: dimensions G (provinces) and N (ages) are hard-coded to (G,N)=(11,10)
    
    Input
    -----
    rescaling_df : pd.DataFrame
        Pandas DataFrame containing all vaccination-induced rescaling values per age (and per province). Output from sciensano.get_vaccination_rescaling_values(). Spatial aggregation depends on this input.
    
    Output
    ------
    __class__ : default function
        Default output function with arguments t (datetime object), which returns spatially stratified parameters E_susc, E_inf, E_hosp
    
    Example use
    -----------
    rescaling_df = sciensano.get_vaccination_rescaling_values(spatial=True)
    E_susc_function = make_vaccination_rescaling_function(rescaling_df).E_susc
    E_inf_function = make_vaccination_rescaling_function(rescaling_df).E_inf
    E_hosp_function = make_vaccination_rescaling_function(rescaling_df).E_hosp
    E_susc_function(pd.Timestamp(2021, 10, 1), 0, 0)
    
    """

    # TODO: hypothetical vaccination schemes
    # TODO: untangle age group 0-18 into 0-12 and 12-18
    
    def __init__(self, update=False, agg=None,
                    age_classes=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'),
                    df_incidences=None, VOC_function=None, vaccine_params=None):

        if update==False:
            # Simply load data
            if agg:
                dir_abs = os.path.join(os.path.dirname(__file__), f"../../../data/interim/sciensano/vacc_rescaling_values_{agg}.pkl")
                df_efficacies = pd.read_pickle(dir_abs).groupby(['date', 'NIS', 'age']).first()
            else:
                dir_abs = os.path.join(os.path.dirname(__file__), "../../../data/interim/sciensano/vacc_rescaling_values_national.pkl")
                df_efficacies = pd.read_pickle(dir_abs).groupby(['date', 'age']).first()
        else:
            # Warn user this may take some time
            warnings.warn("The vaccination rescaling parameters must be updated because a change was made to the desired VOCs or vaccination parameters, this may take some time.", stacklevel=2)
            # Compute population size-normalized relative incidences
            df_incidences = self.compute_relative_incidences(df_incidences, agg)
            # Elongate the dataframe of incidences with 26 weeks in which no vaccines are administered
            df_incidences = self.extend_df_incidences(df_incidences, n_weeks=26)
            # Attach cumulative figures
            df = self.compute_relative_cumulative(df_incidences)
            # Format vaccination parameters 
            vaccine_params, onset, waning = self.format_vaccine_params(vaccine_params)
            # Compute efficacies accouting for onset immunity of vaccines, waning immunity of vaccines and VOCs
            df_efficacies = self.compute_efficacies(df, vaccine_params, VOC_function, onset, waning)
            # Save result
            dir = os.path.join(os.path.dirname(__file__), "../../../data/interim/sciensano/")
            if agg:
                df_efficacies.to_csv(os.path.join(dir, f'vacc_rescaling_values_{agg}.csv'))
                df_efficacies.to_pickle(os.path.join(dir, f'vacc_rescaling_values_{agg}.pkl'))
            else:
                df_efficacies.to_csv(os.path.join(dir, f'vacc_rescaling_values_national.csv'))
                df_efficacies.to_pickle(os.path.join(dir, f'vacc_rescaling_values_national.pkl'))

        # Check if an age conversion is necessary
        age_conv=False
        if len(age_classes) != len(df_efficacies.index.get_level_values('age').unique()):
            df_efficacies = self.age_conversion(df_efficacies, age_classes, agg)
        elif (age_classes != df_efficacies.index.get_level_values('age').unique()).any():
            df_efficacies = self.age_conversion(df_efficacies, age_classes, agg)

        # Visualization (prov)
        #age_group = df_efficacies.index.get_level_values('age').unique()[-1]
        #dates = df_efficacies.index.get_level_values('date').unique()
        #df_efficacies = df_efficacies.groupby(by=['date','NIS']).mean()
        #import matplotlib.pyplot as plt
        #fig,ax=plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(14,12))
        #for NIS in df_efficacies.index.get_level_values('NIS').unique():
        #    ax[0].plot(dates, 1-df_efficacies.loc[(slice(None), NIS),'e_i'])
        #    ax[1].plot(dates, 1-df_efficacies.loc[(slice(None), NIS),'e_s'])
        #    ax[2].plot(dates, 1-df_efficacies.loc[(slice(None), NIS),'e_h'])
        #ax[0].legend(['10000', '20001', '20002', '21000', '30000', '40000', '50000', '60000', '70000', '80000', '90000'], bbox_to_anchor =(1.01, 1.00))
        #ax[0].set_ylim([0,1])
        #ax[1].set_ylim([0,1])
        #ax[2].set_ylim([0,1])
        #ax[0].grid(False)
        #ax[1].grid(False)
        #ax[2].grid(False)
        #ax[0].set_title('e_i')
        #ax[1].set_title('e_s')
        #ax[2].set_title('e_h')
        #from covid19model.visualization.output import _apply_tick_locator 
        #ax[0] = _apply_tick_locator(ax[0])
        #plt.tight_layout()
        #plt.show()

        # Assign efficacy dataset
        self.df_efficacies = df_efficacies

        # Compute some other relevant attributes
        self.available_dates = df_efficacies.index.get_level_values('date').unique()
        self.N = len(age_classes)
        self.agg = agg
        try:
            self.G = len(df_efficacies.index.get_level_values('NIS').unique())
        except:
            self.G = 0

    @lru_cache() # once the function is run for a set of parameters, it doesn't need to compile again
    def __call__(self, t, efficacy):
        """
        Returns rescaling value matrix [G,N] for the requested time t.

        Input
        -----
        t : pd.Timestamp
            Time at which you want to know the rescaling parameter matrix
        rescaling_type : str
            Either 'susc', 'inf' or 'hosp'
            
        Output
        ------
        E : np.array
            Matrix of dimensions (G,N): element E[g,i] is the rescaling factor belonging to province g and age class i at time t
        """

        if efficacy not in self.df_efficacies.columns:
            raise ValueError(
                "valid vaccine efficacies are 'e_s', 'e_i', or 'e_h'.")

        t = pd.Timestamp(t)

        if t < min(self.available_dates):
            # Take unity matrix
            if self.agg:
                E = np.zeros([self.G,self.N])
            else:
                E = np.zeros(self.N)
        
        elif t <= max(self.available_dates):
            # Take interpolation between dates for which data is available
            t_data_first = pd.Timestamp(self.available_dates[np.argmax(self.available_dates >=t)-1])
            t_data_second = pd.Timestamp(self.available_dates[np.argmax(self.available_dates >=t)])
            if self.agg:
                E_values_first = self.df_efficacies.loc[t_data_first, :, :][efficacy].to_numpy()
                E_first = np.reshape(E_values_first, (self.G,self.N))
                E_values_second = self.df_efficacies.loc[t_data_second, :, :][efficacy].to_numpy()
                E_second = np.reshape(E_values_second, (self.G,self.N))
            else:
                E_first = self.rescaling_df.loc[t_data_first, :][efficacy].to_numpy()
                E_second = self.rescaling_df.loc[t_data_second, :][efficacy].to_numpy()
            # linear interpolation
            E = E_first + (E_second - E_first) * (t - t_data_first).total_seconds() / (t_data_second - t_data_first).total_seconds()
            
        elif t > max(self.available_dates):
            # Take last available data point
            t_data = pd.Timestamp(max(self.available_dates))
            if self.agg:
                E_values = self.rescaling_df.loc[t_data, :, :][efficacy].to_numpy()
                E = np.reshape(E_values, (self.G,self.N))
            else:
                E = self.rescaling_df.loc[t_data, :][efficacy].to_numpy()

        return (1-E)
    
    def E_susc(self, t, states, param):
        return self.__call__(t, 'e_s')
    
    def E_inf(self, t, states, param):
        return self.__call__(t, 'e_i')
    
    def E_hosp(self, t, states, param):
        return self.__call__(t, 'e_h')

    @staticmethod
    def compute_relative_incidences(df, agg=None):

        # Compute fractions with dose x using relevant population size
        if agg:
            initN = construct_initN(df.index.get_level_values('age').unique(), agg)
            for age in df.index.get_level_values('age').unique():
                for NIS in df.index.get_level_values('NIS').unique():
                    df.loc[(slice(None), NIS, age, slice(None)),('REL_INCIDENCE')] = df.loc[slice(None), NIS, age, slice(None)]['INCIDENCE'].values / initN.loc[NIS, age]
        else:
            initN = construct_initN(df.index.get_level_values('age').unique())
            for age in df.index.get_level_values('age').unique():
                df.loc[(slice(None), age, slice(None)),('REL_INCIDENCE')] = df.loc[slice(None), age, slice(None)]['INCIDENCE'].values / initN.loc[age]

        # Use more declaritive names for doses
        df.rename(index={'A':'first', 'B':'second', 'C':'one_shot', 'E': 'booster'}, inplace=True)

        return df['REL_INCIDENCE']

    @staticmethod
    def extend_df_incidences(df_incidences, n_weeks=26):
        """A function to extend the series of vaccine incidences with zero administered vaccines for n_weeks"""

        # Construct extended daterange index
        dates = df_incidences.index.get_level_values('date').unique()
        extended_dates = pd.date_range(start=min(dates), end=max(dates)+pd.Timedelta(weeks=n_weeks), freq='W-MON')

        # Define a new series with the extended dates
        iterables=[]
        for index_name in df_incidences.index.names:
            if index_name != 'date':
                iterables += [df_incidences.index.get_level_values(index_name).unique()]
            else:
                iterables += [extended_dates,]
        index = pd.MultiIndex.from_product(iterables, names=df_incidences.index.names)
        extended_df = pd.Series(index=index, name='PREALLOCATED', dtype=float)        

        # Place incidence series in extended series
        extended_df = extended_df.combine_first(df_incidences).fillna(0)

        # Rename
        extended_df = extended_df.rename(df_incidences.name)

        return extended_df

    @staticmethod
    def compute_relative_cumulative(df):

        # Compute cumulative sum over all doses
        levels = list(df.index.names)
        levels.remove("date")
        df_cumsum = df.groupby(by=levels).cumsum().clip(lower=0, upper=1)
        df_cumsum = df_cumsum.reset_index().set_index('date')
        df_cumsum = df_cumsum.rename(columns={'REL_INCIDENCE': 'REL_CUMULATIVE'})

        # Define output dataframe
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'dose':
                iterables += [df.index.get_level_values(index_name).unique()]
            else:
                iterables += [['none', 'partial', 'full', 'boosted'],]
        index = pd.MultiIndex.from_product(iterables, names=df.index.names)
        df_new = pd.DataFrame(index=index, columns=['REL_INCIDENCE', 'REL_CUMULATIVE'], dtype=float)
        df_new_index = df_new.index
        df_new = df_new.reset_index().set_index('date')

        # Remove multiindex from incidences data
        df = df.reset_index().set_index('date')

        ## Compute populations 
        # 'partial' = 'first' - 'second'
        df_new.loc[df_new['dose']=='partial','REL_CUMULATIVE'] = (df_cumsum.loc[df_cumsum['dose']=='first', 'REL_CUMULATIVE'] \
        - df_cumsum.loc[df_cumsum['dose']=='second', 'REL_CUMULATIVE']).clip(lower=0, upper=1)
        # 'full' = 'second' + 'one_shot' - 'booster
        df_new.loc[df_new['dose']=='full','REL_CUMULATIVE'] = (df_cumsum.loc[df_cumsum['dose']=='second','REL_CUMULATIVE'] \
        + df_cumsum.loc[df_cumsum['dose']=='one_shot','REL_CUMULATIVE']).clip(lower=0, upper=1) #- df_cumsum.loc[df_cumsum['dose']=='booster','REL_CUMULATIVE']).clip(lower=0, upper=1)
        # 'boosted' = 'booster'
        df_new.loc[df_new['dose']=='boosted','REL_CUMULATIVE'] = df_cumsum.loc[df_cumsum['dose']=='booster', 'REL_CUMULATIVE'].clip(lower=0, upper=1)
        # 'none' = Rest category. Make sure all exclusive categories adds up to 1.
        df_new.loc[df_new['dose']=='none','REL_CUMULATIVE'] = 1 - df_new.loc[df_new['dose']=='partial','REL_CUMULATIVE'] \
        - df_new.loc[df_new['dose']=='full','REL_CUMULATIVE'] - df_new.loc[df_new['dose']=='boosted','REL_CUMULATIVE']

        ## Make sure incidences wind up in right category
        # full = second + janssen
        df_new.loc[df_new['dose']=='full', 'REL_INCIDENCE'] = df.loc[df['dose']=='second', 'REL_INCIDENCE'] + df.loc[df['dose']=='one_shot', 'REL_INCIDENCE']
        # partial = first
        df_new.loc[df_new['dose']=='partial', 'REL_INCIDENCE'] = df.loc[df['dose']=='first', 'REL_INCIDENCE'] 
        # boosted = booster
        df_new.loc[df_new['dose']=='boosted', 'REL_INCIDENCE'] = df.loc[df['dose']=='booster', 'REL_INCIDENCE'] 
        # none = 0
        df_new.loc[df_new['dose']=='none', 'REL_INCIDENCE'] = 0

        # Reintroduce multiindex
        df = pd.DataFrame(index=df_new_index, columns=['REL_INCIDENCE', 'REL_CUMULATIVE'], data=df_new[['REL_INCIDENCE', 'REL_CUMULATIVE']].values)

        return df

    @staticmethod
    def format_vaccine_params(df):
        """ This function format the vaccine parameters provided by the user in function `covid19model.data.model_parameters.get_COVID19_SEIQRD_VOC_parameters` into a format better suited for the computation in this module."""

        ###################
        ## e_s, e_i, e_h ##
        ###################

        # Define vaccination properties  
        iterables = [df.index.get_level_values('VOC').unique(),['none', 'partial', 'full', 'boosted'], ['e_s', 'e_i', 'e_h']]
        index = pd.MultiIndex.from_product(iterables, names=['VOC', 'dose', 'efficacy'])
        df_new = pd.DataFrame(index=index, columns=['initial', 'best', 'waned'])    

        # No vaccin = zero efficacy
        df_new.loc[(slice(None), 'none', slice(None)),:] = 0

        # Partially vaccinated is initially non-vaccinated + Waning of partially vaccinated to non-vaccinated
        df_new.loc[(slice(None), 'partial', slice(None)),['initial','waned']] = 0

        # Fill in best efficacies using data, waning of full, booster using data
        for VOC in df_new.index.get_level_values('VOC'):
            for dose in df_new.index.get_level_values('dose'):
                for efficacy in df_new.index.get_level_values('efficacy'):
                    df_new.loc[(VOC, dose, efficacy),'best'] = df.loc[(VOC,dose),efficacy]
                    if ((dose == 'full') | (dose =='boosted')):
                        df_new.loc[(VOC, dose, efficacy),'waned'] = df.loc[(VOC,'waned'),efficacy]

        # inital,full = best,partial
        df_new.loc[(slice(None), 'full', slice(None)),'initial'] = df_new.loc[(slice(None), 'partial', slice(None)),'best'].values

        # boosted, initial = full, waned
        df_new.loc[(slice(None), 'boosted', slice(None)),'initial'] = df_new.loc[(slice(None), 'full', slice(None)),'waned'].values

        ############################
        ## onset_immunity, waning ##
        ############################

        onset={}
        waning={}
        for dose in df_new.index.get_level_values('dose'):
            onset.update({dose: df.loc[(df_new.index.get_level_values('VOC')[0], dose), 'onset_immunity']})
            waning.update({dose: df.loc[(df_new.index.get_level_values('VOC')[0], dose), 'waning']})

        return df_new, onset, waning

    @staticmethod
    def waning_delay(delta_t, onset_days, waning_days, E_init, E_best, E_waned):
        """
        Function to compute the vaccine efficacy after delta_t days, accouting for a linear buildup and and subsequent exponential waning of the vaccine's immunity.

        Input
        -----
        days : float
            number of days after the novel vaccination
        onset_days : float
            number of days it takes for the vaccine to take full effect
        waning_days : float
            number of days for the vaccine to wane (on average)
        E_init : float
            vaccine-related rescaling value right before vaccination
        E_best : float
            rescaling value related to the best possible protection by the currently injected vaccine
        E_waned : float
            rescaling value related to the vaccine protection after a waning period.

        Output
        ------
        E_eff : float
            effective rescaling value associated with the newly administered vaccine

        """

        if delta_t < 0:
            return E_init
        elif delta_t < onset_days:
            # Compute weighted average using ramp
            E_eff = (E_best - E_init)/onset_days*delta_t + E_init
            return E_eff
        else:
            # Compute exponential decay factor going from 1 to 0
            halftime = waning_days - onset_days
            tau = halftime/np.log(2)
            f = np.exp(-(delta_t-onset_days)/tau)
            # Compute weighted average
            E_eff = E_waned - f*(E_waned - E_best)
        return E_eff

    def compute_efficacies(self, df, vaccine_params, VOC_function, onset, waning):

        # Pre-allocate a dataframe with the desired format
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Columns
        columns = vaccine_params.index.get_level_values('efficacy').unique().values
        # Desired df
        new_df = pd.DataFrame(index=df.index, columns=columns, dtype=float)
        # Omit multiindex
        new_df_index = new_df.index
        new_df_columns = new_df.columns
        new_df = new_df.reset_index()

        # Omit multiindex on vaccination incidence data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        df_index = df.index
        df_columns = df.columns
        df = df.reset_index()

        # Compute vaccine efficacy per dose
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for dose in new_df['dose'].unique():
            dates = df['date'].unique()
            for date in dates:
                sol = np.zeros([len(vaccine_params.index.get_level_values('efficacy').unique()), len(df.loc[((df['dose']==dose) & (df['date']==dates[0])), 'REL_INCIDENCE'])])
                for inner_date in dates[dates<=date]:
                    # Compute number of days in the past vaccines were administered
                    delta_t = (date-inner_date)/pd.Timedelta(days=1)

                    # Compute VOC average vaccine efficacy at provided date
                    VOC_fraction = VOC_function(pd.Timestamp(date), {}, {})[0,:]
                    E_initial=[]
                    E_best=[]
                    E_waned=[]
                    for efficacy in vaccine_params.index.get_level_values('efficacy').unique():
                        if dose == 'boosted':
                            # Boosted vaccines start from waned 2nd dose, hence 'booster-initial' = 'full-waned'
                            E_initial.append(0)
                            # Maximum booster coverage = diference between 'booster-best' and 'full-waned'='booster_initial' efficacy
                            E_best.append(np.sum(VOC_fraction*vaccine_params.loc[(slice(None), dose, efficacy), 'best'].values) - np.sum(VOC_fraction*vaccine_params.loc[(slice(None), 'full', efficacy), 'waned'].values))
                            # Booster can wane below initial level of 'full-waned'
                            E_waned.append(np.sum(VOC_fraction*vaccine_params.loc[(slice(None), dose, efficacy), 'waned'].values) - np.sum(VOC_fraction*vaccine_params.loc[(slice(None), 'full', efficacy), 'waned'].values))
                        else:
                            E_initial.append(np.sum(VOC_fraction*vaccine_params.loc[(slice(None), dose, efficacy), 'initial'].values))
                            E_best.append(np.sum(VOC_fraction*vaccine_params.loc[(slice(None), dose, efficacy), 'best'].values))
                            E_waned.append(np.sum(VOC_fraction*vaccine_params.loc[(slice(None), dose, efficacy), 'waned'].values))

                    # Compute protection after delta_t days
                    weight = self.waning_delay(delta_t, onset[dose], waning[dose], np.array(E_initial), np.array(E_best), np.array(E_waned))

                    # Multiply weights with doses jabbed delta_t_days in the past
                    if dose == 'boosted':
                        # Booster is 'added' on top of 2nd dose
                        sol = sol + weight[:, np.newaxis]*(df.loc[((df['dose']==dose) & (df['date']==inner_date)), 'REL_INCIDENCE'].values + df.loc[((df['dose']=='full') & (df['date']==inner_date)), 'REL_INCIDENCE'].values)[np.newaxis,:]
                    else:
                        sol = sol + weight[:, np.newaxis]*df.loc[((df['dose']==dose) & (df['date']==inner_date)), 'REL_INCIDENCE'].values[np.newaxis,:]
                
                sol = sol.clip(min=-1,max=1)

                # Perform assignment
                for idx, efficacy in enumerate(vaccine_params.index.get_level_values('efficacy').unique()):
                    new_df.loc[((new_df['date']==date) & (new_df['dose']==dose)), efficacy] = sol[idx,:]

        # Reintroduce multiindeces
        df = pd.DataFrame(index=df_index, columns=df_columns, data=df[df_columns].values)
        new_df = pd.DataFrame(index=new_df_index, columns=new_df_columns, data=new_df[new_df_columns].values)  

        #import matplotlib.pyplot as plt
        #dates = new_df.index.get_level_values('date').unique()
        #age_group = new_df.index.get_level_values('age').unique()[-1]
        #fig,ax=plt.subplots(nrows=3, ncols=1, sharex=True)
        #ax.plot(dates, new_df)
        #plt.show()

        # Take dose-weighted average over time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Initialize dataframe
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'dose':
                iterables += [df.index.get_level_values(index_name).unique()]
        names = list(df.index.names)
        names.remove("dose")        
        index = pd.MultiIndex.from_product(iterables, names=names)
        dose_avg_df = pd.DataFrame(index=index, columns=vaccine_params.index.get_level_values('efficacy').unique().values, dtype=float)
        # Weighted sum of efficacies with cumulative share of population in dose x
        for efficacy in dose_avg_df.columns:
            dose_avg_df[efficacy] = (df['REL_CUMULATIVE']*new_df[efficacy]).groupby(by=dose_avg_df.index.names).sum()

        return dose_avg_df

    def age_conversion(self, df, desired_age_classes, agg):

        # Define a new dataframe with the desired age groups
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'age':
                iterables += [df.index.get_level_values(index_name).unique()]
            else:
                iterables += [desired_age_classes]
        index = pd.MultiIndex.from_product(iterables, names=df.index.names)
        new_df = pd.DataFrame(index=index, columns=df.columns, dtype=float)

        # Loop to the dataseries level and perform demographic weighing
        if agg:
            for date in df.index.get_level_values('date').unique():
                for NIS in df.index.get_level_values('NIS').unique():
                    new_df.loc[date, NIS, slice(None)] = self.demographic_weighing(df.loc[date, NIS, slice(None)], desired_age_classes, agg=agg, NIS=NIS).values
        else:
            for date in df.index.get_level_values('date').unique():
                new_df.loc[date, slice(None)] = self.demographic_weighing(df.loc[date, slice(None)], desired_age_classes).values

        return new_df

    @staticmethod
    def demographic_weighing(data, age_classes, agg=None, NIS=None):
        """ 
        Given an age-stratified series of a (non-cumulative) population property: [age_group_lower, age_group_upper] : property,
        this function can convert the data into another user-defined age-stratification using demographic weighing

        Parameters
        ----------
        data: pd.Series
            A series of age-stratified data. Index must be of type pd.Intervalindex.
        
        age_classes : pd.IntervalIndex
            Desired age groups of the converted table.

        Returns
        -------

        out: pd.Series
            Converted data.
        """

        # Pre-allocate new dataframe
        out = pd.DataFrame(index = age_classes, columns=data.columns, dtype=float)
        if agg:
            out_n_individuals = construct_initN(age_classes, agg).loc[NIS,:].values
            demographics = construct_initN(None,agg).loc[NIS,:].values
        else:
            out_n_individuals = construct_initN(age_classes, agg).values
            demographics = construct_initN(None,agg).values
        # Loop over desired intervals
        for idx,interval in enumerate(age_classes):
            result = []
            for age in range(interval.left, interval.right):
                try:
                    result.append(demographics[age]/out_n_individuals[idx]*data.iloc[np.where(data.index.contains(age))[0][0]])
                except:
                    result.append(0/out_n_individuals[idx]*data.iloc[np.where(data.index.contains(age))[0][0]])
            out.iloc[idx] = sum(result)
        return out
        
###################################
## Google social policy function ##
###################################

class make_contact_matrix_function():
    """
    Class that returns contact matrix based on 4 effention parameters by default, but has other policies defined as well.

    Input
    -----
    df_google : dataframe
        google mobility data
    Nc_all : dictionnary
        contact matrices for home, schools, work, transport, leisure and others

    Output
    ------

    __class__ : default function
        Default output function, based on contact_matrix_4eff

    """
    def __init__(self, df_google, Nc_all):
        self.df_google = df_google.astype(float)
        self.Nc_all = Nc_all
        # Compute start and endtimes of dataframe
        self.df_google_start = df_google.index.get_level_values('date')[0]
        self.df_google_end = df_google.index.get_level_values('date')[-1]
        # Check if provincial data is provided
        self.provincial = None
        if 'NIS' in self.df_google.index.names:
            self.provincial = True
            self.space_agg = len(self.df_google.index.get_level_values('NIS').unique().values)

    @lru_cache() # once the function is run for a set of parameters, it doesn't need to compile again
    def __call__(self, t, eff_home=1, eff_schools=1, eff_work=1, eff_rest = 1, mentality=1,
                       school=None, work=None, transport=None, leisure=None, others=None, home=None):

        """
        t : timestamp
            current date
        eff_... : float [0,1]
            effention parameter to estimate
        school, work, transport, leisure, others : float [0,1]
            level of opening of these sectors
            if None, it is calculated from google mobility data
            only school cannot be None!

        """

        if school is None:
            raise ValueError(
                "Please indicate to which extend schools are open")


        places_var = [work, transport, leisure, others]
        places_names = ['work', 'transport', 'leisure', 'others']
        GCMR_names = ['work', 'transport', 'retail_recreation', 'grocery']

        if self.provincial:
            if t < self.df_google_start:
                return np.ones(self.space_agg)[:,np.newaxis,np.newaxis]*self.Nc_all['total']
            elif self.df_google_start <= t <= self.df_google_end:
                # Extract row at timestep t
                row = -self.df_google.loc[(t, slice(None)),:]/100
            else:
                # Extract last 7 days and take the mean
                row = -self.df_google.loc[(self.df_google_end - pd.Timedelta(days=7)): self.df_google_end, slice(None)].groupby(level='NIS').mean()/100

            # Sort NIS codes from low to high
            row.sort_index(level='NIS', ascending=True, inplace=True)
            # Extract values
            values_dict={}
            for idx,place in enumerate(places_var):
                if place is None:
                    place = 1 - row[GCMR_names[idx]].values
                else:
                    try:
                        test=len(place)
                    except:
                        place = place*np.ones(self.space_agg)     
                values_dict.update({places_names[idx]: place})

            # Schools:
            try:
                test=len(school)
            except:
                school = school*np.ones(self.space_agg)

            # Expand dims on mentality
            if isinstance(mentality,tuple):
                mentality=np.array(mentality)
                mentality = mentality[:, np.newaxis, np.newaxis]

            # Construct contact matrix
            CM = (eff_home*np.ones(self.space_agg)[:, np.newaxis,np.newaxis]*self.Nc_all['home'] + mentality*(
                    (eff_schools*school)[:, np.newaxis,np.newaxis]*self.Nc_all['schools'] +
                    (eff_work*values_dict['work'])[:,np.newaxis,np.newaxis]*self.Nc_all['work'] + 
                    (eff_rest*values_dict['transport'])[:,np.newaxis,np.newaxis]*self.Nc_all['transport'] + 
                    (eff_rest*values_dict['leisure'])[:,np.newaxis,np.newaxis]*self.Nc_all['leisure'] +
                    (eff_rest*values_dict['others'])[:,np.newaxis,np.newaxis]*self.Nc_all['others']) )

        else:
            if t < self.df_google_start:
                return self.Nc_all['total']
            elif self.df_google_start <= t <= self.df_google_end:
                # Extract row at timestep t
                row = -self.df_google.loc[t]/100
            else:
                # Extract last 14 days and take the mean
                row = -self.df_google[-7:-1].mean()/100
            
            # Extract values
            values_dict={}
            for idx,place in enumerate(places_var):
                if place is None:
                    place = 1 - row[GCMR_names[idx]]
                values_dict.update({places_names[idx]: place})  

            # Construct contact matrix
            CM = (eff_home*self.Nc_all['home'] + mentality*(
                    eff_schools*school*self.Nc_all['schools'] +
                    eff_work*values_dict['work']*self.Nc_all['work'] +
                    eff_rest*values_dict['transport']*self.Nc_all['transport'] +
                    eff_rest*values_dict['leisure']*self.Nc_all['leisure'] +
                    eff_rest*values_dict['others']*self.Nc_all['others']) )

        return CM

    def all_contact(self):
        return self.Nc_all['total']

    def all_contact_no_schools(self):
        return self.Nc_all['total'] - self.Nc_all['schools']

    def ramp_fun(self, Nc_old, Nc_new, t, t_start, l):
        """
        t : timestamp
            current simulation time
        t_start : timestamp
            start of policy change
        l : int
            number of additional days after the time delay until full compliance is reached
        """

        return Nc_old + (Nc_new-Nc_old)/l * float( (t-t_start)/pd.Timedelta('1D') )

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
        return Nc_old + (Nc_new-Nc_old)/l * float( (t-t_start-tau_days)/pd.Timedelta('1D') )
    
    ####################
    ## National model ##
    ####################

    def policies_all(self, t, states, param, l1, l2, eff_schools, eff_work, eff_rest, eff_home, mentality):
        '''
        Function that returns the time-dependant social contact matrix Nc for all COVID waves.
        
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
        eff_{location} : float
            "Effectivity" of contacts at {location}. Alternatively, degree correlation between Google mobility indicator and SARS-CoV-2 spread at {location}.
        mentality : float
            Lockdown mentality multipier

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
        t4 = pd.Timestamp('2020-08-03') # Summer lockdown in Antwerp
        t5 = pd.Timestamp('2020-08-24') # End of summer lockdown in Antwerp
        t6 = pd.Timestamp('2020-09-01') # end of summer holidays
        t7 = pd.Timestamp('2020-09-21') # Opening universities

        # Define key dates of winter 2020-2021
        t8 = pd.Timestamp('2020-10-19') # lockdown (1)
        t9 = pd.Timestamp('2020-11-02') # lockdown (2)
        t10 = pd.Timestamp('2020-11-16') # schools re-open
        t11 = pd.Timestamp('2020-12-18') # Christmas holiday starts
        t12 = pd.Timestamp('2021-01-04') # Christmas holiday ends
        t13 = pd.Timestamp('2021-02-15') # Spring break starts
        t14 = pd.Timestamp('2021-02-21') # Spring break ends
        t15 = pd.Timestamp('2021-02-28') # Contact increase in children
        t16 = pd.Timestamp('2021-03-26') # Start of Easter holiday
        t17 = pd.Timestamp('2021-04-18') # End of Easter holiday
        t18 = pd.Timestamp('2021-06-01') # Start of lockdown relaxation
        t19 = pd.Timestamp('2021-07-01') # Start of Summer holiday

        # Define key dates of winter 2021-2022
        t20 = pd.Timestamp('2021-09-01') # End of Summer holiday
        t21 = pd.Timestamp('2021-09-21') # Opening of universities
        t22 = pd.Timestamp('2021-10-01') # Flanders releases all measures
        t23 = pd.Timestamp('2021-11-01') # Start of autumn break
        t24 = pd.Timestamp('2021-11-07') # End of autumn break
        t25 = pd.Timestamp('2021-11-17') # Overlegcommite 1 out of 3
        t26 = pd.Timestamp('2021-12-03') # Overlegcommite 3 out of 3
        t27 = pd.Timestamp('2021-12-20') # Start of Christmass break (one week earlier than normal)

        t28 = pd.Timestamp('2021-12-25') # Christmas
        t29 = pd.Timestamp('2021-12-31') # NYE

        t30 = pd.Timestamp('2022-01-10') # End of Christmass break
        t31 = pd.Timestamp('2022-02-28') # Start of Spring Break
        t32 = pd.Timestamp('2022-03-06') # End of Spring Break
        t33 = pd.Timestamp('2022-04-04') # Start of Easter Break
        t34 = pd.Timestamp('2022-04-17') # End of Easter Break
        t35 = pd.Timestamp('2022-07-01') # Start of summer holidays

        ################
        ## First wave ##
        ################

        if t <= t1:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, school=1)
        elif t1 < t <= t1 + l1_days:
            t = pd.Timestamp(t.date())
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t1, l1)
        elif t1 + l1_days < t <= t2:
            return self.__call__(t, eff_home=eff_home, eff_schools=eff_schools, eff_work=eff_work, eff_rest=eff_rest, mentality=mentality, school=0)
        elif t2 < t <= t3:
            l = (t3 - t2)/pd.Timedelta(days=1)
            r = (t3 - t2)/(t4 - t2)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=0)
            return self.ramp_fun(policy_old, policy_new, t, t2, l)            
        elif t3 < t <= t4:
            l = (t4 - t3)/pd.Timedelta(days=1)
            r = (t3 - t2)/(t4 - t2)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality = 1, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t3, l)  
        elif t4 < t <= t5:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)                                          
        elif t5 < t <= t6:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0)

        ######################      
        ## Winter 2020-2021 ##
        ######################

        elif t6 < t <= t7:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0.7)  
        elif t7 < t <= t8:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)  
        elif t8  < t <= t8 + l2_days:
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
            return self.ramp_fun(policy_old, policy_new, t, t8, l2)
        elif t8 + l2_days < t <= t9:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t9 < t <= t10:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t10 < t <= t11:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1) 
        elif t11 < t <= t12:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t12 < t <= t13:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t13 < t <= t14:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)    
        elif t14 < t <= t15:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t15 < t <= t16:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality,school=1)
        elif t16 < t <= t17:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)                           
        elif t17 < t <= t18:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t18 < t <= t19:
            l = (t19 - t18)/pd.Timedelta(days=1)
            r = (t19 - t18)/(t20 - t18)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=1)
            return self.ramp_fun(policy_old, policy_new, t, t18, l)
        elif t19 < t <= t20:
            l = (t20 - t19)/pd.Timedelta(days=1)
            r = (t19 - t18)/(t20 - t18)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t19, l)

        ######################
        ## Winter 2021-2022 ##
        ######################
        
        elif t20 < t <= t21:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0.7)
        elif t21 < t <= t22:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)    
        elif t22 < t <= t23:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest,  mentality=1, school=1)  
        elif t23 < t <= t24:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0) 
        elif t24 < t <= t25:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)
        elif t25 < t <= t26:
            # Gradual re-introduction of mentality change during overlegcommites
            l = (t26 - t25)/pd.Timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
            return self.ramp_fun(policy_old, policy_new, t, t25, l)
        elif t26 < t <= t27:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t27 < t <= t28:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t28 < t <= t29:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t29 < t <= t30:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t30 < t <= t31:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t31 < t <= t32:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, work=0.5, transport=0.5, leisure=1, others=1,school=0)  
        elif t32 < t <= t33:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, work=1, transport=1, leisure=1, others=1, school=1)           
        elif t33 < t <= t34:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, work=0.7, transport=0.7, leisure=1, others=1, school=0)
        elif t34 < t <= t35:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, work=1, transport=1, leisure=1, others=1, school=1)                                                                                                                                    
        else:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, work=0.7, transport=0.7, leisure=1, others=1, school=0)    

    ###################
    ## Spatial model ##
    ###################

    def policies_all_spatial(self, t, states, param, l1, l2, eff_schools, eff_work, eff_rest, eff_home, mentality):
        '''
        Function that returns the time-dependant social contact matrix Nc for all COVID waves.
        
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
        eff_{location} : float
            "Effectivity" of contacts at {location}. Alternatively, degree correlation between Google mobility indicator and SARS-CoV-2 spread at {location}.
        mentality : float
            Lockdown mentality multipier

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
        t7 = pd.Timestamp('2020-09-21') # Opening universities

        # Define key dates of winter 2020-2021
        t8 = pd.Timestamp('2020-10-19') # lockdown (1)
        t9 = pd.Timestamp('2020-11-02') # lockdown (2)
        t10 = pd.Timestamp('2020-11-16') # schools re-open
        t11 = pd.Timestamp('2020-12-18') # Christmas holiday starts
        t12 = pd.Timestamp('2021-01-04') # Christmas holiday ends
        t13 = pd.Timestamp('2021-02-15') # Spring break starts
        t14 = pd.Timestamp('2021-02-21') # Spring break ends
        t15 = pd.Timestamp('2021-02-28') # Contact increase in children
        t16 = pd.Timestamp('2021-03-26') # Start of Easter holiday
        t17 = pd.Timestamp('2021-04-18') # End of Easter holiday
        t18 = pd.Timestamp('2021-06-01') # Start of relaxations
        t19 = pd.Timestamp('2021-07-01') # Start of Summer holiday
        t20 = pd.Timestamp('2021-08-01') # End of easing on mentality

        # Define key dates of winter 2021-2022
        t21 = pd.Timestamp('2021-09-01') # End of Summer holiday
        t22 = pd.Timestamp('2021-09-21') # Opening of universities
        t23 = pd.Timestamp('2021-10-01') # Flanders releases all measures
        t24 = pd.Timestamp('2021-11-01') # Start of autumn break
        t25 = pd.Timestamp('2021-11-07') # End of autumn break
        t26 = pd.Timestamp('2021-11-17') # Overlegcommite 1 out of 3
        t27 = pd.Timestamp('2021-12-03') # Overlegcommite 3 out of 3
        t28 = pd.Timestamp('2021-12-20') # Start of Christmass break (one week earlier than normal)
        t29 = pd.Timestamp('2022-01-10') # End of Christmass break
        t30 = pd.Timestamp('2022-02-28') # Start of Spring Break
        t31 = pd.Timestamp('2022-03-06') # End of Spring Break
        t32 = pd.Timestamp('2022-04-04') # Start of Easter Break
        t33 = pd.Timestamp('2022-04-17') # End of Easter Break
        t34 = pd.Timestamp('2022-07-01') # Start of summer holidays

        # Manual tweaking is unafortunately needed to make sure the second 2020 wave is correct
        # It is better to tweak the summer of 2020, if not, the summer of 2021 needs to be tweaked..
        idx_F = [0, 1, 4, 5, 8]
        idx_Bxl = [3,]
        idx_W = [2, 6, 7, 9, 10]
        mentality_summer_2020_lockdown = np.array([mentality, 0.5*mentality, # F
                                                2*mentality, # W
                                                2*mentality, # Bxl
                                                0.5*mentality, 2*mentality, # F
                                                3*mentality, 3*mentality, # W
                                                0.2*mentality, # F
                                                1.5*mentality, 2*mentality]) # W

        co_F = 1
        co_W = 1
        co_Bxl = 1
        mentality_relaxation_flanders_2021 = np.array([co_F, co_F, # F
                                                co_W, # W
                                                co_Bxl, # Bxl
                                                co_F, co_F, # F
                                                co_W, co_W, # W
                                                co_F, # F
                                                co_W, co_W]) # W

        ################
        ## First wave ##
        ################

        if t <= t1:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1) 
        elif t1 < t <= t1 + l1_days:
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t1, l1)
        elif t1 + l1_days < t <= t2:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t2 < t <= t3:
            l = (t3 - t2)/pd.Timedelta(days=1)
            r = (t3 - t2)/(t4 - t2)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=0)
            return self.ramp_fun(policy_old, policy_new, t, t2, l)            
        elif t3 < t <= t4:
            l = (t4 - t3)/pd.Timedelta(days=1)
            r = (t3 - t2)/(t4 - t2)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t3, l)  
        elif t4 < t <= t5:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_summer_2020_lockdown), school=0)                                          
        elif t5 < t <= t6:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0)      

        ######################      
        ## Winter 2020-2021 ##
        ######################

        elif t6 < t <= t7:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0.7)  
        elif t7 < t <= t8:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)  
        elif t8  < t <= t8 + l2_days:
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
            return self.ramp_fun(policy_old, policy_new, t, t8, l2)
        elif t8 + l2_days < t <= t9:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t9 < t <= t10:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t10 < t <= t11:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1) 
        elif t11 < t <= t12:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t12 < t <= t13:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t13 < t <= t14:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)    
        elif t14 < t <= t15:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t15 < t <= t16:
            r = 1.35
            mat = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
            mat[idx_Bxl,0:2,0:2] *= r
            mat[idx_W,0:2,0:2] *= r
            return mat #self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t16 < t <= t17:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)                           
        elif t17 < t <= t18:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t18 < t <= t19:
            l = (t19 - t18)/pd.Timedelta(days=1)
            r = (t19 - t18)/(t20 - t18)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=1)
            return self.ramp_fun(policy_old, policy_new, t, t18, l)
        elif t19 < t <= t20:
            l = (t20 - t19)/pd.Timedelta(days=1)
            r = (t19 - t18)/(t20 - t18)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t19, l)

        ######################      
        ## Winter 2021-2022 ##
        ######################        

        elif t20 < t <= t21:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0)
        elif t21 < t <= t22:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0.7)
        elif t22 < t <= t23:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)    
        elif t23 < t <= t24:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_relaxation_flanders_2021), school=1)  
        elif t24 < t <= t25:    
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_relaxation_flanders_2021), school=0)  
        elif t25 < t <= t26:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_relaxation_flanders_2021), school=1)  
        elif t26 < t <= t27:
            # Gradual re-introduction of mentality change during overlegcommites
            l = (t27 - t26)/pd.Timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_relaxation_flanders_2021), school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
            return self.ramp_fun(policy_old, policy_new, t, t26, l)
        elif t27 < t <= t28:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t28 < t <= t29:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t29 < t <= t30:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t30 < t <= t31:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, work=0.5, transport=0.5, leisure=1, others=1,school=0)  
        elif t31 < t <= t32:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, work=1, transport=1, leisure=1, others=1, school=1)           
        elif t32 < t <= t33:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, work=0.7, transport=0.7, leisure=1, others=1, school=0)
        elif t33 < t <= t34:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, work=1, transport=1, leisure=1, others=1, school=1)                                                                                                                                    
        else:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, work=0.7, transport=0.7, leisure=1, others=1, school=0)

    def policies_all_work_only(self, t, states, param, eff_work, l1, l2, mentality):
            '''
            Function that returns the time-dependant social contact matrix of work contacts (Nc_work). 
            
            Input
            -----
            t : Timestamp
                simulation time
            states : xarray
                model states
            param : dict
                model parameter dictionary
            eff_work : float
                "Effectivity" of contacts at work. Alternatively, degree correlation between Google mobility indicator for work and SARS-CoV-2 spread at work.
            mentality : float
                Lockdown mentality multipier

            Returns
            -------
            CM : np.array
                Effective contact matrix (output of __call__ function)
            '''
            t = pd.Timestamp(t.date())

            # Convert compliance l to dates
            l1_days = pd.Timedelta(l1, unit='D')
            l2_days = pd.Timedelta(l2, unit='D')

            ################
            ## First wave ##
            ################

            # Define key dates 
            t1 = pd.Timestamp('2020-03-15') # start of lockdown
            t2 = pd.Timestamp('2020-05-15') # start of relaxation
            t3 = pd.Timestamp('2020-08-10') # end of mentality easing
            t4 = pd.Timestamp('2020-10-19') # start of lockdown
            t5 = pd.Timestamp('2021-06-01') # start of relaxations
            t6 = pd.Timestamp('2021-08-01') # end of easing on mentality
            t7 = pd.Timestamp('2021-11-17') # Overlegcommite 1 out of 3
            t8 = pd.Timestamp('2021-12-03') # Overlegcommite 3 out of 3
            t9 = pd.Timestamp('2022-02-01') # start easing on mentality

            # Define number of contacts
            if t <= t1:
                return self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=1, school=0) 
            elif t1 < t <= t1 + l1_days:
                policy_old = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=1, school=0) 
                policy_new = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=mentality, school=0) 
                return self.ramp_fun(policy_old, policy_new, t, t1, l1)
            elif t1 + l1_days < t <= t2:
                return self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=mentality, school=0) 
            elif t2 < t <= t3:
                l = (t3 - t2)/pd.Timedelta(days=1)
                policy_old = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=mentality, school=0) 
                policy_new = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=1, school=0) 
                return self.ramp_fun(policy_old, policy_new, t, t2, l)
            elif t3 < t <= t4:
                return self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=1, school=0)       

            ######################
            ## Winter 2020-2021 ##
            ######################

            elif t4  < t <= t4 + l2_days:
                policy_old = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=1, school=0)
                policy_new = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=mentality, school=0)
                return self.ramp_fun(policy_old, policy_new, t, t4, l2)
            elif t4 + l2_days < t <= t5:
                return self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=mentality, school=0) 
            elif t5 < t <= t6:
                l = (t6 - t5)/pd.Timedelta(days=1)
                policy_old = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=mentality, school=0) 
                policy_new = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=1, school=0) 
                return self.ramp_fun(policy_old, policy_new, t, t5, l)
            elif t6 < t <= t7:
                return self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=1, school=0)       

            ######################      
            ## Winter 2021-2022 ##
            ######################      

            elif t7 < t <= t8:
                l = (t8 - t7)/pd.Timedelta(days=1)
                policy_old = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=1, school=0) 
                policy_new = self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=mentality, school=0) 
                return self.ramp_fun(policy_old, policy_new, t, t7, l)
            elif t8 < t <= t9:
                return self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=mentality, school=0) 
            else:
                return self.__call__(t, eff_home=0, eff_schools=0, eff_work=eff_work, eff_rest=0, mentality=1, school=0)

##########################
## Seasonality function ##
##########################

class make_seasonality_function():
    """
    Simple class to create functions that controls the season-dependent value of the transmission coefficients. Currently not based on any data, but e.g. weather patterns could be imported if needed.
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
        maxdate = ref_date + pd.Timedelta(days=peak_shift)
        # One period is one year long (seasonality)
        t = (t - pd.to_datetime(maxdate))/pd.Timedelta(days=1)/365
        rescaling = 1 + amplitude*np.cos( 2*np.pi*(t))
        return param*rescaling
    
class make_seasonality_function_NEW():
    """
    NOTE: may replace other seasonality TDPF if deemed better.
    
    Simple class to create functions that controls the season-dependent value of the transmission coefficients. Currently not based on any data, but e.g. weather patterns could be imported if needed.
    """
    def __call__(self, t, states, param, amplitude, peak_shift):
        """
        Default output function. Returns a sinusoid with average value 1.
        
        t : Timestamp
            simulation time
        amplitude : float
            maximum deviation of output with respect to the average (1)
        peak_shift : float
            phase. Number of days after January 1st after which the maximum value of the seasonality rescaling is reached 
        """
        ref_date = pd.to_datetime('2021-01-01')
        # If peak_shift = 0, the max is on the first of January
        maxdate = ref_date + pd.Timedelta(days=peak_shift)
        # One period is one year long (seasonality)
        t = (t - pd.to_datetime(maxdate))/pd.Timedelta(days=1)/365
        rescaling = 1 + amplitude*np.cos( 2*np.pi*(t))
        return rescaling

####################
## Economic model ##
####################

def household_demand_shock(t, states, param, t_start_lockdown, t_end_lockdown, t_end_pandemic, c_s, on_site):
    """
    A time-dependent function to return the household demand shock.

    Parameters
    ----------
    t : pd.timestamp
        current date
    param: np.array
        initialised value of epsilon_S
    states : dict
        Dictionary containing all states of the economic model
    t_start_lockdown : pd.timestamp
        start of lockdown
    t_end_lockdown : pd.timestamp
        end of lockdown
    t_end_pandemic : pd.timestamp
        expected end date of the pandemic
    c_s : np.array
        shock vector
    on_site : np.array
        vector containing 1 if sector output is consumed on-site and 0 if sector output is not consumed on-site

    Returns
    -------
    epsilon_D : np.array
        sectoral household demand shock
    """

    if t < t_start_lockdown:
        return param
    elif ((t >= t_start_lockdown) & (t < t_end_lockdown)):
        return c_s
    elif ((t >= t_end_lockdown) & (t < t_end_pandemic)):
        epsilon = c_s/np.log(100)*np.log(100 - 99*(t-t_end_lockdown)/(t_end_pandemic-t_end_lockdown))
        epsilon[np.where(on_site == 0)] = 0
        return epsilon
    else:
        return param

def labor_supply_shock(t, states, param, t_start_lockdown, t_end_lockdown, l_s):
    """
    A function returning the labor reduction due to lockdown measures.

    Parameters
    ----------
    t : pd.timestamp
        current date
    param: np.array
        initialised value of epsilon_S
    states : dict
        Dictionary containing all states of the economic model
    t_start_lockdown : pd.timestamp
        start of economic lockdown
    t_end_lockdown : pd.timestamp
        end of economic lockdown
    l_s : np.array
        number of unactive workers under lockdown measures (obtained from survey 25-04-2020)
   
    Returns
    -------
    epsilon_S : np.array
        reduction in labor force
        
    """
    if t < t_start_lockdown:
        return param
    elif ((t >= t_start_lockdown) & (t < t_end_lockdown)):
        return l_s
    else:
        return param

def other_demand_shock(t, states, param, t_start_lockdown, t_end_lockdown, t_end_pandemic, f_s):
    """
    A time-dependent function to return the exogeneous demand shock.

    Parameters
    ----------
    t : pd.timestamp
        current date
    param: np.array
        initialised value of epsilon_F
    states : dict
        Dictionary containing all states of the economic model
    t_start_lockdown : pd.timestamp
        start of lockdown
    t_end_lockdown : pd.timestamp
        end of lockdown
    t_end_pandemic : pd.timestamp
        expected end of the pandemic
    f_s : np.array
        exogeneous shock vector

    Returns
    -------
    epsilon_F : np.array
        exogeneous demand shock
    """
    if t < t_start_lockdown:
        return param
    elif ((t >= t_start_lockdown) & (t < t_end_lockdown)):
        return f_s
    else:
        return param

def compute_income_expectations(t, states, param, t_start_lockdown, t_end_lockdown, l_0, l_start_lockdown, rho, L):
    """
    A function to return the expected retained income in the long term of households.

    Parameters
    ----------
    t : pd.timestamp
        current date
    states : dict
        Dictionary containing all states of the economic model
    param : float
        current expected fraction of long term income
    t_start_lockdown : pd.timestamp
        startdate of lockdown
    t_end_lockdown : pd.timestamp
        enddate of lockdown
    l_0 : np.array
        sectoral labour expenditure under business-as-usual
    l_start_lockdown : np.array
        sectoral labour expenditure at start of lockdown
    rho : float
        first order economic recovery time constant
    L : float
        fraction of households believing in an L-shaped economic recovery

    Returns
    -------
    zeta : float
        fraction (0-1) of pre-pandemic income households expect to retain in the long run
    """

    if t < t_start_lockdown:
        zeta = 1
    else:
        zeta_L = 1 - 0.5*(sum(l_0)-l_start_lockdown)/sum(l_0)
        if ((t >= t_start_lockdown) & (t < t_end_lockdown)):
            zeta = zeta_L
        else:
            # first order system
            zeta = zeta_L + (1 - np.exp(-(1-rho)*(t-t_end_lockdown).days))*(1-zeta_L)*L
    return zeta

def government_furloughing(t, states, param, t_start_compensation, t_end_compensation, b_s):
    """
    A function to simulate reimbursement of a fraction b of the income loss by policymakers (f.i. as social benefits, or "tijdelijke werkloosheid")

    Parameters
    ----------
    t : pd.timestamp
        current date
    param: float
        initialised value of b
    t_start_compensation : pd.timestamp
        startdate of compensation
    t_end_lockdown : pd.timestamp
        enddate of compensation
    b_s: float
        fraction of lost labor income furloughed to consumers under 'shock'

    Returns
    -------
    b: float
        fraction of lost labor income compensated
    """
    if t < t_start_compensation:
        return param
    elif ((t >= t_start_compensation) & (t < t_end_compensation)):
        return b_s
    else:
        return param