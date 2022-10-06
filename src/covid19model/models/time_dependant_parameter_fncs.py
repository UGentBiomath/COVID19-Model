import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
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

from covid19model.data.utils import construct_initN

class make_N_vacc_function():
    """A time-dependent parameter function to return the vaccine incidence at each timestep of the simulation
       Includes an example of a "hypothetical" extension of an ongoing booster campaign
    """
    
    def __init__(self,  df_incidences, agg=None, age_classes=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'),
                 hypothetical_function=False):
        
        #####################
        ## Input variables ##
        #####################
        
        # Window length of exponential moving average to smooth vaccine incidence
        filter_length = 31
        # Parameters of hypothetical booster campaign
        weekly_doses = 7*30000
        vacc_order = list(range(len(age_classes)))[::-1]
        stop_idx = len(age_classes)
        refusal = 0.1*np.ones(len(age_classes))
        extend_weeks = 26
        
        ############################################################
        ## Check if Sciensano changed the definition of the doses ##
        ############################################################
        
        df = df_incidences
        if not (df.index.get_level_values('dose').unique() == ['A', 'B', 'C', 'E']).all():
            # This code was built at a point in time when the following doses were in the raw dataset provided by Sciensano:
            # A: 0 --> P, B: P --> F, C: 0 --> F, E: F --> B
            # If you get this error it means Sciensano has changed the definition of these doses
            raise ValueError("The names of the 'doses' in the vaccination dataset have been changed from those the code was designed for ({0})".format(df.index.get_level_values('dose').unique().values))
        
        ####################################
        ## Step 1: Perform age conversion ##
        ####################################

        # Only perform age conversion if necessary
        if len(age_classes) != len(df.index.get_level_values('age').unique()):
            df = self.age_conversion(df, age_classes, agg)
        elif (age_classes != df.index.get_level_values('age').unique()).any():
            df = self.age_conversion(df, age_classes, agg)

        ###############################################
        ## Step 2: Hypothetical vaccination campaign ##
        ###############################################
        
        if hypothetical_function:
            # REMARK: It would be more elegant to have the user supply a "hypothetical function" together with
            # a dictionary of arguments to the __init__ function.
            # However, since I don't really need hypothetical functions at the moment (2022-05), I'm going to park this here without much further ado
            df = self.append_booster_campaign(df, extend_weeks, weekly_doses, vacc_order, stop_idx, refusal)
        
        #####################################################################
        ## Step 3: Convert weekly to daily incidences and smooth using EMA ##
        #####################################################################

        df = self.smooth_incidences(df, filter_length, agg)
        
        #########################
        ## Step 4: Save a copy ##
        ######################### 

        if agg:
            df.to_pickle(os.path.join(os.path.dirname(__file__), f"../../../data/interim/sciensano/vacc_incidence_{agg}.pkl"))
        else:
            df.to_pickle(os.path.join(os.path.dirname(__file__), f"../../../data/interim/sciensano/vacc_incidence_national.pkl"))

        ##################################################
        ## Step 5: Assign relevant (meta)data to object ##
        ##################################################        

        # Dataframe 
        self.df = df
        # Number of spatial patches
        try:
            self.G = len(self.df.index.get_level_values('NIS').unique().values)
        except:
            pass
        # Aggregation    
        self.agg = agg
        # Number of age groups
        self.N = len(age_classes)
        # Number of vaccine doses
        self.D = len(self.df.index.get_level_values('dose').unique().values)
        
        return None
        
    @lru_cache()
    def get_data(self,t):
        """
        Function to extract the vaccination incidence at date 't' from the pd.Dataframe of incidences and format it into the right size
        """
        if self.agg:
            try:
                return np.array(self.df.loc[t,:,:,:].values).reshape( (self.G, self.N, self.D) )
            except:
                return np.zeros([self.G, self.N, self.D])
        else:
            try:
                return np.array(self.df.loc[t,:,:].values).reshape( (self.N, self.D) )
            except:
                return np.zeros([self.N, self.D])
    
    def __call__(self, t, states, param):
        """
        Time-dependent parameter function compatible wrapper for cached function `get_data`
        Returns the vaccination incidence at time "t"
        
        Input
        -----
        
        t : pd.Timestamp
            Current date in simulation
        
        Returns
        -------
        
        N_vacc : np.array
            Number of individuals to be vaccinated at simulation time "t" per [age, (space), dose]
        """
        
        # Convert time to suitable format
        t = pd.Timestamp(t.date())

        return self.get_data(t)
    
    ######################
    ## Helper functions ##
    ######################
    
    @staticmethod
    def smooth_incidences(df, filter_length, agg):
        """
        A function to convert the vaccine incidences dataframe from weekly to daily data and smooth the incidences with an exponential moving average
        
        Inputs
        ------
        df: pd.Series
            Pandas series containing the weekly vaccination incidences, indexed using a pd.Multiindex
            Obtained using the function `covid19model.data.sciensano.get_public_spatial_vaccination_data()`
            Must contain 'date', 'age' and 'dose' as indices for the national model
            Must contain 'date', 'NIS', 'age', 'dose' as indices for the spatial model
        
        filter_length: int
            Window length of the exponential moving average
            
        agg: str or None
            Spatial aggregation level. `None` for the national model, 'prov' for the provincial level, 'arr' for the arrondissement level
        
        Output
        ------
        
        df: pd.Series
            Pandas series containing smoothed daily vaccination incidences, indexed using the same pd.Multiindex as the input
        
        """
        
        # Start- and enddate
        df_start = pd.Timestamp(df.index.get_level_values('date').min())
        df_end = pd.Timestamp(df.index.get_level_values('date').max())
    
        # Make a dataframe with the desired format
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'date':
                iterables += [df.index.get_level_values(index_name).unique()]
            else:
                iterables += [pd.date_range(start=df_start, end=df_end, freq='D'),]
        index = pd.MultiIndex.from_product(iterables, names=df.index.names)
        df_new = pd.Series(index=index, name = df.name, dtype=float)

        # Loop over (NIS),age,dose
        for age in df.index.get_level_values('age').unique():
            for dose in df.index.get_level_values('dose').unique():
                if agg:
                    for NIS in df.index.get_level_values('NIS').unique():
                         # Convert weekly to daily data
                        daily_data = df.loc[slice(None), NIS, age, dose].resample('D').bfill().apply(lambda x : x/7)
                        # Apply an exponential moving average
                        daily_data_EMA = daily_data.ewm(span=filter_length, adjust=False).mean()
                        # Assign to new dataframe
                        df_new.loc[slice(None), NIS, age, dose] = daily_data_EMA.values
                else:
                    # Convert weekly to daily data
                    daily_data = df.loc[slice(None), age, dose].resample('D').bfill().apply(lambda x : x/7)
                    # Apply an exponential moving average
                    daily_data_EMA = daily_data.ewm(span=filter_length, adjust=False).mean()
                    # Assign to new dataframe
                    df_new.loc[slice(None), age, dose] = daily_data_EMA.values
                    
        return df_new
        
    @staticmethod
    def age_conversion(df, desired_age_classes, agg):
        """
        A function to convert a dataframe of vaccine incidences to another set of age groups `desired_age_classes` using demographic weighing
        
        Inputs
        ------
        
         df: pd.Series
            Pandas series containing the vaccination incidences, indexed using a pd.Multiindex
            Obtained using the function `covid19model.data.sciensano.get_public_spatial_vaccination_data()`
            Must contain 'date', 'age' and 'dose' as indices for the national model
            Must contain 'date', 'NIS', 'age', 'dose' as indices for the spatial model  
            
        desired_age_classes: pd.IntervalIndex
            Age groups you want the vaccination incidence to be formatted in
            Example: pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
        
        agg: str or None
            Spatial aggregation level. `None` for the national model, 'prov' for the provincial level, 'arr' for the arrondissement level
           
        Returns
        -------
        
        df_new: pd.Series
            Same pd.Series containing the vaccination incidences, but uses the age groups in `desired_age_classes`
        
        """
        
        
        from covid19model.data.utils import convert_age_stratified_quantity

        # Define a new dataframe with the desired age groups
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'age':
                iterables += [df.index.get_level_values(index_name).unique()]
            else:
                iterables += [desired_age_classes]
        index = pd.MultiIndex.from_product(iterables, names=df.index.names)
        df_new = pd.Series(index=index, name=df.name, dtype=float)

        # Loop to the dataseries level and perform demographic aggregation
        if agg:
            for date in df.index.get_level_values('date').unique():
                for NIS in df.index.get_level_values('NIS').unique():
                    for dose in df.index.get_level_values('dose').unique():
                        data = df.loc[date, NIS, slice(None), dose].reset_index().set_index('age')
                        df_new.loc[date, NIS, slice(None), dose] = convert_age_stratified_quantity(data, desired_age_classes, agg=agg, NIS=NIS).values
        else:
            for date in df.index.get_level_values('date').unique():
                for dose in df.index.get_level_values('dose').unique():
                    data = df.loc[date, slice(None), dose].reset_index().set_index('age')
                    df_new.loc[date, slice(None), dose] = convert_age_stratified_quantity(data, desired_age_classes).values

        return df_new
    
    @staticmethod
    def append_booster_campaign(df, extend_weeks, weekly_doses, vacc_order, stop_idx, refusal):
        """
        A function that appends `extend_weeks` of a hypothetical booster campaign to the dataframe containing the weekly vaccination incidences `df`
        Does not work for the spatial model!
        
        Inputs
        ------
        
        df: pd.Series
            Pandas series containing the weekly vaccination incidences, indexed using a pd.Multiindex
            Obtained using the function `covid19model.data.sciensano.get_public_spatial_vaccination_data()`
            Must contain 'date', 'age' and 'dose' as indices for the national model
        
        extend_weeks: int
            Number of weeks the incidences dataframe should be extended
            
        weekly_doses: int/float
            Number of weekly doses to be administered
        
        vacc_order: list/np.array
            A list containing the age-specific order in which the vaccines should be administered
            Must be the same length as the number of age groups in the model
            f.e. [9 8 .. 1 0] means elderly are prioritized
            
        stop_idx: int
            Number of age groups the algorithm should loop over
            f.e. len(age_classes) means all age groups are vaccine eligible
            f.e. len(age_classes)-1 means the algorithm stops at the last index of vacc_order
        
        refusal: list/np.array
            Vaccine refusal rate in age group i 
            Must be the same length as the number of age groups in the model
        
        Outputs
        -------
        
        df: pd.Series
            Pandas series containing the weekly vaccination indices, extended with a hypothetical booster campaign
            Multiindex from the input series is retained
        """
        
        
        # Index of desired output dose
        dose_idx = 3 # Administering boosters
        
        # Start from the supplied data
        df_out = df  
        
        # Metadata
        N = len(df.index.get_level_values('age').unique())
        D = len(df.index.get_level_values('dose').unique())
        
        # Generate desired daterange
        df_end = pd.Timestamp(df.index.get_level_values('date').max())
        date_range = pd.date_range(start=df_end+pd.Timedelta(days=7), end=df_end+pd.Timedelta(days=7*extend_weeks), freq='W-MON')
        
        # Compute the number of fully vaccinated individuals in every age group
        cumulative_F=[]
        for age_class in df.index.get_level_values('age').unique():
            cumulative_F.append((df.loc[slice(None), age_class, 'B'] + df.loc[slice(None), age_class, 'C']).cumsum().iloc[-1])
        
        # Loop over dates
        weekly_doses0 = weekly_doses
        for date in date_range:
            # Reset the weekly doses
            weekly_doses = weekly_doses0
            # Compute the cumulative number of boosted individuals
            cumulative_B=[]
            for age_class in df_out.index.get_level_values('age').unique():
                cumulative_B.append(df_out.loc[slice(None), age_class, 'E'].cumsum().iloc[-1])
            # Initialize N_vacc
            N_vacc = np.zeros([N,D])
            # Booster vaccination loop
            idx = 0
            while weekly_doses > 0:
                if idx == stop_idx:
                    weekly_doses= 0
                else:
                    doses_needed = (1-refusal[vacc_order[idx]])*cumulative_F[vacc_order[idx]] - cumulative_B[vacc_order[idx]]
                    if doses_needed > weekly_doses:
                        N_vacc[vacc_order[idx],dose_idx] = weekly_doses
                        weekly_doses = 0
                    elif doses_needed >= 0:
                        N_vacc[vacc_order[idx],dose_idx] = doses_needed
                        weekly_doses -= doses_needed
                    else:
                        N_vacc[vacc_order[idx],dose_idx] = 0
                    idx+=1
                    
            # Generate series on date with same multiindex as output
            iterables=[]
            for index_name in df.index.names:
                if index_name != 'date':
                    iterables += [df.index.get_level_values(index_name).unique()]
                else:
                    iterables += [[date],]
            index = pd.MultiIndex.from_product(iterables, names=df.index.names)
            df_new = pd.Series(index=index, name=df.name, dtype=float)
            # Put the result in
            df_new.loc[date, slice(None), slice(None)] = N_vacc.flatten()
            # Concatenate
            df_out = pd.concat([df_out, df_new])
        
        return df_out

from tqdm import tqdm
class make_vaccination_efficacy_function():
    
    ################################################
    ## Generate or load dataframe with efficacies ##
    ################################################
    
    def __init__(self, update=False, agg=None, age_classes=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'),
                    df_incidences=None, vaccine_params=None, VOCs=['WT', 'abc', 'delta']):
    
        if update==False:
            # Simply load data
            from covid19model.data.utils import to_pd_interval
            if agg:
                path = os.path.join(os.path.dirname(__file__), f"../../../data/interim/sciensano/vacc_rescaling_values_{agg}.csv")
                df_efficacies = pd.read_csv(path,  index_col=['date','NIS','age', 'dose', 'VOC'], converters={'date': pd.to_datetime, 'age': to_pd_interval})
            else:
                path = os.path.join(os.path.dirname(__file__), "../../../data/interim/sciensano/vacc_rescaling_values_national.csv")
                df_efficacies = pd.read_csv(path,  index_col=['date','age', 'dose', 'VOC'], converters={'date': pd.to_datetime, 'age': to_pd_interval})
        else:
            # Warn user this may take some time
            warnings.warn("The vaccination rescaling parameters must be updated because a change was made to the vaccination parameters, or the dataframe with incidences was changed/updated. This may take some time.", stacklevel=2)
            # Downsample the incidences dataframe to weekly frequency, if weekly incidences are provided this does nothing. Not tested if the data frequency is higher than one week.
            df_incidences = self.convert_frequency_WMON(df_incidences)
            # Add the one-shot J&J and the second doses together
            df_incidences = self.sum_oneshot_second(df_incidences)
            # Construct dataframe with cumulative sums
            df_cumsum = self.incidences_to_cumulative(df_incidences)
            # Format vaccination parameters 
            vaccine_params, onset, waning = self.format_vaccine_params(vaccine_params)
            # Compute weighted efficacies
            df_efficacies = self.compute_weighted_efficacy(df_incidences, df_cumsum, vaccine_params, waning)
            # Add dummy efficacy for zero doses
            df_efficacies = self.add_efficacy_no_vaccine(df_efficacies)
            # Save result
            dir = os.path.join(os.path.dirname(__file__), "../../../data/interim/sciensano/")
            if agg:
                df_efficacies.to_csv(os.path.join(dir, f'vacc_rescaling_values_{agg}.csv'))
                df_efficacies.to_pickle(os.path.join(dir, f'vacc_rescaling_values_{agg}.pkl'))
            else:
                df_efficacies.to_csv(os.path.join(dir, f'vacc_rescaling_values_national.csv'))
                df_efficacies.to_pickle(os.path.join(dir, f'vacc_rescaling_values_national.pkl'))
            
        # Throw out unwanted VOCs
        df_efficacies = df_efficacies.iloc[df_efficacies.index.get_level_values('VOC').isin(VOCs)]
        
        # Check if an age conversion is necessary
        age_conv=False
        if len(age_classes) != len(df_efficacies.index.get_level_values('age').unique()):
            df_efficacies = self.age_conversion(df_efficacies, age_classes, agg)
        elif (age_classes != df_efficacies.index.get_level_values('age').unique()).any():
            df_efficacies = self.age_conversion(df_efficacies, age_classes, agg)

        # Assign efficacy dataset
        self.df_efficacies = df_efficacies

        # Compute some other relevant attributes
        self.available_dates = df_efficacies.index.get_level_values('date').unique()
        self.n_VOCs = len(df_efficacies.index.get_level_values('VOC').unique())
        self.n_doses = len(df_efficacies.index.get_level_values('dose').unique())
        self.N = len(age_classes)
        self.agg = agg
        try:
            self.G = len(df_efficacies.index.get_level_values('NIS').unique())
        except:
            self.G = 0
        
        return None
    
    ################################################
    ## Convert data to a format model understands ##
    ################################################
    
    @lru_cache()
    def __call__(self, t, efficacy):
        """
        Returns vaccine efficacy matrix of size [N, n_doses, n_VOCs] or [G, N, n_doses, n_VOCs] for the requested time t.

        Input
        -----
        t : pd.Timestamp
            Time at which you want to know the rescaling parameter matrix
        rescaling_type : str
            Either 'e_s', 'e_i' or 'e_h'
            
        Output
        ------
        E : np.array
            Vaccine efficacy matrix at time t.
        """

        if efficacy not in self.df_efficacies.columns:
            raise ValueError(
                "valid vaccine efficacies are 'e_s', 'e_i', or 'e_h'.")

        t = pd.Timestamp(t)

        if t < min(self.available_dates):
            # Take unity matrix
            if self.agg:
                E = np.zeros([self.G, self.N, self.n_doses, self.n_VOCs])
            else:
                E = np.zeros([self.N, self.n_doses, self.n_VOCs, ])

        elif t <= max(self.available_dates):
            # Take interpolation between dates for which data is available
            t_data_first = pd.Timestamp(self.available_dates[np.argmax(self.available_dates >=t)-1])
            t_data_second = pd.Timestamp(self.available_dates[np.argmax(self.available_dates >=t)])
            if self.agg:
                E_values_first = self.df_efficacies.loc[t_data_first, :, :, :,:][efficacy].to_numpy()
                E_first = np.reshape(E_values_first, (self.G,self.N,self.n_doses,self.n_VOCs))
                E_values_second = self.df_efficacies.loc[t_data_second, :, :, :, :][efficacy].to_numpy()
                E_second = np.reshape(E_values_second, (self.G,self.N,self.n_doses,self.n_VOCs))
            else:
                E_values_first = self.df_efficacies.loc[t_data_first, :, :, :][efficacy].to_numpy()
                E_first = np.reshape(E_values_first, (self.N,self.n_doses,self.n_VOCs))
                E_values_second = self.df_efficacies.loc[t_data_second, :, :, :][efficacy].to_numpy()
                E_second = np.reshape(E_values_second, (self.N,self.n_doses,self.n_VOCs))

            # linear interpolation
            E = E_first + (E_second - E_first) * (t - t_data_first).total_seconds() / (t_data_second - t_data_first).total_seconds()
            
        elif t > max(self.available_dates):
            # Take last available data point
            t_data = pd.Timestamp(max(self.available_dates))
            if self.agg:
                E_values = self.df_efficacies.loc[t_data, :, :, :, :][efficacy].to_numpy()
                E = np.reshape(E_values, (self.G,self.N,self.n_doses,self.n_VOCs))
            else:
                E_values = self.df_efficacies.loc[t_data, :, :, :][efficacy].to_numpy()
                E = np.reshape(E_values, (self.N,self.n_doses,self.n_VOCs))

        return (1-E)
    
    def e_s(self, t, states, param):
        return self.__call__(t, 'e_s')
    
    def e_i(self, t, states, param):
        return self.__call__(t, 'e_i')
    
    def e_h(self, t, states, param):
        return self.__call__(t, 'e_h')

    ######################
    ## Helper functions ##
    ######################
    
    @staticmethod
    def convert_frequency_WMON(df):
        """Converts the frequency of the incidences dataframe to 'W-MON'. This is done to avoid using excessive computational resources."""
        groupers=[]
        for index_name in df.index.names:
            if index_name != 'date':
                groupers += [pd.Grouper(level=index_name)]
            else:
                groupers += [pd.Grouper(level='date', freq='W-MON')]
        df = df.groupby(groupers).sum()
        return df

    def compute_weighted_efficacy(self, df, df_cumsum, vaccine_params, waning):
        """ Computes the average protection of the vaccines for every dose and every VOC, based on the vaccination incidence in every age group (and spatial patch) and subject to exponential vaccine waning
        """ 

        VOCs = vaccine_params.index.get_level_values('VOC').unique()
        efficacies=vaccine_params.index.get_level_values('efficacy').unique()
    
        # Define output dataframe
        iterables=[]
        for index_name in df.index.names:
            iterables += [df.index.get_level_values(index_name).unique()]
        iterables.append(VOCs)
        names=list(df.index.names)
        names.append('VOC')
        index = pd.MultiIndex.from_product(iterables, names=names)
        df_efficacies = pd.DataFrame(index=index, columns=efficacies, dtype=float)
        
        # Computational loop
        dates=df.index.get_level_values('date').unique()
        age_classes=df.index.get_level_values('age').unique()
        doses=df.index.get_level_values('dose').unique() 
        
        if not 'NIS' in df.index.names:
            with tqdm(total=len(age_classes)*len(doses)) as pbar:
                for age_class in age_classes:
                    for dose in doses:
                        for date in dates:
                            cumsum = df_cumsum.loc[date, age_class, dose]
                            delta_t=[]
                            f=[]
                            w=pd.DataFrame(index=pd.MultiIndex.from_product([dates[dates<=date], VOCs]), columns=efficacies, dtype=float)
                            for inner_date in dates[dates<=date]:
                                # Compute fraction versus delta_t
                                delta_t.append((date-inner_date)/pd.Timedelta(days=1))
                                if cumsum != 0.0:
                                    f.append(df.loc[inner_date, age_class, dose]/cumsum)
                                else:
                                    if (date-inner_date)/pd.Timedelta(days=1) == 0:
                                        f.append(1.0)
                                    else:
                                        f.append(0)
                                # Compute weight
                                for VOC in VOCs:
                                    for efficacy in efficacies:
                                        waning_days = waning[dose]
                                        E_best = vaccine_params.loc[(VOC, dose, efficacy), 'best']
                                        E_waned = vaccine_params.loc[(VOC, dose, efficacy), 'waned']
                                        weight = self.exponential_waning((date-inner_date)/pd.Timedelta(days=1), waning_days, E_best, E_waned)
                                        w.loc[(inner_date, VOC), efficacy] = weight
                            for VOC in VOCs:
                                for efficacy in efficacies:
                                    # Compute efficacy
                                    df_efficacies.loc[(date, age_class, dose, VOC),efficacy] = np.dot(f,w.loc[(slice(None), VOC), efficacy])
                        pbar.update(1)
        else:
            NISs=df.index.get_level_values('NIS').unique()
            with tqdm(total=len(NISs)*len(age_classes)*len(doses)) as pbar:
                for age_class in age_classes:
                    for NIS in NISs:
                        for dose in doses:
                            for date in dates:
                                cumsum = df_cumsum.loc[date, NIS, age_class, dose]
                                delta_t=[]
                                f=[]
                                w=pd.DataFrame(index=pd.MultiIndex.from_product([dates[dates<=date], VOCs]), columns=efficacies, dtype=float)
                                for inner_date in dates[dates<=date]:
                                    # Compute fraction versus delta_t
                                    delta_t.append((date-inner_date)/pd.Timedelta(days=1))
                                    if cumsum != 0.0:
                                        f.append(df.loc[inner_date, NIS, age_class, dose]/cumsum)
                                    else:
                                        if (date-inner_date)/pd.Timedelta(days=1) == 0:
                                            f.append(1.0)
                                        else:
                                            f.append(0)
                                    # Compute weight
                                    for VOC in VOCs:
                                        for efficacy in efficacies:
                                            waning_days = waning[dose]
                                            E_best = vaccine_params.loc[(VOC, dose, efficacy), 'best']
                                            E_waned = vaccine_params.loc[(VOC, dose, efficacy), 'waned']
                                            weight = self.exponential_waning((date-inner_date)/pd.Timedelta(days=1), waning_days, E_best, E_waned)
                                            w.loc[(inner_date, VOC), efficacy] = weight
                                for VOC in VOCs:
                                    for efficacy in efficacies:
                                        # Compute efficacy
                                        df_efficacies.loc[(date, NIS, age_class, dose, VOC),efficacy] = np.dot(f,w.loc[(slice(None), VOC), efficacy])
                            pbar.update(1)
                                
        return df_efficacies

    @staticmethod
    def add_efficacy_no_vaccine(df):
        """Adds a dummy vaccine dose 'none' to the efficacy dataframe with zero efficacy
        """

        # Define a new dataframe with the desired age groups
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'dose':
                iterables += [df.index.get_level_values(index_name).unique()]
            else:
                iterables += [['none',] + list(df.index.get_level_values(index_name).unique())]
        index = pd.MultiIndex.from_product(iterables, names=df.index.names)
        new_df = pd.DataFrame(index=index, columns=df.columns, dtype=float)
        # Perform a join operation, remove the left columns and fillna with zero
        merged_df = new_df.join(df, how='left', lsuffix='_left', rsuffix='')
        merged_df = merged_df.drop(columns=['e_s_left', 'e_i_left', 'e_h_left']).fillna(0)

        return merged_df

    def age_conversion(self, df, desired_age_classes, agg):
        """
        A function to convert a dataframe of vaccine efficacies to another set of age groups `desired_age_classes` using demographic weighing
        
        Inputs
        ------
        
         df: pd.Series
            Pandas series containing the vaccination incidences, indexed using a pd.Multiindex
            Obtained using the function `covid19model.data.sciensano.get_public_spatial_vaccination_data()`
            Must contain 'date', 'age' and 'dose' as indices for the national model
            Must contain 'date', 'NIS', 'age', 'dose' as indices for the spatial model  
            
        desired_age_classes: pd.IntervalIndex
            Age groups you want the vaccination incidence to be formatted in
            Example: pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
        
        agg: str or None
            Spatial aggregation level. `None` for the national model, 'prov' for the provincial level, 'arr' for the arrondissement level
           
        Returns
        -------
        
        df_new: pd.DataFrame
            Same pd.DataFrame containing the vaccination incidences, but uses the age groups in `desired_age_classes`
        
        """

        from covid19model.data.utils import convert_age_stratified_property

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
            with tqdm(total=len(df.index.get_level_values('date').unique())*len(df.index.get_level_values('NIS').unique())) as pbar:
                for date in df.index.get_level_values('date').unique():
                    for NIS in df.index.get_level_values('NIS').unique():
                        for dose in df.index.get_level_values('dose').unique():
                            for VOC in df.index.get_level_values('VOC').unique():
                                new_df.loc[date, NIS, slice(None), dose, VOC] = convert_age_stratified_property(df.loc[date, NIS, slice(None), dose, VOC], desired_age_classes, agg=agg, NIS=NIS).values
                        pbar.update(1)
        else:
            with tqdm(total=len(df.index.get_level_values('date').unique())) as pbar:
                for date in df.index.get_level_values('date').unique():
                    for dose in df.index.get_level_values('dose').unique():
                        for VOC in df.index.get_level_values('VOC').unique():
                            new_df.loc[(date, slice(None), dose, VOC),:] = convert_age_stratified_property(df.loc[date, slice(None), dose, VOC], desired_age_classes).values
                    pbar.update(1)

        return new_df

    @staticmethod
    def exponential_waning(delta_t, waning_days, E_best, E_waned):
        """
        Function to compute the vaccine efficacy after delta_t days, accouting for exponential waning of the vaccine's immunity.

        Input
        -----
        days : float
            number of days after the novel vaccination
        waning_days : float
            number of days for the vaccine to wane (on average)
        E_best : float
            rescaling value related to the best possible protection by the currently injected vaccine
        E_waned : float
            rescaling value related to the vaccine protection after a waning period.

        Output
        ------
        E_eff : float
            effective rescaling value associated with the newly administered vaccine

        """

        if E_best == E_waned:
            return E_best
        # Exponential waning
        A0 = E_best
        k = -np.log((E_waned)/(E_best))/(waning_days)
        E_eff = A0*np.exp(-k*(delta_t))

        return E_eff
    
    @staticmethod
    def incidences_to_cumulative(df):
        """Computes the cumulative number of vaccines based using the incidences dataframe
        """
        levels = list(df.index.names)
        levels.remove("date")
        df_cumsum = df.groupby(by=levels).cumsum()
        df_cumsum = df_cumsum.reset_index().set_index(list(df.index.names)).squeeze().rename('CUMULATIVE')
        return df_cumsum
    
    @staticmethod
    def sum_oneshot_second(df):
        """ Sums vaccine dose 'B' and 'C' in the Sciensano dataset, corresponding to getting a second dose or getting the one-shot J&J vaccine
        """
        # Add dose 'B' and 'C' together
        df[df.index.get_level_values('dose') == 'B'] = df[df.index.get_level_values('dose') == 'B'].values + df[df.index.get_level_values('dose') == 'C'].values
        # Remove dose 'C'
        df = df.reset_index()
        df = df[df['dose'] != 'C']
        # Reformat dataframe
        df = df.set_index(df.columns[df.columns!='INCIDENCE'].to_list())
        df = df.squeeze()
        # Use more declaritive names for doses
        df.rename(index={'A':'partial', 'B':'full', 'E':'boosted'}, inplace=True)
        
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

        # partial, waned = (1/2) * partial, best (equals assumption that half-life is equal to waning days)
        df_new.loc[(slice(None), 'partial', slice(None)),'waned'] = 0.5*df_new.loc[(slice(None), 'partial', slice(None)),'best'].values

        # full, initial = partial, best
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
            CM = (mentality*(eff_home*np.ones(self.space_agg)[:, np.newaxis,np.newaxis]*self.Nc_all['home'] +
                    (eff_schools*school)[:, np.newaxis,np.newaxis]*self.Nc_all['schools'] +
                    (eff_work*values_dict['work'])[:,np.newaxis,np.newaxis]*self.Nc_all['work'] + 
                    (eff_rest*values_dict['transport'])[:,np.newaxis,np.newaxis]*self.Nc_all['transport'] + 
                    (eff_rest*values_dict['leisure'])[:,np.newaxis,np.newaxis]*self.Nc_all['leisure'] +
                    (eff_rest*values_dict['others'])[:,np.newaxis,np.newaxis]*self.Nc_all['others']) )

        else:
            if t < self.df_google_start:
                row = -self.df_google[0:7].mean()/100
            elif self.df_google_start <= t <= self.df_google_end:
                # Extract row at timestep t
                row = -self.df_google.loc[t]/100
            else:
                # Extract last 7 days and take the mean
                row = -self.df_google[-7:-1].mean()/100
            
            # Extract values
            values_dict={}
            for idx,place in enumerate(places_var):
                if place is None:
                    place = 1 - row[GCMR_names[idx]]
                values_dict.update({places_names[idx]: place})  

            # Construct contact matrix
            CM = (mentality*(eff_home*self.Nc_all['home'] + 
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

        # Assumption eff_schools = eff_work
        eff_schools=eff_work

        t = pd.Timestamp(t.date())
        # Convert compliance l to dates
        l1_days = pd.Timedelta(l1, unit='D')
        l2_days = pd.Timedelta(l2, unit='D')

        # Define key dates of first wave
        t1 = pd.Timestamp('2020-03-16') # start of lockdown
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
            return self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1, school=1)
        elif t1 < t <= t1 + l1_days:
            policy_old = self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1, school=1)
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
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=0.65*mentality, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t8, l2)
        elif t8 + l2_days < t <= t9:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=0.65*mentality, school=0)
        elif t9 < t <= t10:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=0.65*mentality, school=0)
        elif t10 < t <= t11:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1) 
        elif t11 < t <= t12:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t12 < t <= t13:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t13 < t <= t14:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)    
        elif t14 < t <= t15:
            return 1.20*self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t15 < t <= t16:
            return 1.20*self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality,school=1)
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
            l = (t32 - t31)/pd.Timedelta(days=1)
            r = (t32 - t31)/(t33 - t31)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=0)
            return self.ramp_fun(policy_old, policy_new, t, t31, l)
        elif t32 < t <= t33:
            l = (t33 - t32)/pd.Timedelta(days=1)
            r = (t33 - t32)/(t33 - t31)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)
            return self.ramp_fun(policy_old, policy_new, t, t32, l)
        elif t33 < t <= t34:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=0)
        elif t34 < t <= t35:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)                                                                                                                                 
        else:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, work=1, transport=0.7, leisure=1, others=1, school=0)    

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

        # Assumption eff_schools = eff_work
        eff_schools=eff_work

        t = pd.Timestamp(t.date())
        # Convert compliance l to dates
        l1_days = pd.Timedelta(l1, unit='D')
        l2_days = pd.Timedelta(l2, unit='D')

        # Define key dates of first wave
        t1 = pd.Timestamp('2020-03-16') # start of lockdown
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
        t18 = pd.Timestamp('2021-05-15') # Start of relaxations
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
        # Option 1: all relative to mentality
        mentality_summer_2020_lockdown = mentality*np.array([1.25, 1, # F
                                                            1.25, # W
                                                            1.25, # Bxl
                                                            0.50, 1.25, # F
                                                            2, 2, # W
                                                            1, # F
                                                            1, 1]) # W
        # Option 2: either mentality or one
        #mentality_summer_2020_lockdown = np.array([1*mentality, 1*mentality, # F
        #                                            1, # W
        #                                            1*mentality, # Bxl
        #                                            0.75*mentality, 1*mentality, # F
        #                                            1, 1, # W
        #                                            1*mentality, # F
        #                                            1*mentality, 1*mentality]) # W

        co_F = 1
        co_W = 1
        co_Bxl = 1
        mentality_before_relaxation_flanders_2021 = np.array([co_F, co_F, # F
                                                co_W, # W
                                                co_Bxl, # Bxl
                                                co_F, co_F, # F
                                                co_W, co_W, # W
                                                co_F, # F
                                                co_W, co_W]) # W

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
            return self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1, school=1) 
        elif t1 < t <= t1 + l1_days:
            policy_old = self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1, school=1)
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
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=0.65*mentality, school=0)
            return self.ramp_fun(policy_old, policy_new, t, t8, l2)
        elif t8 + l2_days < t <= t9:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=0.65*mentality, school=0)
        elif t9 < t <= t10:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=0.65*mentality, school=0)
        elif t10 < t <= t11:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1) 
        elif t11 < t <= t12:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t12 < t <= t13:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
        elif t13 < t <= t14:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)    
        elif t14 < t <= t15:
            mat = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
            mat[idx_F,:,:] *= 1.0
            mat[idx_Bxl,:,:] *= 1.15
            mat[idx_W,:,:] *= 1.10
            mat[2,:,:] *= 1/1.10
            return mat 
        elif t15 < t <= t16:
            mat = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)
            mat[idx_F,:,:] *= 1.0
            mat[idx_Bxl,:,:] *= 1.15
            mat[idx_W,:,:] *= 1.10
            mat[2,:,:] *= 1/1.10
            return mat 
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
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_before_relaxation_flanders_2021), school=0)
        elif t21 < t <= t22:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_before_relaxation_flanders_2021), school=0.7)
        elif t22 < t <= t23:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_before_relaxation_flanders_2021), school=1)    
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
            l = (t31 - t30)/pd.Timedelta(days=1)
            r = (t31 - t30)/(t32 - t30)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=0)
            return self.ramp_fun(policy_old, policy_new, t, t30, l)
        elif t31 < t <= t32:
            l = (t32 - t31)/pd.Timedelta(days=1)
            r = (t32 - t31)/(t32 - t30)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality + r*(1-mentality), school=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1, school=1)
            return self.ramp_fun(policy_old, policy_new, t, t31, l)
        elif t32 < t <= t33:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=0)
        elif t33 < t <= t34:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality, school=1)                                                                                                                                 
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
            t1 = pd.Timestamp('2020-03-16') # start of lockdown
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
                return self.__call__(t, eff_home=0, eff_schools=0, eff_work=1, eff_rest=0, mentality=1, school=0) 
            elif t1 < t <= t1 + l1_days:
                policy_old = self.__call__(t, eff_home=0, eff_schools=0, eff_work=1, eff_rest=0, mentality=1, school=0) 
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