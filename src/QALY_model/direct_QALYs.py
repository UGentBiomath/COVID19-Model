import os 
import numpy as np
import pandas as pd
from covid19_DTM.data.utils import convert_age_stratified_property
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm
from covid19_DTM.data.utils import construct_initN
import xarray as xr

class life_table_QALY_model():

    def __init__(self, comorbidity_parameters=None):

        default_age_bins = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
        # Build dataframe with default comorbidity_parameters (SMR=1, delta_QoL=0 for all ages)
        iterables = [default_age_bins,['SMR','delta_QoL']]
        index = pd.MultiIndex.from_product(iterables, names=['age_group', 'metric'])
        default_comorbidity_parameters = pd.DataFrame(index=index, columns=['BE',])
        default_comorbidity_parameters.columns.name = 'population'
        default_comorbidity_parameters.loc[(slice(None),'SMR'),:] = 1
        default_comorbidity_parameters.loc[(slice(None),'delta_QoL'),:] = 0

        # Append additional data of comorbid populations
        if isinstance(comorbidity_parameters, pd.DataFrame):

            # Input checks
            if len(comorbidity_parameters.index.names) != 2:
                raise ValueError("Invalid indices provided for input 'comorbidity_parameters': {0}. Only 'age_groups' and 'metric' can be used as index names.".format(list(comorbidity_parameters.index.names)))
            for name in comorbidity_parameters.index.names:
                if name not in ['age_group', 'metric']:
                    raise ValueError("Invalid index name '{0}' for input 'comorbidity_parameters'. Only 'age_groups' and 'metric' can be used as index names.".format(name))
            if not isinstance(comorbidity_parameters.index.get_level_values('age_group'), pd.IntervalIndex):
                    raise ValueError("Index 'age_groups' of input 'comorbidity_parameters' must be of type IntervalIndex")
            if len(comorbidity_parameters.index.get_level_values('metric').unique()) != 2:
                raise ValueError("Invalid metrics provided for input 'comorbidity_parameters': {0}. Only comorbidity metrics permitted are 'SMR' and 'delta_QoL'".format(list(comorbidity_parameters.index.get_level_values('metric').unique())))
            for name in comorbidity_parameters.index.get_level_values('metric').unique():
                if name not in default_comorbidity_parameters.index.get_level_values('metric').unique():
                    raise ValueError("Invalid comorbidity metric '{0}' for input 'comorbidity_parameters'. Only comorbidity metrics permitted are 'SMR' and 'delta_QoL'.".format(name))
            if 'BE' in list(comorbidity_parameters.columns.values):
                raise ValueError("The population name provided in input 'comorbidity_parameters' must not equal 'BE'.")
            # Age conversion
            tmp_comorbidity_parameters = pd.DataFrame(index=default_comorbidity_parameters.index, columns=comorbidity_parameters.columns)
            for metric in comorbidity_parameters.index.get_level_values('metric').unique():
                for population in comorbidity_parameters.columns:
                    tmp_comorbidity_parameters.loc[(slice(None), metric), population] = convert_age_stratified_property(comorbidity_parameters[population].loc[slice(None), metric], default_comorbidity_parameters.index.get_level_values('age_group').unique()).values
            comorbidity_parameters = tmp_comorbidity_parameters
            # Merge
            self.comorbidity_parameters = pd.concat([default_comorbidity_parameters,comorbidity_parameters], axis=1)

        else:
            if not comorbidity_parameters:
                self.comorbidity_parameters = default_comorbidity_parameters
            else:
                raise ValueError("Invalid input type '{0}' for input 'comorbidity_parameters'. Input must of type 'pd.DataFrame'.".format(type(comorbidity_parameters)))

        # Define absolute path
        abs_dir = os.path.dirname(__file__)
        # Import life table (q_x)
        self.life_table = pd.read_csv(os.path.join(abs_dir, '../../data/QALY_model/interim/life_table_model/Life_table_Belgium_2019.csv'),sep=',',index_col=0)
        # Define mu_x explictly to enhance readability of the code
        self.mu_x = self.life_table['mu_x']
        # Define overall Belgian QoL scores (source: ...)
        #self.QoL_Belgium = pd.Series(index=default_age_bins, data=[0.85, 0.85, 0.84, 0.83, 0.805, 0.78, 0.75, 0.72, 0.72])
        age_bins = pd.IntervalIndex.from_tuples([(15,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,105)], closed='left')
        self.QoL_Belgium = pd.Series(index=age_bins, data=[0.85, 0.85, 0.82, 0.78, 0.78, 0.78, 0.66])
        self.QoL_Belgium_func = self.fit_QoL_data()

    def fit_QoL_data(self):
        """ A function to fit an exponential function to the QoL data

        Parameters
        ----------

        self.QoL_Belgium: pd.DataFrame
            Average Quality-of-Life scores of the Belgian population.

        Returns
        -------

        QoL_Belgium_func : func
            fitted exponential function
        """
        # exponential function to smooth binned Qol scores
        QoL_Belgium_func = lambda x,a,b:max(self.QoL_Belgium)-a*x**b

        # objective function to fit exponential function
        def SSE_of_means(theta,QoL_Belgium):
            a,b = theta

            y = QoL_Belgium.values
            y_model = []
            for index in QoL_Belgium.index:
                left = index.left
                right = index.right
                w = right-left
                mean = quad(QoL_Belgium_func,left,right,args=(a,b))[0]/w
                y_model.append(mean)

            return sum((y_model-y)**2)

        # fit exponential function
        sol = minimize(SSE_of_means,x0=(0.00001,2),args=(self.QoL_Belgium))
        a,b = sol.x
        QoL_Belgium_func = lambda x:max(self.QoL_Belgium)-a*x**b
        return QoL_Belgium_func

    def survival_function(self, SMR=1):
        """ A function to compute the probability of surviving until age x

        Parameters
        ----------
        self.mu_x : list or np.array
            Instantaneous death rage at age x 

        SMR : float
            "Standardized mortality ratio" (SMR) is the ratio of observed deaths in a study group to expected deaths in the general population.
            An SMR of 1 corresponds to an average life expectancy, an increase in SMR shortens the expected lifespan.

        Returns
        -------
        S_x : pd.Series
            Survival function, i.e. the probability of surviving up until age x
        """
        # Pre-allocate as np.array
        S_x = np.zeros(len(self.mu_x))
        # Survival rate at age 0 is 100%
        S_x[0] = 1
        # Loop
        for age in range(1,len(self.mu_x)):
            S_x[age] = S_x[age-1]*np.exp(-SMR*self.mu_x[age])
        # Post-allocate as pd.Series object
        S_x = pd.Series(index=range(len(self.mu_x)), data=S_x)
        S_x.index.name = 'x'
        return S_x

    def life_expectancy(self,SMR=1):
        """ A function to compute the life expectancy at age x

        Parameters
        ----------

        SMR : float
            "Standardized mortality ratio" (SMR) is the ratio of observed deaths in a study group to expected deaths in the general population.
            An SMR of 1 corresponds to an average life expectancy, an increase in SMR shortens the expected lifespan.

        Returns
        -------

        LE_x : pd.Series
            Life expectancy at age x
        """

        #calculate q_x
        q_x = (1- np.exp(-SMR*self.mu_x)).values
        q_x[-1] = 1

        l_x = np.zeros(len(q_x))
        l_x[0] = 10^5
        for x in range(1,len(q_x)):
            # Compute number of survivors at age x
            l_x[x] = l_x[x-1]-l_x[x-1]*q_x[x-1]

        # Compute inner sum
        tmp = np.zeros(len(l_x))
        for age in range(len(l_x)-1):
            tmp[age] = 0.5*(l_x[age]+l_x[age+1])
        tmp[-1] = 0.5*l_x[-1]

        LE_x = np.zeros(len(q_x))
        for x in range(len(q_x)):
            # Then normalized sum from x to the end of the table to obtain life expectancy
            LE_x[x] = np.sum(tmp[x:])/l_x[x]
            # Post-allocate to pd.Series object
            LE_x = pd.Series(index=range(len(q_x)), data=LE_x)
            LE_x.index.name = 'x'
        return LE_x
    
    def compute_QALY_D_x(self, SMR=1, r=0.03):
        """ A function to compute the QALY loss upon death at age x

        Parameters
        ----------

        SMR : float
            "Standardized mortality ratio" (SMR) is the ratio of observed deaths in a study group to expected deaths in the general population.
            An SMR of 1 corresponds to an average life expectancy, an increase in SMR shortens the expected lifespan.
        r : float
            discount rate to discount QALYs to occur in the future

        Returns
        -------

        QALY_D_x : pd.Series
        """
        func_to_integrate = lambda i,x,r: self.QoL_Belgium_func(i)/(1+r)**(i-x)

        LE_table = self.life_expectancy(SMR)    
        QALY_D = pd.Series(index=LE_table.index.rename('age'),dtype='float')
        for age in QALY_D.index:
            QALY_D[age] = quad(func_to_integrate,age,age+LE_table[age],args=(age,r))[0]
        return QALY_D
    
    def compute_QALE_x(self, population='BE', SMR_method='convergent'):
        """ A function to compute the quality-adjusted life expectancy at age x

        Parameters
        ----------

        self.mu_x : list or np.array
            Instantaneous death rage at age x 

        self.QoL_Belgium: pd.DataFrame
            Average Quality-of-Life scores of the Belgian population.

        self.comorbidity_parameters: pd.DataFrame
            Dataframe containing SMR and delta_QoL per age group of a user-defined comorbid population.
            By default initialized for population 'BE' with SMR=1 and delta_QoL = 0 for every age group.

        population : string
            Choice of QoL scores and SMR of a comorbid population defined by the user.
            Default option 'BE' uses QoL scores of the Belgian population and an SMR of one, corresponding to not accouting for additional comorbidities.

        SMR_method : string
            Choice of SMR model for remainder of life. Valid options are 'convergent' and 'constant'.
            'convergent' : the SMR gradually converges to SMR=1 by the end of the subjects life.
            If a person is expected to be healthy (SMR<1), this method represents the heuristic that we do not know how healthy this person will be in the future.
            We just assume his "healthiness" converges back to the population average as time goes by.
            'constant' : the SMR used to compute the QALEs remains equal to the expected value for the rest of the subjects life.
            If a person is expected to be healthy (SMR<1), this method assumes the person will remain equally healthy for his entire life.

        Returns
        -------

        QALE_x : pd.Series
            Quality-adjusted ife expectancy at age x
        """

        # Adjust Quality-of-life scores and SMRs
        QoL_population = self.QoL_Belgium + self.comorbidity_parameters.loc[(slice(None),'delta_QoL'), population].values
        SMR_population = self.comorbidity_parameters.loc[(slice(None), 'SMR'), population]

        # Pre-allocate results
        QALE_x = np.zeros(len(self.mu_x))
        # Loop over x
        for x in range(len(self.mu_x)):
            # Pre-allocate dQALY
            dQALE = np.zeros([len(self.mu_x)-x-1])
            # Set age-dependant utility weights to lowest possible
            j=0
            age_limit=QoL_population.index[j].right - 1
            QoL_x=QoL_population.values[j]
            # Calculate the SMR at age x
            if ((SMR_method == 'convergent')|(SMR_method == 'constant')):
                k = np.where(QoL_population.index.contains(x))[0][-1]
                age_limit = QoL_population.index[k].right - 1
                SMR_x = SMR_population.values[k]
            # Loop over years remaining after year x
            for i in range(x,len(self.mu_x)-1):
                # Find the right age bin
                j = np.where(QoL_population.index.contains(i))[0][-1]
                age_limit = QoL_population.index[j].right - 1
                # Choose the right QoL score
                QoL_x = QoL_population.values[j]
                # Choose the right SMR
                if SMR_method == 'convergent':
                    # SMR gradually converges to one by end of life
                    SMR = 1 + (SMR_x-1)*((len(self.mu_x)-1-i)/(len(self.mu_x)-1-x))
                elif SMR_method == 'constant':
                    # SMR is equal to SMR at age x for remainder of life
                    SMR = SMR_x
                # Compute the survival function
                S_x = self.survival_function(SMR)
                # Then compute the quality-adjusted life years lived between age x and x+1
                dQALE[i-x] = QoL_x*0.5*(S_x[i] + S_x[i+1])
            # Sum dQALY to obtain QALY_x
            QALE_x[x] = np.sum(dQALE)
        # Post-allocate to pd.Series object
        QALE_x = pd.Series(index=range(len(self.mu_x)), data=QALE_x)
        QALE_x.index.name = 'x'
        return QALE_x

    def compute_QALY_x(self, population='BE', r=0.03, SMR_method='convergent'):

        """ A function to compute the quality-adjusted life years remaining at age x

        Parameters
        ----------

        self.mu_x : list or np.array
            Instantaneous death rage at age x 

        self.QoL_Belgium: pd.DataFrame
            Average Quality-of-Life scores of the Belgian population.

        self.comorbidity_parameters: pd.DataFrame
            Dataframe containing SMR and delta_QoL per age group of a user-defined comorbid population.
            By default initialized for population 'BE' with SMR=1 and delta_QoL = 0 for every age group.
        population : string
            Choice of QoL scores and SMR of a comorbid population defined by the user.
            Default option 'BE' uses QoL scores of the Belgian population and an SMR of one, corresponding to accounting for average comorbidity.

        r : float
            Discount rate (default 3%)

        SMR_method : string
            Choice of SMR model for remainder of life. Valid options are 'convergent' and 'constant'.
            'convergent' : the SMR gradually converges to SMR=1 by the end of the subjects life.
            If a person is expected to be healthy (SMR<1), this method represents the heuristic that we do not know how healthy this person will be in the future.
            We just assume his "healthiness" converges back to the population average as time goes by.
            'constant' : the SMR used to compute the QALEs remains equal to the expected value for the rest of the subjects life.
            If a person is expected to be healthy (SMR<1), this method assumes the person will remain equally healthy for his entire life.

        Returns
        -------

        QALY_x : pd.Series
            Quality-adjusted life years remaining at age x
        """

        # Adjust Quality-of-life scores and SMRs
        QoL_population = self.QoL_Belgium + self.comorbidity_parameters.loc[(slice(None),'delta_QoL'), population].values
        SMR_population = self.comorbidity_parameters.loc[(slice(None), 'SMR'), population]

        # Pre-allocate results
        QALY_x = np.zeros(len(self.mu_x))
        # Loop over x
        for x in range(len(self.mu_x)):
            # Pre-allocate dQALY
            dQALY = np.zeros([len(self.mu_x)-x-1])
            # Set age-dependant utility weights to lowest possible
            j=0
            age_limit=QoL_population.index[j].right -1
            QoL_x=QoL_population.values[j]
            # Calculate the SMR at age x
            if ((SMR_method == 'convergent')|(SMR_method == 'constant')):
                k = np.where(QoL_population.index.contains(x))[0][-1]
                age_limit = QoL_population.index[k].right - 1
                SMR_x = SMR_population.values[k]
            else:
                raise ValueError("Invalid SMR method")
            # Loop over years remaining after year x
            for i in range(x,len(self.mu_x)-1):
                # Find the right age bin
                j = np.where(QoL_population.index.contains(i))[0][-1]
                age_limit = QoL_population.index[j].right - 1
                # Choose the right QoL score
                QoL_x = QoL_population.values[j]
                # Choose the right SMR
                if SMR_method == 'convergent':
                    # SMR gradually converges to one by end of life
                    SMR = 1 + (SMR_x-1)*((len(self.mu_x)-1-i)/(len(self.mu_x)-1-x))
                elif SMR_method == 'constant':
                    # SMR is equal to SMR at age x for remainder of life
                    SMR = SMR_x
                # Compute the survival function
                S_x = self.survival_function(SMR)
                # Then compute the quality-adjusted life years lived between age x and x+1
                dQALY[i-x] = QoL_x*0.5*(S_x[i] + S_x[i+1])*(1+r)**(x-i)
            # Sum dQALY to obtain QALY_x
            QALY_x[x] = np.sum(dQALY)
        # Post-allocate to pd.Series object
        QALY_x = pd.Series(index=range(len(self.mu_x)), data=QALY_x)
        QALY_x.index.name = 'x'
        return QALY_x

    def bin_QALY_x(self, QALY_x, model_bins=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')):
        """ A function to bin the vector QALY_x according to the age groups in the COVID-19 SEIQRD

        Parameters
        ----------
        QALY_x : np.array
            Quality-adjusted life years remaining at age x

        model_bins : pd.IntervalIndex
            Desired age bins

        Returns
        -------
        QALY_binned: pd.Series
            Quality-adjusted life years lost upon death for every age bin of the COVID-19 SEIQRD model
        """

        # Pre-allocate results vector
        QALY_binned = np.zeros(len(model_bins))
        # Loop over model bins
        for i in range(len(model_bins)):
            # Map QALY_x to model bins
            QALY_binned[i] = np.mean(QALY_x[model_bins[i].left:model_bins[i].right-1])
        # Post-allocate to pd.Series object
        QALY_binned = pd.Series(index=model_bins, data=QALY_binned)
        QALY_binned.index.name = 'age_group'
        return QALY_binned

def bin_data(data, age_groups=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')):
    """ A function to bin data according to the age groups in the COVID-19 SEIQRD

        Parameters
        ----------
        data : pd.Series
            data to be binned (must contain index:age)

        model_bins : pd.IntervalIndex
            Desired age bins

        Returns
        -------
        data_binned: pd.Series
            data for every age bin of the COVID-19 SEIQRD model
        """
    
    level_names = list(data.index.names)
    age_idx = level_names.index('age')
    level_names[age_idx] = 'age_group'
    multi_index = pd.MultiIndex.from_product([age_groups if level == 'age_group' else data.index.get_level_values(level).unique() for level in level_names],names=level_names)
     # Pre-allocate new series
    data_binned = pd.Series(index = multi_index, dtype=float)
    # Extract demographics
    individuals_per_age_group = construct_initN(age_groups)
    individuals_per_age = construct_initN(None)
    # Loop over desired intervals
    for idx in multi_index:
        interval = idx[age_idx]
        result = []
        for age in range(interval.left, interval.right):
            temp_idx = list(idx)
            temp_idx[age_idx] = age
            temp_idx = tuple(temp_idx)

            try:
                result.append(individuals_per_age[age]/individuals_per_age_group[interval]*data.loc[temp_idx])
            except:
                result.append(0)
        data_binned[idx] = sum(result)
    return data_binned

def lost_QALYs(out, AD_non_hospitalised=False, SMR=1, r=0.03, draws=1000):
    """
    This function calculates the expected number of QALYs lost given
    the output of the pandemic model. 
    
    It add the lost QALYs to the given output.
    QALY_D = lost QALYs due COVID-19 deaths
    QALY_NH = lost QALYs due to long-COVID of non-hospitalised patients
    QALY_C = lost QALYs due to long-COVID of cohort patients
    QALY_ICU = lost QALYs due to long-COVID of ICU patients
    
    Parameters
    ----------
    out: xarray
        Output of the pandemic model
    
    AD_non_hospitalised: bool
        If False, it is assumed non-hospitalised patients does not suffer from AD
    
    SMR: float
        Standardised mortality ratio

    r: float
        Discount factor.

    draws: int/float
        Number of draws from distribution of QALY losses due to long-COVID

    Returns
    -------
    out_sup xarray
        Out supplemented with the lost QALYs
    
    Attention
    ---------

    Changing the discount factor is currently only applied to the COVID-19 deaths.
    """

    # Enlarge out to contain at least 200 draws
    if 'draws' not in out.dims:
        out_enlarged = out.expand_dims(dim={'draws':draws})
    else:
        sim_draws = out.dims['draws']
        out_enlarged = xr.concat([out]*int(np.ceil(draws/sim_draws)), dim='draws')

    if AD_non_hospitalised:
        hospitalisation_groups = ['Non-hospitalised','Hospitalised (no IC)','Hospitalised (IC)']
    else:
        hospitalisation_groups = ['Non-hospitalised (no AD)','Hospitalised (no IC)','Hospitalised (IC)']

    # Load average QALY losses
    abs_dir = os.path.dirname(__file__)
    rel_dir = '../../data/QALY_model/interim/long_COVID/'
    file_name = f'average_QALY_losses_per_age_group_SMR{SMR*100:.0f}.csv'
    average_QALY_losses = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1])
    age_groups = average_QALY_losses.index.get_level_values('age_group').unique()

    # Multiply average QALY loss with number of patients
    hospitalisation_abbreviations = ['NH','C','ICU']
    for hospitalisation,hospitalisation_abbreviation in zip(hospitalisation_groups,hospitalisation_abbreviations):

        mean_QALY_losses = []
        for draw in range(out_enlarged.dims['draws']):
            sample = []
            sample_0 = np.random.normal(average_QALY_losses['mean'][hospitalisation,age_groups[0]],
                                        average_QALY_losses['sd'][hospitalisation,age_groups[0]])
            sample.append(sample_0)
            q = norm.cdf(sample_0,loc=average_QALY_losses['mean'][hospitalisation,age_groups[0]],
                         scale = average_QALY_losses['sd'][hospitalisation,age_groups[0]])
            
            for age_group in age_groups[1:]:
                sample.append(norm.ppf(q,loc=average_QALY_losses['mean'][hospitalisation,age_group],
                                       scale=average_QALY_losses['sd'][hospitalisation,age_group]))
            mean_QALY_losses.append(sample)
        mean_QALY_losses = np.array(mean_QALY_losses)[:,np.newaxis,:,np.newaxis]
        out_enlarged[f'QALY_{hospitalisation_abbreviation}'] = out_enlarged[hospitalisation_abbreviation+'_R_in'].cumsum(dim='date')*mean_QALY_losses

    # Calculate QALY losses due COVID death
    Life_table = life_table_QALY_model()
    QALY_D_per_age = Life_table.compute_QALY_D_x(SMR=SMR, r=r)
    QALY_D_per_age_group = bin_data(QALY_D_per_age)

    out_enlarged['QALY_D'] = out_enlarged['D']*np.array(QALY_D_per_age_group)[np.newaxis,np.newaxis,:,np.newaxis]

    return out_enlarged