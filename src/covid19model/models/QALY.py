import os 
import numpy as np
import pandas as pd
from covid19model.data.utils import convert_age_stratified_property

class QALY_model():

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
        self.life_table = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/QALY_model/Life_table_Belgium_2019.csv'),sep=';',index_col=0)
        # Compute the vector mu_x and append to life table
        self.life_table['mu_x']= self.compute_death_rate(self.life_table['q_x'])     
        # Define mu_x explictly to enhance readability of the code
        self.mu_x = self.life_table['mu_x']
        # Define overall Belgian QoL scores (source: ...)
        self.QoL_Belgium = pd.Series(index=default_age_bins, data=[0.85, 0.85, 0.84, 0.83, 0.805, 0.78, 0.75, 0.72, 0.72])

    def compute_death_rate(self, q_x):
        """ A function to compute the force of mortality (instantaneous death rate at age x)

        Parameters
        ----------
        q_x : list or np.array
            Probability of dying between age x and age x+1. 

        Returns
        -------
        mu_x : np.array
            Instantaneous death rage at age x

        """

        # Pre-allocate
        mu_x = np.zeros(len(q_x))
        # Compute first entry
        mu_x[0] = -np.log(1-q_x[0])
        # Loop over remaining entries
        for age in range(1,len(q_x)):
            mu_x[age] = -0.5*(np.log(1-q_x[age])+np.log(1-q_x[age-1]))
        return mu_x

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

        # Compute survival function
        S_x = self.survival_function(SMR)
        # First compute inner sum
        tmp = np.zeros(len(S_x))
        for age in range(len(S_x)-1):
            tmp[age] = 0.5*(S_x[age]+S_x[age+1])
        # Then sum from x to the end of the table to obtain life expectancy
        LE_x = np.zeros(len(S_x))
        for x in range(len(S_x)):
            LE_x[x] = np.sum(tmp[x:])
        # Post-allocate to pd.Series object
        LE_x = pd.Series(index=range(len(self.mu_x)), data=LE_x)
        LE_x.index.name = 'x'
        return LE_x

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

    def append_acute_QALY_losses(self, out, binned_QALY_df):

        # https://link.springer.com/content/pdf/10.1007/s40271-021-00509-z.pdf
        
        ##################
        ## Mild disease ##
        ##################

        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4690729/ --> Table A2 --> utility weight = 0.659
        # https://www.valueinhealthjournal.com/article/S1098-3015(21)00034-6/fulltext --> Table 2 --> 1-0.43=0.57
        out['QALYs_mild'] = out['M']*np.expand_dims(np.expand_dims((self.QoL_df['Belgium']-0.659)/365,axis=1),axis=0)

        #####################
        ## Hospitalization ##
        #####################

        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4690729/ --> Table A2 --> utility weight = 0.514
        # https://www.valueinhealthjournal.com/article/S1098-3015(21)00034-6/fulltext --> Table 2 --> 1-0.50 = 0.50
        out['QALYs_cohort'] = out['C']*np.expand_dims(np.expand_dims((self.QoL_df['Belgium']-0.50)/365,axis=1),axis=0) + out['C_icurec']*np.expand_dims(np.expand_dims((self.QoL_df['Belgium']-0.50)/365,axis=1),axis=0)
        
        # https://www.valueinhealthjournal.com/article/S1098-3015(21)00034-6/fulltext --> Table 2 --> 1-0.60 = 0.40
        out['QALYs_ICU'] = out['ICU']*np.expand_dims(np.expand_dims((self.QoL_df['Belgium']-0.40)/365,axis=1),axis=0)

        ###########
        ## Death ##
        ###########
        m_C_nt = 0.4
        m_ICU_nt = 0.8
        out['QALYs_death'] = out['D']*np.expand_dims(np.expand_dims(binned_QALY_df['D'],axis=1),axis=0)
        out['QALYs_treatment'] = (m_C_nt*out['R_C'] + m_ICU_nt*out['R_C'])*np.expand_dims(np.expand_dims(binned_QALY_df['R'],axis=1),axis=0)
        return out

def lost_QALYs_hospital_care (reduction,granular=False):
    
    """
    This function calculates the expected number of QALYs lost due to a given
    percentage reduction in regular (non Covid-19 related) hospital care. 
    
    The calculation is an approximation based on the reported hospital costs in Belgium per 
    disease group and the average cost per QALY gained per disease group (calculated from cost-effectiveness 
    thresholds reported for the Netherlands)
    
    Parameters
    ----------
    reduction: np.array
        Percentage reduction in hospital care. if granular = True, reduction per disease group
    
    granular: bool
        If True, calculations are performed per disease group. If False, calculations are performed
        on an average basis
    
    Returns
    -------
    lost_QALYs float or pd.DataFrame
        Total number of QALYs lost per day caused by a given reduction in hospital care. 
        if granular = True, results are given per disease group
    
    """
   
    # Import hospital care cost per disease group and cost per QALY
    #cost per qauly (EUR), total spent (mill EUR) 
    hospital_data=pd.read_excel("../../data/interim/QALY_model/hospital_data_qalys.xlsx", sheet_name='hospital_data')
        
    # Average calculations
     
    if granular == False:  
        lost_QALYs = reduction*(hospital_data['total_spent']*1e6/hospital_data['cost_per_qaly']).sum()/365
    else:
        
        data_per_disease=pd.DataFrame(columns=['disease_group','gained_qalys','lost_qalys'])
        data_per_disease['disease_group']=hospital_data['disease_group']
        #Number of QALYs gained per year per disease group
        data_per_disease['gained_qalys']=hospital_data['total_spent']*1000000/hospital_data['cost_per_qaly']
        # Number of QALYs lost per year per disease group
        data_per_disease['lost_qalys']=reduction*data_per_disease['gained_qalys']/365
        lost_QALYs=data_per_disease.copy().drop(columns=['gained_qalys'])
    return lost_QALYs