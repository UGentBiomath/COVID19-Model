import numpy as np
import pandas as pd
import os 

class QALY_model():

    def __init__(self, comorbidity_distribution):
        self.comorbidity_distribution = comorbidity_distribution
        # Define absolute path
        abs_dir = os.path.dirname(__file__)
        # Import life table (q_x)
        self.life_table = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/QALYs/Life_table_Belgium_2019.csv'),sep=';',index_col=0)
        # Compute the vector mu_x and append to life table
        self.life_table['mu_x']= self.compute_death_rate(self.life_table['q_x'])     
        # Define mu_x explictly to enhance readability of the code
        self.mu_x = self.life_table['mu_x'] 
        # Load comorbidity QoL scores for the Belgian population from Lisa Van Wilder
        QoL_Van_Wilder=pd.read_excel(os.path.join(abs_dir,"../../../data/interim/QALYs/De_Wilder_QoL_scores.xlsx"),index_col=0,sheet_name='QoL_scores')
        QoL_Van_Wilder.columns = ['0','1','2','3+']
        idx = QoL_Van_Wilder.index.values
        idx[-1] = '80+'
        QoL_Van_Wilder.index = idx
        self.QoL_Van_Wilder = QoL_Van_Wilder
        # Compute the QoL scores of the studied population
        self.QoL_df = self.build_comorbidity_QoL(self.comorbidity_distribution, QoL_Van_Wilder)
        # Load comorbidity SMR estimates
        SMR_pop_df=pd.read_excel(os.path.join(abs_dir,"../../../data/interim/QALYs/De_Wilder_QoL_scores.xlsx"), index_col=0, sheet_name='SMR')
        SMR_pop_df.columns = ['0','1','2','3+']
        SMR_pop_df.index = idx
        self.SMR_pop_df = SMR_pop_df
        # Compute the SMR of the studied population
        self.SMR_df = self.build_comorbidity_SMR(self.comorbidity_distribution, SMR_pop_df)

    def build_comorbidity_SMR(self, comorbidity_distribution, population_SMR):
        """ A function to compute the Standardized Mortality Ratios (SMRs) in a studied population, based on the comorbidity distribution of the studied population and the comorbidity distribution of the Belgian population

        Parameters
        ----------
        comorbidity_distribution : pd.Dataframe
            A dataframe containing the studied population fraction with x comorbidities.
            This dataframe is the input of the comorbidity-QALY model. The studied population are usually recovered or deceased COVID-19 patients in hospitals.
            The dataframe must have te age group as its index and make use of a pandas multicolumn, where the first level denotes the population (usually R or D, but the code is written to use n populations). 
            The second level denotes the number of comorbidities, which must be equal to 0, 1, 2 or 3+.

        population_SMR : pd.Dataframe
            A dataframe containing the age-stratified SMRs for individuals with 0, 1, 2 or 3+ comorbidities in the general Belgian population.
            Computed using the comorbidity distributions for the general Belgian population obtained from Lisa Van Wilder, and the relative risk of dying by Charslon et. al (computation performed in MS Excel).

        Returns
        -------
        SMR_df: pd.DataFrame
            The weighted Standardized Mortality Ratios (SMRs) in the studied population. 
            An SMR > 1 indicates the studied population is less healthy than the general Belgian population.
        """

        # Extract names of populations
        populations = list(comorbidity_distribution.columns.get_level_values(0).unique())
        # Construct column name vector
        columns = ['group_limit','Belgium']
        columns.extend(populations)
        # Initialize dataframe
        df = pd.DataFrame(index=population_SMR.index, columns=columns)
        # Fill dataframe
        for idx,key in enumerate(df.index):
            for jdx,pop in enumerate(populations):
                df.loc[key][pop] = sum(population_SMR.loc[key]*comorbidity_distribution.loc[key][pop])
            df.loc[key]['Belgium'] = 1
        # Append group limit
        df['group_limit'] = [9, 19, 29, 39, 49, 59, 69, 79, 110]
        return df

    def build_comorbidity_QoL(self, comorbidity_distribution, comorbidity_QoL):
        """ A function to compute the QoL scores in a studied population, based on the comorbidity distribution of the studied population and the QoL scores for 0, 1, 2, 3+ comorbidities for the Belgian population

        Parameters
        ----------
        comorbidity_distribution : pd.Dataframe
            A dataframe containing the studied population fraction with x comorbidities.
            This dataframe is the input of the comorbidity-QALY model. The studied population are usually recovered or deceased COVID-19 patients in hospitals.
            The dataframe must have te age group as its index and make use of a pandas multicolumn, where the first level denotes the population (usually R or D, but the code is written to use n populations). 
            The second level denotes the number of comorbidities, which must be equal to 0, 1, 2 or 3+.

        comorbidity_QoL : pd.Dataframe
            A dataframe containing the age-stratified QoL scores for individuals with 0, 1, 2 or 3+ comorbidities in the general Belgian population.
            Obtained from Lisa Van Wilder.

        Returns
        -------
        QoL_df: pd.DataFrame
            The comorbidity-weighted QoL scores of the studied population. 
        """
        # Extract names of populations
        populations = list(comorbidity_distribution.columns.get_level_values(0).unique())
        # Construct column name vector
        columns = ['group_limit','Belgium']
        columns.extend(populations)
        # Initialize dataframe
        df = pd.DataFrame(index=comorbidity_QoL.index, columns=columns)
        # Fill dataframe
        for idx,key in enumerate(df.index):
            for jdx,pop in enumerate(populations):
                df.loc[key][pop] = sum(comorbidity_distribution.loc[key][pop]*comorbidity_QoL.loc[key])
        df['Belgium'] = [0.85, 0.85, 0.84, 0.83, 0.805, 0.78, 0.75, 0.72, 0.72]
        df['group_limit'] = [9, 19, 29, 39, 49, 59, 69, 79, 110]
        return df

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

    def compute_QALE_x(self, population='Belgium', SMR_method='convergent'):
        """ A function to compute the quality-adjusted life expectancy at age x

        Parameters
        ----------

        self.mu_x : list or np.array
            Instantaneous death rage at age x 

        self.SMR_df : pd.Dataframe
            "Standardized mortality ratio" (SMR) is the ratio of observed deaths in a study group to expected deaths in the general population.
            An SMR of 1 corresponds to an average life expectancy, an increase in SMR shortens the expected lifespan.
            
        self.QoL_df : pd.Dataframe
            Quality-of-life utility weights, as imported from `~/data/interim/QALYs/QoL_scores_Belgium_2018_v3.csv`.
            Must contain two columns: "group_limit" and "QoL_score"

        population : string
            Choice of QoL scores. Valid options are 'Belgium', 'R' and 'D'.
            'Belgium' : Overall QoL scores for the Belgian population by De Wilder et. al and an SMR=1 are applied (this represents average QALY loss)
            'R' : QoL scores and SMR for those recovering from COVID-19 in the hospital (most likely higher quality than average)
            'D' : QoL scores and SMR for those dying from COVID-19 in the hospital (most likely lower quality than average)

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

        # Pre-allocate results
        QALE_x = np.zeros(len(self.mu_x))
        # Loop over x
        for x in range(len(self.mu_x)):
            # Pre-allocate dQALY
            dQALE = np.zeros([len(self.mu_x)-x-1])
            # Set age-dependant utility weights to lowest possible
            j=0
            age_limit=self.QoL_df['group_limit'][j]
            QoL_x=self.QoL_df[population][j]
            # Calculate the SMR at age x
            if ((SMR_method == 'convergent')|(SMR_method == 'constant')):
                k=0
                while x > age_limit:
                    k += 1
                    age_limit = self.QoL_df['group_limit'][k]
                SMR_x = self.SMR_df[population][k]
            # Loop over years remaining after year x
            for i in range(x,len(self.mu_x)-1):
                # Find the right age bin
                while i > age_limit:
                    j += 1
                    age_limit = self.QoL_df['group_limit'][j]
                # Choose the right QoL score
                QoL_x = self.QoL_df[population][j]
                # Choose the right SMR
                if SMR_method == 'convergent':
                    # SMR gradually converges to one by end of life
                    SMR = 1 + (SMR_x-1)*((104-i)/(104-x))
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

    def compute_QALY_x(self, population='Belgium', r=0.03, SMR_method='convergent'):

        """ A function to compute the quality-adjusted life years remaining at age x

        Parameters
        ----------

        self.mu_x : list or np.array
            Instantaneous death rage at age x 

        self.SMR_df : pd.Dataframe
            "Standardized mortality ratio" (SMR) is the ratio of observed deaths in a study group to expected deaths in the general population.
            An SMR of 1 corresponds to an average life expectancy, an increase in SMR shortens the expected lifespan.
            
        self.QoL_df : pd.Dataframe
            Quality-of-life utility weights, as imported from `~/data/interim/QALYs/QoL_scores_Belgium_2018_v3.csv`.
            Must contain two columns: "group_limit" and "QoL_score"

        population : string
            Choice of QoL scores. Valid options are 'Belgium', 'R' and 'D'.
            'Belgium' : Overall QoL scores for the Belgian population by De Wilder et. al and an SMR=1 are applied (this represents average QALY loss)
            'R' : QoL scores and SMR for those recovering from COVID-19 in the hospital (most likely higher quality than average)
            'D' : QoL scores and SMR for those dying from COVID-19 in the hospital (most likely lower quality than average)

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

        # Pre-allocate results
        QALY_x = np.zeros(len(self.mu_x))
        # Loop over x
        for x in range(len(self.mu_x)):
            # Pre-allocate dQALY
            dQALY = np.zeros([len(self.mu_x)-x-1])
            # Set age-dependant utility weights to lowest possible
            j=0
            age_limit=self.QoL_df['group_limit'][j]
            QoL_x=self.QoL_df[population][j]
            # Calculate the SMR at age x
            if ((SMR_method == 'convergent')|(SMR_method == 'constant')):
                k=0
                while x > age_limit:
                    k += 1
                    age_limit = self.QoL_df['group_limit'][k]
                SMR_x = self.SMR_df[population][k]
            # Loop over years remaining after year x
            for i in range(x,len(self.mu_x)-1):
                # Find the right age bin
                while i > age_limit:
                    j += 1
                    age_limit = self.QoL_df['group_limit'][j]
                # Choose the right QoL score
                QoL_x = self.QoL_df[population][j]
                # Choose the right SMR
                if SMR_method == 'convergent':
                    # SMR gradually converges to one by end of life
                    SMR = 1 + (SMR_x-1)*((104-i)/(104-x))
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

    def bin_QALY_x(self, QALY_x, model_bins=['0-9','10-19','20-29','30-39','40-59','50-59','60-69','70-79','80+'], model_bins_UL=[9,19,29,39,49,59,69,79,110]):
        """ A function to bin the vector QALY_x according to the age groups in the COVID-19 SEIQRD

        Parameters
        ----------
        QALY_x : np.array
            Quality-adjusted life years remaining at age x

        Returns
        -------
        QALY_binned: pd.Series
            Quality-adjusted life years lost upon death for every age bin of the COVID-19 SEIQRD model
        """

        # Pre-allocate results vector
        QALY_binned = np.zeros(len(model_bins))
        # Pre-allocate first lower limit
        low_limit = 0
        # Loop over model bins
        for i in range(len(model_bins)):
            # Map QALY_x to model bins
            QALY_binned[i] = np.mean(QALY_x[low_limit:model_bins_UL[i]+1])
            low_limit=model_bins_UL[i]
        # Post-allocate to pd.Series object
        QALY_binned = pd.Series(index=model_bins, data=QALY_binned)
        QALY_binned.index.name = 'age_group'
        return QALY_binned

    def build_binned_QALY_df(self, r=0.03, SMR_method='convergent', model_bins=['0-9','10-19','20-29','30-39','40-59','50-59','60-69','70-79','80+'], model_bins_UL=[9,19,29,39,49,59,69,79,110]):
        # Extract names of populations
        populations = list(self.SMR_df.columns.get_level_values(0).unique())
        # Drop group limit
        populations.remove("group_limit")
        # Initialize empty dataframe
        df = pd.DataFrame()
        # Loop over populations
        for pop in populations:
            QALY_x = self.compute_QALY_x(population=pop, SMR_method='convergent',r=r)
            binned_QALY = self.bin_QALY_x(QALY_x, model_bins, model_bins_UL)
            binned_QALY.name = pop
            df = df.append(binned_QALY)
        return df.T

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
    hospital_data=pd.read_excel("../../data/interim/QALYs/hospital_data_qalys.xlsx", sheet_name='hospital_data')
        
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