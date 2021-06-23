import numpy as np
import pandas as pd

def compute_death_rate(q_x):
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

def compute_survival_function(mu_x, SMR=1):
    """ A function to compute the force of mortality (instantaneous death rate at age x)

    Parameters
    ----------
    mu_x : list or np.array
        Instantaneous death rage at age x 

    SMR : float
        "Standardized mortality ratio" (SMR) is the ratio of observed deaths in a study group to expected deaths in the general population.
        An SMR of 1 corresponds to an average life expectancy, an increase in SMR shortens the expected lifespan.

    Returns
    -------
    S_x : np.array
        Survival function, i.e. the probability of surviving up until age x
    """
    # Pre-allocate
    S_x = np.zeros(len(mu_x))
    # Survival rate at age 0 is 100%
    S_x[0] = 1
    # Loop
    for age in range(1,len(mu_x)):
        S_x[age] = S_x[age-1]*np.exp(-SMR*mu_x[age])
    return S_x

def compute_life_expectancy(S_x):
    """ A function to compute the life expectancy at age x

    Parameters
    ----------

    S_x : list or np.array
        Survival function

    Returns
    -------

    LE_x : np.array
        Life expectancy at age x
    """

    # First compute inner sum
    tmp = np.zeros(len(S_x))
    for age in range(len(S_x)-1):
        tmp[age] = 0.5*(S_x[age]+S_x[age+1])
    # Then sum from x to the end of the table to obtain life expectancy
    LE_x = np.zeros(len(S_x))
    for x in range(len(S_x)):
        LE_x[x] = np.sum(tmp[x:])
    return LE_x

def compute_QALE_x(mu_x, SMR_df, QoL_df, population='Belgium', SMR_method='convergent'):
    """ A function to compute the quality-adjusted life expectancy at age x

    Parameters
    ----------

    mu_x : list or np.array
        Instantaneous death rage at age x 

    SMR_df : pd.Dataframe
        "Standardized mortality ratio" (SMR) is the ratio of observed deaths in a study group to expected deaths in the general population.
        An SMR of 1 corresponds to an average life expectancy, an increase in SMR shortens the expected lifespan.
        
    QoL_df : pd.Dataframe
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

    QALE_x : np.array
        Quality-adjusted ife expectancy at age x
    """

    # Pre-allocate results
    QALE_x = np.zeros(len(mu_x))
    # Loop over x
    for x in range(len(mu_x)):
        # Pre-allocate dQALY
        dQALE = np.zeros([len(mu_x)-x-1])
        # Set age-dependant utility weights to lowest possible
        j=0
        age_limit=QoL_df['group_limit'][j]
        QoL_x=QoL_df['Van Wilder', population][j]
        # Calculate the SMR at age x
        if ((SMR_method == 'convergent')|(SMR_method == 'constant')):
            k=0
            while x > age_limit:
                k += 1
                age_limit = QoL_df['group_limit'][k]
            SMR_x = SMR_df[population][k]
        # Loop over years remaining after year x
        for i in range(x,len(mu_x)-1):
            # Find the right age bin
            while i > age_limit:
                j += 1
                age_limit = QoL_df['group_limit'][j]
            # Choose the right QoL score
            QoL_x = QoL_df['Van Wilder', population][j]
            # Choose the right SMR
            if SMR_method == 'convergent':
                # SMR gradually converges to one by end of life
                SMR = 1 + (SMR_x-1)*((104-i)/(104-x))
            elif SMR_method == 'constant':
                # SMR is equal to SMR at age x for remainder of life
                SMR = SMR_x
            # Compute the survival function
            S_x = compute_survival_function(mu_x, SMR)
            # Then compute the quality-adjusted life years lived between age x and x+1
            dQALE[i-x] = QoL_x*0.5*(S_x[i] + S_x[i+1])
        # Sum dQALY to obtain QALY_x
        QALE_x[x] = np.sum(dQALE)
    return QALE_x

def compute_QALY_x(mu_x, SMR_df, QoL_df, population='Belgium', r=0.03, SMR_method='convergent'):

    """ A function to compute the quality-adjusted life years remaining at age x

    Parameters
    ----------

    mu_x : list or np.array
        Instantaneous death rage at age x 

    SMR_df : pd.Dataframe
        "Standardized mortality ratio" (SMR) is the ratio of observed deaths in a study group to expected deaths in the general population.
        An SMR of 1 corresponds to an average life expectancy, an increase in SMR shortens the expected lifespan.
        
    QoL_df : pd.Dataframe
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

    QALY_x : np.array
        Quality-adjusted life years remaining at age x
    """

    # Pre-allocate results
    QALY_x = np.zeros(len(mu_x))
    # Loop over x
    for x in range(len(mu_x)):
        # Pre-allocate dQALY
        dQALY = np.zeros([len(mu_x)-x-1])
        # Set age-dependant utility weights to lowest possible
        j=0
        age_limit=QoL_df['group_limit'][j]
        QoL_x=QoL_df['Van Wilder', population][j]
        # Calculate the SMR at age x
        if ((SMR_method == 'convergent')|(SMR_method == 'constant')):
            k=0
            while x > age_limit:
                k += 1
                age_limit = QoL_df['group_limit'][k]
            SMR_x = SMR_df[population][k]
        # Loop over years remaining after year x
        for i in range(x,len(mu_x)-1):
            # Find the right age bin
            while i > age_limit:
                j += 1
                age_limit = QoL_df['group_limit'][j]
            # Choose the right QoL score
            QoL_x = QoL_df['Van Wilder', population][j]
            # Choose the right SMR
            if SMR_method == 'convergent':
                # SMR gradually converges to one by end of life
                SMR = 1 + (SMR_x-1)*((104-i)/(104-x))
            elif SMR_method == 'constant':
                # SMR is equal to SMR at age x for remainder of life
                SMR = SMR_x
            # Compute the survival function
            S_x = compute_survival_function(mu_x, SMR)
            # Then compute the quality-adjusted life years lived between age x and x+1
            dQALY[i-x] = QoL_x*0.5*(S_x[i] + S_x[i+1])*(1+r)**(x-i)
        # Sum dQALY to obtain QALY_x
        QALY_x[x] = np.sum(dQALY)
    return QALY_x

def compute_QALY_binned(QALY_x):
    """ A function to bin the vector QALY_x

    Parameters
    ----------

    Returns
    -------
    QALY_binned: np.array
        Quality-adjusted life years lost upon death for every age bin of the COVID-19 SEIQRD model
    """

    # Define bins: default 10 year decades of COVID-19 SEIRD
    model_bins = ['0-9','10-19','20-29','30-39','40-59','50-59','60-69','70-79','80+']
    model_bins_UL = [9,19,29,39,49,59,69,79,110]
    # Pre-allocate results vector
    QALY_binned = np.zeros(len(model_bins))
    # Pre-allocate first lower limit
    low_limit = 0
    # Loop over model bins
    for i in range(len(model_bins)):
        # Map QALY_x to model bins
        QALY_binned[i] = np.mean(QALY_x[low_limit:model_bins_UL[i]+1])
        low_limit=model_bins_UL[i]
    return QALY_binned

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

def append_acute_QALY_losses(out,QoL_df,lost_QALY_pp):

    # https://link.springer.com/content/pdf/10.1007/s40271-021-00509-z.pdf
    
    ##################
    ## Mild disease ##
    ##################

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4690729/ --> Table A2 --> utility weight = 0.659
    # https://www.valueinhealthjournal.com/article/S1098-3015(21)00034-6/fulltext --> Table 2 --> 1-0.43=0.57
    out['QALYs_mild'] = out['M']*np.expand_dims(np.expand_dims((QoL_df['QoL_score']-0.659)/365,axis=1),axis=0)

    #####################
    ## Hospitalization ##
    #####################

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4690729/ --> Table A2 --> utility weight = 0.514
    # https://www.valueinhealthjournal.com/article/S1098-3015(21)00034-6/fulltext --> Table 2 --> 1-0.50 = 0.50
    out['QALYs_cohort'] = out['C']*np.expand_dims(np.expand_dims((QoL_df['QoL_score']-0.50)/365,axis=1),axis=0) + out['C_icurec']*np.expand_dims(np.expand_dims((QoL_df['QoL_score']-0.50)/365,axis=1),axis=0)
    
    # https://www.valueinhealthjournal.com/article/S1098-3015(21)00034-6/fulltext --> Table 2 --> 1-0.60 = 0.40
    out['QALYs_ICU'] = out['ICU']*np.expand_dims(np.expand_dims((QoL_df['QoL_score']-0.40)/365,axis=1),axis=0)

    ###########
    ## Death ##
    ###########
    m_no_treatment = 0.40
    out['QALYs_death'] = out['D']*np.expand_dims(np.expand_dims(lost_QALY_pp,axis=1),axis=0)
    out['QALYs_treatment'] = (1-m_no_treatment)*out['R_hosp']*np.expand_dims(np.expand_dims(lost_QALY_pp,axis=1),axis=0)
    return out

def append_QALYs_gained_hospital_treatment(out,lost_QALY_pp):
    out['QALYs_treatment'] = (1-m_no_treatment)*out['R_hosp']*np.expand_dims(np.expand_dims(lost_QALY_pp,axis=1),axis=0)
    return out

def QALY2xarray(out,QALY_binned):
    """
    This function computes age-stratified QALYs lost due to COVID-19.
    To this end, the simulation output is multiplied with the number of QALYs lost per person.
    
    Parameters
    ----------
    out: xarray dataset
        Simulation output. A state with the number of deaths must be present as 'D'.

    lost_QALY_pp: pd.DataFrame
        QALYs lost when a person of a certain age (within the age bins of the model) dies.

    Returns
    -------
    out: xarray dataset
        Simulation output with new data variable 'QALYs lost'.
    
    """
    m_no_treatment = 0.40 # average ventilation percentage: https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(21)00045-6/fulltext
    # alternative: 90% of ICU patients is intubated
    return out.assign(variables={'QALY_death': out['D']*np.expand_dims(QALY_binned,axis=1),
                                'QALY_treatment': (1-m_no_treatment)*out['R_hosp']*np.expand_dims(QALY_binned,axis=1)})