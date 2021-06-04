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

def compute_survival_function(mu_x, s=1):
    """ A function to compute the force of mortality (instantaneous death rate at age x)

    Parameters
    ----------
    mu_x : list or np.array
        Instantaneous death rage at age x 

    s : float
        "Standardized mortality ratio" is the ratio of observed deaths in a study group to expected deaths in the general population.
        Generally s>=1, where an increase in s corresponds to a shorter expected life span, for instance due to comorbidity.

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
        S_x[age] = S_x[age-1]*np.exp(-s*mu_x[age])
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

def compute_QALE(S_x, QoL_df, q_CM=0):
    """ A function to compute the quality-adjusted life expectancy at age x

    Parameters
    ----------

    S_x : list or np.array
        Survival function

    QoL_df : pd.Dataframe
        Quality-of-life utility weights, as imported from `~/data/interim/QALYs/QoL_scores_Belgium_2018_v3.csv`.
        Must contain two columns: "group_limit" and "QoL_score"

    q_CM : float [0-1]
        Additional quality-of-life utility loss due to a comorbidity

    Returns
    -------

    QALE_x : np.array
        Quality-adjusted ife expectancy at age x
    """

    # Pre-allocate
    dQALE = np.zeros(len(S_x))
    QALE_x = np.zeros(len(S_x))
    # Set quality-of-life utility to first age group
    j=0
    QoL_x=QoL_df['QoL_score'][0]
    age_limit=QoL_df['group_limit'][0] 
    # Loop over ages
    for age in range(len(S_x)-1):
        # Check if we're in a new age bin
        if age > age_limit:
            j += 1
            QoL_x = QoL_df['QoL_score'][j]
            age_limit = QoL_df['group_limit'][j]
        # Then compute the quality-adjusted life years lived between age x and x+1
        dQALE[age] = (QoL_x-q_CM)*0.5*(S_x[age] + S_x[age+1])
    # Loop again over all ages
    for age in range(len(S_x)):
        QALE_x[age] = np.sum(dQALE[age:])

    return QALE_x

def compute_QALY(S_x, QoL_df, q_CM=0, r=0.03):

    """ A function to compute the quality-adjusted life years remaining at age x

    Parameters
    ----------

    S_x : list or np.array
        Survival function

    QoL_df : pd.Dataframe
        Quality-of-life utility weights, as imported from `~/data/interim/QALYs/QoL_scores_Belgium_2018_v3.csv`.
        Must contain two columns: "group_limit" and "QoL_score"

    q_CM : float [0-1]
        Additional quality-of-life utility loss due to a comorbidity

    r : float
        Discount rate (default 3%)

    Returns
    -------

    QALY_x : np.array
        Quality-adjusted life years remaining at age x
    """

    # Pre-allocate
    dQALY = np.zeros(len(S_x))
    QALY_x = np.zeros(len(S_x))
    # Set quality-of-life utility to first age group
    j=0
    QoL_x=QoL_df['QoL_score'][0]
    age_limit=QoL_df['group_limit'][0] 
    # Loop over ages
    for age in range(len(S_x)-1):
        # Check if we're in a new age bin
        if age > age_limit:
            j += 1
            QoL_x = QoL_df['QoL_score'][j]
            age_limit = QoL_df['group_limit'][j]
        # Then compute the quality-adjusted life years lived between age x and x+1
        dQALY[age] = (QoL_x-q_CM)*0.5*(S_x[age] + S_x[age+1])*(1+r)**(0-age)
    # Loop again over all ages
    for age in range(len(S_x)):
        QALY_x[age] = np.sum(dQALY[age:])
    return QALY_x

def compute_QALY_binned(QALY_x):
    # Define bins: default 10 year decades of COVID-19 SEIRD
    bins = ['0-9','10-19','20-29','30-39','40-59','50-59','60-69','70-79','80+']
    bins_UL = [9,19,29,39,49,59,69,89,110]
    #Calculate QALY_x per age bin of COVID-19 SEIRD model
    low_limit=0
    QALY_binned = np.zeros(len(bins))
    for i in range(len(bins)):
        QALY_binned[i] = np.mean(QALY_x[low_limit:bins_UL[i]+1])
        low_limit=bins_UL[i]

    return QALY_binned


#%%
def lost_QALY_hospital_care (reduction,granular=False):
    
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
        Total number of QALYs lost per year (!) caused by a given reduction in hospital care. 
        if granular = True, results are given per disease group
    
    """
   
    # Import hospital care cost per disease group and cost per QALY
    #cost per qauly (EUR), total spent (mill EUR) 
    hospital_data=pd.read_excel("../../data/interim/QALYs/hospital_data_qalys.xlsx", sheet_name='hospital_data')
        
    # Average calculations
     
    if granular == False:
        
        #total_cost=hospital_data['total_spent'].sum()*1000000 # Total amount spent in regular health care per year
        #avg_cost_per_qaly=hospital_data['cost_per_qaly'].mean() # Average cost per QALY gained
        #total_gained_QALYs=total_cost/avg_cost_per_qaly # Total number of QALYs gained per year (EUR/EUR/QALY = QALY )    
        lost_QALYs = reduction*(hospital_data['total_spent']*1e6/hospital_data['cost_per_qaly']).sum()
        #lost_QALYs= reduction*total_gained_QALYs # Lost qalys per year
    else:
        
        data_per_disease=pd.DataFrame(columns=['disease_group','gained_qalys','lost_qalys'])
        data_per_disease['disease_group']=hospital_data['disease_group']
        #Number of QALYs gained per year per disease group
        data_per_disease['gained_qalys']=hospital_data['total_spent']*1000000/hospital_data['cost_per_qaly']
        # Number of QALYs lost per year per disease group
        data_per_disease['lost_qalys']=reduction*data_per_disease['gained_qalys']
        
        lost_QALYs=data_per_disease.copy().drop(columns=['gained_qalys'])
    
    return lost_QALYs

#%%

def get_QALY_parameters (input_table,input_QoL,SMR,qCM,r):

    """
    This functions calculates age-stratified QALYs lost associated with the death of a person.
    The output is used as a parameter of main model calculations
    
    Parameters
    ----------
    input_table: pd.DataFrame
        Base life table information. x, q(x)

    input_QoL: pd.DataFrame
        Quality weights for age groups

    SMR: float
        Standarized Mortality ratio

    qCM: float
        Adjustment paramter to account for additional impact on quality of life
    
    r: float
        Discount rate

    Returns
    -------
    life_table: pd.DataFrame
        QALYs lost when a person of a certain age (within the age bins of the model) dies.
    
    """  
    
    #Generate compleate life table from input data
    life_table=create_life_table(input_table,input_QoL,SMR,qCM,r)

    #Deaths per age group dataframe initialization
    deaths_input=pd.DataFrame({'age_group':['0-9','10-19','20-29','30-39','40-59','50-59','60-69','70-79','80+'],
                          'group_limit':[9,19,29,39,49,59,69,89,110],
                          'deaths':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]})                  
    
    #Calculate age-stratified lost QALYs per person
    low_limit=0
    
    for i in range(len(deaths_input['group_limit'])):
        avg_dQALY=np.mean(life_table.loc[low_limit:deaths_input.loc[i,'group_limit'],'QALY_x'])
        deaths_input.loc[i,'lost_QALY_pp']=avg_dQALY
        low_limit=deaths_input.loc[i,'group_limit']+1
        
    lost_QALY_pp=np.array(deaths_input['lost_QALY_pp']) 
    
    return lost_QALY_pp

def QALY2xarray(out,lost_QALY_pp):
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
    return out.assign(variables={'QALY_death': out['D']*np.expand_dims(lost_QALY_pp,axis=1),
                                'QALY_treatment': (1-m_no_treatment)*out['R_hosp']*np.expand_dims(lost_QALY_pp,axis=1)})