import numpy as np
import pandas as pd

def create_life_table(input_table,input_QoL,SMR,qCM,r):
    
    """
    Life tables are used to calculate the number of QALYs lost if a person dies at a given age.
    That information can then be used to calculate the total number of QALYs lost.
    It is not necessary that deaths are caused by a specific reason

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
    life_table: Dataframe with calculation results. 
                             
    """
    
    life_table=input_table
    QoL=input_QoL
    # Instantaneous death rate: Probability of death at age x
    life_table.loc[0,'d_x']=-np.log(1-life_table.loc[0,'q_x'])
    for i in range(1,len(life_table['q_x'])):
        life_table.loc[i,'d_x']=-0.5*(np.log(1-life_table.loc[i-1,'q_x'])+np.log(1-life_table.loc[i,'q_x'])) 
    # life_table['d_x']=-np.log(1-life_table['q_x']) 
    
    # Number of people who survive to age x per 100000 hab.
    #This is 100000 at x=0 by definition
    life_table.loc[0,'l_x']=100000
    for i in range(1,len(life_table['q_x'])):
        life_table.loc[i,'l_x']=life_table.loc[i-1,'l_x']*np.exp(-life_table.loc[i-1,'d_x']*SMR)
        #Years lived between x and (x+1) per 100000 hab. 
        life_table.loc[i-1,'L_x']=(life_table.loc[i-1,'l_x']+life_table.loc[i,'l_x'])/2
        
           
    QoL_x=QoL.loc[0,'QoL_score']# Quality of life index for each age group
    age_limit=QoL.loc[0,'group_limit'] # Limit of the age group
    j=0 #Alternative index for QoL indices search
    for i in range(len(life_table['l_x'])):
        
        #Life expectancy at age x
        life_table.loc[i,'LE_x']=np.sum(life_table.loc[i:len(life_table['L_x'])-1,'L_x'])/life_table.loc[i,'l_x']
        #Check which QoL index to use
        if i >age_limit:
            j+=1
            QoL_x=QoL.loc[j,'QoL_score']
            age_limit=QoL.loc[j,'group_limit']
            
        #Quality adjusted years lived between x and (x+1) per 100000 hab    
        life_table.loc[i,'QAL_x']=life_table.loc[i,'L_x']*QoL_x*qCM
        
    for i in range(len(life_table['l_x'])):
        #Quality adjusted life expectancy
        life_table.loc[i,'QALE_x']=np.sum(life_table.loc[i:len(life_table['L_x'])-1,'QAL_x'])/life_table.loc[i,'l_x']
       
        dQAL_x=np.zeros(len((life_table['l_x'])))
        for u in range(len(life_table['l_x'])-1):
            #Discounted lived years
            dQAL_x[u]=life_table.loc[u,'QAL_x']/(1+r)**(u-i) 
        #QALYs 
        life_table.loc[i,'QALY_x']=np.sum(dQAL_x[i:])/life_table.loc[i,'l_x']
        life_table=life_table.copy()

    return life_table

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
        
        total_cost=hospital_data['total_spent'].sum()*1000000 # Total amount spent in regular health care per year
        avg_cost_per_qaly=hospital_data['cost_per_qaly'].mean() #Average cost per QALY gained
        total_gained_QALYs=total_cost/avg_cost_per_qaly #Total number of QALYs gained per year (EUR/EUR/QALY = QALY )    
        
        lost_QALYs= reduction*total_gained_QALYs # Lost qalys per year
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
    return out.assign(variables={'QALYs_lost': out['D']*np.expand_dims(lost_QALY_pp,axis=1)})