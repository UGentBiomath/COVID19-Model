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