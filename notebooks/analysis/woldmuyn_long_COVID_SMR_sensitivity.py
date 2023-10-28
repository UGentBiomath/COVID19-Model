"""
This script contains the calculation of the average QALY loss for different SMRs
and combines it with the dynamic tranmsission model to perform a sensitivity of SMR on the distribution of QALYs
lost due to covid-19 deaths and long-COVID

optional parameter: -d --draws : determines how many draws of QoL score and f_AD as well as how many DTM simulations (default 200)
""" 

__author__      = "Wolf Demuynck"
__copyright__   = "Copyright (c) 2022 by W. Demuynck, BIOMATH, Ghent University. All Rights Reserved."

from covid19_DTM.models.utils import initialize_COVID19_SEIQRD_hybrid_vacc
from QALY_model.direct_QALYs import life_table_QALY_model, bin_data
Life_table = life_table_QALY_model()
import numpy as np
import os
import multiprocessing as mp
import pandas as pd
import xarray as xr
import argparse
import emcee
from tqdm import tqdm
from scipy.integrate import quad
from scipy.optimize import minimize

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--draws", help="QALY draws and DTM simulations", default=2)
    args = parser.parse_args()

    draws = int(args.draws)

    ###############
    ## Load data ##
    ###############

    print('\n(1) Loading data\n')

    abs_dir = os.path.dirname(__file__)
    rel_dir = '../../data/QALY_model/raw/long_COVID'

    # --------------- #
    # Prevalence data #
    # --------------- #

    severity_groups = ['Mild','Moderate','Severe-Critical']
    hospitalisation_groups = ['Non-hospitalised','Cohort','ICU']

    # raw prevalence data per severity group
    prevalence_data_per_severity_group = pd.read_csv(os.path.join(abs_dir,rel_dir,'Long_COVID_prevalence.csv'),index_col=[0,1])

    # severity distribution in raw data
    severity_distribution= pd.DataFrame(data=np.array([[(96-6)/(338-172),(145-72)/(338-172),((55+42)-(52+42))/(338-172)],
                                                [(6-0)/(172-42),(72-0)/(172-42),((52+42)-(0+42))/(172-42)],
                                                [0,0,1]]),
                                                columns=severity_groups,index=hospitalisation_groups)
    severity_distribution.index.name='hospitalisation'

    # convert data per severity group to data per hospitalisation group
    index = pd.MultiIndex.from_product([hospitalisation_groups,prevalence_data_per_severity_group .index.get_level_values('Months').unique()])
    prevalence_data_per_hospitalisation_group = pd.Series(index=index,dtype='float')
    for hospitalisation,month in index:
        prevalence = sum(prevalence_data_per_severity_group.loc[(slice(None),month),:].values.squeeze()*severity_distribution.loc[hospitalisation,:].values)
        prevalence_data_per_hospitalisation_group[(hospitalisation,month)]=prevalence

    # -------- #
    # QoL data #
    # -------- #

    LE_table = Life_table.life_expectancy()

    # reference QoL scores
    age_bins = pd.IntervalIndex.from_tuples([(15,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,LE_table.index.values[-1])], closed='left')
    QoL_Belgium = pd.Series(index=age_bins, data=[0.85, 0.85, 0.82, 0.78, 0.78, 0.78, 0.66])

    # QoL decrease due to long-COVID
    mean_QoL_decrease_hospitalised = 0.24
    mean_QoL_decrease_non_hospitalised = 0.19
    sd_QoL_decrease_hospitalised =  0.41/np.sqrt(174) #(0.58-0.53)/1.96
    sd_QoL_decrease_non_hospitalised =  0.33/np.sqrt(1146) #(0.66-0.64)/1.96
    QoL_difference_data = pd.DataFrame(data=np.array([[mean_QoL_decrease_non_hospitalised,mean_QoL_decrease_hospitalised,mean_QoL_decrease_hospitalised],
                                                    [sd_QoL_decrease_non_hospitalised,sd_QoL_decrease_hospitalised,sd_QoL_decrease_hospitalised]]).transpose(),
                                                    columns=['mean','sd'],index=['Non-hospitalised','Cohort','ICU'])

    # ------- #
    # results #
    # ------- #

    data_result_folder = '../../data/QALY_model/interim/long_COVID/'
    fig_result_folder = '../../results/QALY_model/direct_QALYs/prepocessing/'

    # Verify that the paths exist and if not, generate them
    for directory in [data_result_folder, fig_result_folder]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    ################
    ## Prevalence ##
    ################

    print('\n(2) Calculate prevalence\n')
    print('\n(2.1) Estimate tau\n')

    # objective function to minimize
    def WSSE_no_pAD(tau,x,y):
        y_model = np.exp(-x/tau)
        SSE = sum((y_model-y)**2)
        WSSE = sum((1/y)**2 * (y_model-y)**2)
        return WSSE

    def WSSE(theta,x,y):
        tau,p_AD = theta
        y_model = p_AD + (1-p_AD)*np.exp(-x/tau)
        SSE = sum((y_model-y)**2)
        WSSE = sum((1/y)**2 * (y_model-y)**2)
        return WSSE

    # minimize objective function to find tau
    taus = pd.Series(index=hospitalisation_groups+['Non-hospitalised (no AD)'],dtype='float')
    p_ADs = pd.Series(index=hospitalisation_groups+['Non-hospitalised (no AD)'],dtype='float')

    for hospitalisation in hospitalisation_groups+['Non-hospitalised (no AD)']:
        
        if hospitalisation == 'Non-hospitalised (no AD)':
            x = prevalence_data_per_hospitalisation_group.loc['Non-hospitalised'].index.values
            y = prevalence_data_per_hospitalisation_group.loc['Non-hospitalised'].values.squeeze()

            sol = minimize(WSSE_no_pAD,x0=3,args=(x,y))
            tau = sol.x[0]
            p_AD = 0
        else:
            x = prevalence_data_per_hospitalisation_group.loc[hospitalisation].index.values
            y = prevalence_data_per_hospitalisation_group.loc[hospitalisation].values.squeeze()

            sol = minimize(WSSE,x0=(3,min(y)),args=(x,y))
            tau = sol.x[0]
            p_AD = sol.x[1]
            
        p_ADs[hospitalisation] = p_AD
        taus[hospitalisation] = tau

    print('\n(1.2.2) MCMC to estimate f_AD\n')
    # objective functions for MCMC
    def WSSE(theta,x,y):
        tau,p_AD = theta
        y_model = p_AD + (1-p_AD)*np.exp(-x/tau)
        SSE = sum((y_model-y)**2)
        WSSE = sum((1/y)**2 * (y_model-y)**2)
        return WSSE

    def log_likelihood(theta, tau, x, y):
        p_AD = theta[0]
        y_model = p_AD + (1-p_AD)*np.exp(-x/tau)
        SSE = sum((y_model-y)**2)
        WSSE = sum((1/y)**2 * (y_model-y)**2)
        return -WSSE

    def log_prior(theta,p_AD_bounds):
        p_AD = theta[0]
        if p_AD_bounds[0] < p_AD < p_AD_bounds[1]:
            return 0.0
        else:
            return -np.inf

    def log_probability(theta, tau, x, y, bounds):
        lp = log_prior(theta,bounds)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta,tau, x, y)
        if np.isnan(ll):
            return -np.inf
        return lp + ll

    # run MCMC
    samplers = {}
    p_AD_summary = pd.DataFrame(index=hospitalisation_groups,columns=['mean','sd','lower','upper'],dtype='float64')

    for hospitalisation in hospitalisation_groups:
        
        x = prevalence_data_per_hospitalisation_group.loc[hospitalisation].index.values
        y = prevalence_data_per_hospitalisation_group.loc[hospitalisation].values.squeeze()

        tau = taus[hospitalisation]
        p_AD = p_ADs[hospitalisation]
        
        nwalkers = 32
        ndim = 1
        pos = p_AD + p_AD*1e-1 * np.random.randn(nwalkers, ndim)

        bounds = (0,1)
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(tau,x,y,bounds)
        )
        samplers.update({hospitalisation:sampler})
        sampler.run_mcmc(pos, 20000, progress=True)

        flat_samples = sampler.get_chain(discard=1000, thin=30, flat=True)
        p_AD_summary['mean'][hospitalisation] = np.mean(flat_samples,axis=0)
        p_AD_summary['sd'][hospitalisation] = np.std(flat_samples,axis=0)
        p_AD_summary['lower'][hospitalisation] = np.quantile(flat_samples,0.025,axis=0)
        p_AD_summary['upper'][hospitalisation] = np.quantile(flat_samples,0.975,axis=0)

    print(p_AD_summary)

    #########
    ## QoL ##
    #########

    print('\n(3) Fit exponential curve to QoL scores\n')

    QoL_Belgium_func = Life_table.QoL_Belgium_func

    ################################
    ## Define simulation settings ##
    ################################

    # Number of simulations
    N=draws
    # Number of neg. binomial draws/ simulation
    K=10
    # Number of cpu's
    processes=int(mp.cpu_count()/2)
    # Number of age groups
    age_stratification_size=10
    # End of simulation
    end_sim='2021-12-31'
    # Confidence level used to visualise model fit
    conf_int=0.05

    ##########################
    ## Initialize the model ##
    ##########################

    print(f'\n(4) Initialize model')

    model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_hybrid_vacc(age_stratification_size=age_stratification_size, start_date='2020-03-15')

    warmup = float(BASE_samples_dict['warmup'])
    dispersion = float(BASE_samples_dict['dispersion'])
    start_sim = BASE_samples_dict['start_calibration']

    #########################
    ## Perform simulations ##
    #########################

    from covid19_DTM.models.draw_functions import draw_fnc_COVID19_SEIQRD_hybrid_vacc as draw_function

    print(f'\n(5) Simulating COVID19_SEIQRD_hybrid_vacc '+str(N)+' times')
    out_sim = model.sim([start_sim,end_sim], warmup=warmup, N=N, samples=BASE_samples_dict, draw_function=draw_function)

    #######################
    ## Average QALY loss ##
    #######################

    index = pd.Index(np.arange(0.1,2.1,0.1),name='SMR')
    columns = ['mean','lower','upper']
    SMR_sensitivity_AD = pd.DataFrame(index=index,columns=columns)
    SMR_sensitivity_no_AD = pd.DataFrame(index=index,columns=columns)
    for i, SMR in enumerate(np.linspace(1/1.5,1.5,10)):
    #for i, SMR in enumerate(np.arange(0.1,2.1,0.1)):
        print(f'\n---------\nSMR:{SMR}\n---------\n')
        print(f'\n(6.{i}) Calculate average QALY losses\n')

        LE_table = Life_table.life_expectancy(SMR=SMR)
        hospitalisation_groups = ['Non-hospitalised','Cohort','ICU']

        # Pre-allocate new multi index series with index=hospitalisation,age,draw
        multi_index = pd.MultiIndex.from_product([hospitalisation_groups+['Non-hospitalised (no AD)'],np.arange(draws),LE_table.index.values],names=['hospitalisation','draw','age'])
        average_QALY_losses_per_age = pd.Series(index = multi_index, dtype=float)

        prevalence_func = lambda t,tau, p_AD: p_AD + (1-p_AD)*np.exp(-t/tau)

        # QALY loss func for fixed QoL after but beta is absolute difference and changing over time due to decreasing QoL reference
        def QALY_loss_func(t,tau,p_AD,age,QoL_after):
            beta = QoL_Belgium_func(age+t/12)-QoL_after
            return prevalence_func(t,tau,p_AD) * max(0,beta)

        # Calculate average QALY loss for each age 'draws' times
        for idx,(hospitalisation,draw,age) in enumerate(tqdm(multi_index)):
            LE = LE_table[age]*12

            # use the same samples for beta and p_AD to calculate average QALY loss for each age
            if age==0:
                if hospitalisation == 'Non-hospitalised (no AD)':
                    p_AD = 0
                    beta = np.random.normal(QoL_difference_data.loc['Non-hospitalised']['mean'],
                                        QoL_difference_data.loc['Non-hospitalised']['sd'])
                else:
                    p_AD = np.random.normal(p_AD_summary.loc[hospitalisation]['mean'],
                                            p_AD_summary.loc[hospitalisation]['sd'])
                    if p_AD <0:
                        p_AD = 0
                    beta = np.random.normal(QoL_difference_data.loc[hospitalisation]['mean'],
                                            QoL_difference_data.loc[hospitalisation]['sd'])
                    
                tau = taus[hospitalisation]

            # calculate the fixed QoL after getting infected for each age
            QoL_after = QoL_Belgium_func(age)-beta
            # integrate QALY_loss_func from 0 to LE  
            QALY_loss = quad(QALY_loss_func,0,LE,args=(tau,p_AD,age,QoL_after))[0]/12 
            average_QALY_losses_per_age[idx] = QALY_loss

        # bin data
        average_QALY_losses_per_age_group = bin_data(average_QALY_losses_per_age)

        #######################
        ## QALY calculations ##
        #######################

        print(f'\n(7.{i}) Calculating QALYs')
        for AD, out in zip((True,False),(out_sim.copy(),out_sim.copy())):
            if AD:
                hospitalisation_groups = ['Non-hospitalised','Cohort','ICU']
            else:
                hospitalisation_groups = ['Non-hospitalised (no AD)','Cohort','ICU']
            hospitalisation_abbreviations = ['NH','C','ICU']
        
            for hospitalisation,hospitalisation_abbreviation in zip(hospitalisation_groups,hospitalisation_abbreviations):
                mean_QALY_losses = average_QALY_losses_per_age_group.loc[hospitalisation].unstack().values[:,np.newaxis,:,np.newaxis]
                out[f'QALY_{hospitalisation_abbreviation}'] = out[hospitalisation_abbreviation+'_R_in'].cumsum(dim='date')*mean_QALY_losses

            # Calculate QALY losses due COVID death
            QALY_D_per_age = Life_table.compute_QALY_D_x(SMR=SMR)
            QALY_D_per_age_group = bin_data(QALY_D_per_age)
            out['QALY_D'] = out['D']*np.array(QALY_D_per_age_group)[np.newaxis,np.newaxis,:,np.newaxis]

            if AD:
                out_AD = out
            else:
                out_no_AD = out

        ################
        # Save results #
        ################
        print(f'\n(8.{i}) Save results')

        abs_dir = os.getcwd()
        result_folder = '../../results/QALY_model/direct_QALYs/analysis/'

        # Verify that the paths exist and if not, generate them
        if not os.path.exists(os.path.join(abs_dir,result_folder)):
            os.makedirs(os.path.join(abs_dir,result_folder))

        def get_lower(x):
            return np.quantile(x,0.025)
        def get_upper(x):
            return np.quantile(x,0.975)
        def get_sd(x):
            return np.std(x)

        for file_name, SMR_sensitivity, out in zip(('SMR_sensitivity_AD.csv','SMR_sensitivity_no_AD.csv'),(SMR_sensitivity_AD,SMR_sensitivity_no_AD),(out_AD,out_no_AD)):
            QALYs_LC = sum(out[['QALY_NH','QALY_C','QALY_ICU']].isel(date=-1).sum(['age_groups','doses']).values())
            QALYs_D = out['QALY_D'].isel(date=-1).sum(['age_groups','doses']).values

            ratios = QALYs_LC/QALYs_D
            mean_ratio = np.mean(ratios).values
            lower_ratio = np.quantile(ratios,0.025)
            upper_ratio = np.quantile(ratios,0.975)

            SMR_sensitivity['mean'][SMR] = mean_ratio
            SMR_sensitivity['lower'][SMR] = lower_ratio
            SMR_sensitivity['upper'][SMR] = upper_ratio

            SMR_sensitivity.to_csv(os.path.join(abs_dir,result_folder,file_name))

        if SMR in [1/1.5,1,1.5]:
            age_groups = out_AD.coords['age_groups'].values
            age_group_labels = ['0-12','12-18','18-25','25-35','35-45','45-55','55-65','65-75','75-85','85+']

            states = ['QALY_D','QALY_NH', 'QALY_C', 'QALY_ICU']
            titles = ['Deaths','Non-hospitalised', 'Cohort', 'ICU']
            QALYs = {'Non-hospitalised (no AD)':'QALY_NH','Non-hospitalised (AD)':'QALY_NH','Cohort':'QALY_C','ICU':'QALY_ICU','Deaths':'QALY_D'}

            # plot data
            for scenario,out in zip(['no_AD','AD'],[out_no_AD,out_AD]):
                mean_QALYs = pd.DataFrame(index=pd.Index(age_group_labels),columns=titles)
                for state, label in zip(states,titles):
                    mean = out[state].mean('draws').sum('doses')[-1].values
                    mean_QALYs[label] = mean

                mean_QALYs.to_csv(os.path.join(result_folder,f'Long_COVID_plotdata_{scenario}.csv'))

            # summarise results in table
            index = pd.Index(age_group_labels+['Total'])
            columns = ['Non-hospitalised (no AD)', 'Non-hospitalised (AD)', 'Cohort', 'ICU', 'Deaths','Total (no AD)', 'Total (AD)']
            QALY_table = pd.DataFrame(index=index,columns=columns)

            # QALY per age group per hospitalisation group
            for age_group,age_group_label in zip(age_groups,age_group_labels):
                for column in ['Non-hospitalised (no AD)', 'Non-hospitalised (AD)', 'Cohort', 'ICU', 'Deaths']:
                    if column == 'Non-hospitalised (AD)':
                        out = out_AD
                    else:
                        out = out_no_AD

                    out_slice = out[QALYs[column]].sel({'age_groups':age_group}).sum('doses')[slice(None),-1].values
                    mean = out_slice.mean()
                    sd = np.std(out_slice)
                    lower = np.quantile(out_slice,0.025)
                    upper = np.quantile(out_slice,0.975)

                    QALY_table[column][age_group_label] = f'{mean:.0f}\n({lower:.0f};{upper:.0f})'
                    
            # Total QALY per hospitalisation group        
            for column in ['Non-hospitalised (no AD)', 'Non-hospitalised (AD)', 'Cohort', 'ICU', 'Deaths']:
                if column == 'Non-hospitalised (AD)':
                    out = out_AD
                else:
                    out = out_no_AD
                
                out_slice = out[QALYs[column]].sum('age_groups').sum('doses')[slice(None),-1].values
                mean = out_slice.mean()
                sd = np.std(out_slice)
                lower = np.quantile(out_slice,0.025)
                upper = np.quantile(out_slice,0.975)

                QALY_table[column]['Total'] = f'{mean:.0f}\n({lower:.0f};{upper:.0f})'

            # Total QALY per age group
            for total_label,out in zip(['Total (no AD)', 'Total (AD)'],[out_no_AD,out_AD]):
                total = np.zeros(out.dims['draws'])
                for age_group,age_group_label in zip(age_groups,age_group_labels):
                    total_per_age_group = np.zeros(out.dims['draws'])
                    for state in states:
                        total_per_age_group+=out[state].sel({'age_groups':age_group}).sum('doses')[slice(None),-1].values
                    total += total_per_age_group

                    mean = total_per_age_group.mean()
                    sd = np.std(total_per_age_group)
                    lower = np.quantile(total_per_age_group,0.025)
                    upper = np.quantile(total_per_age_group,0.975)

                    QALY_table[total_label][age_group_label] = f'{mean:.0f}\n({lower:.0f};{upper:.0f})'
                
                mean = total.mean()
                sd = np.std(total)
                lower = np.quantile(total,0.025)
                upper = np.quantile(total,0.975)
                
                QALY_table[total_label]['Total'] = f'{mean:.0f}\n({lower:.0f};{upper:.0f})'

            QALY_table.to_csv(os.path.join(result_folder,f'Long_COVID_summary_SMR{SMR*100:.0f}.csv'))
