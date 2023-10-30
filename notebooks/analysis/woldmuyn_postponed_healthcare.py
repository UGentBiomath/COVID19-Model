"""
This script uses the calibrated postponed healthcare models (queuing, (constrained) PI) to calculate QALY loss
Results are saved to results/PHM/analysis

""" 

__author__      = "Wolf Demuynck"
__copyright__   = "Copyright (c) 2022 by W. Demuynck, BIOMATH, Ghent University. All Rights Reserved."

import json
import argparse
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import xarray as xar
import math
from covid19_DTM.data.sciensano import get_sciensano_COVID19_data
from QALY_model.postponed_healthcare_models import draw_fcn 

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--N", help="simulation runs", default=50)
args = parser.parse_args()
N = int(args.N)

##############################################################
## From here it is a copy paste from the calibration script ##
##############################################################

# The code below loads all data and sets up the model
from pySODM.models.base import ODE
from functools import lru_cache
from datetime import datetime

use_covid_data = True

########
# Data #
########

abs_dir = os.path.dirname(__file__)
rel_dir = '../../data/QALY_model/interim/postponed_healthcare'
# baseline
file_name = 'MZG_baseline.csv'
types_dict = {'APR_MDC_key': str, 'week_number': int, 'day_number':int}
baseline = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict).squeeze()['mean']
# raw data
file_name = 'MZG_2016_2021.csv'
types_dict = {'APR_MDC_key': str}
raw = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2,3],dtype=types_dict).squeeze()
raw = raw.groupby(['APR_MDC_key','date']).sum()
raw.index = raw.index.set_levels([raw.index.levels[0], pd.to_datetime(raw.index.levels[1])])
# normalised data
file_name = 'MZG_2020_2021_normalized.csv'
types_dict = {'APR_MDC_key': str}
normalised = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True)
MDC_keys = normalised.index.get_level_values('APR_MDC_key').unique().values
# COVID-19 data
covid_data, _ , _ , _ = get_sciensano_COVID19_data(update=False)
new_index = pd.MultiIndex.from_product([pd.to_datetime(normalised.index.get_level_values('date').unique()),covid_data.index.get_level_values('NIS').unique()])
covid_data = covid_data.reindex(new_index,fill_value=0)
df_covid_H_in = covid_data['H_in'].loc[:,40000]
df_covid_H_tot = covid_data['H_tot'].loc[:,40000]
df_covid_dH = df_covid_H_tot.diff().fillna(0)
# mean hospitalisation length
file_name = 'MZG_residence_times.csv'
types_dict = {'APR_MDC_key': str, 'age_group': str, 'stay_type':str}
residence_times = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,]).squeeze()
mean_residence_times = residence_times.groupby(by=['APR_MDC_key']).mean()

#####################
## Smooth raw data ##
#####################

# filter settings
window = 15
order = 3

# Define filter #TODO: move to seperate file
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# copy data
raw_smooth = pd.Series(0, index=raw.index, name='n_patients')
# smooth
for MDC_key in MDC_keys:
    raw_smooth.loc[MDC_key,slice(None)] = savitzky_golay(raw.loc[MDC_key,slice(None)].values,window,order)
raw = raw_smooth

##################
## Define model ##
##################

class QueuingModel(ODE):

    states = ['W','H','H_adjusted','H_norm','R','NR','X']
    parameters = ['X_tot','f_UZG','covid_H','alpha','post_processed_H']
    stratified_parameters = ['A','gamma','epsilon','sigma']
    dimensions = ['MDC']
    
    @staticmethod
    def integrate(t, W, H, H_adjusted, H_norm, R, NR, X, X_tot, f_UZG, covid_H, alpha, post_processed_H, A, gamma, epsilon, sigma):
        X_new = (X_tot-alpha*covid_H-sum(H - (1/gamma*H))) * (sigma*(A+W))/sum(sigma*(A+W))

        W_to_H = np.where(W>X_new,X_new,W)
        W_to_NR = epsilon*(W-W_to_H)
        A_to_H = np.where(A>(X_new-W_to_H),(X_new-W_to_H),A)
        A_to_W = A-A_to_H
        
        dX = -X + X_new - W_to_H - A_to_H
        dW = A_to_W - W_to_H - W_to_NR
        dH = A_to_H + W_to_H - (1/gamma*H) 
        dR = (1/gamma*H)
        dNR = W_to_NR

        dH_adjusted = -H_adjusted + post_processed_H[0]
        dH_norm = -H_norm + post_processed_H[1]
        
        return dW, dH, dH_adjusted, dH_norm, dR, dNR, dX

##################
## Define TDPFs ##
##################

class get_A():
    def __init__(self, baseline, mean_residence_times):
        self.baseline = baseline
        self.mean_residence_times = mean_residence_times
        
    def A_wrapper_func(self, t, states, param):
        return self.__call__(t)
    
    @lru_cache()
    def __call__(self, t):
        A = (self.baseline.loc[(slice(None),t.isocalendar().week)]/self.mean_residence_times).values
        return A 

class get_covid_H():

    def __init__(self, covid_data, baseline, hospitalizations):
        self.covid_data = covid_data
        self.baseline_04 = baseline['04']
        self.hospitalizations_04 = hospitalizations.loc[('04', slice(None))]

    def H_wrapper_func(self, t, states, param, f_UZG):
        return self.__call__(t, f_UZG)

    def __call__(self, t, f_UZG):
        if use_covid_data:
            try:
                covid_H = self.covid_data.loc[t]*f_UZG
            except:
                covid_H = 0

        else:
            covid_H = max(self.hospitalizations_04.loc[t] - self.baseline_04.loc[t.isocalendar().week],0)
        
        return covid_H 
    
class H_post_processing():

    def __init__(self, baseline, MDC_sim):
        self.MDC = np.array(MDC_sim)
        self.baseline = baseline

    def H_post_processing_wrapper_func(self, t, states, param, covid_H):
        H = states['H']
        return self.__call__(t, H, covid_H)

    def __call__(self, t, H, covid_H):
        H_adjusted = np.where(self.MDC=='04',H+covid_H,H)
        H_norm = H_adjusted/self.baseline.loc[slice(None),t.isocalendar().week]
        return (H_adjusted,H_norm)
    
#################
## Setup model ##
#################

def init_queuing_model(start_date, MDC_sim, wave='first'):
    # Define model parameters, initial states and coordinates
    if wave == 'first':
        epsilon = [0.157, 0.126, 0.100, 0.623,
                0.785, 0.339, 0.318, 0.561,
                0.266, 0.600, 0.610, 0.562,
                0.630, 0.534, 0.488, 0.469,
                0.570, 0.612, 0.276, 0.900,
                0.577, 0.163, 0.865, 0.708,
                0.494, 0.557, 0.463, 0.538]
        sigma = [0.229, 0.816, 6.719, 1.023,
                1.251, 1.089, 1.695, 8.127,
                1.281, 1.063, 1.425, 2.819,
                1.899, 1.240, 8.057, 2.657,
                6.685, 2.939, 2.674, 1.011,
                3.876, 7.467, 0.463, 1.175,
                1.037, 1.181, 3.002, 6.024]
        alpha = 5.131
    elif wave == 'second':
        epsilon = [0.176,0.966,0.000,0.498,
                   0.868,0.648,0.000,0.000,
                   0.674,0.593,0.458,0.728,
                   0.666,0.499,0.623,0.777,
                   0.076,0.152,0.200,0.793,
                   0.031,0.100,0.272,0.385,
                   0.551,0.288,0.464,0.518]              
        sigma = [0.344,3.652,6.948,2.831,
                 2.079,5.101,5.865,8.442,
                 6.215,4.911,7.272,4.729,
                 4.415,2.914,5.240,4.299,
                 4.013,5.091,2.000,3.628,
                 0.333,6.550,1.468,6.147,
                 3.184,1.728,6.982,8.739]
        alpha = 5.106
    else:
        epsilon = np.ones(len(MDC_sim))*0.1
        sigma = np.ones(len(MDC_sim))
        alpha = 5

    # Parameters
    gamma =  mean_residence_times.loc[MDC_sim].values
    f_UZG = 0.13
    X_tot = 1049
    start_date_string = start_date.strftime('%Y-%m-%d')
    covid_H = df_covid_H_tot.loc[start_date_string]*f_UZG
    H_init = raw.loc[(MDC_sim, start_date)].values
    H_init_normalized = (H_init/baseline.loc[((MDC_sim,start_date.isocalendar().week))]).values
    A = H_init/mean_residence_times.loc[MDC_sim]
    params={'A':A,'covid_H':covid_H,'alpha':alpha,'post_processed_H':(H_init,H_init_normalized),'X_tot':X_tot, 'gamma':gamma, 'epsilon':epsilon, 'sigma':sigma,'f_UZG':f_UZG}
    # Initial states
    init_states = {'H': H_init, 'H_norm': np.ones(len(MDC_sim)), 'H_adjusted': H_init}
    coordinates={'MDC': MDC_sim}
    # TDPFs
    daily_hospitalizations_func = get_A(baseline,mean_residence_times.loc[MDC_sim]).A_wrapper_func
    covid_H_func = get_covid_H(df_covid_H_tot,baseline,raw).H_wrapper_func
    post_processing_H_func = H_post_processing(baseline,MDC_sim).H_post_processing_wrapper_func

    # Initialize model
    model = QueuingModel(init_states,params,coordinates,
                          time_dependent_parameters={'A': daily_hospitalizations_func,'covid_H':covid_H_func,'post_processed_H':post_processing_H_func})
    return model

#################
## Setup model ##
#################

model = init_queuing_model(datetime(2020,3,1), MDC_keys, 'first')



################################################
## From here it is back to an original script ##
################################################

#############
# QALY data #
#############

file_name = 'hospital_yearly_QALYs.csv'
hospital_yearly_QALYs = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0],na_values='/')

###########
# Samples #
###########

rel_dir = '../../data/QALY_model/interim/postponed_healthcare/model_parameters'
## Load samples
# Two calibrations
#samples = {'first_wave': {}, 'second_wave': {}}
#for file_name,period in zip(['calibrate_first_SAMPLES.json', 'calibrate_second_SAMPLES.json'], ['first_wave', 'second_wave']):
#    f = open(os.path.join(abs_dir,rel_dir,file_name))
#    samples[period].update(json.load(f))
# One calibration
file_name = 'calibrate_2020_SAMPLES.json'
f = open(os.path.join(abs_dir,rel_dir,file_name))
samples = json.load(f)

#########
# setup #
#########

start_calibration_first = pd.to_datetime('2020-01-01')
end_calibration_first = pd.to_datetime('2020-09-01')
start_calibration_second = pd.to_datetime('2020-09-01')
end_calibration_second = pd.to_datetime('2021-03-01')

MDC_plot = ['03', '04', '05']
dates = pd.date_range('2020-01-01','2021-12-31')

result_folder = os.path.join(abs_dir,'../../results/QALY_model/postponed_healthcare/analysis/')
if not os.path.exists(os.path.join(abs_dir,result_folder)):
        os.makedirs(os.path.join(abs_dir,result_folder))

# mean absolute error, outs kan lijst zijn met out_first en out_second voor de totale MAE te bereken, beetje messy
def MAE(data,outs,MDC_key):
    AE = 0
    for out in outs:
        start_date = out.date[0].values
        end_date = out.date[-1].values
        date_range = pd.date_range(start_date,end_date)

        y_data = data.loc[(date_range,MDC_key)]
        y_model = out['H_norm'].sel(date=date_range,MDC=MDC_key).mean('draws')
        AE += sum(abs(y_model-y_data))
    
    start_date = outs[0].date[0].values
    end_date = outs[-1].date[-1].values
    date_range = pd.date_range(start_date,end_date)
    return AE/len(date_range)

# functie die een plot maakt van de data en de 3 gekalibreerde modellen
def plot_model_outputs(plot_name,outs,plot_start,plot_end,start_calibration,end_calibration,MDC_plot):
    plot_time = pd.date_range(plot_start,plot_end)

    box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    fig,axs = plt.subplots(len(MDC_plot),3,sharex=True,sharey='row',figsize=(6,1.3*len(MDC_plot)))
    axs[0,0].set_title('Queuing model')
    axs[0,1].set_title('Constrained PI Model')
    axs[0,2].set_title('PI Model')

    for i,MDC_key in enumerate(MDC_plot):
        axs[i,0].set_ylabel(MDC_key)
        for j,out in enumerate(outs):

            mean_fit = out['H_norm'].sel(date=plot_time,MDC=MDC_key).mean('draws')
            lower_fit = out['H_norm'].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.025)
            upper_fit = out['H_norm'].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.975)
            mae = MAE(postponed_healthcare.hospitalizations_normalized_smooth,[out],MDC_key)
            axs[i,j].text(0.05, 0.95, f'MAE={mae:.3f}', transform=axs[i,j].transAxes, fontsize=8,verticalalignment='top', bbox=box_props,)

            # data
            axs[i,j].plot(plot_time,postponed_healthcare.hospitalizations_normalized_smooth.loc[plot_time,MDC_key], label='Filtered data', alpha=0.7,linewidth=1)
            axs[i,j].fill_between(plot_time,postponed_healthcare.hospitalizations_normalized_lower_smooth.loc[plot_time,MDC_key],
                                            postponed_healthcare.hospitalizations_normalized_upper_smooth.loc[plot_time,MDC_key], alpha=0.2, label='95% CI on data')
            # sim
            axs[i,j].plot(plot_time,mean_fit, color='black', label='Model output',linewidth=1,alpha=0.7)
            axs[i,j].fill_between(plot_time,lower_fit,upper_fit,color='black', alpha=0.2, label='95% CI on model output')
            
            # fancy plot
            axs[i,j].set_xticks(pd.date_range(plot_start+pd.Timedelta('60D'),plot_end-pd.Timedelta('60D'),periods=2))
            axs[i,j].grid(False)
            axs[i,j].tick_params(axis='both', which='major', labelsize=8)
            axs[i,j].tick_params(axis='both', which='minor', labelsize=8)
            axs[i,j].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5)
            if hasattr(start_calibration,'__iter__'):
                for start in start_calibration:
                    axs[i,j].axvline(x = start, color = 'gray', linestyle = 'dashed', alpha=0.5)
            else:
                axs[i,j].axvline(x = start_calibration, color = 'gray', linestyle = 'dashed', alpha=0.5) 
            if hasattr(end_calibration,'__iter__'):
                for end in end_calibration:
                    axs[i,j].axvline(x = end, color = 'gray', linestyle = 'dashed', alpha=0.5)
            else:  
                axs[i,j].axvline(x = end_calibration, color = 'gray', linestyle = 'dashed', alpha=0.5) 

    handles, plot_labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles=handles,labels=plot_labels,bbox_to_anchor =(0.5,-0.04), loc='lower center',fancybox=False, shadow=False,ncol=5)

    fig.tight_layout()
    fig.savefig(os.path.join(result_folder,plot_name),dpi=600,bbox_inches='tight')

if __name__=='__main__':

    #########################
    # simulations and plots #
    #########################
    print('1) Simulating and plotting')

    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    processes = 1
    # simulate and plot first wave
    start_date = start_calibration_first
    end_date = end_calibration_second

    queuing_model,constrained_PI_model,PI_model = postponed_healthcare.init_models(start_date)
    out_queuing = queuing_model.sim([start_date,end_date],N=N, samples=samples['first_wave']['queuing_model'], draw_function=draw_fcn,tau=1, processes=processes)
    out_constrained_PI = constrained_PI_model.sim([start_date,end_date], N=N, samples=samples['first_wave']['constrained_PI'], draw_function=draw_fcn,method='LSODA', processes=processes)
    out_PI = PI_model.sim([start_date,end_date], N=N, samples=samples['first_wave']['PI'], draw_function=draw_fcn,method='LSODA', processes=processes)
    outs_first = [out_queuing,out_constrained_PI,out_PI]
    
    plot_model_outputs('postponed_healthcare_calibrations_first_wave.pdf',outs_first,start_date,end_date,start_calibration_first,end_calibration_first,MDC_plot)
    
    # simulate and plot second wave
    start_date = start_calibration_second
    end_date = pd.to_datetime(dates[-1])

    queuing_model,constrained_PI_model,PI_model = postponed_healthcare.init_models(start_date)
    out_queuing = queuing_model.sim([start_date,end_date],N=N, samples=samples['second_wave']['queuing_model'], draw_function=draw_fcn,tau=1,processes=processes)
    out_constrained_PI = constrained_PI_model.sim([start_date,end_date], N=N, samples=samples['second_wave']['constrained_PI'], draw_function=draw_fcn,method='LSODA',processes=processes)
    out_PI = PI_model.sim([start_date,end_date], N=N, samples=samples['second_wave']['PI'], draw_function=draw_fcn,method='LSODA',processes=processes)
    outs_second = [out_queuing,out_constrained_PI,out_PI]

    plot_model_outputs('postponed_healthcare_calibrations_second_wave.pdf',outs_second,start_date,end_date,start_calibration_second,end_calibration_second,MDC_plot)

    outs_first_trimmed = [out.sel(date=pd.date_range(start_calibration_first,end_calibration_first-pd.Timedelta('1D'))) for out in outs_first]

    start_date = start_calibration_first
    end_date = pd.to_datetime(dates[-1])
    outs_full = [xar.concat([out_first,out_second],dim='date') for out_first,out_second in zip(outs_first_trimmed,outs_second)]
    plot_model_outputs('postponed_healthcare_calibrations_second_wave_all_MDC_full_period.pdf',outs_full,start_date,end_date,[start_calibration_first,start_calibration_second],[end_calibration_first,end_calibration_second],MDC_keys)

    ################
    #  fit per MDC #
    ################
    # maakt tabel uit appendix die gekalibreerde parameters mee geeft en bijhorende MAE score
    # MAE score voor de calibration period, maar ook totaal met de 2 kalibraties gecombineerd + periode na 2e golf

    print('2) Save fit per MDC')

    def extract_parameters_for_MDC(MDC_key,samples_dict):
        MDC_idx = MDC_keys.index(MDC_key)

        pars = samples_dict['parameters']
        
        calibrated_parameters = []
        
        for param in pars:
            if hasattr(samples_dict[param][0],'__iter__'):
                samples = samples_dict[param][MDC_idx]
            else:
                samples = samples_dict[param]
            mean = np.mean(samples)
            lower = np.quantile(samples,0.025)
            upper = np.quantile(samples,0.975)

            calibrated_parameters.append(f'{mean:.2E}\n({lower:.2E};{upper:.2E})')
        return calibrated_parameters
    
    models = ['queuing_model','constrained_PI','PI']
    MAEs = pd.DataFrame(index=MDC_keys,columns=models)
    for model,outs in zip(models,zip(outs_first_trimmed,outs_second)):
        model_fit_summary = {}

        # calibrated parameters per MDC
        for out, period in zip(outs,['first_wave','second_wave']):
            samples_dict = samples[period][model]
            parameters = samples_dict['parameters']

            start_calibration = pd.to_datetime(samples_dict['start_calibration'])
            end_calibration = pd.to_datetime(samples_dict['end_calibration'])
            calibration_period = pd.date_range(start_calibration,end_calibration-pd.Timedelta('1D'))

            model_fit = pd.DataFrame(index=MDC_keys,columns=parameters+['MAE'])
            for MDC_key in MDC_keys: 
                model_fit.loc[MDC_key][parameters] = extract_parameters_for_MDC(MDC_key,samples_dict)
                mae = MAE(postponed_healthcare.hospitalizations_normalized_smooth.loc[calibration_period,MDC_key],
                        [out.sel(date=calibration_period)],MDC_key)     
                model_fit.loc[MDC_key]['MAE'] = f'{mae:.3f}'
            model_fit_summary[period]=model_fit.drop(columns=['MAE'])
            model_fit.to_csv(os.path.join(result_folder,model+'_'+period+'_fit.csv'))

        # overall MAE per MDC
        for MDC_key in MDC_keys:
            mae = MAE(postponed_healthcare.hospitalizations_normalized_smooth,outs,MDC_key)     
            MAEs.loc[MDC_key][model] = f'{mae:.3f}'
        model_fit_summary['MAE']=MAEs[model]

        # save result in dataframe, csv
        model_fit_summary = pd.concat(model_fit_summary, axis=1)
        model_fit_summary.to_csv(os.path.join(result_folder,model+'_fit_summary.csv')) 

    MAEs.to_csv(os.path.join(result_folder,'MAE.csv'))  

    #########################
    # QALY loss calculation #
    #########################

    # !WARNING MESSY CODE!
    # makes summary table with for each MDC reduction with CI and associated QALY loss with CI
    # Hierin nog foutje voor CI, nu lower and upper WTP genomen, maar hieruit moet eigenlijk gesampled worden

    print('3) Calculate QALYs')
    models = ['queuing_model','constrained_PI','PI']
    multi_index = pd.MultiIndex.from_product([MDC_keys+['total','total (no negatives)'],models+['data']],names=['disease_group','model'])
    reductions = pd.DataFrame(index=multi_index,columns=['mean','lower','upper'])
    QALYs = pd.DataFrame(index=multi_index,columns=['mean','lower','upper'])
    result_table = pd.DataFrame(index=multi_index,columns=['reduction','QALY'])

    for model,out in zip(models,outs_full):
        delta_t = pd.Timedelta((out.date[-1]-out.date[0]).values)/pd.Timedelta('1D')
        date_range = pd.date_range(out.date[0].values,out.date[-1].values)

        mean_total_QALY_loss,lower_total_QALY_loss,upper_total_QALY_loss = 0,0,0
        mean_total_QALY_loss_noN,lower_total_QALY_loss_noN,upper_total_QALY_loss_noN = 0,0,0
        mean_total_difference,lower_total_difference,upper_total_difference = 0,0,0
        mean_total_difference_noN,lower_total_difference_noN,upper_total_difference_noN = 0,0,0
        total_reference, total_reference_noN = 0,0
        
        for MDC_key in MDC_keys:
            # MDC specific
            reference = sum(postponed_healthcare.hospitalizations_baseline_mean_smooth.loc[(date_range,MDC_key)].values/postponed_healthcare.mean_residence_times[MDC_key])
            #if model == 'queuing_model':
            #    mean_difference += out['NR'].sel(MDC=MDC_key).mean('draws').values[-1]
            #    lower_difference += out['NR'].sel(MDC=MDC_key).quantile(dim='draws', q=0.025).values[-1]
            #    upper_difference += out['NR'].sel(MDC=MDC_key).quantile(dim='draws', q=0.975).values[-1]
            #else:
            mean_difference  = reference-sum(postponed_healthcare.hospitalizations_baseline_mean_smooth.loc[(date_range,MDC_key)].values*out['H_norm'].sel(MDC=MDC_key).mean('draws').values/postponed_healthcare.mean_residence_times[MDC_key])
            upper_difference = reference-sum(postponed_healthcare.hospitalizations_baseline_mean_smooth.loc[(date_range,MDC_key)].values*out['H_norm'].sel(MDC=MDC_key).quantile(dim='draws', q=0.025).values/postponed_healthcare.mean_residence_times[MDC_key])
            lower_difference = reference-sum(postponed_healthcare.hospitalizations_baseline_mean_smooth.loc[(date_range,MDC_key)].values*out['H_norm'].sel(MDC=MDC_key).quantile(dim='draws', q=0.975).values/postponed_healthcare.mean_residence_times[MDC_key])

            mean_r = mean_difference/reference
            lower_r = lower_difference/reference
            upper_r = upper_difference/reference

            delta_t = pd.Timedelta((outs[1].date[-1]-outs[0].date[0]).values)/pd.Timedelta('1D')
            mean_QALY = delta_t/365*mean_r*hospital_yearly_QALYs.loc[MDC_key]['yearly_QALYs_mean']
            lower_QALY = delta_t/365*lower_r*hospital_yearly_QALYs.loc[MDC_key]['yearly_QALYs_lower']
            upper_QALY = delta_t/365*upper_r*hospital_yearly_QALYs.loc[MDC_key]['yearly_QALYs_upper']

            # totals
            total_reference += reference

            mean_total_difference += mean_difference
            lower_total_difference += lower_difference
            upper_total_difference += upper_difference

            if not math.isnan(mean_QALY):
                mean_total_QALY_loss += mean_QALY
            if not math.isnan(lower_QALY):
                lower_total_QALY_loss += lower_QALY
            if not math.isnan(upper_QALY):
                upper_total_QALY_loss += upper_QALY

            if mean_difference > 0:
                total_reference_noN += reference

                mean_total_difference_noN += mean_difference
                lower_total_difference_noN += lower_difference
                upper_total_difference_noN += upper_difference

                if not math.isnan(mean_QALY):
                    mean_total_QALY_loss_noN += mean_QALY
                if not math.isnan(lower_QALY):
                    lower_total_QALY_loss_noN += lower_QALY
                if not math.isnan(upper_QALY):
                    upper_total_QALY_loss_noN += upper_QALY

            # writing data
            reductions.loc[(MDC_key,model)]['mean'] = mean_r 
            reductions.loc[(MDC_key,model)]['lower'] = lower_r 
            reductions.loc[(MDC_key,model)]['upper'] = upper_r
            result_table.loc[(MDC_key,model)]['reduction'] = f'{mean_r*100:.2f} ({lower_r*100:.2f};{upper_r*100:.2f})'

            QALYs.loc[(MDC_key,model)]['mean'] = mean_QALY 
            QALYs.loc[(MDC_key,model)]['lower'] = lower_QALY 
            QALYs.loc[(MDC_key,model)]['upper'] = upper_QALY
            result_table.loc[(MDC_key,model)]['QALY'] = f'{mean_QALY:.0f} ({lower_QALY:.0f};{upper_QALY:.0f})'

        # calculate total reduction
        for (mean_difference,lower_difference, upper_difference),(mean_QALY_loss,lower_QALY_loss, upper_QALY_loss), reference, total in zip(((mean_total_difference,lower_total_difference,upper_total_difference),
                                                                                                                                            (mean_total_difference_noN,lower_total_difference_noN,upper_total_difference_noN)),
                                                                                                                                            ((mean_total_QALY_loss,lower_total_QALY_loss,upper_total_QALY_loss),
                                                                                                                                            (mean_total_QALY_loss_noN,lower_total_QALY_loss_noN,upper_total_QALY_loss_noN)),
                                                                                                                                            (total_reference,total_reference_noN),
                                                                                                                                            ('total','total (no negatives)')):
            mean_r = mean_difference/reference
            lower_r = lower_difference/reference
            upper_r = upper_difference/reference

            reductions.loc[(total,model)]['mean'] = mean_r 
            reductions.loc[(total,model)]['lower'] = lower_r 
            reductions.loc[(total,model)]['upper'] = upper_r
            result_table.loc[(total,model)]['reduction'] = f'{mean_r*100:.2f} ({lower_r*100:.2f};{upper_r*100:.2f})'

            QALYs.loc[(total,model)]['mean'] = mean_QALY_loss 
            QALYs.loc[(total,model)]['lower'] = lower_QALY_loss 
            QALYs.loc[(total,model)]['upper'] = upper_QALY_loss
            result_table.loc[(total,model)]['QALY'] = f'{mean_QALY_loss:.0f} ({lower_QALY_loss:.0f};{upper_QALY_loss:.0f})'

    # data
    date_range = pd.date_range(dates[0],dates[-1])
    mean_total_QALY_loss,lower_total_QALY_loss,upper_total_QALY_loss = 0,0,0
    mean_total_QALY_loss_noN,lower_total_QALY_loss_noN,upper_total_QALY_loss_noN = 0,0,0
    mean_total_difference, mean_total_difference_noN = 0,0
    mean_total_reference,lower_total_reference,upper_total_reference = 0,0,0 
    mean_total_reference_noN,lower_total_reference_noN,upper_total_reference_noN = 0,0,0 
    for MDC_key in MDC_keys:
        # MDC specific
        mean_reference = sum(postponed_healthcare.hospitalizations_baseline_mean_smooth.loc[(date_range,MDC_key)].values/postponed_healthcare.mean_residence_times[MDC_key])
        lower_reference = sum(postponed_healthcare.hospitalizations_baseline_lower_smooth.loc[(date_range,MDC_key)].values/postponed_healthcare.mean_residence_times[MDC_key])
        upper_reference = sum(postponed_healthcare.hospitalizations_baseline_upper_smooth.loc[(date_range,MDC_key)].values/postponed_healthcare.mean_residence_times[MDC_key])

        mean_difference = mean_reference-sum(postponed_healthcare.hospitalizations_smooth.loc[(date_range,MDC_key)]/postponed_healthcare.mean_residence_times[MDC_key])
        
        mean_r = mean_difference/mean_reference
        lower_r = mean_difference/upper_reference
        upper_r = mean_difference/lower_reference

        delta_t = pd.Timedelta((dates[-1]-dates[0]))/pd.Timedelta('1D')
        mean_QALY = delta_t/365*mean_r*hospital_yearly_QALYs.loc[MDC_key]['yearly_QALYs_mean']
        lower_QALY = delta_t/365*lower_r*hospital_yearly_QALYs.loc[MDC_key]['yearly_QALYs_lower']
        upper_QALY = delta_t/365*upper_r*hospital_yearly_QALYs.loc[MDC_key]['yearly_QALYs_upper']
        
        # writing data
        reductions.loc[(MDC_key,'data')]['mean'] = mean_r 
        reductions.loc[(MDC_key,'data')]['lower'] = lower_r 
        reductions.loc[(MDC_key,'data')]['upper'] = upper_r
        result_table.loc[(MDC_key,'data')]['reduction'] = f'{mean_r*100:.2f} ({lower_r*100:.2f};{upper_r*100:.2f})'

        QALYs.loc[(MDC_key,'data')]['mean'] = mean_QALY 
        QALYs.loc[(MDC_key,'data')]['lower'] = lower_QALY 
        QALYs.loc[(MDC_key,'data')]['upper'] = upper_QALY
        result_table.loc[(MDC_key,'data')]['QALY'] = f'{mean_QALY:.0f} ({lower_QALY:.0f};{upper_QALY:.0f})'

        # totals
        mean_total_reference += mean_reference
        lower_total_reference += lower_reference
        upper_total_reference += upper_reference

        mean_total_difference += mean_difference

        if not math.isnan(mean_QALY):
            mean_total_QALY_loss += mean_QALY
        if not math.isnan(lower_QALY):
            lower_total_QALY_loss += lower_QALY
        if not math.isnan(upper_QALY):
            upper_total_QALY_loss += upper_QALY
        
        if mean_difference > 0:
            mean_total_reference_noN += mean_reference
            lower_total_reference_noN += lower_reference
            upper_total_reference_noN += upper_reference

            mean_total_difference_noN += mean_difference

            if not math.isnan(mean_QALY):
                mean_total_QALY_loss_noN += mean_QALY
            if not math.isnan(lower_QALY):
                lower_total_QALY_loss_noN += lower_QALY
            if not math.isnan(upper_QALY):
                upper_total_QALY_loss_noN += upper_QALY 

    # calculate total reduction
    for (mean_reference,lower_reference, upper_reference),(mean_QALY_loss,lower_QALY_loss, upper_QALY_loss), difference, total in zip(((mean_total_reference,lower_total_reference,upper_total_reference),
                                                                                                                                            (mean_total_reference_noN,lower_total_reference_noN,upper_total_reference_noN)),
                                                                                                                                            ((mean_total_QALY_loss,lower_total_QALY_loss,upper_total_QALY_loss),
                                                                                                                                            (mean_total_QALY_loss_noN,lower_total_QALY_loss_noN,upper_total_QALY_loss_noN)),
                                                                                                                                            (mean_total_difference,mean_total_difference_noN),
                                                                                                                                            ('total','total (no negatives)')):
        mean_r = difference/mean_reference
        lower_r = difference/lower_reference
        upper_r = difference/upper_reference

        reductions.loc[(total,'data')]['mean'] = mean_r 
        reductions.loc[(total,'data')]['lower'] = lower_r 
        reductions.loc[(total,'data')]['upper'] = upper_r
        result_table.loc[(total,'data')]['reduction'] = f'{mean_r*100:.2f} ({lower_r*100:.2f};{upper_r*100:.2f})'

        QALYs.loc[(total,'data')]['mean'] = mean_QALY_loss 
        QALYs.loc[(total,'data')]['lower'] = lower_QALY_loss 
        QALYs.loc[(total,'data')]['upper'] = upper_QALY_loss
        result_table.loc[(total,'data')]['QALY'] = f'{mean_QALY_loss:.0f} ({lower_QALY_loss:.0f};{upper_QALY_loss:.0f})'
        
    reductions.to_csv(os.path.join(result_folder,'hospitalization_reductions.csv'))
    QALYs.to_csv(os.path.join(result_folder,'postponed_healthcare_QALY_loss.csv'))
    result_table.to_csv(os.path.join(result_folder,'postponed_healthcare_summary.csv'))