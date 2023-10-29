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
from matplotlib import font_manager
import xarray as xar
import math
from QALY_model.postponed_healthcare_models import Postponed_healthcare_models_and_data, draw_fcn
postponed_healthcare = Postponed_healthcare_models_and_data()

parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--hpc", help="dissable CMU font type", action="store_true")
parser.add_argument("-n", "--N", help="simulation runs", default=50)
args = parser.parse_args()

N = int(args.N)
hpc = args.hpc

########
# Data #
########
abs_dir = os.path.dirname(__file__)

rel_dir = '../../data/QALY_model/interim/postponed_healthcare'
file_name = 'hospital_yearly_QALYs.csv'
hospital_yearly_QALYs = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0],na_values='/')

###########
# Samples #
###########

rel_dir = '../../data/QALY_model/interim/postponed_healthcare/model_parameters'

# Load samples
file_names_first = ['queuing_model_SAMPLES_first.json',
                    'constrained_PI_SAMPLES_first.json',
                    'PI_SAMPLES_first.json']
file_names_second = ['queuing_model_SAMPLES_second.json',
                     'constrained_PI_SAMPLES_second.json',
                     'PI_SAMPLES_second.json']
sample_names = ['queuing_model','constrained_PI','PI']

samples = {'first_wave':{},'second_wave':{}}
for period,file_names in zip(samples.keys(),[file_names_first,file_names_second]):
    for file_name,sample_name in zip(file_names,sample_names):
        f = open(os.path.join(abs_dir,rel_dir,file_name))
        samples[period].update({sample_name:json.load(f)})

#########
# setup #
#########

start_calibration_first = pd.to_datetime('2020-01-01')
end_calibration_first = pd.to_datetime('2020-08-01')
start_calibration_second = pd.to_datetime('2020-08-01')
end_calibration_second = pd.to_datetime('2021-03-01')

MDC_keys = postponed_healthcare.all_MDC_keys
MDC_plot = ['03', '04', '05']
dates = pd.date_range('2020-01-01','2021-12-31')

if not hpc:
    label_font = font_manager.FontProperties(family='CMU Sans Serif',
                                    style='normal', 
                                    size=10)
    legend_font = font_manager.FontProperties(family='CMU Sans Serif',
                                    style='normal', 
                                    size=8)
else:
    label_font = font_manager.FontProperties(style='normal', size=10)
    legend_font = font_manager.FontProperties(style='normal', size=8)

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
    axs[0,0].set_title('Queuing model',font=label_font)
    axs[0,1].set_title('Constrained PI Model',font=label_font)
    axs[0,2].set_title('PI Model',font=label_font)

    for i,MDC_key in enumerate(MDC_plot):
        axs[i,0].set_ylabel(MDC_key,font=label_font)
        for j,out in enumerate(outs):

            mean_fit = out['H_norm'].sel(date=plot_time,MDC=MDC_key).mean('draws')
            lower_fit = out['H_norm'].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.025)
            upper_fit = out['H_norm'].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.975)
            mae = MAE(postponed_healthcare.hospitalizations_normalized_smooth,[out],MDC_key)
            axs[i,j].text(0.05, 0.95, f'MAE={mae:.3f}', transform=axs[i,j].transAxes, fontsize=8,verticalalignment='top', bbox=box_props, font=legend_font)

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
    fig.legend(handles=handles,labels=plot_labels,bbox_to_anchor =(0.5,-0.04), loc='lower center',fancybox=False, shadow=False,ncol=5, prop=legend_font)

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