"""
This script combines the output of the pandemic model with the average QALY losses due to long-COVID and COVID-19 deaths
Visualizations are saved to results/preprocessing/QALY/long_COVID

""" 

__author__      = "Wolf Demuynck"
__copyright__   = "Copyright (c) 2022 by W. Demuynck, BIOMATH, Ghent University. All Rights Reserved."

import argparse
from covid19_DTM.models.utils import initialize_COVID19_SEIQRD_hybrid_vacc
from QALY_model.direct_QALYs import lost_QALYs, life_table_QALY_model
life_table = life_table_QALY_model()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import font_manager
import os
import multiprocessing as mp
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--SMR", help="SMR", default=1)
parser.add_argument("-n", "--N", help="DTM simulations", default=50)
args = parser.parse_args()

SMR = float(args.SMR)
N = int(args.N)

# From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
colorscale_okabe_ito = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
                        "green" : "#009E73", "yellow" : "#F0E442",
                        "blue" : "#0072B2", "red" : "#D55E00",
                        "pink" : "#CC79A7", "black" : "#000000"}

if __name__ == '__main__':
    ################################
    ## Define simulation settings ##
    ################################

    # Number of neg. binomial draws/ simulation
    K=10
    # Number of cpu's
    processes=int(mp.cpu_count()/2)
    # Number of age groups
    age_stratification_size=10
    # End of simulation
    end_sim='2021-01-01'
    # Confidence level used to visualise model fit
    conf_int=0.05

    ##########################
    ## Initialize the model ##
    ##########################

    print('\n1) Initialize model')

    model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_hybrid_vacc(age_stratification_size=age_stratification_size, start_date='2020-03-15')

    warmup = float(BASE_samples_dict['warmup'])
    dispersion = float(BASE_samples_dict['dispersion'])
    start_sim = BASE_samples_dict['start_calibration']

    #########################
    ## Perform simulations ##
    #########################

    from covid19_DTM.models.draw_functions import draw_fnc_COVID19_SEIQRD_hybrid_vacc as draw_function

    print('\n2) Simulating COVID19_SEIQRD_hybrid_vacc '+str(N)+' times')
    out = model.sim([start_sim,end_sim], warmup=warmup, processes=processes, N=N, samples=BASE_samples_dict, draw_function=draw_function)

    #######################
    ## QALY calculations ##
    #######################

    print('\n3) Calculating QALYs')
    out_AD = lost_QALYs(out,AD_non_hospitalised=True,SMR=SMR)
    out_no_AD = lost_QALYs(out,AD_non_hospitalised=False,SMR=SMR)

    ####################
    ## Visualisations ##
    ####################

    print('\n4) Visualise results')

    abs_dir = os.path.dirname(__file__)
    result_folder = os.path.join(abs_dir,'../../results/QALY_model/direct_QALYs/analysis/')
    
    # Barplot (addition TWA)
    states = ['QALY_NH', 'QALY_C', 'QALY_ICU','QALY_D']
    titles = ['Non-hospitalised', 'Non-hospitalised (no IC)', 'Non-hospitalised (IC)','Deaths']
    colors = ['green','yellow','red','black']

    age_groups=['0-12','12-18','18-25','25-35','35-45','45-55','55-65','65-75','75-85','85+']

    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(8.3,0.25*11.7))
    for ax,scenario,out in zip(axes, ['no_AD','AD'], [out_no_AD,out_AD]):
      # group data
      data = [out['QALY_D'].mean(dim="draws").sum(dim='doses').isel(date=-1).values/initN*100000,
              out['QALY_NH'].mean(dim="draws").sum(dim='doses').isel(date=-1).values/initN*100000,
              out['QALY_C'].mean(dim="draws").sum(dim='doses').isel(date=-1).values/initN*100000,
              out['QALY_ICU'].mean(dim="draws").sum(dim='doses').isel(date=-1).values/initN*100000
              ]
      # settings
      colors = ['grey', colorscale_okabe_ito['green'], colorscale_okabe_ito['orange'], colorscale_okabe_ito['red']]
      hatches = ['....','////','\\\\\\','||||']
      labels = ['Deaths', 'Non-hospitalised', 'Hospitalised (no IC)', 'Hospitalised (IC)']
      # plot data
      bottom=np.zeros(len(age_groups))
      for d, color, hatch, label in zip(data, colors, hatches, labels):
        p = ax.bar(age_groups, d, color=color, hatch=hatch, label=label, bottom=bottom, linewidth=0.25, alpha=0.7)
        bottom += d
      ax.grid(False)
      ax.tick_params(axis='both', which='major', labelsize=10)
      ax.tick_params(axis='x', which='major', rotation=30)
      ax.set_ylim([0,3200])
    axes[0].legend(loc=2, framealpha=1, fontsize=10)
    axes[0].set_ylabel('QALYs lost per 100K inhab.', size=10)

    plt.tight_layout()
    fig.savefig(result_folder+'QALY_losses_per_age_group.pdf')

    # Wolf's code starts here
    label_font = font_manager.FontProperties(family='CMU Sans Serif',
                                    style='normal', 
                                    size=10)
    legend_font = font_manager.FontProperties(family='CMU Sans Serif',
                                    style='normal', 
                                    size=8)

    # Verify that the paths exist and if not, generate them
    if not os.path.exists(os.path.join(abs_dir,result_folder)):
        os.makedirs(os.path.join(abs_dir,result_folder))

    states = ['QALY_D','QALY_NH', 'QALY_C', 'QALY_ICU']
    titles = ['Deaths','Non-hospitalised', 'Cohort', 'ICU']
    QALYs = {'Non-hospitalised (no AD)':'QALY_NH','Non-hospitalised (AD)':'QALY_NH','Cohort':'QALY_C','ICU':'QALY_ICU','Deaths':'QALY_D'}

    colors = ['black','green','orange','red']
    palette = cm.get_cmap('tab10').colors
    palette_colors = {'black':palette[7],'green':palette[2],'orange':palette[1],'red':palette[3]}

    # These twolines may have come from the master branch
    age_groups = out_AD.coords['age_groups'].values
    age_group_labels = ['0-12','12-18','18-25','25-35','35-45','45-55','55-65','65-75','75-85','85+']

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
        
        QALY_table[total_label]['Total'] = f'{mean:.0f}\n({lower:.0f}; {upper:.0f})'

    QALY_table.to_csv(os.path.join(result_folder,f'Long_COVID_summary_SMR{SMR*100:.0f}.csv'))