"""
This script combines the output of the pandemic model with the average QALY losses due to long-COVID and COVID-19 deaths
Visualizations are saved to results/preprocessing/QALY/long_COVID

""" 

__author__      = "Wolf Demuynck"
__copyright__   = "Copyright (c) 2022 by W. Demuynck, BIOMATH, Ghent University. All Rights Reserved."

from covid19model.models.utils import output_to_visuals
from covid19model.models.utils import initialize_COVID19_SEIQRD_hybrid_vacc
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.models.QALY import lost_QALYs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

if __name__ == '__main__':
    ################################
    ## Define simulation settings ##
    ################################

    # Number of simulations
    N=10
    # Number of neg. binomial draws/ simulation
    K=20
    # Number of cpu's
    processes=4
    # Number of age groups
    age_stratification_size=10
    # End of simulation
    end_sim='2021-07-01'
    # Confidence level used to visualise model fit
    conf_int=0.05

    ##########################
    ## Initialize the model ##
    ##########################

    print('\n1) Initialize model')

    model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_hybrid_vacc(age_stratification_size=age_stratification_size)

    warmup = float(BASE_samples_dict['warmup'])
    dispersion = float(BASE_samples_dict['dispersion'])
    start_sim = BASE_samples_dict['start_calibration']

    #########################
    ## Perform simulations ##
    #########################

    print('\n2) Simulating COVID19_SEIQRD_hybrid_vacc '+str(N)+' times')
    out = model.sim([start_sim,end_sim], warmup=warmup, processes=processes, N=N, samples=BASE_samples_dict)

    #######################
    ## QALY calculations ##
    #######################

    print('\n3) Calculating QALYs')
    out_AD = lost_QALYs(out,AD_non_hospitalised=True)
    out_no_AD = lost_QALYs(out,AD_non_hospitalised=False)

    ####################
    ## Visualisations ##
    ####################

    print('\n4) Visualise results')

    abs_dir = os.path.dirname(__file__)
    result_folder = '../../../results/analysis/QALY/long_COVID'

    states = ['QALY_NH', 'QALY_C', 'QALY_ICU','QALY_D']
    titles = ['Non-hospitalised', 'Cohort', 'ICU','Deaths']
    colors = ['green','yellow','red','black']

    for scenario,out in zip(['no_AD','AD'],[out_no_AD,out_AD]):
        
      # With confidence interval
      df_2plot = output_to_visuals(out, states, alpha=dispersion, n_draws_per_sample=K, UL=1-conf_int*0.5, LL=conf_int*0.5)
      simtime = out['date'].values

      fig,axs = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(10,10))
      axs=axs.reshape(-1)
      for ax, QALYs, title, color in zip(axs, states,titles,colors):

          ax.plot(df_2plot[QALYs,'mean'],'--', color=color)
          ax.fill_between(simtime, df_2plot[QALYs,'lower'], df_2plot[QALYs,'upper'],alpha=0.20, color = color)

          ax = _apply_tick_locator(ax)
          ax.set_title(title,fontsize=20)
          ax.set_ylabel('lost QALYs')
          ax.grid(False)

      plt.subplots_adjust(hspace=0.5)
      fig.savefig(os.path.join(abs_dir,result_folder,f'QALY_losses_{scenario}.png'))

      # QALYS per age group
      Palette=cm.get_cmap('tab10_r', initN.size).colors
      age_group=['0-12','12-18','18-25','25-35','35-45','45-55','55-65','65-75','75-85','85+']

      fig, axs = plt.subplots(4,figsize=(10,10),sharex=True)
      axs=axs.reshape(-1)
      for ax, QALYs, title, color in zip(axs,states,titles,colors):
        ax.stackplot(simtime,out[QALYs].mean(dim="draws").sum(dim='doses'),linewidth=3, labels=age_group, colors=Palette, alpha=0.8)
        ax.set_title(title,fontsize=20)
        ax.set_ylabel('lost QALYs')
        ax = _apply_tick_locator(ax) 
        ax.grid(False)
      axs[0].legend(fancybox=True, frameon=True, framealpha=1, fontsize=15,title='Age Group', loc="upper left", bbox_to_anchor=(1,1)) 

      plt.subplots_adjust(hspace=0.5)
      fig.savefig(os.path.join(abs_dir,result_folder,f'QALY_losses_per_age_group_{scenario}.png'))