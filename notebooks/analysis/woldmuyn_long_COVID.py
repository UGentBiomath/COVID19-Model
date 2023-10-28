"""
This script combines the output of the pandemic model with the average QALY losses due to long-COVID and COVID-19 deaths
Visualizations are saved to results/preprocessing/QALY/long_COVID

""" 

__author__      = "Wolf Demuynck"
__copyright__   = "Copyright (c) 2022 by W. Demuynck, BIOMATH, Ghent University. All Rights Reserved."

from covid19_DTM.models.utils import output_to_visuals
from covid19_DTM.models.utils import initialize_COVID19_SEIQRD_hybrid_vacc
from covid19_DTM.visualization.output import _apply_tick_locator 
from covid19_DTM.models.QALY import lost_QALYs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
colorscale_okabe_ito = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
                        "green" : "#009E73", "yellow" : "#F0E442",
                        "blue" : "#0072B2", "red" : "#D55E00",
                        "pink" : "#CC79A7", "black" : "#000000"}

if __name__ == '__main__':
    ################################
    ## Define simulation settings ##
    ################################

    # Number of simulations
    N=10
    # Number of neg. binomial draws/ simulation
    K=20
    # Number of cpu's
    processes=18
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
    out_AD = lost_QALYs(out,AD_non_hospitalised=True)
    out_no_AD = lost_QALYs(out,AD_non_hospitalised=False)

    ####################
    ## Visualisations ##
    ####################

    print('\n4) Visualise results')

    abs_dir = os.path.dirname(__file__)
    result_folder = '../../results/covid19_DTM/analysis/QALY/long_COVID'

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
      ax.set_ylim([0,7000])
    axes[0].legend(loc=2, framealpha=1, fontsize=10)
    axes[0].set_ylabel('QALYs lost per 100K inhab.', size=10)

    plt.tight_layout()
    plt.show()
    fig.savefig('QALY_losses_per_age_group.pdf')
    plt.close()

    for scenario,out in zip(['no_AD','AD'],[out_no_AD,out_AD]):

      # With confidence interval
      df_2plot = output_to_visuals(out, states, alpha=dispersion, n_draws_per_sample=K, UL=1-conf_int*0.5, LL=conf_int*0.5)
      simtime = out['date'].values

      fig,axs = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(12,10))
      axs=axs.reshape(-1)
      for ax, QALYs, title, color in zip(axs, states,titles,colors):

          ax.plot(df_2plot[QALYs,'mean'],'--', color=color)
          ax.fill_between(simtime, df_2plot[QALYs,'lower'], df_2plot[QALYs,'upper'],alpha=0.20, color = color)

          ax = _apply_tick_locator(ax)
          ax.set_title(title,fontsize=20)
          ax.set_ylabel('lost QALYs')
          ax.grid(False)

      plt.subplots_adjust(hspace=0.5)
      fig.savefig(os.path.join(abs_dir,result_folder,f'QALY_losses_{scenario}.pdf'))

      # QALYS per age group
      Palette=cm.get_cmap('tab10_r', initN.size).colors
      age_group=['0-12','12-18','18-25','25-35','35-45','45-55','55-65','65-75','75-85','85+']

      fig, axs = plt.subplots(4,figsize=(10,10),sharex=True)
      axs=axs.reshape(-1)
      for ax, QALYs, title, color in zip(axs,states,titles,colors):

        ax.stackplot(simtime,np.transpose(out[QALYs].mean(dim="draws").sum(dim='doses').values),linewidth=3, labels=age_group, colors=Palette, alpha=0.8)
        ax.set_title(title,fontsize=20)
        ax.set_ylabel('lost QALYs')
        ax = _apply_tick_locator(ax) 
        ax.grid(False)
      axs[0].legend(fancybox=True, frameon=True, framealpha=1, fontsize=15,title='Age Group', loc="upper left", bbox_to_anchor=(1,1)) 

      plt.subplots_adjust(hspace=0.5)
      fig.savefig(os.path.join(abs_dir,result_folder,f'QALY_losses_per_age_group_{scenario}.pdf'))