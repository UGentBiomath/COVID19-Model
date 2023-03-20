############################
## Load required packages ##
############################

import os
import json
import datetime
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
# COVID-19 code
from EPNM.models.utils import initialize_model
from EPNM.data.NBB import get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand
# pySODM code
from covid19_DTM.visualization.output import _apply_tick_locator 
from covid19_DTM.models.utils import output_to_visuals
from covid19_DTM.models.utils import load_samples_dict
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-d", "--date", help="Calibration date")
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-p", "--processes", help="Number of cpus used to perform computation", default=int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2)), type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save figures",action='store_true')
args = parser.parse_args()

###############
## Load data ##
###############

data_employment = get_employment_survey()
data_revenue = get_revenue_survey()
data_GDP = get_synthetic_GDP()
data_B2B_demand = get_B2B_demand()

#########################
## Simulation settings ##
#########################

# Start- and enddate simulation
start_sim = data_GDP.index.get_level_values('date').unique().min()
end_sim = data_GDP.index.get_level_values('date').unique().min()
# Confidence level used to visualise model fit
conf_int = 0.05

#############################
## Load samples dictionary ##
#############################

# Path where figures and results should be stored
fig_path = f'../../results/EPNM/calibrations/'
# Path where MCMC samples should be saved
samples_path = f'../../data/EPNM/interim/model_parameters/calibrations/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

samples_dict = load_samples_dict(samples_path+str(args.agg)+'_'+str(args.identifier) + '_SAMPLES_' + str(args.date) + '.json', age_stratification_size=age_stratification_size)

##########################
## Initialize the model ##
##########################

parameters, model = initialize_model()
from EPNM.models.draw_functions import draw_function

########################
## Simulate the model ##
########################

out = model.sim([start_sim, end_sim], N=args.n_samples, draw_function=draw_function, samples=samples_dict, processes=args.processes)
simtime = out['date'].values

###################
## Synthetic GDP ##
###################

