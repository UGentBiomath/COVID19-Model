import os
import datetime
import pandas as pd
import numpy as np

def get_interaction_matrices():
    """Extract interaction matrices from data/raw/polymod folder 

    This function returns the total number of individuals in ten year age bins in the Belgian population and the interaction matrices Nc at home, at work, in schools, on public transport, during leisure activities and during other activities.

    Returns
    -----------
    initN : np.array
        number of Belgian individuals, regardless of sex, in ten year age bins
    Nc_home :  np.array (9x9)
        number of daily contacts at home of individuals in age group X with individuals in age group Y
    Nc_work :  np.array (9x9)
        number of daily contacts in the workplace of individuals in age group X with individuals in age group Y 
    Nc_schools :  np.array (9x9)
        number of daily contacts in schools of individuals in age group X with individuals in age group Y 
    Nc_transport :  np.array (9x9)
        number of daily contacts on public transport of individuals in age group X with individuals in age group Y 
    Nc_leisure :  np.array (9x9)
        number of daily contacts during leisure activities of individuals in age group X with individuals in age group Y
    Nc_others :  np.array (9x9)
        number of daily contacts in other places of individuals in age group X with individuals in age group Y 
    Nc_total :  np.array (9x9)
        total number of daily contacts of individuals in age group X with individuals in age group Y, calculated as the sum of all the above interaction

    Notes
    ----------
    The interaction matrices are extracted from the polymod dataset `contacts.Rdata`: https://lwillem.shinyapps.io/socrates_rshiny/.
    The demographic data was retreived from https://statbel.fgov.be/en/themes/population/structure-population

    Example use
    -----------
    initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = get_interaction_matrices()
    """

    # Data source
    Nc_home = np.loadtxt("../../data/raw/polymod/interaction_matrices/Belgium/BELhome.txt", dtype='f', delimiter='\t')
    Nc_work = np.loadtxt("../../data/raw/polymod/interaction_matrices/Belgium/BELwork.txt", dtype='f', delimiter='\t')
    Nc_schools = np.loadtxt("../../data/raw/polymod/interaction_matrices/Belgium/BELschools.txt", dtype='f', delimiter='\t')
    Nc_transport = np.loadtxt("../../data/raw/polymod/interaction_matrices/Belgium/BELtransport.txt", dtype='f', delimiter='\t')
    Nc_leisure = np.loadtxt("../../data/raw/polymod/interaction_matrices/Belgium/BELleisure.txt", dtype='f', delimiter='\t')
    Nc_others = np.loadtxt("../../data/raw/polymod/interaction_matrices/Belgium/BELothers.txt", dtype='f', delimiter='\t')
    Nc_total = np.loadtxt("../../data/raw/polymod/interaction_matrices/Belgium/BELtotal.txt", dtype='f', delimiter='\t')
    initN = np.loadtxt("../../data/raw/polymod/interaction_matrices/Belgium/BELagedist_10year.txt", dtype='f', delimiter='\t')

    return initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total
