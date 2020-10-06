import os
import datetime
import pandas as pd
import numpy as np

def get_interaction_matrices(spatial=None):
    """Extract interaction matrices and demographic data from `data/raw/polymod` folder

    This function returns the total number of individuals in ten year age bins in the Belgian population and the interaction matrices Nc at home, at work, in schools, on public transport, during leisure activities and during other activities.
    
    The total number of individuals may be spatially stratified per municipality, per arrondissement, or per province (including Brussels-Capital). The final value in the list is the total.
    
    Parameters
    ----------
    spatial : string
        either 'mun', 'arr', or 'prov'. Default is None

    Returns
    -----------
    initN : np.array (10)
        number of Belgian individuals, regardless of sex, in ten year age bins, including the total number
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

    abs_dir = os.path.dirname(__file__)
    polymod_path = os.path.join(abs_dir, "../../../data/raw/polymod/")

    # Data source
    Nc_home = np.loadtxt(os.path.join(polymod_path, "interaction_matrices/Belgium/BELhome.txt"), dtype='f', delimiter='\t')
    Nc_work = np.loadtxt(os.path.join(polymod_path, "interaction_matrices/Belgium/BELwork.txt"), dtype='f', delimiter='\t')
    Nc_schools = np.loadtxt(os.path.join(polymod_path, "interaction_matrices/Belgium/BELschools.txt"), dtype='f', delimiter='\t')
    Nc_transport = np.loadtxt(os.path.join(polymod_path, "interaction_matrices/Belgium/BELtransport.txt"), dtype='f', delimiter='\t')
    Nc_leisure = np.loadtxt(os.path.join(polymod_path, "interaction_matrices/Belgium/BELleisure.txt"), dtype='f', delimiter='\t')
    Nc_others = np.loadtxt(os.path.join(polymod_path, "interaction_matrices/Belgium/BELothers.txt"), dtype='f', delimiter='\t')
    Nc_total = np.loadtxt(os.path.join(polymod_path, "interaction_matrices/Belgium/BELtotal.txt"), dtype='f', delimiter='\t')
    
    # Total population per age class, and per NIS code
    if not spatial:
        # Sum over all municipalities
        initN_df = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/demographic/initN_mun.csv'), index_col='NIS')
        initN = initN_df.sum().values.astype(float)
    
    else:
        if spatial not in ['mun', 'arr', 'prov']:
            raise ValueError(
                        "spatial stratification '{0}' is not legitimate. Possible spatial "
                        "stratifications are 'mun', 'arr', 'prov'".format(spatial)
                    )
        initN_data = '../../../data/interim/demographic/initN_' + spatial + '.csv'
        initN_df = pd.read_csv(os.path.join(abs_dir, initN_data), index_col='NIS')
        initN_df = initN_df.sort_index()
        initN = initN_df.values.astype(float)[:,:-1]

    return initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total
