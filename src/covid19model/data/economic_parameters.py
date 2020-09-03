import os
import pandas as pd
import numpy as np

def get_economic_parameters():
    """
    Extracts and returns the parameters for the economic model

    This function returns a dictionary with all parameters needed to run the economic model.

    Returns
    -------

    pars_dict : dictionary
        contains the values of all economic parameters

        Parameters
        ------------
        IO: input-output matrix
        x_0 : sectoral output during business-as-usual
        c_0 : household demand during business-as-usual
        f_0 : other final demand during business-as-usual
        n : desired stock
        c_s : consumer demand shock during lockdown
        f_s : other final demand shock during lockdown
        l_0 : sectoral employees during business-as-usual
        l_s : sectoral employees during lockdown
        C : matrix of crictical inputs

    Example use
    -----------
    parameters = get_economic_parameters()
    """

    abs_dir = os.path.dirname(__file__)
    par_interim_path = os.path.join(abs_dir, "../../../data/interim/economical/")

    # Initialize parameters dictionary
    pars_dict = {}

    return pars_dict
