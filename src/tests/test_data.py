
import pytest

import pandas as pd

from covid19model.data.sciensano import get_sciensano_COVID19_data
from covid19model.data.mobility import get_google_mobility_data

def test_sciensano_output():
    # check the characteristics of the sciensano data loda function
    df = get_sciensano_COVID19_data(update=False)
    # set of variable names as output
    assert set(df.columns) == set(["H_tot", "ICU_tot", "H_in", "H_out", "H_tot_cumsum", "D_tot", "D_25_44", "D_45_64", "D_65_74", "D_75_84", "D_85+"])
    # index is a datetime index with daily frequency
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freq == 'D'

def test_google_output():
    # check the characteristics of the sciensano data loda function
    df = get_google_mobility_data(update=False)
    # set of variable names as output
    assert set(df.columns) == set(['retail_recreation', 'grocery',
                                   'parks', 'transport', 'work',
                                   'residential'])
    # index is a datetime index with daily frequency
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freq == 'D'