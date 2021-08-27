
import pytest

import pandas as pd

from covid19model.data.sciensano import get_sciensano_COVID19_data
from covid19model.data.mobility import get_google_mobility_data

def test_sciensano_output():
    # check the characteristics of the sciensano data loda function
    df = get_sciensano_COVID19_data(update=False)
    # set of variable names as output
    assert set(df.columns) == set(['H_tot', 'ICU_tot', 'H_in', 'H_out', 'D_tot', 'D_25_44', 'D_45_64',
       'D_65_74', 'D_75_84', 'D_85+', 'C_tot', 'C_0_9', 'C_10_19', 'C_20_29',
       'C_30_39', 'C_40_49', 'C_50_59', 'C_60_69', 'C_70_79', 'C_80_89',
       'C_90+', 'V1_tot', 'V1_00_11', 'V1_12_15', 'V1_16_17', 'V1_18_24',
       'V1_25_34', 'V1_35_44', 'V1_45_54', 'V1_55_64', 'V1_65_74', 'V1_75_84',
       'V1_85+', 'V2_tot', 'V2_00_11', 'V2_12_15', 'V2_16_17', 'V2_18_24',
       'V2_25_34', 'V2_35_44', 'V2_45_54', 'V2_55_64', 'V2_65_74', 'V2_75_84',
       'V2_85+', 'VJ&J_tot', 'VJ&J_00_11', 'VJ&J_12_15', 'VJ&J_16_17',
       'VJ&J_18_24', 'VJ&J_25_34', 'VJ&J_35_44', 'VJ&J_45_54', 'VJ&J_55_64',
       'VJ&J_65_74', 'VJ&J_75_84', 'VJ&J_85+'])
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
