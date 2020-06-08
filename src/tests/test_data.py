
import pytest

from covid19model.data.sciensano import get_sciensano_COVID19_data

def test_sciensano_output():
    # check the characteristics of the sciensano data loda function
    df = get_sciensano_COVID19_data(update=False)
    # set of variable names as output
    assert set(df.columns) == set(["H_tot", "ICU_tot", "H_in", "H_out", "H_tot_cumsum"])
    # index is a datetime index with daily frequency
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freq == 'D'

