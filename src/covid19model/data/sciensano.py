import pandas as pd
import numpy as np


def get_sciensano_data():
    """
    Function to update the available data on hospitalisation cases (including ICU).
    The data is extracted from Sciensano database: https://epistat.wiv-isp.be/covid/
    Data is reported as showed in: https://epistat.sciensano.be/COVID19BE_codebook.pdf

    Output:
    * initial – initial date of records: string 'YYYY-MM-DD'
    * data – list with total number of patients as [hospital, ICUvect]:
        * hospital - total number of hospitalised patients : array
        * ICUvect - total number of hospitalised patients in ICU: array

    Utilisation: use as [hospital, ICUvect] = getData()
    """

    # Data source
    url = 'https://epistat.sciensano.be/Data/COVID19BE.xlsx'

    # Extract hospitalisation data from source
    df = pd.read_excel(url, sheet_name="HOSP")

    # Date of initial records
    initial = df.astype(str)['DATE'][0]

    # Resample data from all regions and sum all values for each date
    data = df.loc[:,['DATE','TOTAL_IN','TOTAL_IN_ICU']]
    data = data.resample('D', on='DATE').sum()
    hospital = np.array([data.loc[:,'TOTAL_IN'].tolist()]) # export as array
    ICU = np.array([data.loc[:,'TOTAL_IN_ICU'].tolist()]) # export as array

    # List of time datapoints
    index = pd.date_range(initial, freq='D', periods=ICU.size)
    #data.index # equivalently from dataframe index
    # List of daily numbers of ICU and hospitaliside patients
    data = [initial,ICU,hospital]
    return [index, data]