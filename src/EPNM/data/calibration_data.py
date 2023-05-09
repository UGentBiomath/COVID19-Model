import os
import numpy as np
import pandas as pd
from EPNM.data.parameters import get_model_parameters
from EPNM.data.utils import get_sector_labels, get_sectoral_conversion_matrix

# Set path to interim data folder
abs_dir = os.path.dirname(__file__)
par_interim_path = os.path.join(abs_dir, "../../../data/EPNM/interim/")

def get_NAI_value_added(NACE10=True, relative=True):
    """
    Loads the NAI value added data located in `data/EPNM/interim/calibration_data/NAI_value_added.csv`
    Source: https://www.nbb.be/doc/ts/publications/economicreview/2021/ecorevii2021.pdf

    Parameters
    ==========

    relative: bool
        Relative or absolute data.

    Returns
    =======

    df: pd.Series
        Contains the value added in services (NACE10) during the SARS-CoV-2 pandemic. Two index levels: dates and NACE10 sector
    """
    if NACE10==False:
        # Get raw dataframe
        df = pd.read_csv(os.path.join(par_interim_path,"calibration_data/NAI_value_added.csv"), header=[0],index_col=[1], parse_dates=True)
        # Drop quarter column
        df = df.drop(columns=['quarter'])
        # Stack columns into multiindex
        df = df.stack()
        # Assign a name to the index levels
        df.index = df.index.rename(['date', 'NACE10/NACE21'])
        # Multiply with sectoral output 'x' as proxy for revenue
        if relative == False:
            parameters = get_model_parameters()
            x_NACE10 = np.matmul(get_sectoral_conversion_matrix('NACE21_NACE10'), np.matmul(get_sectoral_conversion_matrix('NACE38_NACE21'), np.matmul(get_sectoral_conversion_matrix('NACE64_NACE38'), parameters['x_0'])))
            x_NACE21 = np.matmul(get_sectoral_conversion_matrix('NACE38_NACE21'), np.matmul(get_sectoral_conversion_matrix('NACE64_NACE38'), parameters['x_0']))
            labels = get_sector_labels('NACE10')
            for sector in df.index.get_level_values('NACE10/NACE21').unique():
                if ((sector != 'O-P') & (sector != 'Q')):
                    i = labels.index(sector)
                    df.loc[slice(None), sector] = df.loc[slice(None), sector].values*x_NACE10[i]
            # O-P/Q manually (NACE10: O-P-Q)
            labels = get_sector_labels('NACE21')
            df.loc[slice(None), 'O-P'] = df.loc[slice(None), 'O-P'].values*(x_NACE21[labels.index('O')] + x_NACE21[labels.index('P')])
            df.loc[slice(None), 'Q'] = df.loc[slice(None), 'Q'].values*x_NACE21[labels.index('Q')]
    else:
        # Get raw dataframe
        df = pd.read_csv(os.path.join(par_interim_path,"calibration_data/NAI_value_added_NACE10.csv"), header=[0],index_col=[1], parse_dates=True)
        # Drop quarter column
        df = df.drop(columns=['quarter'])
        # Stack columns into multiindex
        df = df.stack()
        # Assign a name to the index levels
        df.index = df.index.rename(['date', 'NACE10'])
        # Multiply with sectoral output 'x' as proxy for revenue
        if relative == False:
            parameters = get_model_parameters()
            x_NACE10 = np.matmul(get_sectoral_conversion_matrix('NACE21_NACE10'), np.matmul(get_sectoral_conversion_matrix('NACE38_NACE21'), np.matmul(get_sectoral_conversion_matrix('NACE64_NACE38'), parameters['x_0'])))
            labels = get_sector_labels('NACE10')
            for sector in df.index.get_level_values('NACE10').unique():
                i = labels.index(sector)
                df.loc[slice(None), sector] = df.loc[slice(None), sector].values*x_NACE10[i]

    return df

def get_revenue_survey(relative=True):
    """
    Loads the revenue survey data located in `data/EPNM/interim/calibration_data/ERMG_revenue_survey.xlsx`

    Parameters
    ==========

    relative: bool
        Relative or absolute data.

    Returns
    =======

    df: pd.Series
        Contains the self-reported sectoral (NACE64) revenue declines during the SARS-CoV-2 pandemic. Two index levels: dates and NACE64 sector
    """

    # Get raw dataframe
    df = pd.read_excel(os.path.join(par_interim_path,"calibration_data/ERMG_revenue_survey.xlsx"), header=[0],index_col=[0])
    # Drop all Nan rows
    df.dropna(inplace=True)
    # Arrange into a multiindex
    df = df.stack()
    # Assign a name to the index levels
    df.index = df.index.rename(['NACE64', 'date'])
    # Reconstruct multiindex
    new_index = pd.MultiIndex.from_product([df.index.get_level_values('NACE64').unique(), df.index.get_level_values('date').unique()], names=('NACE64','date'))
    df = pd.Series(df.values, index=new_index, name='revenue_decline')
    # Convert to percentage reductions
    df = (100+df)/100
    df = df.swaplevel()
    df = df.groupby(by=['date', 'NACE64']).sum()
    # Multiply with sectoral output 'x' as proxy for revenue
    if relative == False:
        parameters = get_model_parameters()
        labels = get_sector_labels('NACE64')
        for sector in df.index.get_level_values('NACE64').unique():
            if sector == 'BE':
                df.loc[slice(None), 'BE'] = df.loc[slice(None), 'BE'].values*np.sum(parameters['x_0'])
            else:
                i = labels.index(sector)
                df.loc[slice(None), sector] = df.loc[slice(None), sector].values*parameters['x_0'][i]
    return df

def get_employment_survey(relative=True):
    """
    Loads the employment survey data located in `data/EPNM/interim/calibration_data/ERMG_temporary_unemployment.xlsx`

    Parameters
    ==========

    relative: bool
        Relative or absolute data.

    Returns
    =======

    df: pd.Series
        Contains the sectoral (NACE64) number of employees on temporary unemployment during the SARS-CoV-2 pandemic. Two index levels: dates and NACE64 sector
    """
    # Get raw dataframe
    df = pd.read_excel(os.path.join(par_interim_path,"calibration_data/ERMG_temporary_unemployment.xlsx"), header=[0],index_col=[0])
    # Drop all Nan columns
    df.dropna(axis=1, inplace=True)
    # Arrange into a multiindex
    df = df.stack()
    # Assign a name to the index levels
    df.index = df.index.rename(['date', 'NACE64'])
    # Convert to percentage reductions
    df = (100-df)/100
    # Series name
    df.name = 'temporary_unemployment'
    # Multiply with sectoral labor income 'l' as proxy
    if relative == False:
        parameters = get_model_parameters()
        labels = get_sector_labels('NACE64')
        for sector in df.index.get_level_values('NACE64').unique():
            if sector == 'BE':
                df.loc[slice(None), 'BE'] = df.loc[slice(None), 'BE'].values*np.sum(parameters['l_0'])
            else:
                i = labels.index(sector)
                df.loc[slice(None), sector] = df.loc[slice(None), sector].values*parameters['l_0'][i]
    return df

def get_synthetic_GDP(relative=True):
    """
    Loads the synthetic GDP data located in `data/EPNM/interim/calibration_data/NBB_synthetic_GDP.xlsx`

    Parameters
    ==========

    relative: bool
        Relative or absolute data.

    Returns
    =======

    df: pd.Series
        Contains the sectoral (NACE64) synthetic GDP during the SARS-CoV-2 pandemic. Two index levels: dates and NACE64 sector
    """
    # Get raw dataframe
    df = pd.read_excel(os.path.join(par_interim_path,"calibration_data/NBB_synthetic_GDP.xlsx"), header=[0],index_col=[0])
    # Drop all Nan columns
    df.dropna(axis=1, inplace=True)
    # Arrange into a multiindex
    df = df.stack()
    # Assign a name to the index levels
    df.index = df.index.rename(['date', 'NACE64'])
    # Convert to percentage reductions
    df = (100+df)/100
    # Series name
    df.name = 'synthetic_GDP'
    # Multiply with sectoral output 'x' as proxy for revenue
    if relative == False:
        parameters = get_model_parameters()
        labels = get_sector_labels('NACE64')
        for sector in df.index.get_level_values('NACE64').unique():
            if sector == 'BE':
                df.loc[slice(None), 'BE'] = df.loc[slice(None), 'BE'].values*np.sum(parameters['x_0'])
            else:
                i = labels.index(sector)
                df.loc[slice(None), sector] = df.loc[slice(None), sector].values*parameters['x_0'][i]
    return df

def get_B2B_demand(relative=True):
    """
    Loads the B2B demand data located in `data/EPNM/interim/calibration_data/WoW_Growths.xlsx`

    Parameters
    ==========

    relative: bool
        Relative or absolute data.

    Returns
    =======

    df: pd.Series
        Contains the sectoral (NACE21) synthetic GDP during the SARS-CoV-2 pandemic. Two index levels: dates and NACE21 sector
    """
    # Get raw dataframe
    df = pd.read_excel(os.path.join(par_interim_path,"calibration_data/WoW_Growths.xlsx"), sheet_name='SECTORAL_GROWTHS_REL_100', header=[0],index_col=[0])
    # Reset index
    df = df.reset_index()
    # Convert year+week to datetime format
    df['formatted_date'] = df.year * 1000 + df.week * 10 + 0
    df['date'] = pd.to_datetime(df['formatted_date'], format='%Y%W%w')
    del df['formatted_date']
    del df['week']
    del df['year']
    df = df.set_index(df['date'])
    del df['date']
    # Arrange into a multiindex
    df = df.stack()
    # Assign a name to the index levels
    df.index = df.index.rename(['date', 'NACE21'])
    # Series name
    df.name = 'B2B demand'
    # Multiply with orders 'O_j' as proxy for B2B demand
    if relative == False:
        parameters = get_model_parameters()
        labels = get_sector_labels('NACE21')
        O_j = np.matmul(get_sectoral_conversion_matrix('NACE38_NACE21'), np.matmul(get_sectoral_conversion_matrix('NACE64_NACE38'), parameters['O_j']))
        for sector in df.index.get_level_values('NACE21').unique():
            if sector != 'U':
                i = labels.index(sector)
                df.loc[slice(None), sector] = df.loc[slice(None), sector].values*O_j[i]
    return df/100