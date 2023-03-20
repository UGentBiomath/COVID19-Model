import os
import pandas as pd

# Set path to interim data folder
abs_dir = os.path.dirname(__file__)
par_interim_path = os.path.join(abs_dir, "../../../data/EPNM/interim/")

def get_revenue_survey():
    """
    Loads the revenue survey data located in `data/EPNM/interim/calibration_data/ERMG_revenue_survey.xlsx`

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

    return df

def get_employment_survey():
    """
    Loads the employment survey data located in `data/EPNM/interim/calibration_data/ERMG_temporary_unemployment.xlsx`

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
    return df

def get_synthetic_GDP():
    """
    Loads the synthetic GDP data located in `data/EPNM/interim/calibration_data/NBB_synthetic_GDP.xlsx`

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
    return df

def get_B2B_demand():
    """
    Loads the B2B demand data located in `data/EPNM/interim/calibration_data/WoW_Growths.xlsx`

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
    return df