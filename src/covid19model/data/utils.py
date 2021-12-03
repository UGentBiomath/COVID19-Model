import os
import pandas as pd
import numpy as np

def construct_initN(age_classes=None, spatial=None):
    """
    Returns the initial number of susceptibles conform the user-defined age groups and spatial aggregation.

    Parameters
	----------

    age_classes : pd.IntervalIndex
        Desired age groups in the model, initialize as follows:
        age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,110)], closed='left')
        Alternatively: None --> no grouping in age bins but data/age
    spatial : string
        Can be either None (default), 'mun', 'arr' or 'prov' for various levels of geographical stratification. Note that
        'prov' contains the arrondissement Brussels-Capital. When 'test' is chosen, the mobility matrix for the test scenario is provided:
        mobility between Antwerp, Brussels-Capital and Ghent only (all other outgoing traffic is kept inside the home arrondissement).

    Returns
    -------

    initN : np.array (size: n_spatial_patches * n_age_groups)
        Number of individuals per age group and per geographic location. Used to initialize the number of susceptibles in the model.

    """

    abs_dir = os.path.dirname(__file__)

    if spatial == 'mun':
        age_struct = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demographic/age_structure_per_mun.csv'))
    elif spatial == 'arr':
        age_struct = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demographic/age_structure_per_arr.csv'))
    elif spatial == 'prov':
        age_struct = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demographic/age_structure_per_prov.csv'))
    else:
        age_struct = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demographic/age_structure_per_prov.csv'))

    if age_classes is not None:
        age_struct['age_class'] = pd.cut(age_struct.age, bins=age_classes)
        age_piramid = age_struct.groupby(['NIS','age_class']).sum().reset_index()
        initN = age_piramid.pivot(index='NIS', columns='age_class', values='number')
    else:
        age_piramid = age_struct.groupby(['NIS','age']).sum().reset_index()
        initN = age_piramid.pivot(index='NIS', columns='age', values='number')
        initN = initN.fillna(0)
        
    initN.columns = initN.columns.astype(str)

    if spatial:
        return initN
    else:
        return initN.sum(axis=0)

def convert_age_stratified_property(data, age_classes):
    """ 
    Given an age-stratified series of a (non-cumulative) population property: [age_group_lower, age_group_upper] : property,
    this function can convert the data into another user-defined age-stratification using demographic weighing

    Parameters
    ----------
    data: pd.Series
        A series of age-stratified data. Index must be of type pd.Intervalindex.
    
    age_classes : pd.IntervalIndex
        Desired age groups of the converted table.

    Returns
    -------

    out: pd.Series
        Converted data.
    """

    # Pre-allocate new series
    out = pd.Series(index = age_classes)
    out_n_individuals = construct_initN(age_classes)
    # Extract demographics for all ages
    demographics = construct_initN(None,None)
    # Loop over desired intervals
    for idx,interval in enumerate(age_classes):
        result = []
        for age in range(interval.left, interval.right):
            try:
                result.append(demographics[age]/out_n_individuals[idx]*data.iloc[np.where(data.index.contains(age))[0][0]])
            except:
                result.append(0/out_n_individuals[idx]*data.iloc[np.where(data.index.contains(age))[0][0]])
        out.iloc[idx] = sum(result)
    return out

def convert_age_stratified_quantity(data, age_classes, spatial=None, NIS=None):
        """ 
        Given an age-stratified series of some quantity: [age_group_lower, age_group_upper] : quantity,
        this function can convert the data into another user-defined age-stratification using demographic weighing

        Parameters
        ----------
        data: pd.Series
            A series of age-stratified vaccination incidences. Index must be of type pd.Intervalindex.
        
        age_classes : pd.IntervalIndex
            Desired age groups of the vaccination dataframe.

        spatial: str
            Spatial aggregation: prov, arr or mun
        
        NIS : str
            NIS code of consired spatial element

        Returns
        -------

        out: pd.Series
            Converted data.
        """

        # Pre-allocate new series
        out = pd.Series(index = age_classes, dtype=float)
        # Extract demographics
        if spatial: 
            data_n_individuals = construct_initN(data.index, spatial).loc[NIS,:].values
            demographics = construct_initN(None, spatial).loc[NIS,:].values
        else:
            data_n_individuals = construct_initN(data.index, spatial).values
            demographics = construct_initN(None, spatial).values
        # Loop over desired intervals
        for idx,interval in enumerate(age_classes):
            result = []
            for age in range(interval.left, interval.right):
                try:
                    result.append(demographics[age]/data_n_individuals[data.index.contains(age)]*data.iloc[np.where(data.index.contains(age))[0][0]])
                except:
                    result.append(0/data_n_individuals[data.index.contains(age)]*data.iloc[np.where(data.index.contains(age))[0][0]])
            out.iloc[idx] = sum(result)
        return out
