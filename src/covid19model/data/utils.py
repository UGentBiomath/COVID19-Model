import pandas as pd
from covid19model.data.model_parameters import construct_initN

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