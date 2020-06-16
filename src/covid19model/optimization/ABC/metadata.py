# -*- coding: utf-8 -*-
"""
Tools to add metadata in scripts/notebooks.

Created on Wed Jan  8 12:18:11 2020

@author: Lander De Visscher
"""

# =============================================================================
# %% IMPORTS
# =============================================================================
import types

# =============================================================================
# %% FUNCTION DEFINITIONS
# =============================================================================
def list_module_versions(glob, return_dict = False):
    """
    Prints the versions of all loaded modules/packages in a script or notebook.

    Parameters
    ----------
    glob : dict 
        output of the globals() function call.
    return_dict : bool, optional
        Parameter to decide if function should return versions dict. The default is False.    
    
    Returns
    -------
    versions_dict : dict, optional
        Dict with module names as keys and version numbers as values.

    """
    versions_dict = {}
    
    for name, val in glob.items():
        if isinstance(val, types.ModuleType) and "__version__" in val.__dict__:
            print(val.__name__, val.__version__)
            versions_dict[val.__name__] = val.__version__
            
    if return_dict is True:
        return versions_dict

def get_metadata_from_attributes(Object, skip_attributes = None, custom_classes = None):
    """
    Get metadata dict from attributes of an object.
    

    Parameters
    ----------
    Object : 
        object from which the attributes are.
    skip_attributes : list, optional
        If given, these attributes are skipped (next to the methods of the class of Object). 
        The default is None.
    custom_classes : dict, optional
        Dict where keys are classes and values are functions that specify how 
        objects of this class should be stored in metadata_dict. The default is None.

    Returns
    -------
    metadata_dict : dict
        dict where keys-values are attributes from Object.

    """
    if skip_attributes is None:
        skip_attributes = []
    skip_attributes += dir(type(Object)) # methods of class will be skipped as attributes
    
    if custom_classes is None:
        custom_classes = {}
    metadata_dict = {}
    for a in dir(Object):
        if a not in skip_attributes:
            a_val = getattr(Object,a)
            if a_val is None:
                metadata_dict[a] = "None"
            elif type(a_val) in custom_classes:
                # treat class as specified in custom_classes dict
                metadata_dict[a] = custom_classes[type(a_val)](a_val)
            elif callable(a_val):
                # only get docstrings from callables
                metadata_dict[a] = a_val.__doc__
            else:
                metadata_dict[a] = a_val
    return metadata_dict