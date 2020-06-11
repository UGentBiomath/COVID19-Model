# -*- coding: utf-8 -*-
"""
Tools to store results of simulations.

Created on Mon Feb 10 11:47:00 2020

@author: Administrator
"""


# =============================================================================
# %% IMPORTS
# =============================================================================
import os
from errno import EEXIST
import h5py

import numpy as np

# =============================================================================
# %% FUNCTION DEFINITIONS
# =============================================================================

def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line.
    Source: https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory

    Parameters
    ----------
    mypath : string
        path of directories that have to be created.

    Returns
    -------
    None.

    """
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else: raise

def check_extension(path, ext):
    """
    Asserts that the given path ends with a specified extension.

    Parameters
    ----------
    path : str
        Path to file.
    ext : str
        File extension.

    Returns
    -------
    path : str
        Path with specified extension.

    """
    if not path.endswith(ext):
        if ext[0] == ".":
            path += ext
        else:
            path += "."+ext
    return path
    
def write_results_hdf5(filename,
                       metadata,
                       separate_metadata = True,
                       group = "/",
                       **results):
    """
    Writes simulation results to a hdf5 file. If the given filename refers to 
    an existing file, then the data are appended, otherwise a new file is 
    created.

    Parameters
    ----------
    filename : str
        Path to file.
    metadata : dict
        Metadata of the results.
    group : str, optional
        If given, this specifies a group in the hdf5 file, where the results have to be stored. 
        If None is given, the results are stored in the main file and the 
        metadata are stored as attributes of a "metadata" group.
        The default is root.
    **results : ndarrays
        arrays that have to be written to the hdf5 file.

    Returns
    -------
    filename : str
        Path to hdf5 file.

    """
    # Write (on) file
    #==========================================================================
    filename = check_extension(filename,"hdf5")
    with h5py.File(filename, mode = "a") as file:
        try: 
            # group doesn't exsist yet
            results_group = file.create_group(group)
        except ValueError:
            # group already exists
            results_group = file[group]
        
        if separate_metadata:
            # Add metadata to a group named ""metadata"
            #------------------------------------------------------
            try:
                # group doesn't exsist yet -> make new metadata subgroup
                metadata_group = results_group.create_group("metadata")
            except ValueError:
                # group already exists 
                metadata_group = results_group["metadata"] 
            add_metadata_as_attributes(metadata_group, metadata)
        
        # Add results to file
        #------------------------------------------------------
        for resultname, result in results.items():
            results_group.create_dataset(resultname,data=result)
            if not separate_metadata:
                try: 
                    add_metadata_as_attributes(results_group, metadata[resultname]) 
                except KeyError:
                    add_metadata_as_attributes(results_group, metadata)
    return filename

def add_metadata_as_attributes(metadata_group, metadata_dict):
    """
    Adds metadata to group in hdf5 file.
    
    If metadata is a dict of dicts, then separate subgroups are created for 
    which the items of the subdicts are stored as attributes.

    Parameters
    ----------
    metadata_group : hdf5 group
        group where metadata are stored as attributes.
    metadata_dict : dict
        dict with names and values of metadata.

    Returns
    -------
    None.

    """
    for mdg_name, mdg_val in metadata_dict.items():
                if isinstance(mdg_val, dict): # if dict in dict: create subgroup
                    metadata_subgroup = metadata_group.create_group(mdg_name)
                    add_metadata_as_attributes(metadata_subgroup, mdg_val)
                else: 
                    try: 
                        metadata_group.attrs[mdg_name] = mdg_val
                    except TypeError as te:
                        print(mdg_name)
                        print(mdg_val)
                        raise te

def load_results_hdf5(filename, result_names, group = "/"):
    """
    Loads results, stored in hdf5 file.

    Parameters
    ----------
    filename : string
        Path to file.
    result_names : list
        List contraining the names (`str`) of the datasets which have to be loaded.
    group : string or Nonetype, optional
        Group in which the datasets are stored.
        If None, the results are being looked for at the root.
        The default is None.

    Returns
    -------
    results : list
        Datasets loaded from the hdf5 file.

    """
    filename = check_extension(filename,"hdf5") # asserts that the filename ends with the hdf5 extension
    results = []
    with h5py.File(filename, mode = "r") as file:
        for dataset in result_names:
            results.append(file[group][dataset][()]) # get values from dataset
    return results

def load_group_metadata_hdf5(filename, group = "/"):
    """
    Loads specific metadata of hdf5 file as a dict.

    Parameters
    ----------
    filename : string
        Path to file.
    group : string or Nonetype, optional
        Group in which the datasets are stored.
        If None, the results are being looked for at the root.
        The default is root.

    Returns
    -------
    dict
        Loaded metadata.

    """
    filename = check_extension(filename,"hdf5") # asserts that the filename ends with the hdf5 extension
    with h5py.File(filename, mode = "r") as file:
        return dict(file[group].attrs)
    

    
def show_contents_hdf5(filename, 
                       groups = None, 
                       f_md_ndarray = lambda a:"{}".format(type(a))):
    """
    Prints the contents of an hdf5 file, created with `write_results_hdf5` where 
    separate_metadata == True.
    Parameters
    ----------
    filename : string
        Path to file..
    groups : list or Nonetype, optional
        Groups that have to be printed. If None, al groups in the file are printed. 
        The default is None.
    f_md_ndarray : function, optional
        Function to format ndarray in the metadata. 
        The default is lambda a:"{}".format(type(a)).

    Returns
    -------
    None.

    """
    def print_attributes(group, f_md_ndarray):
        """
        Auxilary function.
        """
        attr = dict(group.attrs)
        for a,val in attr.items():
            if isinstance(val, np.ndarray):
                print(f"\t- {a:15s}:\t"+f_md_ndarray(val))
            else:
                print(f"\t- {a:15s}:\t{val}")
    
    
    filename = check_extension(filename,"hdf5") # asserts that the filename ends with the hdf5 extension
    line_len = 50
    with h5py.File(filename, mode = "r") as file:
        # header: file name
        #-------------------
        print()
        print(filename)
        print("_"*line_len, end = "\n\n\n")
        if groups is None:
            groups = ["/"] # if no groups are given: groups = root
        for g in groups: 
            if g != "/":
                print("#"*line_len)
                print(g)
                print("#"*line_len)
            
            group = file[g]
            
            print("\nMetadata")
            print("="*line_len)
            
            metadata_group = group["metadata"]
            print_attributes(metadata_group,f_md_ndarray)
            
            for subgroup in metadata_group:
                print("-"*line_len)
                print("|-"+subgroup)
                print("-"*line_len)
                print_attributes(metadata_group[subgroup],f_md_ndarray)
                    
            print("\nDatasets")
            print("="*line_len)
            datasets = {}
            for obj in group:
                if isinstance(group[obj], h5py._hl.dataset.Dataset):
                    try:
                        obj_splitted = obj.split("_")
                        assert len(obj_splitted) > 1
                        prefix = "_".join(obj_splitted[0:-1])
                        suffix = obj_splitted[-1]
                        datasets[prefix].append(suffix)
                    except KeyError:
                        datasets[prefix] = [suffix]
                    except AssertionError:
                        datasets[obj] = []
            for ds in datasets:
                try:
                    suffixes = sorted(datasets[ds], key = int)
                    suffixes = ", ".join(suffixes)
                    print(f"{ds}_[{suffixes}]")
                except ValueError:
                    print(f"{ds}")
                



            
def show_all_contents_hdf5(filename):
    """
    Prints contents of hdf5 file.

    Parameters
    ----------
    filename : str
        Path to hdf5 file.

    Returns
    -------
    None.

    """
        
    filename = check_extension(filename,"hdf5") # asserts that the filename ends with the hdf5 extension
    line_len = 50
    
    def print_attrs(name, obj):
        """
        Auxilary function, used in visititems method.

        """
        name_split = name.split("/")
        level = len(name_split)
        if isinstance(obj, h5py._hl.dataset.Dataset):
            name_print = f"{name_split[-1]} : ndarray shape {obj.shape}, type {obj.dtype}"
        else: 
            name_print = name_split[-1]
        print("|  "*(level-1)+"|-", name_print)
        for key, val in obj.attrs.items():
            print("|  "*level+"|---", f"{key:15s}\t: {val}")
            
    
    print(filename)
    print("="*line_len)
    with h5py.File(filename, mode = "r") as file:
        file.visititems(print_attrs)