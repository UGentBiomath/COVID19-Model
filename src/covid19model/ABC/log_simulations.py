# -*- coding: utf-8 -*-
"""
Author : Lander De Visscher

Date : 18/03/2019

Description :  
    Homebrew functions to log parameter values during simulations.
"""

import logging
import types
import git

# =============================================================================
# %% Version controle functions
# =============================================================================

########################  FUNCTION DEFINITIONS  ###############################
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



def get_last_git_commit():
    """
    Get the details of the current git commit.

    Returns
    -------
    branch, str
        name of active brach.
    commit_name : str
        summary of commit.
    commit_hash : str
        hash of commit.

    """
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch
    commit_name = branch.commit.summary
    commit_hash = str(branch.commit)
    return str(branch), commit_name, commit_hash
    
# =============================================================================
# %% Logger functions
# =============================================================================

def initialise_logger(name, 
                      logfile, 
                      file_format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                      print_to_console = False,
                      console_format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                      level = "INFO"):
    """
    Initialise a logger object (from the logging module), to write logs to 
    a specified logfile.

    Parameters
    ----------
    name : str
        logger name.
    logfile : str
        name/path to the log file.
        
    file_format_str : str, optional
        Specifies the format of the log messages to the log file 
        (see documentation of logging module). 
        The default is '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.
        
    print_to_console : bool, optional
        Switch to print log messages to console. The default is True.
    console_format_str : str, optional
        When `print_to_console == True`, this specifies the format of the log 
        messages to the console (see documentation of logging module). 
        The default is '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.
    
    Returns
    -------
    logger : logger object

    """
    # basic configurations
    #=====================
    logging.basicConfig(level=getattr(logging,level))
    logger = logging.getLogger(name)
    
    # file log handler
    #=================
    file_log_handler = logging.FileHandler(logfile)
    file_log_handler.setFormatter(logging.Formatter(file_format_str))
    logger.addHandler(file_log_handler)
    
    # console log handler
    #====================
    if print_to_console:
        console_log_handler = logging.StreamHandler()
        console_log_handler.setFormatter(logging.Formatter(console_format_str))
        logger.addHandler(console_log_handler)
    
    return logger

