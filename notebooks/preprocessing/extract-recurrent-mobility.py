"""
This script contains all necessary code to extract the recurrent work mobility matrix from the 2011 census, which is usable by the spatial BIOMATH COVID-19 SEIRD model.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import numpy as np
import pandas as pd

