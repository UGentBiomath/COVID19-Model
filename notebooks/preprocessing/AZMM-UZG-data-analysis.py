"""
This script contains all necessary code to extract and convert the patients data from hospitals AZ Maria Middelares and Ghent University hospital into parameters usable by the BIOMATH COVID-19 SEIRD model.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import numpy as np
import pandas as pd
