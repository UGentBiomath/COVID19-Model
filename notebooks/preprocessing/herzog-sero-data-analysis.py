"""
This script contains all necessary code to extract and convert the serological data from Herzog into a timeseries for model calibration.
The raw data are located in the `~/data/raw/sero/` folder and were downloaded from https://zenodo.org/record/4665373#.YH13QHUzaV4.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

# -------------------------------------------------
# Load and format Sciensano hospital survey dataset
# -------------------------------------------------

df = pd.read_csv('../../data/raw/sero/serology_covid19_belgium_round_1_to_7_v20210406.csv', parse_dates=True)

print(df.head())
