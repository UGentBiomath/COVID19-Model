"""
This script extracts and formats the recurrent work mobility matrix from the 2011 census, which is usable by the spatial BIOMATH COVID-19 SEIRD model.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import numpy as np
import pandas as pd

# --------------------
# Load the census data
# --------------------

abs_dir = os.getcwd()
rel_dir = os.path.join(abs_dir, '../../data/interim/census_2011/Pop_LPW_NL_25FEB15_delete_unknown.xlsx')
df = pd.read_excel(rel_dir, sheet_name="Tabel1_2011")

# -----------
# Format data
# -----------

codes=df['00.24 - Werkende bevolking volgens geslacht, verblijfplaats en plaats van tewerkstelling'].loc[5:1942].dropna().values
codes_int = [int(i) for i in codes]
mobility_df=pd.DataFrame(np.zeros([len(codes),len(codes)]),index=codes,columns=codes)
names = df.iloc[5:,1].dropna().values
rows=[]
for i in df['00.24 - Werkende bevolking volgens geslacht, verblijfplaats en plaats van tewerkstelling'].loc[5:1942].dropna().index:
    rows.append(df.iloc[i+2,4:-1].values)

matrix = np.zeros([len(rows),len(rows)])
for j in range(len(rows)):
    matrix[j,:]=rows[j]

mobility_df=pd.DataFrame(matrix,index=codes_int,columns=codes_int)
mobility_df.head()

idx_arrondisement=[]
for idx in mobility_df.index:
    if ((str(idx)[-3:] == '000') & (len(str(idx)) != 4) & (str(idx)[-4:] != '0000') & (str(idx)[0] != '0')):
        idx_arrondisement.append(idx)

# ---------------------
# Save formatted matrix
# ---------------------

mobility_df.loc[idx_arrondisement,idx_arrondisement].to_csv('../../data/interim/census_2011/recurrent_mobility.csv', index=True)