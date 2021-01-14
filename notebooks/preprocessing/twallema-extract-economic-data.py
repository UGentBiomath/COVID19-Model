"""
This script contains all necessary code to extract and convert the data needed for the UGent production network model.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import numpy as np
import pandas as pd

# ------------------------
# NACE Conversion matrices
# ------------------------

# NACE 21 to NACE 10

columns = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
index = ['A','B, C, D, E','F','G-H-I','J','K','L','M-N','O, P, Q','R, S, T']

NACE_21to10_mat = np.zeros([10,20])
NACE_21to10_mat[0,0] = 1
NACE_21to10_mat[1,1:5] = 1
NACE_21to10_mat[2,5] = 1
NACE_21to10_mat[3,6:9] = 1
NACE_21to10_mat[4,9] = 1
NACE_21to10_mat[5,10] = 1
NACE_21to10_mat[6,11] = 1
NACE_21to10_mat[7,12:14] = 1
NACE_21to10_mat[8,14:17] = 1
NACE_21to10_mat[9,17:20] = 1

NACE21to10 = pd.DataFrame(data=NACE_21to10_mat,columns=columns,index=index)
NACE21to10.head()

# NACE 38 to NACE 21

columns = ['AA','BB','CA','CB','CC','CD','CE','CF','CG','CH','CI','CJ','CK','CL','CM','DD','EE',
            'FF','GG','HH','II','JA','JB','JC','KK','LL','MA','MB','MC','NN','OO','PP','QA','QB',
            'RR','SS','TT']
index = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']

NACE_38to21_mat = np.zeros([20,37])
NACE_38to21_mat[0,0] = 1
NACE_38to21_mat[1,1] = 1
NACE_38to21_mat[2,2:15] = 1
NACE_38to21_mat[3,15] = 1
NACE_38to21_mat[4,16] = 1
NACE_38to21_mat[5,17] = 1
NACE_38to21_mat[6,18] = 1
NACE_38to21_mat[7,19] = 1
NACE_38to21_mat[8,20] = 1
NACE_38to21_mat[9,21:24] = 1
NACE_38to21_mat[10,24] = 1
NACE_38to21_mat[11,25] = 1
NACE_38to21_mat[12,26:29] = 1
NACE_38to21_mat[13,29] = 1
NACE_38to21_mat[14,30] = 1
NACE_38to21_mat[15,31] = 1
NACE_38to21_mat[16,32:34] = 1
NACE_38to21_mat[17,34] = 1
NACE_38to21_mat[18,35] = 1
NACE_38to21_mat[19,36] = 1

NACE38to21 = pd.DataFrame(data=NACE_38to21_mat,columns=columns,index=index)
NACE38to21.head()

# NACE 64 to NACE 38

index = ['AA','BB','CA','CB','CC','CD','CE','CF','CG','CH','CI','CJ','CK','CL','CM','DD','EE',
            'FF','GG','HH','II','JA','JB','JC','KK','LL','MA','MB','MC','NN','OO','PP','QA','QB',
            'RR','SS','TT']

IO_df = pd.read_excel("../../data/raw/economical/input-output.xlsx", sheet_name='tbl_8',index_col=[0], header=[0])
codes64 = list(IO_df.index.values[1:-19])
codes64[-1] = '97-98'
codes64.remove('68a')
codes64[codes64.index('68_')]='68'
codes64 = np.array(codes64)
columns = codes64

NACE_64to38_mat = np.zeros([37,63])
NACE_64to38_mat[0,0:3] = 1
NACE_64to38_mat[1,3] = 1
NACE_64to38_mat[2,4] = 1
NACE_64to38_mat[3,5] = 1
NACE_64to38_mat[4,6:9] = 1
NACE_64to38_mat[5,9] = 1
NACE_64to38_mat[6,10] = 1
NACE_64to38_mat[7,11] = 1
NACE_64to38_mat[8,12:14] = 1
NACE_64to38_mat[9,14:16] = 1
NACE_64to38_mat[10,16] = 1
NACE_64to38_mat[11,17] = 1
NACE_64to38_mat[12,18] = 1
NACE_64to38_mat[13,19:21] = 1
NACE_64to38_mat[14,21:23] = 1
NACE_64to38_mat[15,23] = 1
NACE_64to38_mat[16,24:26] = 1
NACE_64to38_mat[17,26] = 1
NACE_64to38_mat[18,27:30] = 1
NACE_64to38_mat[19,30:35] = 1
NACE_64to38_mat[20,35] = 1
NACE_64to38_mat[21,36:38] = 1
NACE_64to38_mat[22,38] = 1
NACE_64to38_mat[23,39] = 1
NACE_64to38_mat[24,40:43] = 1
NACE_64to38_mat[25,43] = 1
NACE_64to38_mat[26,44:46] = 1
NACE_64to38_mat[27,46] = 1
NACE_64to38_mat[28,47:49] = 1
NACE_64to38_mat[29,49:53] = 1
NACE_64to38_mat[30,53] = 1
NACE_64to38_mat[31,54] = 1
NACE_64to38_mat[32,55] = 1
NACE_64to38_mat[33,56] = 1
NACE_64to38_mat[34,57:59] = 1
NACE_64to38_mat[35,59:62] = 1
NACE_64to38_mat[36,62] = 1

NACE64to38 = pd.DataFrame(data=NACE_64to38_mat,columns=columns,index=index)
NACE64to38.head()

# WIOD 55 to NACE 64

abs_dir = os.getcwd()
rel_dir = os.path.join(abs_dir, '../../data/raw/economical/IHS_Markit_results_compact.csv')
IHS_df = pd.read_csv(rel_dir,header=[0],index_col=[0])
index = IHS_df.index.values[:]

WIOD55toNACE64_mat = np.zeros([55,63])
for i in range(49):
    WIOD55toNACE64_mat[i,i]=1
WIOD55toNACE64_mat[49,49:53]=1
WIOD55toNACE64_mat[50,53]=1
WIOD55toNACE64_mat[51,54]=1
WIOD55toNACE64_mat[52,55:57]=1
WIOD55toNACE64_mat[53,57:62]=1
WIOD55toNACE64_mat[54,62]=1

WIOD55toNACE64 = pd.DataFrame(data=WIOD55toNACE64_mat,columns=columns,index=index)
WIOD55toNACE64.tail()

# Write all matrices to excel file

with pd.ExcelWriter('../../data/interim/economical/conversion_matrices.xlsx') as writer:
    NACE21to10.to_excel(writer, sheet_name='NACE 21 to NACE 10')
    NACE38to21.to_excel(writer, sheet_name='NACE 38 to NACE 21')
    NACE64to38.to_excel(writer, sheet_name='NACE 64 to NACE 38')
    WIOD55toNACE64.to_excel(writer, sheet_name='NACE 64 to WIOD 55')