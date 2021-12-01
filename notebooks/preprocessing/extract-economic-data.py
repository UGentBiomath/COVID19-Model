"""
This script contains all necessary code to extract and convert the data needed to setup the UGent production network model.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import requests
import zipfile
import shutil
import numpy as np
import pandas as pd

##################################
## Download all necessary files ##
##################################

print('1) Downloading data ')

# Pichler et al. 2020
# ~~~~~~~~~~~~~~~~~~~

if not os.path.isfile('../../data/raw/economical/IHS_Markit_results_compact.csv') or not os.path.isfile('../../data/raw/economical/WIOD_shockdata.csv') or not os.path.isfile('../../data/raw/economical/IHS_Markit_results_compact.csv'):
    # Make a temporary folder
    if not os.path.exists('../../data/raw/economical/tmp'): # if the directory does not exist
        os.makedirs('../../data/raw/economical/tmp') # make the directory
    # Download data
    url = 'https://zenodo.figshare.com/ndownloader/files/22754936'
    r = requests.get(url,stream=True)
    with open('../../data/raw/economical/tmp/pichler_etal.zip', 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
    # Unzip directory
    with zipfile.ZipFile('../../data/raw/economical/tmp/pichler_etal.zip',"r") as zip_ref:
        zip_ref.extractall('../../data/raw/economical/tmp/pichler_etal')
    # Place files we want in raw folder
    os.system('cp ../../data/raw/economical/tmp/pichler_etal/covid19inputoutput/data/IHS_Markit_results_compact.csv ../../data/raw/economical/IHS_Markit_results_compact.csv')  
    os.system('cp ../../data/raw/economical/tmp/pichler_etal/covid19inputoutput/data/table_ratio_inv_go.csv ../../data/raw/economical/table_ratio_inv_go.csv')
    os.system('cp ../../data/raw/economical/tmp/pichler_etal/covid19inputoutput/data/WIOD_shockdata.csv ../../data/raw/economical/WIOD_shockdata.csv')  
    # Delete temporary folder
    shutil.rmtree('../../data/raw/economical/tmp')

# Federaal Planbureau
# ~~~~~~~~~~~~~~~~~~~
if not os.path.isfile('../../data/raw/economical/input-output.xlsx'):
    url = 'https://www.plan.be/databases/io2015/vr64_en_20181217.xlsx'
    r = requests.get(url)
    with open('../../data/raw/economical/input-output.xlsx', 'wb') as f:
        f.write(r.content)

IO_df = pd.read_excel('../../data/raw/economical/input-output.xlsx', sheet_name='tbl_8',index_col=[0], header=[0])

# Census 2011
# ~~~~~~~~~~~
if not os.path.isfile('../../data/raw/census_2011/census_arbeidsmarkt_nl_24oct14.xlsx'):
    url = 'https://www.census2011.be/download/census_arbeidsmarkt_nl_24oct14.xlsx'
    r = requests.get(url)
    with open('../../data/raw/census_2011/census_arbeidsmarkt_nl_24oct14.xlsx', 'wb') as f:
        f.write(r.content)

census_df = pd.read_excel('../../data/raw/census_2011/census_arbeidsmarkt_nl_24oct14.xlsx', sheet_name="Tabel3_2011")

print('done\n')

print('2) Formatting data ')

########################################
## Construct NACE Conversion matrices ##
########################################

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

###################################################
## Input-output matrix: Federaal Planning Bureau ##
###################################################

# Extract IO table
IO = IO_df.values[1:-19,1:-10]
IO=np.delete(IO,45,axis=0) # remove row 68a (adding not necessary since row 68a contains zeros only)
IO[1:-19:44] = IO[1:-19:44] + IO[1:-19:45] # Add column 68a to column 68_
IO=np.delete(IO,45,axis=1)
IO_new = pd.DataFrame(data=IO,columns=codes64,index=codes64)
IO_new.head()
# Write formatted IO table
IO_new.to_csv('../../data/interim/economical/IO_NACE64.csv', index=True)

# Sectoral output under business-as-usual
x_0 = IO_df.values[1:-19,-1]
x_0[43] = x_0[43] + x_0[44] # Confirm with Koen or Gert that this needs to be added togheter
x_0 = np.delete(x_0,44) 

# Household demand during business-as-usual
c_0 = IO_df.values[1:-19,-9]
c_0[43] = c_0[43] + c_0[44] # Confirm with Koen or Gert that this needs to be added togheter
c_0 = np.delete(c_0,44)

# Other final demand
f_0 = np.sum(IO_df.values[1:-19,-8:-1],axis=1)
# "Changes in inventories and acquisition less disposals of valuables" can be negative
# Do I just sum the columns?
f_0[43] = f_0[43] + f_0[44] # Confirm with Koen or Gert that this needs to be added togheter
f_0 = np.delete(f_0,44)

###################
## Desired stock ##
###################

# Read
nj_df = pd.read_csv("../../data/raw/economical/table_ratio_inv_go.csv", index_col=[0], header=[0])
nj55 = nj_df['ratio_all_inv_go_monthly'].values[:-1]*30
# Convert to NACE64
nj64 = np.zeros(63)
for i in range(nj55.size):
    nj64[WIOD55toNACE64_mat[i,:] == 1] = nj55[i]

#####################
## Critical inputs ##
#####################

# Read and format
abs_dir = os.getcwd()
rel_dir = os.path.join(abs_dir, '../../data/raw/economical/IHS_Markit_results_compact.csv')
IHS_df = pd.read_csv(rel_dir,header=[0],index_col=[0])
IHS_df.fillna(0)
IHS_mat = IHS_df.values
new_last_column= np.expand_dims(np.append(IHS_mat[-1,:],1),axis=1)
IHS_mat = np.append(IHS_mat,new_last_column,axis=1)
IHS_df['T']=new_last_column

# Convert to NACE64
NACE64toWIOD55_mat = WIOD55toNACE64_mat

IHS_mat = IHS_df.fillna(0).values
# Convert to all entries  before sector N using the conversion matrix
new64_mat = np.zeros([63,63])
for j in range(49): # row 49 is sector N
    new64 = np.zeros(63)
    orig55 = IHS_mat[j,:]
    for i in range(orig55.size):
        new64[NACE64toWIOD55_mat[i,:] == 1] = orig55[i]
    new64_mat[j,:]=new64
    
# Now we're at row N
# First convert row N using the matrix
new64 = np.zeros(63)
orig55 = IHS_mat[49,:]
for i in range(orig55.size):
    new64[NACE64toWIOD55_mat[i,:] == 1] = orig55[i]
    new64_mat[49:53,:]=new64
# Then modify the diagonal (i.e. no dependency of sector 77 to sector 78 etc.)
new64_mat[49,50:53]=0
new64_mat[50,49]=0
new64_mat[50,51:53]=0
new64_mat[51,49:51]=0
new64_mat[51,52:53]=0
new64_mat[52,49:52]=0

# Then go on from index 50 to 51 (sectors 84 and 85)
for j in range(50,52): # 49 or 63
    new64 = np.zeros(63)
    orig55 = IHS_mat[j,:]
    for i in range(orig55.size):
        new64[NACE64toWIOD55_mat[i,:] == 1] = orig55[i]
    new64_mat[j+3,:]=new64

# Now we're at sectors Q (index 52)
# First convert row Q using the conversion matrix
new64 = np.zeros(63)
orig55 = IHS_mat[52,:]
for i in range(orig55.size):
    new64[NACE64toWIOD55_mat[i,:] == 1] = orig55[i]
    new64_mat[55:57,:]=new64
# Then modify the diagonal (i.e. no dependency of sector 93 to sector 94 etc.)
new64_mat[55,56]=0
new64_mat[56,55]=0

# Now we're at sectors R_S (index 54)
# First convert row R_S using the conversion matrix
new64 = np.zeros(63)
orig55 = IHS_mat[53,:]
for i in range(orig55.size):
    new64[NACE64toWIOD55_mat[i,:] == 1] = orig55[i]
    new64_mat[57:62,:]=new64
# Then modify the diagonal (i.e. no dependency of sector 93 to sector 94 etc.)
new64_mat[57,58:62]=0
new64_mat[58,57]=0
new64_mat[58,59:62]=0
new64_mat[59,57:59]=0
new64_mat[59,60:62]=0
new64_mat[60,57:60]=0
new64_mat[60,61:62]=0
new64_mat[61,57:61]=0

# Convert and insert row T using the conversion matrix
new64 = np.zeros(63)
orig55 = IHS_mat[54,:]
for i in range(orig55.size):
    new64[NACE64toWIOD55_mat[i,:] == 1] = orig55[i]
    new64_mat[62,:]=new64

columns = codes64
index = codes64
IHS_critical = pd.DataFrame(data = new64_mat, index=index, columns=columns)
IHS_critical.to_csv('../../data/interim/economical/IHS_critical_NACE64.csv', index=True)

###########################
## Consumer demand shock ##
###########################

# Read 
ed_df = pd.read_csv("../../data/raw/economical/WIOD_shockdata.csv", index_col=[0], header=[0])
ed55 = ed_df['demand.shock.household'].values
# Convert to NACE 64
ed64 = np.zeros(63)
for i in range(nj55.size):
    ed64[WIOD55toNACE64_mat[i,:] == 1] = ed55[i]

########################
## Other demand shock ##
########################

# Read and compute
no_shock = ed_df['fdemand.other'].values
shocked = ed_df['fdemand.other.shocked'].values
fd55 = np.array(-(1 - shocked/no_shock)*100)
# Convert to NACE64
fd64 = np.zeros(63)
for i in range(fd55.size):
    fd64[WIOD55toNACE64_mat[i,:] == 1] = fd55[i]

#######################
## Labor income: NBB ##
#######################

# Business-as-usual
# ~~~~~~~~~~~~~~~~~

sectoral64_df = pd.read_excel("../../data/raw/economical/Employees_NACE64.xlsx",sheet_name = 'Binnenlands concept - A64', index_col=[0], header=[0])
l0_64 = sectoral64_df.values[7:-1,-1]

# Lockdown survey
# ~~~~~~~~~~~~~~~

# Read
sectoral38_lockdown_df = pd.read_excel("../../data/raw/economical/Employees_25-04-2020_NACE38.xlsx",sheet_name = 'Formated data', index_col=[0], header=[0])
l_lockdown38 = (sectoral38_lockdown_df['telework'] + sectoral38_lockdown_df['mix telework-workplace'] + sectoral38_lockdown_df['at workplace']).values[1:]
# Convert to NACE 64
# Telework
telework38 = sectoral38_lockdown_df['telework'].values[1:]
telework64 = np.zeros(63)
for i in range(l_lockdown38.size):
    telework64[NACE_64to38_mat[i,:] == 1] = telework38[i]

# Mix
mix38 = sectoral38_lockdown_df['mix telework-workplace'].values[1:]
mix64 = np.zeros(63)
for i in range(l_lockdown38.size):
    mix64[NACE_64to38_mat[i,:] == 1] = mix38[i]

# Workplace
workplace38 = sectoral38_lockdown_df['at workplace'].values[1:]
workplace64 = np.zeros(63)
for i in range(l_lockdown38.size):
    workplace64[NACE_64to38_mat[i,:] == 1] = workplace38[i]

# Absent
absent38 = sectoral38_lockdown_df['absent'].values[1:]
absent64 = np.zeros(63)
for i in range(l_lockdown38.size):
    absent64[NACE_64to38_mat[i,:] == 1] = absent38[i]

########################
## Group in dataframe ##
########################

tuples = [('Business-as-usual', 'Sectoral output (M€)'),
            ('Business-as-usual', 'Household demand (M€)'),
            ('Business-as-usual', 'Other demand (M€)'),
            ('Business-as-usual', 'Desired stock (days)'),
            ('Business-as-usual', 'Employees (x1000)'),
            ('Lockdown', 'Consumer demand shock (%)'),
            ('Lockdown', 'Other demand shock (%)'),
            ('Lockdown', 'Telework (%)'),
            ('Lockdown', 'Mix (%)'),
            ('Lockdown', 'Workplace (%)'),
            ('Lockdown', 'Absent (%)'),]

colnames = pd.MultiIndex.from_tuples(tuples, names=['',''])

data = {
    ('Business-as-usual','Sectoral output (M€)'): list(x_0),
    ('Business-as-usual','Household demand (M€)'): list(c_0),
    ('Business-as-usual','Other demand (M€)'): list(f_0),
    ('Business-as-usual','Desired stock (days)'): list(nj64),
    ('Business-as-usual','Employees (x1000)'): list(l0_64),
    ('Lockdown', 'Consumer demand shock (%)'): list(ed64*100),
    ('Lockdown', 'Other demand shock (%)'): list(fd64),
    ('Lockdown', 'Telework (%)'): list(telework64),
    ('Lockdown', 'Mix (%)'): list(mix64),
    ('Lockdown', 'Workplace (%)'): list(workplace64),
    ('Lockdown', 'Absent (%)'): list(absent64),
}
df = pd.DataFrame(data=data,columns=colnames,index=codes64)
df.head()

df.to_csv('../../data/interim/economical/others.csv', index=True)

#################
## Census 2011 ##
#################

codes=census_df['00.55 - Werkende bevolking van belgische en vreemde nationaliteit naar geslacht en economische sector'].loc[5:1943].dropna().values
codes_int = [int(i) for i in codes]

names = census_df.iloc[5:,1].dropna().values
rows=[]
for i in census_df['00.55 - Werkende bevolking van belgische en vreemde nationaliteit naar geslacht en economische sector'].loc[5:1943].dropna().index:
    intra_row = np.array([])
    intra_row = np.append(intra_row,census_df.iloc[i+2,3]) # A
    intra_row = np.append(intra_row,census_df.iloc[i+2,5:10]) # B, C, D, F
    intra_row = np.append(intra_row,census_df.iloc[i+2,11:17]) # G, H, I, J, K , L
    intra_row = np.append(intra_row,census_df.iloc[i+2,18:20]) # M, N
    intra_row = np.append(intra_row,census_df.iloc[i+2,21:24]) # O, P, Q
    intra_row = np.append(intra_row,census_df.iloc[i+2,25:29]) # R, S, T, U
    rows.append(intra_row)
matrix = np.zeros([len(rows),21])
for j in range(len(rows)):
    matrix[j,:]=rows[j]
    
economic_df=pd.DataFrame(matrix,index=codes_int,columns=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U'])
economic_df.index.name = 'NIS'

idx_arrondisement=[]
for idx in economic_df.index:
    if ((str(idx)[-3:] == '000') & (len(str(idx)) != 4) & (str(idx)[-4:] != '0000') & (str(idx)[0] != '0')):
        idx_arrondisement.append(idx)

economic_df.loc[idx_arrondisement,:].to_csv('../../data/interim/economical/census2011_NACE21.csv', index=True)

############################################
## Economic Risk Management Group Surveys ##
############################################


print('done\n')