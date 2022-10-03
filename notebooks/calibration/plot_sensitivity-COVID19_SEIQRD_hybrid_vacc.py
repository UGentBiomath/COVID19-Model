import argparse
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.visualization.sensitivity import plot_sobol_indices_bar, plot_sobol_indices_circular

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Sensitivity analysis filename")
args = parser.parse_args()

# Load data
results_folder = '../../results/calibrations/COVID19_SEIQRD/national/others/sobol_sensitivity/'
results_name = args.filename
S1ST = pd.read_excel(results_folder+results_name, sheet_name='S1ST', index_col=[0])
S2 = pd.read_excel(results_folder+results_name, sheet_name='S2', index_col=[0])

# Hardcoded labels
labels = ['$\\beta$', '$\\omega$', '$a$', '$d_{a}$','$h$','$A_{s}$','$\\zeta$','$M$', '$\Omega_{work}$', '$\Omega_{rest}$']

# Bar plot
fig,ax=plt.subplots(figsize=(12,4))
ax = plot_sobol_indices_bar(S1ST, ax, labels)
plt.tight_layout()
plt.show()
plt.close()

# Circular plot
fig=plot_sobol_indices_circular(S1ST, S2, labels)
plt.tight_layout()
plt.show()
plt.close()

