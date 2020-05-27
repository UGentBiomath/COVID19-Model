# BIOMATH COVID19-Model
I broke this code intentionally
*Original code by Ryan S. McGee. Modified by T.W. Alleman in consultation with the BIOMATH research unit headed by prof. Ingmar Nopens.*

Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University.

## Introduction

Our code implements a SEIRS infectious disease dynamics model with extensions to model the effect of quarantining detected cases. Our code allows to perform Monte Carlo simulations, calibrate model parameters and calculate *optimal* government policies using a model predictive controller (MPC). A white paper and source code of our previous work can be found on the [BIOMATH website](https://biomath.ugent.be/covid-19-outbreak-modelling-and-control).

Check the [documentation website](https://ugentbiomath.github.io/COVID19-Model/) for more information about the code and the models.

## Demo

A demo of the model can be found [here](notebooks/templates/SEIRSAgeModel_demo.ipynb). This notebook can also be run in the browser through binder,

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UGentBiomath/COVID19-Model/master?filepath=src%2FSEIRSAgeModel_demo.ipynb)

## Installation

The information needed to install the required packages, can be found [here](https://ugentbiomath.github.io/COVID19-Model/installation.html) on the documentation website.

## Acknowledgements

- The repository setup is a derived version of the [data science cookiecutter](https://drivendata.github.io/cookiecutter-data-science/) providing a consistent structure.
- Original code by Ryan S. McGee
