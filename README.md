# BIOMATH COVID19-Model
*Original code by Ryan S. McGee. Modified by T.W. Alleman in consultation with the BIOMATH research unit headed by prof. Ingmar Nopens.*

Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University.

Our code implements a SEIRS infectious disease dynamics models with extensions to model the effect quarantining detected cases. Our code allows to quickly perform Monte Carlo simulations, calibrate model parameters and calculate an *optimal* government policies using a model predictive controller (MPC). A white paper and souce code of our previous work can be found on the [BIOMATH website](https://biomath.ugent.be/covid-19-outbreak-modelling-and-control). 


## Demo
A demo of the model can be found [here](src/SEIRSAgeModel_demo.ipynb). This notebook can also be run in the browser through binder,

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UGentBiomath/COVID19-Model/master?filepath=src%2FSEIRSAgeModel_demo.ipynb)


## Model highlights

### Model dynamics
We use an extended version of the SEIR model to model the disease at a higher resolution. This classic SEIR model splits the population into different categories, i.e. susceptible, exposed, infected and removed. We break down the latter two categories in super mild (asymptotic), mild, heavy and critical for the infected part of the population, whereas the removed population indicates the immune and dead fraction. Parameters values are (for now) based on Chinese covid-19 literature but we are seeking to closer collaborate with Belgian hospitals as more data becomes available. The dynamics are modeled using two frameworks: 1) deterministic and 2) stochastic. The deterministic model can fluently run age-structured (metapopulation) simulations naturally by changing the initial conditions.  The stochastic model class is under construction but will be updated soon.

<img src="figs/flowchart2.jpg" alt="drawing" width="400"/>

### Additional capabilities
As of now (18/04/2020), the SEIRSAgeModel (deterministic model implementation with possiblity of age-structuring) contains 7 functions which can be grouped in two parts: 1) functions to run and visualise simulations and 2) functions to perform parameter estimations and visualse the results. 3) functions to optimize future policies using model predictive control (MPC).  Also, scenario specific functions will be added over the course of next week. 

<img src="figs/SEIRSAgeModel.jpg"
     alt="class"
     width="700"     
     style="float: left; margin-right: 500px;" /> 

## How to use the model

A Jupyter Notebooks containing all scientific details and a tutorial is available in the /src folder.

## Future work

### Model dynamics

Future work will include a modification of the model dynamics according to the flowchart below. We believe these allow to simulate even more realisticly. Available data from Belgian hospitals will be used.

<img src="figs/flowchart3_1.jpg" alt="drawing" width="800"/>

<img src="figs/flowchart3_2.jpg" alt="drawing" width="600"/>

### Stochastic model
Implementing the class-based functions available in the SEIRSAgeModel for Monte-Carlo sampling, calibration and model predictive control in the stochastic model framework. Investigate if the Ryan McGee-implementation of the stochastic model can be modified to include age-structuring.

### Scenario-specific functions
 We will implement a function which uses the past policies to quickly recreate the Belgian ICU and hospitalisation curves up-to-date. This function will be used to quickly propose MPC optimal policies and to perform scenario analysis about the future.