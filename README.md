# BIOMATH COVID19-Model

Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University.

## Introduction

### Aim of the doctoral dissertation

During the past two years, drastic social restrictions were taken to safeguard our healthcare system. It is likely that SARS-CoV-2, much like seasonal flu, will become endemic and this implies there might be a more severe outbreak sometime in the future. Further, it’s not a question if there will be a next pandemic, but rather when there will be one. Given the important role epidemiological models have played in the decision making process during the 2020-2021 SARS-CoV-2 pandemic, more research on the topic is warranted to increase our pandemic preparedness.

During the past two years, we have developed two compartmental epidemiological models for SARS-CoV-2 in Belgium and used them to support pandemic policy. Although these models can be used to make accurate forecasts on key epidemiological parameters, their scope is too narrow to provide policy makers with comprehensive advice. Indeed, we have to acknowledge that there are a multitude of societal costs associated with a pandemic, such as increased feelings of anxiety and depression; sector closure, loss of income; postponement of non-COVID treatment and long-term health issues caused by long-COVID. Read more in the following [technical report](https://www.researchgate.net/publication/360342734_Covid-19_from_model_prediction_to_model_predictive_control).

For that purpose, we have implemented a macro-economic input-output (IO) model to make forecasts on profit, GDP and employment and a health economic model able to quantify the direct and indirect impacts in quality-adjusted life-years (QALYs). By combining these models with our epidemiological models, an approximate societal cost can be computed for every epidemiological trajectory. Then, Model Predictive Control (MPC) can be used to find the policy associated with the smallest societal cost. To the best of our knowledge, such triade of models currently does not exist. The aforementioned models have been implemented but important questions on their coupling and use remain unanswered. Prospecting students are invited to contact the promotor or tutor as the topic can be tailored to their interests.

### Current state of the project (2022-05-03)

The epidemiological models are in an advanced stage of development, all details regarding the national model were [published](https://doi.org/10.1016/j.epidem.2021.100505) in peer-reviewed literature. Both models, a nation-level model and a spatially-explicit model were used to provide policymakers with [advice](https://www.researchgate.net/publication/356289190_Effect_of_non-pharmaceutical_interventions_on_the_number_of_new_hospitalizations_during_the_fall_2021_Belgian_COVID-19_wave_-_version_11) during the 2020-2022 SARS-CoV-2 pandemic. Our health economic QALY model is implemented and capable of computing QALY losses caused by mild symptoms, hospitalizations and deaths of the epidemiological model output. The aim is to, before September 2022, add losses due to long-COVID and acquired disabilities to the QALY module. Ethical clearance was granted by the UZ Ghent to analyze a dataset on the amount of non-COVID treatment postponed during the pandemic. The goal is to correlate the degree postponement of non-COVID care to the spread of SARS-CoV-2 and add these to the QALY model as well. The macro-economic model is implemented but must still be calibrated on relevant data. The MPC is implemented and was already used on a "toy model".

## Installation

The information needed to install the required packages, can be found on the [documentation website](https://ugentbiomath.github.io/COVID19-Model/installation.html) on the documentation website.

## Acknowledgements

- The repository setup is a derived version of the [data science cookiecutter](https://drivendata.github.io/cookiecutter-data-science/) providing a consistent structure.
-  I want to thank Bram De Jaegher, Daan Van Hauwermeiren, Stijn Van Hoey and Joris Van den Bossche for their help in maintaining the GitHub repo, coding the visualizations and for teaching me the basics of object-oriented programming in Python.
