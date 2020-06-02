.. covid19-model documentation master file, created by
   sphinx-quickstart on Mon May  4 17:31:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to covid19-model's documentation
========================================

.. _biomath_website : https://biomath.ugent.be/covid-19-outbreak-modelling-and-control

Our code implements a SEIRS infectious disease dynamics model with extensions to model the effect of quarantining detected cases. Our code allows to perform Monte Carlo simulations,
calibrate model parameters and calculate *optimal* government policies using a model predictive controller (MPC). A white paper and source code of our previous
work can be found on the `BIOMATH website <biomath_website>`_.

Demo
----

A demo of the model can be found [here](notebooks/templates/SEIRSAgeModel_demo.ipynb). This notebook can also be run in the browser through binder,

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/UGentBiomath/COVID19-Model/master

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   installation
   collaborate


.. toctree::
   :caption: User guide
   :maxdepth: 1

   Background <models>
   Run the model <application.md>
   Future work <roadmap>

.. toctree::
   :caption: Developer guide
   :maxdepth: 1

   Contributing <contributing>
   License <license>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

