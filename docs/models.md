## Model highlights

### Model dynamics

We use an extended version of the SEIR model to model the disease at a higher resolution. This classic SEIR model splits the population into different categories, i.e. susceptible, exposed, infected and removed. We break down the latter two categories in super mild (asymptotic), mild, heavy and critical for the infected part of the population, whereas the removed population indicates the immune and dead fraction. Parameters values are (for now) based on Chinese covid-19 literature but we are seeking to closer collaborate with Belgian hospitals as more data becomes available. The dynamics are modeled using two frameworks: 1) deterministic and 2) stochastic. The deterministic model can fluently run age-structured (metapopulation) simulations naturally by changing the initial conditions.  The stochastic model class is under construction but will be updated soon.

<img src="_static/figs/flowchartAll.jpg" alt="drawing" width="400"/>

### Additional capabilities

As of now (20/04/2020), the *SEIRSAgeModel* (deterministic model implementation with inherent age-structuring) contains 7 functions which can be grouped into three parts: 1) functions to run and visualise simulations, 2) functions to perform parameter estimations and visualise the results and 3) functions to optimize future policies using model predictive control (MPC).  Also, scenario specific functions will be added over the course of next week.
<img src="_static/figs/SEIRSAgeModel.jpg"
     alt="class"
     width="700"
     style="float: left; margin-right: 500px;" />

As of now (20/04/2020), the *SEIRSNetworkModel* contains 5 functions which can be grouped into two parts: 1) functions to run and visualise simulations and 2) functions to perform parameter estimations and visualise the results. Implementing the model predictive controller is straightforward and can easily be done. However, the optimisation problem is really difficult and requires thousands of function evaluations. Given the large amount of computational resources required to run just one stochastic simulation, it is highly unlikely that the model predictive controller will ever be used to optimize government policy. The MPC functions will be implemented and their usefullness assessed after a calibration is performed. Also, scenario specific functions will be added over the course of next week.

