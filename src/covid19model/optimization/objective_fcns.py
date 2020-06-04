import numpy as np
import xarray

def SSE(BaseModel,thetas,data,states,parNames,weights,checkpoints=None):

    """
    A function to return the sum of squared errors given a model prediction and a dataset

    Parameters
    -----------
    BaseModel: model object
        correctly initialised model to be fitted to the dataset
    thetas: np.array
        vector containing estimated parameter values
    thetas: list
        names of parameters to be fitted
    data: list
        list containing dataseries
    states: list
        list containg the names of the model states to be fitted to data
    weights: np.array
        weight of every dataseries

    Returns
    -----------
    SSE : float64
        total sum of squared errors

    Example use
    -----------
    SSE = SSE(model,thetas,data,parNames,positions,weights)
    """

    # ~~~~~~~~~~~~
    # Input checks
    # ~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # assign estimates to correct variable
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    i = 0
    for param in parNames:
        if param == 'extraTime': # don't know if there's a way to make this function more general due to the 'extraTime', can this be abstracted in any way?
            setattr(BaseModel,param,int(round(thetas[i])))
        else:
            setattr(BaseModel,param,thetas[i])
        i = i + 1
    
    # ~~~~~~~~~~~~~~
    # Run simulation
    # ~~~~~~~~~~~~~~
    # number of dataseries
    n = len(data) 
    # Compute simulation time
    data_length =[]
    for i in range(n):
        data_length.append(data[i].size)
    T = max(data_length)+BaseModel.extraTime-1
    # Perform simulation
    out=BaseModel.sim(T,checkpoints=checkpoints)

    # -------------
    # calculate SSE
    # -------------
    ymodel=[]
    SSE = 0
    for i in range(n):
        som = 0
        # sum required states
        for j in range(len(states[i])):
            som = som + out[states[i][j]].sum(dim="stratification").values
        ymodel.append(som[BaseModel.extraTime:])
        # calculate quadratic error
        print(ymodel[i].shape,data[i].shape)
        SSE = SSE + weights[i]*sum((ymodel[i]-data[i])**2)
    return SSE

