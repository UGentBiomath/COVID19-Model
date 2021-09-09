import pandas as pd
import numpy as np

class MPC():

    def __init__(self, model, control_handles): #NOTE: include output + function to match here?
        self.model = model
        self.control_handles = control_handles 
        self.validate_control_handles()
    
    def validate_control_handles(self):
        """ Check if the control_handles variables passed by the user is a list containing only parameters of the model
        """
        # Check if the control_handles variable is a list
        if not isinstance(self.control_handles, list) :
            raise ValueError(
                "The control handles are of type '{0}' but must be of "
                "type '<class 'list'>'".format(type(self.control_handles)))
        # Check if the control_handles are model parameters
        for ch in self.control_handles:
            if ch not in self.model.parameters:
                raise ValueError(
                    "The specified control handle '{0}' is not an "
                    "existing model parameter".format(ch))

    def construct_horizon(self, values, L, parameter_name, t_start=0):
        """ A function to construct a time-dependent parameter function for a given control handle
            If the control handle was already a pre-defined model time-dependent parameter function, than the controller is engaged at timestep t_start 
            The function was built and tested to work with pd.Timestamp/pd.Datetime or with floats as time unit

            Parameters
            ----------
            values : list
                Contains the discrete values of the control handles over the prediction horizon
            L : integer
                Length in days of one discrete control interval
            parameter_name : str
                Parameter name of the control handle
            t_start : float or pd.datetime or pd.Timestamp
                Time at which the TDPF should switch from the pre-defined TDPF to the policy proposed by the MPC controller

            Output
            ------
            horizon: function
                Time-dependent parameter function containing the prediction horizon of the MPC controller

            Example
            -------
            >>> values = [3, 8, 5]
            >>> L = 7
            >>> function = MPC.construct_horizon(values, L, 'Nc', pd.to_datetime('2020-05-01'))
            >>> print(function(pd.Timestamp('2020-05-02'),{},5))
            Return a value of 3
        """

        # Construct vector with timesteps of policy changes, starting at time 0
        policy_nodes = []
        for i in range(len(values)):
            policy_nodes.append(L*i)

        # Convert policy_nodes to dates if necessary (should allow this code to work both for timestep or dates)
        try:
            policy_nodes = [t_start + pd.Timedelta(days=policy_node) for policy_node in policy_nodes]
        except:
            policy_nodes = [t_start + policy_node for policy_node in policy_nodes]

        # Based on parameter name, find out if the parameter is a TDPF
        if parameter_name in self.model.time_dependent_parameters:
            # Get the TDPF by using the parameter_name
            for idx, (key,value) in enumerate(self.model.time_dependent_parameters.items()):
                if key == parameter_name:
                    TDPF = value
                    TDPF_parameter_names = self.model._function_parameters[idx]
                    TDPF_parameter_values = [self.model.parameters[x] for x in TDPF_parameter_names]
            TDPF_kwargs = dict(zip(TDPF_parameter_names,TDPF_parameter_values))

            def horizon(t, states, param, **TDPF_kwargs): # *TDPF_parameter_values --> may have to be moved to the horizon function call
                if t <= t_start:
                    return TDPF(t, states, param, **TDPF_kwargs)
                else:
                    return values[[index for index,value in enumerate(policy_nodes) if value <= t][-1]]

            return horizon

        else:
            def horizon(t, states, param):
                return values[[index for index,value in enumerate(policy_nodes) if value <= t][-1]]

            return horizon


    def constructHorizon(self,thetas,parNames,policy_period):
        # from length of theta list and number of parameters, length of horizon can be calculated
        N = int(len(thetas)/len(parNames))
        # Time
        t = []
        for i in range(N-1):
            t.append(policy_period*(i+1))
        checkpoints = {'t': t}
        # Initialise empty list for every control handle
        for i in range(len(parNames)):
            checkpoints.update({parNames[i] : []})
        # Append to list
        for i in range(len(parNames)):
            if parNames[i] == 'Nc':
                for j in range(0,N):
                    if j == 0:
                        setattr(self, parNames[i],numpy.array([thetas[i*N+j]]))
                    else:
                        checkpoints[parNames[i]].append(numpy.array([thetas[i*N + j]]))
            else:
                for j in range(0,N):
                    if j == 0:
                        setattr(self, parNames[i],numpy.array([thetas[i*N+j]]))
                    else:
                        checkpoints[parNames[i]].append(numpy.array([thetas[i*N + j]]))
        return(checkpoints)

    def calcMPCsse(self,thetas,parNames,setpoints,positions,weights,policy_period,P):
        # ------------------------------------------------------
        # Building the prediction horizon checkpoints dictionary
        # ------------------------------------------------------

        # from length of theta list and number of parameters, length of horizon can be calculated
        N = int(len(thetas)/len(parNames))

        # Build prediction horizon
        thetas_lst=[]
        for i in range(len(parNames)):
            for j in range(0,N):
                thetas_lst.append(thetas[i*N + j])
            for k in range(P-N):
                thetas_lst.append(thetas[i*N + j])
        chk = self.constructHorizon(thetas_lst,parNames,policy_period)

        # ------------------
        # Perform simulation
        # ------------------

        # Set correct simtime
        T = chk['t'][-1] + policy_period
        # run simulation
        self.reset()
        self.sim(T,checkpoints=chk)
        # tuple the results, this is necessary to use the positions index
        out = (self.sumS,self.sumE,self.sumI,self.sumA,self.sumM,self.sumCtot,self.sumICU,self.sumR,self.sumD,self.sumSQ,self.sumEQ,self.sumAQ,self.sumMQ,self.sumRQ)

        # ---------------
        # Calculate error
        # ---------------
        error = 0
        ymodel =[]
        for i in range(len(setpoints)):
            som = 0
            for j in positions[i]:
                som = som + numpy.mean(out[j],axis=1).reshape(numpy.mean(out[j],axis=1).size,1)
            ymodel.append(som.reshape(som.size,1))
            # calculate error
        for i in range(len(ymodel)):
            error = error + weights[i]*(ymodel[i]-setpoints[i])**2
        SSE = sum(error)[0]
        return(SSE)

    def optimizePolicy(self,parNames,bounds,setpoints,positions,weights,policy_period=7,N=6,P=12,disp=True,polish=True,maxiter=100,popsize=20):
        # -------------------------------
        # Run a series of checks on input
        # -------------------------------
        # Check if parNames, bounds, setpoints and positions are lists
        if type(parNames) is not list or type(bounds) is not list or type(setpoints) is not list or type(positions) is not list:
            raise Exception('Datatype of arguments parNames, bounds, setpoints and positions must be lists. Lists are made by wrapping whatever datatype in square brackets [].')
        # Check that length of parNames is equal to the length of bounds
        if len(parNames) is not len(bounds):
            raise Exception('The number of controlled parameters must match the number of bounds given to function MPCoptimize.')
        # Check that length of setpoints is equal to length of positions
        if len(setpoints) is not len(positions):
            raise Exception('The number of output positions must match the number of setpoints names given to function MPCoptimize.')
        # Check that all parNames are actual model parameters
        possibleNames = ['beta', 'sigma', 'Nc', 'zeta', 'a', 'm', 'h', 'c','mi','da','dm','dc','dmi','dICU','dICUrec','dmirec','dhospital','m0','maxICU','totalTests',
                        'psi_FP','psi_PP','dq']
        i = 0
        for param in parNames:
            # For params that don't have given checkpoint values (or bad value given),
            # set their checkpoint values to the value they have now for all checkpoints.
            if param not in possibleNames:
                raise Exception('The parametername provided by user in position {} of argument parNames is not an actual model parameter. Please check its spelling.'.format(i))
            i = i + 1

        # ----------------------------------------------------------------------------------------
        # Convert bounds vector to an appropriate format for scipy.optimize.differential_evolution
        # ----------------------------------------------------------------------------------------
        scipy_bounds=[]
        for i in range(len(parNames)):
            for j in range(N):
                scipy_bounds.append((bounds[i][0],bounds[i][1]))

        # ---------------------
        # Run genetic algorithm
        # ---------------------
        #optim_out = scipy.optimize.differential_evolution(self.calcMPCsse, scipy_bounds, args=(parNames,setpoints,positions,weights,policy_period,P),disp=disp,polish=polish,workers=-1,maxiter=maxiter, popsize=popsize,tol=1e-18)
        #theta_hat = optim_out.x
        p_hat, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = pso.pso(self.calcMPCsse, scipy_bounds, args=(parNames,setpoints,positions,weights,policy_period,P), swarmsize=popsize, maxiter=maxiter,
                                                                                    processes=multiprocessing.cpu_count(),minfunc=1e-9, minstep=1e-9,debug=True, particle_output=True)
        theta_hat = p_hat
        # ---------------------------------------------
        # Assign optimal policy to SEIRSAgeModel object
        # ---------------------------------------------
        self.optimalPolicy = theta_hat
        return(theta_hat)

    def plotOptimalPolicy(self,parNames,setpoints,policy_period,asymptomatic=False,mild=False,filename=None,getfig=False):
        # Construct checkpoints dictionary using the optimalPolicy list
        # Mind that constructHorizon also sets self.Parameters to the first optimal value of every control handle
        # This is done because the first checkpoint cannot be at time 0.
        checkpoints=self.constructHorizon(self.optimalPolicy,parNames,policy_period)
        # First run the simulation
        self.reset()
        self.sim(T=len(checkpoints['t'])*checkpoints['t'][0],checkpoints=checkpoints)

        # Then perform plot
        fig, ax = plt.subplots()
        if asymptomatic is not False:
            ax.plot(self.tseries,numpy.mean(self.sumA,axis=1),color=blue)
            ax.fill_between(self.tseries, numpy.percentile(self.sumA,90,axis=1), numpy.percentile(self.sumA,10,axis=1),color=blue,alpha=0.2)
        if mild is not False:
            ax.plot(self.tseries,numpy.mean(self.sumM,axis=1),color=green)
            ax.fill_between(self.tseries, numpy.percentile(self.sumM,90,axis=1), numpy.percentile(self.sumM,10,axis=1),color=green,alpha=0.2)
        H = self.sumCtot + self.sumICU
        ax.plot(self.tseries,numpy.mean(H,axis=1),color=orange)
        ax.fill_between(self.tseries, numpy.percentile(H,90,axis=1), numpy.percentile(H,10,axis=1),color=orange,alpha=0.2)
        icu = self.sumMi + self.sumICU
        ax.plot(self.tseries,numpy.mean(icu,axis=1),color=red)
        ax.fill_between(self.tseries, numpy.percentile(icu,90,axis=1), numpy.percentile(icu,10,axis=1),color=red,alpha=0.2)
        ax.plot(self.tseries,numpy.mean(self.sumD,axis=1),color=black)
        ax.fill_between(self.tseries, numpy.percentile(self.sumD,90,axis=1), numpy.percentile(self.sumD,10,axis=1),color=black,alpha=0.2)
        if mild is not False and asymptomatic is not False:
            legend_labels = ('asymptomatic','mild','hospitalised','ICU','dead')
        elif mild is not False and asymptomatic is False:
            legend_labels = ('mild','hospitalised','ICU','dead')
        elif mild is False and asymptomatic is not False:
            legend_labels = ('asymptomatic','hospitalised','ICU','dead')
        elif mild is False and asymptomatic is False:
            legend_labels = ('hospitalised','ICU','dead')
        ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1,1))
        ax.set_xlabel('days')
        ax.set_ylabel('number of patients')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # enable the grid
        plt.grid(True)
        # To specify the number of ticks on both or any single axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if filename is not None:
            plt.savefig(filename,dpi=600,bbox_inches='tight')
        if getfig:
            return fig, ax
        else:
            plt.show()

    def mergeDict(self,T,dict1, dict2):
        # length of dict1 is needed later on
        orig_len = len(dict1['t'])
        merged = {}
        # add latest simulation time to dict2
        end = T
        #end = dict1['t'][-1]
        for i in range(len(dict2['t'])):
            dict2['t'][i] = dict2['t'][i]+end
        # merge dictionaries by updating
        temp = {**dict2, **dict1}
        # loop over all key-value pairs
        for key,value in temp.items():
            if key in dict1 and key in dict2:
                for i in range(len(dict2[key])):
                    value.append(dict2[key][i])
                merged[key] = value
            elif key in dict1 and not key in dict2:
                if key != 'Nc':
                    for i in range(len(dict2['t'])):
                        dict1[key].append(getattr(self,key))
                    merged[key] = dict1[key]
                else:
                    for i in range(len(dict2['t'])):
                        dict1[key].append(getattr(self,key))
                    merged[key] = dict1[key]
            elif key in dict2 and not key in dict1:
                if key != 'Nc':
                    for i in range(orig_len):
                        dict2[key].insert(0,getattr(self,key))
                    merged[key] = dict2[key]
                else:
                    for i in range(orig_len):
                        dict2[key].insert(0,getattr(self,key))
                    merged[key] = dict2[key]
        return(merged)
