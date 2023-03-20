import math
import pandas as pd
import numpy as np
import rbfopt
import multiprocessing as mp
from ..optimization import pso

class MPC():

    def __init__(self, model, control_handles_dict): #NOTE: include output + function to match here?
        self.model = model
        self.control_handles_dict = control_handles_dict
        self._validate_control_handles()
        self._assign_TDPF()
        self.model._validate_time_dependent_parameters()


    def _assign_TDPF(self):
        # Construct dummy TDPF in right format
        def dummy_TDPF(t, states, param):
            return param
        # If control handle is not a TDPF yet, initialize the dummmy TDPF
        for ch in self.control_handles_names:
            if not ch in self.model.time_dependent_parameters:
                self.model.time_dependent_parameters.update({ch: dummy_TDPF})
        

    def _validate_control_handles(self):
        """ Check if the control_handles_dict passed by the user has the right format
        """
        # Perform checks on format of control_handles_dict
        if not isinstance(self.control_handles_dict, dict):
            raise ValueError(
                "The control handles dictionary is of type '{0}' but must be a dict".format(type(self.control_handles_dict)))
        # Extract the names of the control handles
        self.control_handles_names = list(self.control_handles_dict.keys())
        # Check if every entry contains another dict with keys: continuous, bounds_values
        for ch in self.control_handles_names:
            if not set(['continuous', 'bounds_values']).issubset(set(list(self.control_handles_dict[ch].keys()))):
                raise ValueError(
                    "The control handles dictionary must contain the properties 'continuous' and 'bounds_values'"
                )
            for prop, value in self.control_handles_dict[ch].items():
                if prop == 'continuous':
                    if not isinstance(self.control_handles_dict[ch][prop], bool):
                        raise ValueError(
                            "Control handle '{0}' property '{1}' must be of type 'bool'!".format(ch,prop)
                        )
                elif prop == 'bounds_values':
                    if self.control_handles_dict[ch]['continuous'] == True:
                        if not isinstance(self.control_handles_dict[ch][prop], tuple):
                            raise ValueError(
                            "Control handle '{0}' property '{1}' must be a tuple containing the upper and lower bounds".format(ch,prop)
                            )
                        if len(self.control_handles_dict[ch][prop]) != 2:
                            raise ValueError(
                            "Control handle '{0}' property '{1}' must be a tuple of length 2 (containing the upper and lower bounds)".format(ch,prop)
                            )                            
                    else:
                        if not isinstance(self.control_handles_dict[ch][prop], list):
                            raise ValueError(
                            "Control handle '{0}' property '{1}' must be a list containing the discrete values control handle '{2}' can assume".format(ch,prop,ch)
                            )               
        # Check if the control_handles names are model parameters
        for ch in self.control_handles_names:
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

        for idx, (key,value) in enumerate(self.model.time_dependent_parameters.items()):
            if key == parameter_name:
                TDPF = value
                TDPF_parameter_names = self.model._function_parameters[idx]
                TDPF_parameter_values = [self.model.parameters[x] for x in TDPF_parameter_names]
        TDPF_kwargs = dict(zip(TDPF_parameter_names,TDPF_parameter_values))

        def horizon(t, states, param, **TDPF_kwargs): # *TDPF_parameter_values --> may have to be moved to the horizon function call
            if t < t_start:
                return TDPF(t, states, param, **TDPF_kwargs)
            else:
                return values[[index for index,value in enumerate(policy_nodes) if value <= t][-1]]

        return horizon
    
    def thetas_to_control_handles(self, thetas):
        """ A function to convert the pso-specific 'thetas' format into key-value pairs of the control handles
        """

        # Compute length of the control horizon
        N = int(len(thetas)/len(list(self.control_handles_dict.keys())))

        # Reshape thetas to size: n_control handles * length_control_horizon, where the order of the control handles is assumed the same as the one provided in the control_handles_dict
        thetas = np.reshape(thetas, (len(list(self.control_handles_dict.keys())), N) )

        values={}
        for idx,ch in enumerate(self.control_handles_names):
            if self.control_handles_dict[ch]['continuous'] == True:
                ### Append to dictionary
                values.update({ch : list(thetas[idx,:])})
            else:
                ### Discrete : convert pso estimate to discrete values (then --> construct_horizon) 
                #### Loop over estimates
                converted_thetas=[]
                for theta in thetas[idx,:]:
                    ##### Make a list containing the corresponding values
                    try:
                        converted_thetas.append(self.control_handles_dict[ch]['bounds_values'][math.floor(theta)])
                    except:
                        converted_thetas.append(self.control_handles_dict[ch]['bounds_values'][int(math.floor(theta)-1)])
                ### Append to dictionary
                values.update({ch : converted_thetas})
        return values

    def run_theta_wrapper(self, thetas, *run_args):
        """ This function converts the pso/rbfopt estimate thetas into a key-value dictionary of the control horizon,
            which is the required input format for the MPC algorithm"""

        # Convert thetas into control horizon dictionary
        control_handles = self.thetas_to_control_handles(thetas)

        return self.run(control_handles, *run_args)

    def run(self, control_handles, L, P, t_start_controller, t_start_simulation, cost_function, cost_function_args, simulation_kwargs):

        #####################################################
        ## Convert control horizon into prediction horizon ##
        #####################################################

        # Extend control horizon into prediction horizon
        for key,value in control_handles.items():
            N = len(value)
            value += value[-1] * (P-N)

        ############################################
        ## Build and assign the horizons as TDPFs ##
        ############################################

        for ch,ph in control_handles.items():
            self.model.time_dependent_parameters.update({ch : self.construct_horizon(ph, L, ch, t_start_controller)})

        ########################
        ## Perform simulation ##
        ########################

        simout = self.model.sim(t_start_controller + pd.Timedelta(days=L*P), start_date=t_start_simulation, **simulation_kwargs)

        ##################
        ## Compute cost ##
        ##################

        return sum(cost_function(simout, control_handles, t_start_controller, *cost_function_args ))

    def rbfopt_optimize(self, L, N, P, t_start_controller, t_start_simulation, cost_function, max_evals, *cost_function_args, **simulation_kwargs):

        #######################################################################
        ## Construct vector of lower and upper bounds, construct type vector ##
        #######################################################################

        t = []
        lower = []
        upper = []

        for idx, (ch, properties_dict) in enumerate(self.control_handles_dict.items()):
            if properties_dict['continuous'] == True:
                t += ['R'] * N
                lower += [properties_dict['bounds_values'][0]] * N
                upper += [properties_dict['bounds_values'][1]] * N
            else:
                t += ['C'] * N
                lower +=  [0] * N
                upper = [len(properties_dict['bounds_values'])] * N

        #############################
        ## Wrap objective function ##
        #############################

        run_theta_wrapper = self.run_theta_wrapper
        def obj_fun(theta):
            return run_theta_wrapper(theta, L, P, t_start_controller, t_start_simulation, cost_function, cost_function_args, simulation_kwargs)

        ##########################
        ## Perform optimization ##
        ##########################

        blackbox = rbfopt.RbfoptUserBlackBox(N*len(self.control_handles_dict), lower, upper, t, obj_fun)
        settings = rbfopt.RbfoptSettings(max_evaluations=max_evals)
        algorithm = rbfopt.RbfoptAlgorithm(settings,blackbox)
        objval, thetas, itercount, evalcount, fast_evalcount = algorithm.optimize()

        ######################################
        ## Retrieve the corresponding TDPFs ##
        ######################################

        # Control horizon dictionary
        thetas_values = self.thetas_to_control_handles(thetas)
        # Conversion into prediction horizon dictionary
        for key,value in thetas_values.items():
            value +=  value[-1] * (P-N) 
        # Conversion into prediction horizon TDPFs
        thetas_TDPF={}
        for ch,ph in thetas_values.items():
            thetas_TDPF.update({ch : self.construct_horizon(ph, L, ch, t_start_controller)})

        return thetas, thetas_values, thetas_TDPF

    def pso_optimize(self, L, N, P, t_start_controller, t_start_simulation, cost_function, maxiter, popsize, *cost_function_args, **simulation_kwargs):
        
        ########################################
        ## Construct bounds for pso optimizer ##
        ########################################

        bounds=[]
        for idx, (ch, properties_dict) in enumerate(self.control_handles_dict.items()):
            if properties_dict['continuous'] == True:
                bounds += N * [(properties_dict['bounds_values'][0], properties_dict['bounds_values'][1])]
            else:
                bounds += N * [(0, len(properties_dict['bounds_values']))]

        ##########################
        ## Perform optimization ##
        ##########################

        thetas, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = pso.optim(self.run_theta_wrapper, bounds,
                    args=(L, P, t_start_controller, t_start_simulation, cost_function, cost_function_args, simulation_kwargs),
                    swarmsize=popsize, maxiter=maxiter, minfunc=1e-9, minstep=1e-9, debug=True, particle_output=True)
        
        ######################################
        ## Retrieve the corresponding TDPFs ##
        ######################################

        # Control horizon dictionary
        thetas_values = self.thetas_to_control_handles(thetas)
        # Conversion into prediction horizon dictionary
        for key,value in thetas_values.items():
            value += (P-N) * value[-1]
        # Conversion into prediction horizon TDPFs
        thetas_TDPF={}
        for ch,ph in thetas_values.items():
            thetas_TDPF.update({ch : self.construct_horizon(ph, L, ch, t_start_controller)})

        return thetas, thetas_values, thetas_TDPF


    def cost_economic(self, model_output, control_handles, t_start, L, model_output_costs):
        """A cost function where a cost can be associated with any model state and with any control handle"""

        ###########################
        ## Cost of model outputs ##
        ###########################
        
        for idx,(state, cost) in enumerate(model_output_costs.items()):
            # Reduce the output dimensions to 'time only' by default
            ymodel = model_output[state]
            for dimension in model_output.dims:
                if dimension != 'time':
                    if dimension == 'draws':
                        ymodel = ymodel.mean(dim=dimension)
                    else:
                        ymodel = ymodel.sum(dim=dimension)

            if idx == 0:
                cost_lst = np.zeros(len(ymodel.sel(time=slice(t_start,None)).values))

            cost_lst = cost_lst + ymodel.sel(time=slice(t_start,None)).values*cost

        #############################
        ## Cost of control handles ##
        #############################   
        
        # Loop over the control handles --> provide these as dict to the function
        for idx, (ch, values) in enumerate(control_handles.items()):
            # Check if the control handle is continuous or discrete
            if self.control_handles_dict[ch]['continuous'] == True:
                for policy_idx in range(len(values)):
                    # Find the right elements
                    cost_lst[policy_idx*L:(policy_idx+1)*L] = cost_lst[policy_idx*L:(policy_idx+1)*L] + self.control_handles_dict[ch]['costs']*values[policy_idx]
            else:
                for policy_idx in range(len(values)):
                    cost_lst[policy_idx*L:(policy_idx+1)*L] = cost_lst[policy_idx*L:(policy_idx+1)*L] + self.control_handles_dict[ch]['costs'][math.floor(np.where([np.allclose(matrix, values[policy_idx]) for matrix in self.control_handles_dict[ch]['bounds_values']])[0])]
        
        return cost_lst

    def cost_setpoint(self, model_output, control_handles, t_start, states, setpoints, weights):
        """ A generic function to drive the values of 'states' to 'setpoints'
            Automatically performs a dimension reduction in the xarray model output until only time remains
            The sum-of-squared errors in this objective function is minimized if the desired 'states' go to the values 'setpoints'
        """
        ##################
        ## Input checks ##
        ##################
        
        if not isinstance(states, list):
            raise ValueError("Input argument states must be of type list instead of '{0}'".format(type(states)))
        if not isinstance(setpoints, list):
            raise ValueError("Input argument setpoints must be of type list instead of '{0}'".format(type(setpoints)))    
        if not isinstance(weights, list):
            raise ValueError("Input argument weights must be of type list instead of '{0}'".format(type(weights)))      
        if ((len(states) != len(setpoints)) | (len(states) != len(weights))):
            raise ValueError("The lengths of 'states', 'setpoints' and 'weights' must be equal")
            
        ######################
        ## Cost calculation ##
        ######################
        
        for idx, state in enumerate(states):
            # Reduce the output dimensions to 'time only' by default
            ymodel = model_output[state]
            for dimension in model_output.dims:
                if dimension != 'time':
                    if dimension == 'draws':
                        ymodel = ymodel.mean(dim=dimension)
                    else:
                        ymodel = ymodel.sum(dim=dimension)

            if idx == 0:
                SSE = np.zeros(len(ymodel.sel(time=slice(t_start,None)).values))

            # Compute SSE
            SSE = SSE + weights[idx]*((ymodel.sel(time=slice(t_start,None)).values - np.ones(len(ymodel.sel(time=slice(t_start,None)).values))*setpoints[idx])**2)
        return SSE