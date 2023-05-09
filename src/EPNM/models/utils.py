from EPNM.models.model import Economic_Model
from EPNM.data.parameters import get_model_parameters
from EPNM.data.utils import get_sector_labels
from EPNM.models.TDPF import labor_supply_shock, household_demand_shock, other_demand_shock, government_furloughing, compute_income_expectations

def initialize_model(shocks='alleman', prodfunc='half_critical'):
    """
    A function to initialize the economic production network model

    Inputs
    ======

    shocks: str
        Use supply and demand shocks from 'Pichler' or 'Alleman'

    prodfunc: str
        Type of Partially-Binding Leontief function (default: half critical). Options: Leontief, Strongly Critical, Half Critical, Weakly Critical

    Returns
    =======

    params: dictionary
        Dictionary containing all model parameters. Obtained using EPNM.data.parameters.get_model_parameters().

    model: pySODM model
        EPNM model
    """

    # Load parameters
    params = get_model_parameters(shocks=shocks)
    params.update({'prodfunc': prodfunc})
    
    # Load initial states
    initial_states = {'x': params['x_0'],
                    'c': params['c_0'],
                    'c_desired': params['c_0'],
                    'f': params['f_0'],
                    'd': params['x_0'],
                    'l': params['l_0'],
                    'O': params['O_j'],
                    'S': params['S_0']}

    # Coordinates and TDPF
    coordinates = {'NACE64': get_sector_labels('NACE64'), 'NACE64_star': get_sector_labels('NACE64')}
    time_dependent_parameters = {'epsilon_S': labor_supply_shock,
                                'epsilon_D': household_demand_shock,
                                'epsilon_F': other_demand_shock,
                                'b': government_furloughing,
                                'zeta': compute_income_expectations}

    # Initialize the model
    model = Economic_Model(initial_states, params, coordinates=coordinates, time_dependent_parameters=time_dependent_parameters)

    return params, model