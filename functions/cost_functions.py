from ._model_helper import CostFunctionHelper
import numpy as np

### --- Initialise CostFunctionHelper --- ###

# This class will store all cost_functions implemented
_fh = CostFunctionHelper()

### --- Cost Functions --- ###

@_fh.add_cost_function
def mean_squared_error(y:np.array):
    """
    Mean squared error for a series of target values

    Args:
        y (np.array): Series of the target variable
    """
    return(np.mean(y-np.mean(y))**2)

