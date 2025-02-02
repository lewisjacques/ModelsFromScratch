import pandas as pd
import numpy as np

### --- Cost Functions --- ###

def mean_squared_error(y:pd.Series):
    """
    Mean squared error for a series of target values

    Args:
        y (pd.Series): Series of the target variable
    """
    return(np.mean(y-np.mean(y))**2)

