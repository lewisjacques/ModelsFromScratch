from ._model_helper import CostFunctionHelper
import numpy as np

### --- Initialise CostFunctionHelper --- ###

# This class will store all cost_functions implemented
_fh = CostFunctionHelper()

### --- Cost Functions --- ###

@_fh.add_cost_function
def mean_squared_error_variance(y:np.array):
    """
    Mean squared error for a series of target values. 
    Essentially calculating the variance in a set of y values,
    where the variance is to be minimised at each split of a
    decision tree

    MSE =  1/n sum(i=1:N) { (yi - ybar)^2 }

    Args:
        y (np.array): Series of the target variable
    """
    cost = np.mean(y-np.mean(y))**2
    return(cost)

@_fh.add_cost_function
def binary_cross_entropy(y:np.array, y_pred:np.array, eps:float=1e-15):
    """
    Log-loss cost function used primarily in logistic modelling. This
    function is highly penalising for incorrect predictions

    L = - 1/n sum(i=1:N) { (yi * log(yhat_i)) + ((1 - yi) * log(1 - yhat_i) }

    The prediction yhati should never be 0 or 1, but in the case that it is due
    to an extremely certain prediction, account for a potential error in the log
    function by adding the error value epsilon

    Args:
        y (np.array): Series of the target variable
        y_pred (np.array): Series of predicted values of y
        eps (float, optional): Epsilon error value Defaults to 1e-15.
    """

    # y < eps (0) then eps, y > 1-eps (1) then 1-eps
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Compute binary cross-entropy loss
    cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return(cost)
