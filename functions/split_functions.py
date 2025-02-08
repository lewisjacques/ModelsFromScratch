from ._model_helper import SplitFunctionHelper
from typing import Callable
import pandas as pd

### --- Initialise SplitFunctionHelper --- ###

# This class will store all split_functions implemented
_fh = SplitFunctionHelper()

### --- Split Functions --- ###

@_fh.add_split_function
def best_split_numerical(
    X:pd.DataFrame, 
    y:pd.Series,
    cost_function:Callable
) -> tuple:
    """
    Find the best split for a group of numerical columns. Returning
    the feature to split on along with the threshold to take

    Args:
    X (pd.DataFrame): Data frame of features and their values
    y (pd.Series): Target variable for the same dataset
    cost_function (Callable): As named

    Returns:
        tuple: (feature, threshold)
    """

    # All features
    features = X.columns
    n_samples = X.shape[0]

    # Current best splits and thresholds
    best_feature, best_threshold, best_mse = None, None, float("inf")

    # Iterate through each feature and potential threshold
    for f in features:
        unique_f_values = X.loc[:,f].unique()
        # Take all potential values as thresholds for now
        for threshold in unique_f_values:
            # Find the values either side of the split
            left_mask = X.loc[:,f] <= threshold
            right_mask = ~left_mask

            # Avoid empty splits on either side by checking true counts
            if sum(left_mask) == 0 or sum(right_mask) == 0:
                continue

            # Calculate the MSE on either side of the split
            # The lower the variance the higher the homogeneity 
            left_mse = cost_function(y[left_mask])
            right_mse = cost_function(y[right_mask])
            # Weighted mse dependent on split sizes
            total_mse = ((sum(left_mask)*left_mse) + (sum(right_mask)*right_mse))/n_samples

            # Check if this MSE outperforms current
            if total_mse < best_mse:
                best_feature, best_threshold, best_mse = f, threshold, total_mse
    
    return best_feature, best_threshold