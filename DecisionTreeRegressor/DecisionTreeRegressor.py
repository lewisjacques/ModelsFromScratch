from split_functions import best_split_numerical
import pandas as pd
import numpy as np

class DecisionTreeRegressor:
    #! Wrap the split and cost functions in decorators to improve
    SPLIT_FUNCTIONS = ("best_split_numerical",)
    COST_FUNCTIONS = ("mse")

    def __init__(
        self,
        split_function="best_split_numerical",
        cost_function="mse"
    ):
        """
        Class for a Decision Tree Regressor

        We want to recursively split the decision tree until one of
        two stopping conditions are met
            - One node has too few samples
            - The max depth has been reached
        """

        # Set stopping conditions
        self.min_samples = 5
        self.max_depth = 10

        # Set function choices
        assert split_function in self.SPLIT_FUNCTIONS, \
            'Please select a relevant split function'
        self.split_function = split_function
        assert cost_function in self.COST_FUNCTIONS, \
            'Please select a relevant cost function'
        self.cost_function = cost_function

        self.tree = None

    def build_tree(self, X:pd.DataFrame, y:pd.Series, depth=1):
        # Check depth stopping condition
        if depth > self.max_depth or len(y) < self.min_samples:
            # Return the average value of the y-vals at the leaf node
            return(np.mean(y))

        # Get split under current parameters
        top_feature, thresh = best_split_numerical(X,y)

        # Set masks
        new_left_mask = X[:, top_feature] <= thresh
        new_right_mask = ~new_left_mask
        # Filter x and y
        new_left_x, new_left_y = X[new_left_mask], y[new_left_mask]
        new_right_x, new_right_y = X[new_right_mask], y[new_right_mask]

        return({
            "feature": top_feature,
            "threshold": thresh,
            "left": self.build_tree(new_left_x, new_left_y, depth+1),
            "right": self.build_tree(new_right_x, new_right_y, depth+1)
        })