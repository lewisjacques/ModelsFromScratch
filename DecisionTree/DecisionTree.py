from functions import function_dict
import pandas as pd
import numpy as np

class DecisionTreeModel:
    def __init__(
        self,
        split_function:str="best_split_numerical",
        cost_function:str="mean_squared_error",
        model_type:str="regression",
        threshold:str|None=None
    ):
        """
        Class for a Decision Tree Regressor

        We want to recursively split the decision tree until one of
        two stopping conditions are met
            - One node has too few samples
            - The max depth has been reached

        Args:
            split_function (str, optional): function to determine how we split nodes.
                Defaults to "best_split_numerical".
            cost_function (str, optional): function to minimise at each split. 
                Defaults to "mse".
            model_type (str, optional): classifier|regression. 
                Defaults to "regression".
            threshold (str | None, optional): Required if model_type is classifier.
                Defaults to None.
        """

        # Check requested model type
        assert model_type in ("regression","classifier")
        # If model type is classifier, threshold must be set
        if model_type == "classifier":
            assert threshold is not None, \
                "If model-type is classification, threshold for a true assignment must be provided"
            self.threshold = float(threshold)

        assert split_function in function_dict["split_functions"].keys(), \
            "Split function not yet implemented"
        assert cost_function in function_dict["cost_functions"].keys()
        # Initialise verified functions within arguments
        self.split_function = function_dict["split_functions"][split_function]
        self.cost_function = function_dict["cost_functions"][cost_function]

        # Set stopping conditions and initialise tree
        self.min_samples = 5
        self.max_depth = 10
        self.model_type = model_type
        self.tree = None

    def build_tree(self, X:pd.DataFrame, y:pd.Series, depth=0) -> dict:
        """
        Build the decision tree based on class parameters

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            depth (int, optional): Current tree depth. Defaults to 1.

        Returns:
            dict: Tree in the form of a dictionary
        """
        # Check depth stopping condition
        if depth > self.max_depth or len(y) < self.min_samples:
            if self.model_type == "regression":
                # Return the average value of the y-vals at the leaf node
                return(np.mean(y))
            elif self.model_type == "classifier":
                return(np.mean(y) > self.threshold)
            else:
                # Shouldn't be here
                return(None)

        # Get split under current parameters
        top_feature, thresh = self.split_function(
            X,y,self.cost_function
        )

        # Set masks
        new_left_mask = X.loc[:, top_feature] <= thresh
        new_right_mask = ~new_left_mask
        # Filter x and y
        new_left_x, new_left_y = X.loc[new_left_mask], y[new_left_mask]
        new_right_x, new_right_y = X.loc[new_right_mask], y[new_right_mask]

        node = {
            "feature": top_feature,
            "threshold": thresh,
            "left": self.build_tree(new_left_x, new_left_y, depth+1),
            "right": self.build_tree(new_right_x, new_right_y, depth+1)
        }

        # If this is the first call (root node), store it in self.tree
        if depth == 0:
            self.tree = node
        # Return the node for recursive calls
        return node  
    
    def predict_one(self, X:tuple) -> float:
        """
        Navigate the class tree to return a single prediction
        based on the features provided

        Args:
            X (tuple): Named tuple for features of a single row

        Returns:
            float: Target variable, y
        """

        assert self.tree is not None, "Build tree before predicting variables"
        # Set current_node to be the core tree
        current_node = self.tree

        # Essentially checking if we're at a leaf node
        while isinstance(current_node, dict):
            node_feature = self.tree["feature"]
            node_thresh = self.tree["threshold"]
            
            # Select the value from the features based on the current node
            # Named tuple should be accessed with X._col_ but use getattr for dynamism
            node_feature_val = getattr(X, node_feature)
            if node_feature_val <= node_thresh:
                current_node = current_node["left"]
            else:
                current_node = current_node["right"]

        # Return the leaf node which should now be a float
        return(current_node)

    def predict(self, X:pd.DataFrame) -> np.array:
        """
        Make predictions for all samples

        Args:
            X (pd.DataFrame): All rows for all features

        Returns:
            np.array: Array of y, target variables
        """
        
        return np.array([self.predict_one(x) for x in X.itertuples(index=False)])