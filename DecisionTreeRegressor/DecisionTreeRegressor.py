import pandas as pd
import numpy as np

#! Add split and cost functions in their own classes

class DecisionTreeRegressor:
    def __init__(self):
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
        self.tree = None

    def build_tree(self, X:pd.DataFrame, y:pd.Series, depth=1):
        # Check depth stopping condition
        if depth > self.max_depth or len(y) < self.min_samples:
            # Return the average value of the y-vals at the leaf node
            return(np.mean(y))

        # Get split under current parameters
        top_feature, thresh = self.best_split_numerical(X,y)

        new_left_mask = X[:, top_feature] <= thresh
        new_right_mask = ~new_left_mask


        new_left_x, new_left_y = X[new_left_mask], y[new_left_mask]
        new_right_x, new_right_y = X[new_right_mask], y[new_right_mask]

        return({
            "feature": top_feature,
            "threshold": thresh,
            "left": self.build_tree(new_left_x, new_left_y, depth+1),
            "right": self.build_tree(new_right_x, new_right_y, depth+1)
        })

    ### --- Cost Functions --- ###

    @staticmethod
    def mean_squared_error(y:pd.Series):
        """
        Mean squared error for a series of target values

        Args:
            y (pd.Series): Series of the target variable
        """
        return(np.mean(y-np.mean(y))**2)
    
    ### --- Split Functions --- ###

    def best_split_numerical(self, X:pd.DataFrame, y:pd.Series) -> tuple:
        """
        Find the best split for a group of numerical columns. Returning
        the feature to split on along with the threshold to take

        Args:
        X (pd.DataFrame): Data frame of features and their values
        y (pd.Series): Target variable for the same dataset

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
            unique_f_values = X[:,f].unique()
            # Take all potential values as thresholds for now
            for threshold in unique_f_values:
                # Find the values either side of the split
                left_mask = X[:,f] <= threshold
                right_mask = ~left_mask

                # Avoid empty splits on either side by checking true counts
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                # Calculate the MSE on either side of the split
                # The lower the variance the higher the homogeneity 
                left_mse = self.mean_squared_error(y[left_mask])
                right_mse = self.mean_squared_error(y[right_mask])
                # Weighted mse dependent on split sizes
                total_mse = ((sum(left_mask)*left_mse) + (sum(right_mask)*right_mse))/n_samples

                # Check if this MSE outperforms current
                if total_mse < best_mse:
                    best_feature, best_threshold, best_mse = f, threshold, total_mse
        
        return best_feature, best_threshold