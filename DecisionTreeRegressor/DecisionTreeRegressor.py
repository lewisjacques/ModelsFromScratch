import pandas as pd
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, X:pd.DataFrame, y:pd.Series):
        """
        Class for a Decision Tree Regressor

        Args:
            X (pd.DataFrame): Data frame of features and their values
            y (pd.Series): Target variable for the same dataset
        """

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

    def best_split_numerical(self) -> tuple:
        """
        Find the best split for a group of numerical columns. Returning
        the feature to split on along with the threshold to take

        Returns:
            tuple: (feature, threshold)
        """

        # All features
        features = self.X.columns
        n_samples = self.X.shape[0]

        # Current best splits and thresholds
        best_feature, best_threshold, best_mse = None, None, float("inf")

        # Iterate through each feature and potential threshold
        for f in features:
            unique_f_values = self.X[:,f].unique()
            # Take all potential values as thresholds for now
            for threshold in unique_f_values:
                # Find the values either side of the split
                left_mask = self.X[:,f] <= threshold
                right_mask = ~left_mask

                # Avoid empty splits on either side by checking true counts
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                # Calculate the MSE on either side of the split
                # The lower the variance the higher the homogeneity 
                left_mse = self.mean_squared_error(self.y[left_mask])
                right_mse = self.mean_squared_error(self.y[right_mask])
                # Weighted mse dependent on split sizes
                total_mse = ((sum(left_mask)*left_mse) + (sum(right_mask)*right_mse))/n_samples

                # Check if this MSE outperforms current
                if total_mse < best_mse:
                    best_feature, best_threshold, best_mse = f, threshold, total_mse
        
        return((best_feature, best_threshold))