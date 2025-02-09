import pandas as pd
import numpy as np

def convert_bool(s:pd.Series, true_flag=1) -> pd.Series:
    """
    Convert series flags to boolean

    Args:
        s (pd.Series): series to change data type to boolean
        true_flag (int, optional): Current value of truth. Defaults to 1.

    Returns:
        pd.Series: Updated series with True, False values
    """
    return(pd.Series(np.array([v==true_flag for v in s])))

def one_hot_encode(df:pd.DataFrame, cat_cols:list) -> pd.DataFrame:
    """
    One hot encode categorical columns so they can be handled numerically

    Args:
        df (pd.DataFrame): Main dataframe
        cat_cols (list): Categorical columns

    Returns:
        pd.DataFrame: New dataframe with one hot encoded features
    """
    return

def split_test_train(
    X:pd.DataFrame,
    y:pd.Series,
    test_size:float=0.2,
    seed:int|None=None
):
    """
    Function to split the dataset into train and test segments

    Args:
        X (pd.DataFrame): Predictor features
        y (pd.Series): Target variable
        test_size (float, optional): Proportion of test. Defaults to 0.2.
        seed (int | None, optional): Enable reproductibility. Defaults to None.
    """

    if seed is not None:
        np.random.seed(seed)
    # Assign a new list of random indexes to be assigned to the dataframe
    new_ind = np.random.permutation(len(X))
    test_count = int(len(X)*test_size)
    # Assign test and training indexes
    test_idx, train_idx = new_ind[:test_count], new_ind[test_count:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return((X_train, X_test, y_train, y_test))