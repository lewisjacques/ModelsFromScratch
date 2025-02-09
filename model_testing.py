from DecisionTree.DecisionTree import DecisionTreeModel
from postprocessing import PostProcess
from preprocessing import convert_bool, one_hot_encode, split_test_train
import numpy as np

import pandas as pd

# Import amphibian data
amphibian_df = pd.read_csv("/Users/ljw/Projects/ModelsFromScratch/DataSources/Amphibians/dataset.csv", delimiter=";", skiprows=1)

# Handle boolean columns for the frog presence
frog_cols = ["Green frogs","Brown frogs","Common toad","Fire-bellied toad","Tree frog","Common newt","Great crested newt"]
for col in frog_cols:
    amphibian_df[col] = convert_bool(amphibian_df[col])

# Handle categorical columns
cat_cols = []
#! amphibian_df[col] = one_hot_encode(amphibian_df, cat_cols)

# Non-categorical variables
X = amphibian_df.loc[:, ["SR","OR","RR","BR"]+cat_cols]
y = amphibian_df.loc[:,"Common toad"]

# Split test and train data using preprocessing
X_train, X_test, y_train, y_test = split_test_train(X,y,seed=1)

# Initialise the decision tree based on the training data
dt = DecisionTreeModel(
    model_type="classifier",
    threshold=0.8
)
dt.build_tree(X_train,y_train)

# Predict the values in the test data
y_pred = dt.predict(X_test)

# Initialise PostProcessing to determine the success of the model
pp = PostProcess(y_test.to_numpy(), y_pred)
pp.model_review()
