from DecisionTree.DecisionTree import DecisionTreeModel
import numpy as np
import pandas as pd

# Import amphibian data
amphibian_df = pd.read_csv("/Users/ljw/Projects/ModelsFromScratch/DataSources/Amphibians/dataset.csv", delimiter=";", skiprows=1)

# Convert frog flags to boolean
def convert_bool(s:pd.Series, true_flag=1):
    return(pd.Series(np.array([v==true_flag for v in s])))

frog_cols = ["Green frogs","Brown frogs","Common toad","Fire-bellied toad","Tree frog","Common newt","Great crested newt"]
for col in frog_cols:
    amphibian_df[col] = convert_bool(amphibian_df[col])

X = amphibian_df.loc[:, ["SR","NR","TR","VR","SUR1","SUR2","SUR3","UR","FR","OR","RR","BR","MR","CR"]]
y = amphibian_df.loc[:,"Common toad"]

# Initialise the decision tree
dt = DecisionTreeModel()
dt.build_tree(X,y)