#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
import os
import numpy as np


def merge_datafiles(files:list):
    """
    Merges the data files together.
    
    params:
        files: The files that contain the data.
    """
    
    dfs = [pd.read_csv(f, sep=",") for f in files]
    concat_df = pd.concat(dfs, axis=0)
    return concat_df


def normalize_sim_scores(df:pd.DataFrame, max_value:int=5):
    """
    Normalize continous similarity scorings by dividing through the maximum value.
    
    params:
        df: The dataframe that contains all the data.
        max_value: The maximum possible value for a dataset.
    """
    
    # normalize by dividing through the maxmimum value, which is 5 for this dataset
    df["regression_label"] = df["regression_label"].map(lambda l: float(l)/5)
    df.to_csv("Merged_dataset.csv", index=False)


def split_data(df_orig:pd.DataFrame, train_ratio=0.8) -> tuple:
    """
    Split the dataframe with train_ratio, 1-train_ratio split into training and test set.
    
    params:
        df_orig: The original dataframe containing the complete data.
        train_ratio: The ratio of training data compared to the complete data.
    """

    mask = np.random.rand(len(df_orig)) < train_ratio
    train = df_orig[mask]
    test = df_orig[~mask]
    return (train, test)


def write_split(data_split:tuple):
    """
    Creates train and test directory and saves the data splits into them.
    
    params:
        data_split: tuple t where t[0] contains the training data and t[1] contains the test data.
    """
    
    dirs = ["TRAIN", "TEST"]
    for d in dirs:
        os.makedirs(d) 

    data_split[0].to_csv(dirs[0] + "/" + dirs[0] + "_" + f, index=False)
    data_split[1].to_csv(dirs[1] + "/" + dirs[1] + "_" + f, index=False)


# Read in relevant data
files = [f for f in os.listdir("../../data/fine_tune/Argument_Facet_Similarity") if f.endswith(".csv")]
# Merge data
merged_df = merge_datafiles(files)
# Normalize the similarity scores to the range [0,1]
normalize_sim_scores(merged_df)
# Create data split
data_split = split_data(merged_df)
# Write data splits into files
write_split(data_split)

