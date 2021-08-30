#!/usr/bin/env python
# coding: utf-8


# imports
import pandas as pd
import numpy as np
from IPython.core.debugger import set_trace
import torch
from collections import namedtuple


get_ipython().system('jupyter nbconvert --output-dir="../python-code" --to python Rating_Matrix_Handler.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True')



class Rating_Matrix_Handler():
    """
    A class that deals with all Rating-Matrix related issues like merging and masking rating-matrices.
    """
    
    def __init__(self, train_rating_matrix:pd.DataFrame, test_rating_matrix:pd.DataFrame, validation_rating_matrix:pd.DataFrame=None):
        """
        Params:
            train_rating_matrix (pd.DataFrame): The training rating_matrix on which the TLMF algorithm will be trained upon.
            validation_rating_matrix (pd.DataFrame): The validation rating_matrix on which the TLMF algorithm can be validated on.
            test_rating_matrix (pd.DataFrame): The test rating_matrix on which the TLMF algorithm will be tested upon.
        """
        self.train_rating_matrix = train_rating_matrix
        self.validation_rating_matrix = validation_rating_matrix
        self.is_validation_set_available = self.validation_rating_matrix is not None 
        self.test_rating_matrix = test_rating_matrix
        self.validation_eval_indices = None
        self.test_eval_indices = None
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
        
    def merge_rating_matrices(self, dim:int=1, mode:str="Test") -> None:
        """
        Left-joins the training rating-matrix together on identical users with either the test-matrix or the validation-matrix(depending on the value of mode).

        Params:
            dim (int, optional): The dimension along which rating matrices are merged. Defaults to 1.
            mode (str, optional): The mode of df2. It can either be "Test" or "Validation". Defaults to "Test".
        """
        # Assertions
        assert dim >= 0, "Dimension must be non-negative."
        
        if mode=="Test":    
            df = self.test_rating_matrix
            self.test_eval_indices = self.get_eval_indices(df)
        elif mode=="Validation":
            df = self.validation_rating_matrix
            self.validation_eval_indices = self.get_eval_indices(df)
            
        # Get all non-na column indices for each username.
        eval_indices = self.get_eval_indices(df)
        # Join the matrices on the username column, keep all usernames that were already in the training matrix. Replace all values of the joined table with NaN as they have to be predicted later.
        df_nan = df.copy()
        df_nan.loc[:, df_nan.columns != "username"] = np.nan
        self.final_rating_matrix = self.train_rating_matrix.copy() 
        self.final_rating_matrix = self.final_rating_matrix.merge(right=df_nan, how="left", on="username")
        # Drop the username column as it is non-numeric and can't be converted to a tensor.
        self.final_rating_matrix.drop(labels=["username"], axis=1, inplace=True)
        # The same for the joined matrix as the username column contains non-na values but will not be evaluated.
        df.drop(labels=["username"], axis=1, inplace=True)
        # Set the datatypes of the rating matrix to float16 to save memory and speed up computation while keeping the nan-values (not possible for integer datatype). 
        self.final_rating_matrix = torch.from_numpy(self.final_rating_matrix.values).to(torch.float16).to(self.device)
        
    def get_eval_indices(self, df:pd.DataFrame, mode:str="Test") -> dict:
        """
        Get all indices that are not NaN of the provided dataframe. These indices are later used to evaluate recommender systems on.

        Params:
            df (pd.DataFrame): Dataframe whose non-null indices have to be found.
        Returns:
            dict: A dictionary containg a username as key associated with a numpy-array containing all the indices of the non-na columns for that username.
        """        
        # Get all not-null indices from the dataframe
        mask_idxs = np.argwhere(~pd.isna(df.values))
        # Perform a group by on the username - row - index.
        unique_username_rows = np.unique(mask_idxs[:,0], return_index=True)[1][1:]
        mask_idxs_grouped = np.split(mask_idxs[:,1], unique_username_rows)
        # Combine the row indices with all the non - na column - indices for this row
        username_column_agg = np.array(list(zip(list(df["username"]), mask_idxs_grouped)), dtype=object)
        # Exclude the username column from the non - na values, which is the first one
        username_column_agg = {a[0]: a[1][1:]-1 for a in username_column_agg}
        return username_column_agg



train = pd.read_csv("../../data/T1_T2/train.csv")
test = pd.read_csv("../../data/T1_T2/test.csv")
rmh = Rating_Matrix_Handler(train_rating_matrix=train, test_rating_matrix=test)
rmh.merge_rating_matrices()
rmh.final_rating_matrix

