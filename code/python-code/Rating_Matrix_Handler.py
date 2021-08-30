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
        
    def merge_rating_matrices(self, *rating_matrices:pd.DataFrame, dim:int=1) -> None:
        """
        Merges different rating-matrices together on identical users e.g. a training and a test rating - matrix.
        
        Params:
            *rating_matrices: An arbitrary number of rating_matrices to be combined along the given dimension.
            dim (int): The dimension along which rating matrices are merged. Defaults to 1, as the set of user present in both rating matrices are evaluated. 
        """
        # Assertions
        assert len(rating_matrices) > 0, "No rating-matrix was provided."
        assert dim >= 0, "Dimension must be non-negative."
        dims = []
        for i, r in enumerate(rating_matrices):
            dims.append(len(r.shape))
            assert dim <= dims[i], "f The given dimension - value {dim} is too big for the dimensions of the given rating-matrices."
        assert len(set(dims)) == 1, "All rating matrices need to have the same number of dimensions."
        
        if len(rating_matrices) == 1:
            print("Only one rating matrix provided, returning it without change.")
            return rating_matrices[0]
        else:
            # Only keep data for users that are present in both rating matrices.
            final_rating_matrix = rating_matrices[0]
            self.test_eval_indices = self.get_eval_indices(rating_matrices[1])
            for username in self.test_eval_indices.keys():
                # Set a tuple containing the old coordinates of the user-index at index 0 and the new coordinates of the same user at index 1, -1 because the username column will later be deleted
                self.test_eval_indices[username] = (self.test_eval_indices[username], (final_rating_matrix[final_rating_matrix["username"]==username].index.values.astype(int)[0], self.test_eval_indices[username][1]-1))
            
            # Join the matrices on the username column, keep all usernames that were already in the training matrix    
            final_rating_matrix = final_rating_matrix.merge(right=rating_matrices[1], how="left", on="username")
            
            # Drop the username column as it is non-numeric and can't be converted to a tensor.
            final_rating_matrix.drop(labels=["username"], axis=1, inplace=True)
            # Set the datatypes of the rating matrix to float16 to save memory and speed up computation while keeping the nan-values (not possible for integer datatype). 
            self.final_rating_matrix =  torch.from_numpy(final_rating_matrix.values).to(torch.float16).to(self.device)
          
    def get_eval_indices(self, df:pd.DataFrame) -> dict:
        """
        Get all indices that are not NaN of the provided dataframe. These indices are later used to evaluate recommender systems on.

        Args:
            df (pd.DataFrame): Dataframe whose non-null indices have to be found.
        Returns:
            dict: A dictionary containg a username as key associated with a numpy-array containing all the indices of the non-na columns for that username.
        """
        # Get all not-null indices from the dataframe
        mask_idxs = np.argwhere(~pd.isna(test.values))
        # Perform a group by on the username - row - index.
        unique_username_rows = np.unique(mask_idxs[:,0], return_index=True)[1][1:]
        mask_idxs_grouped = np.split(mask_idxs[:,1], unique_username_rows)
        # Combine the row indices with all the non - na column - indices for this row
        username_column_agg = np.array(list(zip(list(df["username"]), mask_idxs_grouped)))
        # Exclude the username column from the non - na values, which is the first one
        username_column_agg = {a[0]: a[1][1:] for a in username_column_agg}
        return username_column_agg



train = pd.read_csv("../../data/T1_T2/train.csv")
test = pd.read_csv("../../data/T1_T2/test.csv")
rmh = Rating_Matrix_Handler(train_rating_matrix=train, test_rating_matrix=test)
# rmh.get_distinct_args()
# rmh.select_distinct_args()
rmh.merge_rating_matrices(rmh.train_rating_matrix, rmh.test_rating_matrix)
rmh.test_eval_indices

