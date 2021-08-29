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
        self.validation_mask_indices = None
        self.test_mask_indices = None
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
        
    def get_distinct_args(self) -> None:
        """
        Get all arguments from the validation (if available) and test-set, such that only arguments are added to the final rating matrix that were not already in the training - set.
        """
        # Get all training arguments and exclude the username as the first column of the dataframe
        train_args = set([arg for arg in self.train_rating_matrix.columns][1:])
        # Check if validation set is available
        if self.is_validation_set_available:
            # Get all validation arguments and exclude the username as the first column of the dataframe
            validation_args = set([arg for arg in self.validation_rating_matrix.columns][1:])
            validation_args = list(validation_args.difference(train_args))
        else:
            validation_args = None
        # Get all test arguments and exclude the username as the first column of the dataframe
        test_args = set([arg for arg in self.test_rating_matrix.columns][1:])
        test_args = list(test_args.difference(train_args))
        Args = namedtuple("Args", ["train", "validation", "test"])
        self.distinct_args = Args(train=train_args, validation=validation_args, test=test_args)
    
    def select_distinct_args(self) -> None:
        """
        Prune the validation and test rating matrix to their distinct arguments.
        """
        username_col = ["username"]
        self.test_rating_matrix = self.test_rating_matrix[username_col + self.distinct_args.test]
        if self.is_validation_set_available:
            self.validation_rating_matrix = self.validation_rating_matrix[username_col + self.distinct_args.validation]

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
            for r in (rating_matrices[1:]):
                final_rating_matrix = final_rating_matrix.merge(right=r, how="inner", on="username")
            
            # Drop the username column as it is non-numeric and can't be converted to a tensor.
            # set_trace()
            final_rating_matrix.drop(labels=["username"], axis=1, inplace=True)
            # set_trace()
            # Set the datatypes of the rating matrix to float16 to save memory and speed up computation while keeping the nan-values (not possible for integer datatype). 
            self.final_rating_matrix =  torch.from_numpy(final_rating_matrix.values).to(torch.float16).to(self.device)
          
    def get_masking_indices(self, df:pd.DataFrame) -> torch.tensor:
        """
        Get all indices that are not NaN of the provided dataframe.

        Args:
            df (pd.DataFrame): Dataframe whose non-null indices have to be found.

        Returns:
            torch.tensor: A torch tensor containing all non-null indices of the dataframe df.
        """
        # Get all not-null indices from the dataframe
        mask_idxs =  np.argwhere(~pd.isna(df.values))
        return torch.from_numpy(mask_idxs).int().to(self.device)




train = pd.read_csv("../../data/T1_T2/train.csv")
test = pd.read_csv("../../data/T1_T2/test.csv")
rmh = Rating_Matrix_Handler(train_rating_matrix=train, test_rating_matrix=test)
rmh.get_distinct_args()
rmh.select_distinct_args()
rmh.merge_rating_matrices(rmh.train_rating_matrix, rmh.test_rating_matrix)
rmh.final_rating_matrix.dtype

