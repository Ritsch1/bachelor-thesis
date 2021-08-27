#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
import numpy as np
from IPython.core.debugger import set_trace
import torch


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
        self.test_rating_matrix = test_rating_matrix
        self.validation_mask_indices = None
        self.test_mask_indices = None
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
        
    def suffixing_rating_matrices(self) -> None:
        """
        Suffixing the columns of test and validation rating matrices with "_test" and "_validation", respectively to differentiate between different identical arguments but at different timesteps.
        
        """
        # Suffixing columns
        self.test_rating_matrix.columns = [c + "_test" if c != "username" else c for c in self.test_rating_matrix.columns]
        # Check if validation rating matrix is available
        if self.validation_rating_matrix is not None:
            self.validation_rating_matrix = [c + "_validation" if c != "username" else c for c in self.validation_rating_matrix]     

    def merge_rating_matrices(self, *rating_matrices:pd.DataFrame, dim:int=1) -> None:
        """
        Merges different rating-matrices together e.g. a training and a test rating - matrix.
        
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
            final_rating_matrix = pd.merge(rating_matrices, on="username")
            # Drop the username column as it is non-numeric and can't be converted to a tensor.
            final_rating_matrix.drop(labels=["username"], axis=1, inplace=True)
            # Set the datatypes of the rating matrix to uint8 (unsigned 8-bit integer) to save memory and speed up computation. The numerical values in the rating matrices are in the range of [0,6]
            self.final_rating_matrix =  torch.from_numpy(final_rating_matrix.values).uint8().to(self.device)
    
    @staticmethod
    def get_masking_indices(df:pd.DataFrame) -> torch.tensor:
        """
        Get all indices that are not_null of the provided dataframe.

        Args:
            df (pd.DataFrame): Dataframe whose non-null indices have to be found.

        Returns:
            torch.tensor: A torch tensor containing all non-null indices of the dataframe df.
        """
        # Get all not-null indices from the dataframe
        mask_idxs =  np.argwhere(~np.isnan(df.values))
        return torch.from_numpy(mask_idxs).float()


train = pd.read_csv("../../data/T1_T2/train.csv")
test = pd.read_csv("../../data/T1_T2/test.csv")
rmh = Rating_Matrix_Handler(train_rating_matrix=train, test_rating_matrix=test)
rmh.suffixing_rating_matrices()
rmh.test_rating_matrix

