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
    
    def __init__(self, train_rating_matrix:torch.tensor, test_rating_matrix:torch.tensor, validation_rating_matrix:torch.tensor=None):
        """
        Params:
            train_rating_matrix (torch.tensor): The training rating_matrix on which the TLMF algorithm will be trained upon.
            validation_rating_matrix (torch.tensor): The validation rating_matrix on which the TLMF algorithm can be validated on.
            test_rating_matrix (torch.tensor): The test rating_matrix on which the TLMF algorithm will be tested upon.
        """
        self.training_rating_matrix = training_rating_matrix
        self.validation_rating_matrix = validation_rating_matrix
        self.test_rating_matrix = test_rating_matrix
        self.validation_mask_indices = None
        self.test_mask_indices = None
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
        
    def merge_rating_matrices(self, *rating_matrices:pd.DataFrame, dim:int=0) -> torch.tensor:
        """
        Merges different rating-matrices together e.g. a training and a test rating - matrix.
        
        Params:
            *rating_matrices: An arbitrary number of rating_matrices to be combined along the given dimension.
            dim (int): The dimension along which rating matrices are merged.
            
        Returns:
            A torch tensor that combines all rating-matrices to one final rating-matrix.
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
            return rating_matrices[0]
        else:
            return torch.cat(rating_matrices, dim)
    
    @staticmethod
    def get_masking_indices(self, df:pd.DataFrame) -> torch.tensor:
        """
        Get all indices that are not_null of the provided dataframe.

        Args:
            df (pd.DataFrame): Dataframe whose non-null indices have to be found.

        Returns:
            torch.tensor: A torch tensor containing all non-null indices of the dataframe df.
        """
        mask_idxs =  np.where(pd.notna())
        return torch.from_numpy(mask_idxs).float().to(self.device)
        

