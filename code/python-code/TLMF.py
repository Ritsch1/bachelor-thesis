#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
import numpy as np
from IPython.core.debugger import set_trace
import torch


get_ipython().system('jupyter nbconvert --output-dir="../python-code" --to python TLMF.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True')


class TLMF():
    """
    A class that represents the Two-Level-Matrix-Factorization (TLMF).
    """
    
    def __init__(self, wtmf:WTMF, train_rating_matrix:torch.tensor, validation_rating_matrix:torch.tensor, test_rating_matrix:torch.tensor):
        """
        Params:
            wtmf (WTMF): A wtmf (Weighted Text Matrix Factorization) - object. This object represents the first level of the TLMF.
            train_rating_matrix (torch.tensor): The training rating_matrix on which the TLMF algorithm is trained upon.
            validation_rating_matrix (torch.tensor): The validation rating_matrix on which the TLMF algorithm is validated on.
            test_rating_matrix (torch.tensor): The test rating_matrix on which the TLMF algorithm is tested upon.
        """
        self.wtmf = wtmf
        self.training_rating_matrix = training_rating_matrix
        self.validation_rating_matrix = validation_rating_matrix
        self.test_rating_matrix = test_rating_matrix
        self.validation_mask_indices = None
        self.test_mask_indices = None
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
        
    def mask_rating_matrix(self, rating_matrix:torch.tensor, mask_indices:torch.tensor) -> torch.tensor:
        pass

