#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
import numpy as np
import torch
import subprocess
import warnings
warnings.filterwarnings('ignore')


# Export notebook as python script to the ../python-code folder
subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python Rating_Matrix_Handler.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


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
        self.validation_eval_indices = None
        self.test_eval_indices = None
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
        
    def create_test_user_mapping(self, train:pd.DataFrame, test:pd.DataFrame) -> dict:
        """
        Create a mapping of a test users name to his position in the train rating matrix.

        Params:
            train (pd.DataFrame): The train rating matrix.
            test (pd.DataFrame): The test rating matrix.

        Returns:
            dict: Mapping of test user name to position index in the train rating matrix.
        """
        mapping = {user:int(np.argwhere((train["username"]==user).values)) for user in test["username"]}
        return mapping
        
    def create_torch_rating_matrix(self) -> None:
        """
        Creates the final rating matrix as torch tensor that is to be trained on.
        
        Params:
            df (pd.DataFrame): Either the test or validation dataframe for which the evaluation indices are calculated.
            mode (str, optional): The mode for which the evaluation indices of the test or validation matrix are calculated. 
            It can either be "Test" or "Validation". Defaults to "Test".
        """    
        self.final_rating_matrix = self.train_rating_matrix.copy() 
        self.final_rating_matrix_w_usernames = self.final_rating_matrix.copy()
        # Drop the username column as it is non-numeric and can't be converted to a tensor.
        self.final_rating_matrix.drop(labels=["username"], axis=1, inplace=True)
        # Set the datatypes of the rating matrix to float16 to save memory and speed up computation while keeping the nan-values (not possible for integer datatype). 
        self.final_rating_matrix = torch.from_numpy(self.final_rating_matrix.values).to(torch.float16).to(self.device)
        
    def get_eval_indices(self, df:pd.DataFrame, mode:str="test") -> None:
        """
        Get all indices that are not NaN of the provided dataframe. These indices are later used to evaluate the recommender systems on,
        either during training on the validation set or during testing on the test dataset.

        Params:
            df (pd.DataFrame): Dataframe whose non-null indices have to be found.
        """        
        # Get all not-null indices from the dataframe
        mask_idxs = np.argwhere(~pd.isna(df.values))
        # Build dictionary of unique row-ids associated with sets that will contain the non-na column-ids
        userid_ratings = {id:set() for id in np.unique(mask_idxs[:,0])}
        # Add all non-na column indices to the corresponding row-ids
        for entry in mask_idxs:
            # Exclude the username column-index from the non - na values, which is the index 0. It is not part of the evaluation
            if entry[1] == 0:
                continue
            # All added column - indices have to be decremented by 1, as the username-column is deleted and they are shifted one index to the left
            userid_ratings[entry[0]].add(entry[1]-1)
        # Use a tuples consisting of (username, row_id) as keys for cross-referencing later on
        username_ratings = {(df.loc[username]["username"], username):ratings for username, ratings in userid_ratings.items()}
        # Cast the set-values to numpy-arrays for later filtering the column-indices depending on the task
        username_ratings = {username:np.array(list(ratings)) for username,ratings in username_ratings.items()}
        if mode == "test":
            self.test_eval_indices = username_ratings
        else:
            self.validation_eval_indices = username_ratings


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
validation = pd.read_csv(validation_path)
rmh = Rating_Matrix_Handler(train_rating_matrix=train, test_rating_matrix=test, validation_rating_matrix=validation)
rmh.create_torch_rating_matrix()
rmh.get_eval_indices(test)
rmh.get_eval_indices(validation, "validation")

