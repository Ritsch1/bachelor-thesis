#!/usr/bin/env python
# coding: utf-8

# imports
import numpy as np
import pandas as pd
import subprocess
import torch
from IPython.core.debugger import set_trace


# Export notebook as python script to the ../python-code folder
subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python Majority.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


class MajorityVoter():
    
    def __init__(self, rmh, task:str="Conviction") -> None:
        self.rmh_ = rmh
        self.task_ = task
        
    def calculate_predictions(self) -> None:
        """
        Calculate the mode for each item if it is the Conviction task, else calculate the mean for each item.
        """
        if self.task_ == "Conviction":
            # Drop username column and get columns corresponding to the task
            rating_matrix = self.rmh_.train_rating_matrix.drop("username", axis=1).values[:,1::2]
            num_cols = rating_matrix.shape[1]
            self.item_means = {}
            for i in range(num_cols):
                values, counts = np.unique(rating_matrix[:,i][~np.isnan(rating_matrix[:,i])], return_counts=True)
                self.item_means[i] = values[np.argmax(counts)]
        else:
            # Drop username column and get columns corresponding to the task
            rating_matrix = self.rmh_.train_rating_matrix.drop("username", axis=1).values[:,::2]
            num_cols = rating_matrix.shape[1]
            self.item_means = {i: np.nanmean(rating_matrix[:,i]) for i in range(num_cols)}
            
    def evaluate(self) -> float:
        
        if self.task_ == "Conviction":
            # Get odd-indexed arguments that correspond to conviction arguments in the range [0,1]        
            test_eval_indices_copy = {user:items[items % 2 == 1] for user,items in self.rmh_.test_eval_indices.items()}
            # To match the indices of the training, integer divide all odd indices by 2 to map them to the correct index
            for key, value in test_eval_indices_copy.items():
                test_eval_indices_copy[key] = value // 2 
            # Get rid of the username column in the test-rating -matrix for converting only numerical values into a pytorch tensor
            test_rating_matrix_copy = self.rmh_.test_rating_matrix.drop(["username"], axis=1)
            # Trim the original test_rating_matrix to the conviction columns only
            trimmed_test_rating_matrix = torch.index_select(torch.from_numpy(test_rating_matrix_copy.values).to(torch.float16), 1, torch.arange(1, test_rating_matrix_copy.shape[1], 2))
            # Calculate the mean-accuracy for the Prediction of Conviction (PoC) - task 
            mean_acc = 0.0
            # Variable for counting the correct 0/1 prediction
            count_equality = 0
            for username, test_samples in test_eval_indices_copy.items():
                # The actual username of the user
                username_str = username[0]
                # The row-index in the test set of that user
                user_idx_test = username[1]
                # Get the row-index for the user in the latent user-vector
                user_idx_pred = self.rmh_.final_rating_matrix_w_usernames[self.rmh_.final_rating_matrix_w_usernames["username"]==username_str].index[0]
                for arg_idx in test_samples:
                    # Look up the true value
                    true_value = trimmed_test_rating_matrix[user_idx_test][arg_idx]
                    prediction = round(self.item_means[arg_idx])
                    # If the prediction is correct, increment the counter
                    if  true_value == prediction:
                        count_equality += 1
                # Normalize by the number of test samples for this user
                mean_acc += count_equality / len(test_samples)
                # Set the count equality to 0 for the next user
                count_equality = 0
            # Normalize the error by the number of users in the test-set
            mean_acc /= len(test_eval_indices_copy)
        
            return mean_acc
        
        else:
            #Get even-indexed arguments that correspond to weight arguments in the range [0,6]  
            test_eval_indices_copy = {user:items[items % 2 == 0] for user,items in self.rmh_.test_eval_indices.items()}
            # To match the indices of the training, integer divide all odd indices by 2 to map them to the correct index
            for key, value in test_eval_indices_copy.items():
                test_eval_indices_copy[key] = value // 2
            # Get rid of the username column in the test-rating -matrix for proper indexing
            test_rating_matrix_copy = self.rmh_.test_rating_matrix.drop(["username"], axis=1) 
            # Trim the original test_rating_matrix to the weight columns only
            trimmed_test_rating_matrix = torch.index_select(torch.from_numpy(test_rating_matrix_copy.values).to(torch.float16), 1, torch.arange(0, test_rating_matrix_copy.shape[1], 2))
            # Calculate the averaged root mean squared error for the Prediction of Weight (PoW) - task
            rmse_error = 0.0
            # Variable for measuring the distance of the true value and the prediction
            prediction_distance = 0.0
            for username, test_samples in test_eval_indices_copy.items():
                # The actual username of the user
                username_str = username[0]
                # The row-index in the test set of that user
                user_idx_test = username[1]
                # Get the row-index for the user in the latent user-vector
                user_idx_pred = self.rmh_.final_rating_matrix_w_usernames[self.rmh_.final_rating_matrix_w_usernames["username"]==username_str].index[0]
                for arg_idx in test_samples:
                    # Look up the true value
                    true_value = trimmed_test_rating_matrix[user_idx_test][arg_idx]
                    prediction = round(self.item_means[arg_idx])
                    prediction_distance += (true_value - prediction)**2
                # Normalize by the number of test samples for this user     
                rmse_error += (prediction_distance / len(test_samples))
                # Set the prediction distance to 0 for the next user
                prediction_distance = 0
            # Normalize the prediction_distance by the number of users in the test-set
            rmse_error /= len(test_eval_indices_copy)
            
            return rmse_error
        


# Parameters for executing the Rating-Matrix-Handler notebook
timepoint = "T1_T2"
train_path = f"../../data/{timepoint}/train.csv"
test_path  = f"../../data/{timepoint}/test.csv"
validation_path = f"../../data/{timepoint}/validation.csv"
get_ipython().run_line_magic('run', 'Rating_Matrix_Handler.ipynb')


mv = MajorityVoter(rmh, task="Weight")
mv.calculate_predictions()
mv.evaluate()

