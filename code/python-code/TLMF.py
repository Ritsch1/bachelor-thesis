#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
import numpy as np
from IPython.core.debugger import set_trace
import torch
import matplotlib.pyplot as plt
import subprocess


subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python TLMF.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


class TLMF():
    """
    A class that represents the Two-Level-Matrix-Factorization (TLMF).
    """
    
    def __init__(self, wtmf, rmh):
        """
        Params:
            wtmf (WTMF): A wtmf (Weighted Text Matrix Factorization) - object. This object represents the first level of the TLMF. It contains the argument similarity matrix which is used in TLMF.
            rmh (Rating_Matrix_Handler): A Rating_Matrix_Handler object that contains the final rating matrix that consists of the training data - entires as well as the masked test data - entries and which
            is used for optimizing the TLMF on. It also contains the indices of the original test-set in order to perform evaluation.
        """
        self.wtmf = wtmf
        self.rmh = rmh
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)

    def train(self, d:int=20, training_iterations:int=50, random_seed:int=1, print_frequency:int=1, r:float=0.05, l:float=0.01, alpha:float=0.2, n:int=10) -> [float]:
        """
        Use stochastic gradient descent to find the two optimal latent factor matrices U (Users), I (Items) 
        that minimizes the difference between the predicted values of the dot product of user-vectors and item-vectors compared to the known ratings in the user-item matrix.
        
        Params:
            d (int, optional): The embedding dimension of the user and item vectors within the latent factor matrices. Defaults to 20.
            training_iterations (int, optional): The number of training iterations over known user-item ratings. Defaults to 50.
            random_seed (int, optional):  Random seed that is used to intialize the latent factor matrices. Defaults to 1.. Defaults to 1.
            print_frequency (int, optional): The frequency in which the training error is printed w.r.t. to the iterations. Defaults to 1.
            r (float, optional): The regularization factor that controls the overfitting of the model. Defaults to 0.05.
            l (float, optional): The learning rate, a parameter that determines how heavily the vectors of the latent factor matrices are updated in every iteration. Defaults to 0.01.
            alpha (float, optional): The parameter that controls the influence of the semantic relations of the items in the prediction. Defaults to 0.02.
            n (int, optional): The n most similar items to item i that are considered for any prediction in which item i is involved.
            
        Returns:
            [float]: A list containing the error values for every iteration.
        """
        # Set random seed for reproducability
        torch.manual_seed(random_seed)
        # Randomly initialize the latent factor matrices U(ser) and I(tems)
        self.U = torch.rand([self.rmh.final_rating_matrix.shape[0], d]).to(self.device)
        self.I = torch.rand([self.rmh.final_rating_matrix.shape[1], d]).to(self.device)
        
        # Error - variable keep track of it for later visualization 
        error = []
        error_cur = 0.0
        frobenius_norm = torch.linalg.matrix_norm

        # Get non-na indices of rating - matrix to train TLMF on
        training_indices = ~torch.isnan(self.rmh.final_rating_matrix).nonzero().to(self.device)

        for iteration in range(1, training_iterations+1):
            for idx in training_indices:
                # Get the index of the current user
                user = idx[0]
                # Get the index of the current argument within the argument similarity matrix
                arg = idx[1]
                # Map the column index to the corresponding argument - similarity - matrix column index, as an argument in the rating matrix has two columns (one for Conviction and one for Weight)
                lookup_arg_idx = arg/2 if arg % 2 == 0 else arg//2  
                # Get the column- indices of the n items that are most similar to the current item in the argument similarity matrix
                most_sim_indices = torch.topk(self.wtmf.similarity_matrix[lookup_arg_idx], n, dim=0, sorted=False)[1]
                
                # Calculate the sum of similarities over all n most-similar args
                sim_sum_scaled = 0.0
                for sim_idx in most_sim_indices:
                    sim_sum_scaled += self.wtmf.similarity_matrix[lookup_arg_idx][sim_idx] * self.I[sim_idx]
                    
                sim_sum_scaled = self.I[arg] - sim_sum_scaled
                sim_sum_scaled = alpha/2 * (sim_sum_scaled.matmul(sim_sum_scaled.T))
                     
                prediction = self.U[user].matmul(self.I.T[:,arg])
                true_value = self.rmh.final_rating_matrix[user][arg]
                difference = true_value - prediction
                error_cur = (difference)**2 + (r/2 * (frobenius_norm(self.U) + frobenius_norm(self.I))) + sim_sum_scaled

                # Save old value of the user - vector for updating the item - vector (TODO: Not sure if I should already use the updated user-vector for updating the item vector)
                old_user_vector = self.U[user]

                # Update the user-vector
                self.U[user] += l * ((difference) * self.I[arg] - r * self.U[user])
                
                # Calculate the similarity sum components to update the item latent vector
                sim_sum = 2 * sim_sum_scaled
                sim_sum2 = []
                sim_sum3 = 0.0
                for neighbor_item in most_sim_indices:
                    similarity_neighbor_to_orig_item = self.wtmf.similarity_matrix[neighbor_item]
                    for neighbor_neighbor_item in torch.topk(self.wtmf.similarity_matrix[neighbor_item], n, dim=0, sorted=False)[1]:
                        sim_sum3 += self.wtmf.similarity_matrix[neighbor_item][neighbor_neighbor_item] * self.I[neighbor_neighbor_item]
                        
                    sim_sum3 = self.I[neighbor_item] - sim_sum3
                    similarity_neighbor_to_orig_item * sim_sum3
                    similarity_neighbor_to_orig_item *= alpha
                    sim_sum2.append(sim_sum + similarity_neighbor_to_orig_item)
                
                sim_sum = sum(sim_sum2)
                
                # Update the item-vector        
                self.I[arg] += l * ((difference) * self.U[user] - l * self.I[arg] - alpha * sim_sum)


            error.append(error_cur)
            error_cur = 0.0
            
            # Print out error w.r.t print-frequency
            if iteration % print_frequency == 0:
                print(f"Training - Error:{error[iteration]:.2f}\tCurrent Iteration: {iteration}\\{training_iterations}")

    def evaluate(self, task:str="conviction") -> tuple:
        """
        Params:
            task (str): The task on which the TLMF-model will be evaluated on. Can either be "conviction" or "weight". Defaults to "conviction".
        
        Returns:
            tuple: A tuple consisting of he error of the TLMF-model on the test set on index 0 and the specific task on index 1. 
        """
        # Assertions
        assert(task != "conviction" and task != "weight"), f"Unkown task - description {task} was passed."
        
        # Filter the evaluation indices based on the task
        if task == "conviction":        
            self.rmh.test_eval_indices = {user:items[items % 2 == 1] for user,items in self.rmh.test_eval_indices.items()}
            # Calculate the mean-accuracy test-error for the Prediction of Conviction (PoC) - task 
            mean_acc_error = 0.0
            # Variable for counting the correct 0/1 prediction
            count_equality = 0
            for username, test_samples  in self.rmh.test_eval_indices.keys():
                user_idx = self.rmh.final_rating_matrix_w_usernames.loc[self.rmh.final_rating_matrix_w_usernames["username"] == username]
                for arg_idx in test_samples:
                    # If the prediction is correct, increment the counter
                    count_equality += 1 if self.rmh.test_rating_matrix.loc[user_idx, arg_idx] == round(self.U[user_idx].matmul(self.I[arg_idx].T), 0) else count_equality
                # Normalize by the number of test samples for this user
                mean_acc_error += count_equality / len(test_samples)
                # Set the count equality to 0 for the next user
                count_equality = 0
            # Normalize the error by the number of users in the test-set
            mean_acc_error /= len(self.rmh.test_eval_indices)
        
            return mean_acc_error
        
        else:
            self.rmh.test_eval_indices = {user:items[items % 2 == 0] for user,items in self.rmh.test_eval_indices.items()}
            # Calculate the averaged root mean squared error for the Prediction of Weight (PoW) - task
            rmse_error = 0.0
            # Variable for measuring the distance of the true value and the prediction
            prediction_distance = 0.0
            for username, test_samples  in self.rmh.test_eval_indices.keys():
                user_idx = self.rmh.final_rating_matrix_w_usernames.loc[self.rmh.final_rating_matrix_w_usernames["username"] == username]
                for arg_idx in test_samples:
                    # If the prediction is correct, increment the counter
                    prediction_distance += (self.rmh.test_rating_matrix.loc[user_idx, arg_idx] - round(self.U[user_idx].matmul(self.I[arg_idx].T), 0))**2 
                    # Normalize by the number of test samples for this user
                rmse_error += prediction_distance / len(test_samples)
                # Set the prediction distance to 0 for the next user
                prediction_distance = 0
            # Normalize the prediction_distance by the number of users in the test-set
            rmse_error /= len(self.rmh.test_eval_indices)
            
            return rmse_error, task
        
    def plot_training_error(self, error:[float], **kwargs) -> None:
        """
        Plots the training error for every training iteration.
        
        Params:
            error (list): A list of error - values that correspond to each training iteration of the WTMF - algorithm.    
            **kwargs: Arbitrary many keyword arguments to customize the plot. E.g. color, linewidth or title.
        """        
        plt.plot([i for i in range(1, len(error)+1)], error)
        for k in kwargs.keys():
            # Invoke the function k of the plt - module to customize the plot
            getattr(plt, k) (kwargs[k])
        
        plt.show()


tlmf = TLMF(wtmf, rmh)
train_error = tlmf.train(d=20, training_iterations=50, random_seed=1, print_frequency=1, r=0.05, l=0.01, alpha=0.2, n=10)
tlmf.plot_training_error(train_error, title="TLMF Objective function error", xlabel="Iterations", ylabel="Training error")
test_result = tlmf.evaluate()
print(f"Error for task {test_result[1]} : {test_result[0]}")

