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
    
    def __init__(self, wtmf, rmh, mode:str="Conviction"):
        """
        Params:
            wtmf (WTMF): A wtmf (Weighted Text Matrix Factorization) - object. This object represents the first level of the TLMF. It contains the argument similarity matrix which is used in TLMF.
            rmh (Rating_Matrix_Handler): A Rating_Matrix_Handler object that contains the final rating matrix that consists of the training data - entires as well as the masked test data - entries and which
            is used for optimizing the TLMF on. It also contains the indices of the original test-set in order to perform evaluation.
            mode (str, optional): The task on which the tlmf-model is trained. Depending on the task it is trained on other subsets of data. Can take values ['Conviction','Weight']
            Defaults to 'Conviction'.
        """
        self.wtmf = wtmf
        self.rmh = rmh
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
        # Assertions
        assert(mode == "Conviction" or mode == "Weight"), f"Unkown task - description {mode} was passed."
        self.mode_ = mode

    def train(self, d:int=20, training_iterations:int=50, random_seed:int=1, print_frequency:int=1, r:float=0.05, lambda_:float=0.01, alpha:float=0.2, n:int=10, mode="Conviction") -> [float]:
        """
        Use stochastic gradient descent to find the two optimal latent factor matrices U (Users), I (Items) 
        that minimizes the difference between the predicted values of the dot product of user-vectors and item-vectors compared to the known ratings in the user-item matrix.
        
        Params:
            d (int, optional): The embedding dimension of the user and item vectors within the latent factor matrices. Defaults to 20.
            training_iterations (int, optional): The number of training iterations over known user-item ratings. Defaults to 50.
            random_seed (int, optional):  Random seed that is used to intialize the latent factor matrices. Defaults to 1.. Defaults to 1.
            print_frequency (int, optional): The frequency in which the training error is printed w.r.t. to the iterations. Defaults to 1.
            r (float, optional): The regularization factor that controls the overfitting of the model. Defaults to 0.05.
            lambda_ (float, optional): The learning rate, a parameter that determines how heavily the vectors of the latent factor matrices are updated in every iteration. Defaults to 0.01.
            alpha (float, optional): The parameter that controls the influence of the semantic relations of the items in the prediction. Defaults to 0.02.
            n (int, optional): The n most similar items to item i that are considered for any prediction in which item i is involved.
            
        Returns:
            [float]: A list containing the error values for every iteration.
        """
        if mode=="Conviction":
            # Select every second column to only contain the conviction columns with values in the range [0,1]
            # Get all relevant column-indices
            idxs = torch.arange(1, self.rmh.final_rating_matrix.shape[1], 2)
            trimmed_rating_matrix = torch.index_select(self.rmh.final_rating_matrix, 1, idxs)
        elif mode=="Weight":
            # Select every second column to only contain the conviction columns with values in the range [0,1]
            # Get all relevant column-indices
            idxs = torch.arange(0, self.rmh.final_rating_matrix.shape[1], 2)           
            trimmed_rating_matrix = torch.index_select(self.rmh.final_rating_matrix, 1, idxs)
        
        # Set random seed for reproducability
        torch.manual_seed(random_seed)
        # Randomly initialize the latent factor matrices U(ser) and I(tems)
        self.U = torch.rand([trimmed_rating_matrix.shape[0], d]).to(self.device)
        self.I = torch.rand([trimmed_rating_matrix.shape[1], d]).to(self.device)
        
        # Error - variable: keep track of each error in every iteration for later visualization 
        error = []
        error_cur = 0.0
        frobenius_norm = torch.linalg.matrix_norm

        # Get non-na indices of rating - matrix to train TLMF on
        training_indices = (~torch.isnan(trimmed_rating_matrix)).nonzero().to(torch.int).to(self.device)

        for iteration in range(training_iterations):
            for idx in training_indices:
                # Get the index of the current user
                user = idx[0]
                # Get the index of the current argument within the argument similarity matrix
                arg = idx[1]
                # Get the column- indices of the n items that are most similar to the current item in the argument similarity matrix
                most_sim_indices = torch.topk(self.wtmf.similarity_matrix[arg], n, dim=0, sorted=False)[1]
                
                # Calculate the sum of similarities over all n most-similar args
                sim_sum_scaled = 0.0    
                for index, sim_value in enumerate(self.wtmf.similarity_matrix[arg]):
                    sim_sum_scaled += sim_value * self.I[index]
                    
                sim_sum_scaled = self.I[arg] - sim_sum_scaled
                sim_sum_scaled = alpha/2 * (sim_sum_scaled.matmul(sim_sum_scaled.T))
                     
                prediction = self.U[user].matmul(self.I.T[:,arg])
                true_value = trimmed_rating_matrix[user][arg]
                difference = abs(true_value - prediction)
                error_cur += (difference)**2 + (r/2 * (frobenius_norm(self.U) + frobenius_norm(self.I))) + sim_sum_scaled

                # Save old value of the user - vector for updating the item - vector (TODO: Not sure if I should already use the updated user-vector for updating the item vector)
                old_user_vector = self.U[user]

                # Update the user-vector
                self.U[user] += lambda_ * ((difference) * self.I[arg] - r * self.U[user])
                
                # Calculate the similarity sum components to update the item latent vector
                # sim_sum = 2 * sim_sum_scaled
                # sim_sum2 = []
                # sim_sum3 = 0.0
                # for neighbor_item in most_sim_indices:
                #     similarity_neighbor_to_orig_item = self.wtmf.similarity_matrix[lookup_arg_idx][neighbor_item]
                #     for neighbor_neighbor_item in torch.topk(self.wtmf.similarity_matrix[neighbor_item], n, dim=0, sorted=False)[1]:
                #        sim_sum3 += self.wtmf.similarity_matrix[neighbor_item][neighbor_neighbor_item] * self.I[neighbor_neighbor_item]
                        
                #     sim_sum3 = self.I[neighbor_item] - sim_sum3
                #     sim_sum3 *= similarity_neighbor_to_orig_item                    
                #     sim_sum3 *= alpha
                #     sim_sum2.append(sim_sum + sim_sum3)
                
                # sim_sum = sum(sim_sum2)
                
                # Update the item-vector        
                #self.I[arg] += lambda_ * ((difference) * self.U[user] - r * self.I[arg] - alpha * sim_sum)   
                
            error.append(error_cur)
            error_cur = 0.0
            
            # Print out error w.r.t print-frequency
            if iteration % print_frequency == 0:
                print(f"Training - Error:{error[iteration]:.2f}\tCurrent Iteration: {iteration+1}\\{training_iterations}")
                
        return error

    def evaluate(self) -> float:
        """
        Returns:
            float: A float that represents the error of the TLMF-model on the test set. 
        """
        
        # Filter the evaluation indices based on the task
        if self.mode_ == "Conviction":        
            self.rmh.test_eval_indices = {user:items[items % 2 == 1] for user,items in self.rmh.test_eval_indices.items()}
            # To match the indices of the training, integer divide all odd indices by 2 to map them to the correct index
            for key, value in self.rmh.test_eval_indices.items():
                self.rmh.test_eval_indices[key] = value // 2 
            # Get rid of the username column in the test-rating -matrix for proper indexing
            self.rmh.test_rating_matrix.drop(["username"], axis=1, inplace=True)
            # Trim the original test_rating_matrix to the conviction columns only
            trimmed_test_rating_matrix = torch.index_select(torch.from_numpy(self.rmh.test_rating_matrix.values).to(torch.float16), 1, torch.arange(1, self.rmh.test_rating_matrix.shape[1], 2))
            # Calculate the mean-accuracy test-error for the Prediction of Conviction (PoC) - task 
            mean_acc_error = 0.0
            # Variable for counting the correct 0/1 prediction
            count_equality = 0
            for username, test_samples  in self.rmh.test_eval_indices.items():
                user_idx = username[1]
                for arg_idx in test_samples:
                    # If the prediction is correct, increment the counter
                    true_value = trimmed_test_rating_matrix[user_idx][arg_idx]
                    prediction = torch.round(self.U[user_idx].matmul(self.I[arg_idx].T))
                    print(f"True value: {true_value}, Prediction: {prediction}")
                    count_equality += 1 if  true_value == prediction  else count_equality
                # Normalize by the number of test samples for this user
                mean_acc_error += count_equality / len(test_samples)
                # Set the count equality to 0 for the next user
                count_equality = 0
            # Normalize the error by the number of users in the test-set
            mean_acc_error /= len(self.rmh.test_eval_indices)
        
            return mean_acc_error
        
        elif self.mode_=="Weight":
            self.rmh.test_eval_indices = {user:items[items % 2 == 0] for user,items in self.rmh.test_eval_indices.items()}
            # To match the indices of the training, integer divide all odd indices by 2 to map them to the correct index
            for key, value in self.rmh.test_eval_indices.items():
                self.rmh.test_eval_indices[key] = value // 2
            # Get rid of the username column in the test-rating -matrix for proper indexing
            self.rmh.test_rating_matrix.drop(["username"], axis=1, inplace=True) 
            # Trim the original test_rating_matrix to the weight columns only
            trimmed_test_rating_matrix = torch.index_select(torch.from_numpy(self.rmh.test_rating_matrix.values).to(torch.float16), 1, torch.arange(0, self.rmh.test_rating_matrix.shape[1], 2))
            # Calculate the averaged root mean squared error for the Prediction of Weight (PoW) - task
            rmse_error = 0.0
            # Variable for measuring the distance of the true value and the prediction
            prediction_distance = 0.0
            for username, test_samples  in self.rmh.test_eval_indices.items():
                user_idx = username[1]
                for arg_idx in test_samples:
                    # If the prediction is correct, increment the counter
                    prediction_distance += (trimmed_test_rating_matrix[user_idx][arg_idx] - torch.round(self.U[user_idx].matmul(self.I[arg_idx].T)))**2 
                    # Normalize by the number of test samples for this user
                rmse_error += prediction_distance / len(test_samples)
                # Set the prediction distance to 0 for the next user
                prediction_distance = 0
            # Normalize the prediction_distance by the number of users in the test-set
            rmse_error /= len(self.rmh.test_eval_indices)
            
            return rmse_error
        
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


k=10
training_iterations=2
weight=0.05
gamma=0.01
random_seed=11
print_frequency=1
get_ipython().run_line_magic('run', 'WTMF.ipynb')


# Parameters for executing the Rating-Matrix-Handler notebook
train_path = f"C:\\Users\\Rico\\Desktop\\Diverses\\bachelorarbeit\\bachelor-thesis\\data\\T1_T2\\train.csv"
test_path  = f"C:\\Users\\Rico\\Desktop\\Diverses\\bachelorarbeit\\bachelor-thesis\\data\\T1_T2\\test.csv"
get_ipython().run_line_magic('run', 'Rating_Matrix_Handler.ipynb')


wtmf=wtmf
rmh=rmh
d=20
training_iterations=10
random_seed=12
print_frequency=1
r=0.02
l=0.01
alpha=0.01
n=3


tlmf = TLMF(wtmf, rmh, mode="Conviction")
train_error = tlmf.train(d, training_iterations, random_seed    , print_frequency, r, l, alpha, n, tlmf.mode_)
tlmf.plot_training_error(train_error, title="TLMF Objective function error", xlabel="Iterations", ylabel="Training error")
test_result = tlmf.evaluate()
print(f"Error for the {tlmf.mode_} task: {test_result}")

