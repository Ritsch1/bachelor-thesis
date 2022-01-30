#!/usr/bin/env python
# coding: utf-8


# imports
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import subprocess


# Export notebook as python script to the ../python-code folder
subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python AutoRec.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


class AutoRec(torch.nn.Module):
    """
    Implementation of the AutoRec autoencoder model.
    """    
    def __init__(self, input_dim:int, hidden_layer_dim:int, mask_value:int, learning_rate:float, 
                 epochs:int, rmh, task:str="Conviction", random_seed:int=42):
        """
        Params:
            input_dim (int): The input dimension of the user or item vector.
            hidden_layer_dim (int): The size of neurons in the hidden layer. 
            mask_value (int): The value that is used for masking missing values in the input.
            learning_rate (float): The learning rate that is initially used to update the weights of the network.
            epochs (int): The number of times the data passes through the network in training.
            task (str, optional): The task that the AutoRec model is trained on, can be "Conviction" or "Weight". Defaults to "Conviction".
            random_seed (int, optional): The value that is used to set the random state of the model. Important for reproducing the results. Defaults to 42.
        """
        super(AutoRec, self).__init__()
        
        # Variables to perform assertions on
        checklist = [input_dim, hidden_layer_dim, epochs]
        assert all([v > 0 for v in checklist + [learning_rate]]) & all([type(v) == int for v in checklist]),         "Input dimension, epochs and hidden layer dimension need to be positive integers."
        
        # Initialize GPU for computation if available
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
        
        # Setting all configurable random seeds for reproducability
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        
        self.mask_value = mask_value
        self.input_dim = input_dim
        
        # Creating linear layers
        self.inputLayer = torch.nn.Linear(input_dim, hidden_layer_dim, bias=True, device=self.device)
        # input dimension equals output dimensions in autoencoders
        self.outputLayer = torch.nn.Linear(hidden_layer_dim, input_dim, bias=True, device=self.device)
        
        self.learning_rate = learning_rate
        self.epochs = epochs 
        self.rmh = rmh
        self.task = task
       
        # Prepare the rating matrix
        self.X = torch.nan_to_num(self.rmh.final_rating_matrix, nan=self.mask_value)
        # Slice the rating matrix based on the task
        self.X = self.X[:,1::2].to(self.device) if self.task == "Conviction" else self.X[:,::2].to(self.device)
        self.X = self.X.type(torch.FloatTensor)
       
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Calculate a reconstruction of the input tensor.

        Params:
            x (torch.Tensor): Masked user or item - vector.

        Returns:
            torch.Tensor: Reconstruction of the masked user or item - vector.
        """
        sigmoid = torch.nn.Sigmoid()
        identity = torch.nn.Identity()
        
        hidden_layer_result = sigmoid(self.inputLayer(x))
        x_reconstruction = self.outputLayer(hidden_layer_result)
        # Mask the gradient during training
        if self.training:
            masked_positions = (x == self.mask_value)
            for i, mask in enumerate(masked_positions):
                if mask:
                    self.inputLayer.weight[0][i].detach() 
                    self.inputLayer.weight[1][i].detach() 
            
        return x_reconstruction
    
    def train(self) -> list:
        """
        Train the AutoRec model.

        Params:
            X (torch.Tensor): The masked training dataset consisting of item or user vectors.
        Returns:
            List of error values for each iteration.
        """
        error = 0.0
        errors = []
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        
        i = 0
        while i < self.epochs:
            i += 1
            for x in self.X.T:
                x_reconstruction = self.forward(x)
                
                idx_non_masked_entries = torch.where(~(x == self.mask_value))[0]
                num_non_masked_entries = len(idx_non_masked_entries)
                # If all entries of a column are null 
                if num_non_masked_entries == 0:
                    continue
                else:
                  loss = torch.div(torch.sum(torch.pow((x[idx_non_masked_entries] - x_reconstruction[idx_non_masked_entries]), 2)), num_non_masked_entries) 
                  error += loss
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  # Unmask the masked gradients after backpropagation
                  torch.set_grad_enabled(True)
            
            print(f"Training Error: {error:.2f}\tCurrent Iteration {i}/{self.epochs}")
            errors.append(float(error))
            error = 0.0
        return errors
       
    def evaluate(self, mode:str="validation") -> float:
        """
        Evaluate the AutoRec - model.
        
        Params:
            mode (str, optional): A string representing the dataset (validation, test) on which to evaluate on.
        
        Returns:
            float: A number that represents the test - error of the AutoRec-model on the test set. In the case of the
            'Conviction' task it is the mean - accuracy error. In the case of the 'Weight' task it is the RMSE.
        """
        
        trues, preds = [], []
        # Set model into evaluation mode
        self.eval = True

        if mode == "validation":
            test_user_mapping = self.rmh.create_test_user_mapping(self.rmh.train_rating_matrix, self.rmh.validation_rating_matrix)
            if self.task == "Conviction":
                test_eval_indices_conviction = {user:items[items % 2 == 1] for user,items in self.rmh.validation_eval_indices.items()}
                # To match the indices of the training, integer divide all odd indices by 2 to map them to the correct index
                for key, value in test_eval_indices_conviction.items():
                    test_eval_indices_conviction[key] = value // 2
                # Get rid of the username column in the test-rating -matrix for converting only numerical values into a pytorch tensor
                test_rating_matrix_copy = self.rmh.validation_rating_matrix.drop(["username"], axis=1)
                # Trim the original validation rating_matrix to the conviction columns only
                trimmed_test_rating_matrix = torch.index_select(torch.from_numpy(test_rating_matrix_copy.values).to(torch.float16), 1, torch.arange(1, test_rating_matrix_copy.shape[1], 2))
            elif self.task == "Weight":
                test_eval_indices_weight = {user:items[items % 2 == 0] for user,items in self.rmh.validation_eval_indices.items()}
                # To match the indices of the training, integer divide all odd indices by 2 to map them to the correct index
                for key, value in test_eval_indices_weight.items():
                    test_eval_indices_weight[key] = value // 2
                # Get rid of the username column in the test-rating -matrix for converting only numerical values into a pytorch tensor
                test_rating_matrix_copy = self.rmh.validation_rating_matrix.drop(["username"], axis=1)
                # Trim the original test_rating_matrix to the conviction columns only
                trimmed_test_rating_matrix = torch.index_select(torch.from_numpy(test_rating_matrix_copy.values).to(torch.float16), 1, torch.arange(0, test_rating_matrix_copy.shape[1], 2))

        else:
            # Calculate position of test user in train input vector 
            test_user_mapping = self.rmh.create_test_user_mapping(self.rmh.train_rating_matrix, self.rmh.test_rating_matrix)
            if self.task == "Conviction":
                test_eval_indices_conviction = {user:items[items % 2 == 1] for user,items in self.rmh.test_eval_indices.items()}
                # To match the indices of the training, integer divide all odd indices by 2 to map them to the correct index
                for key, value in test_eval_indices_conviction.items():
                    test_eval_indices_conviction[key] = value // 2
                # Get rid of the username column in the test-rating -matrix for converting only numerical values into a pytorch tensor
                test_rating_matrix_copy = self.rmh.test_rating_matrix.drop(["username"], axis=1)
                # Trim the original test_rating_matrix to the conviction columns only
                trimmed_test_rating_matrix = torch.index_select(torch.from_numpy(test_rating_matrix_copy.values).to(torch.float16), 1, torch.arange(1, test_rating_matrix_copy.shape[1], 2))
            elif self.task == "Weight":
                test_eval_indices_weight = {user:items[items % 2 == 0] for user,items in self.rmh.test_eval_indices.items()}
                # To match the indices of the training, integer divide all odd indices by 2 to map them to the correct index
                for key, value in test_eval_indices_weight.items():
                    test_eval_indices_weight[key] = value // 2
                # Get rid of the username column in the test-rating -matrix for converting only numerical values into a pytorch tensor
                test_rating_matrix_copy = self.rmh.test_rating_matrix.drop(["username"], axis=1)
                # Trim the original test_rating_matrix to the conviction columns only
                trimmed_test_rating_matrix = torch.index_select(torch.from_numpy(test_rating_matrix_copy.values).to(torch.float16), 1, torch.arange(0, test_rating_matrix_copy.shape[1], 2))

        if self.task == "Conviction":
            mean_acc = 0.0
            # Variable for counting the correct 0/1 prediction
            count_equality = 0
            for username, test_samples in test_eval_indices_conviction.items():
                for arg_idx in test_samples:
                    # Look up true value
                    true_value = int(trimmed_test_rating_matrix[username[1]][arg_idx])
                    # Get prediction for the row in which the test user is located in the training set
                    prediction = int(torch.round(self.forward(self.X[:,arg_idx])[test_user_mapping[username[0]]]))
                    trues.append(int(true_value))
                    preds.append(int(prediction))
                    # If the prediction is correct, increment the counter
                    if  true_value == prediction:
                        count_equality += 1
                # Normalize by the number of test samples for this user
                mean_acc += count_equality / len(test_samples)
                # Set the count equality to 0 for the next user
                count_equality = 0
            # Normalize the error by the number of users in the test-set
            mean_acc /= len(test_eval_indices_conviction)
            print(f"Accuracy: {mean_acc:.3f}")
            return np.array(trues), np.array(preds)
        
        else: 
            # Calculate the averaged root mean squared error for the Prediction of Weight (PoW) - task
            rmse_error = 0.0
            # Variable for measuring the distance of the true value and the prediction
            prediction_distance = 0.0
            for username, test_samples in test_eval_indices_weight.items():
                for arg_idx in test_samples:
                    # Look up the true value
                    true_value = trimmed_test_rating_matrix[username[1]][arg_idx]
                    prediction = torch.round((self.forward(self.X[:,arg_idx])[test_user_mapping[username[0]]]))
                    trues.append(int(true_value))
                    preds.append(int(prediction))
                    prediction_distance += float(torch.pow(true_value - prediction, 2))
                # Normalize by the number of test samples for this user     
                rmse_error += (prediction_distance / len(test_samples))
                # Set the prediction distance to 0 for the next user
                prediction_distance = 0
            # Normalize the prediction_distance by the number of users in the test-set
            rmse_error /= len(test_eval_indices_weight)
            print(f"RMSE: {rmse_error:.3f}")
            
            return np.array(trues), np.array(preds)


autorec = AutoRec(**i_autorec_config)
results = autorec.train()
graphics.plot_training_error(error=results, title="AutoRec Objective function error", xlabel="Iterations", ylabel="Error")
trues, preds = autorec.evaluate("test")

get_ipython().run_line_magic('run', 'MetricHelper.ipynb')
print(mh.compute_average_metrics())

