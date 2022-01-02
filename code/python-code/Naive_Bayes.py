#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
import numpy as np
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import subprocess


subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python Naive_Bayes.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


class Naive_Bayes_CF():
    """
    Class representing a Naive - Bayes classifier implementation for the collaborative filterting setting of recommender systems.
    """
    def __init__(self, rmh, is_task_conviction:bool=True, laplacian_smoothing:float=0.0):
        self.rmh_ = rmh
        self.is_task_conviction = is_task_conviction
        self.laplacian_smoothing_ = laplacian_smoothing
        arg_range = [x for x in range(324,400) if x != 397]
        if is_task_conviction:
            self.possibles_classes = set([0,1])
            self.test_eval_indices = {user:items[items % 2 == 1] for user,items in self.rmh_.test_eval_indices.items()}
            self.task_test_rating_matrix = self.rmh_.test_rating_matrix[["username"] + [f"statement_attitude_{i}" for i in arg_range]]
            self.task_train_rating_matrix = self.rmh_.final_rating_matrix_w_usernames[["username"] + [f"statement_attitude_{i}" for i in arg_range]]
        else:
            self.possibles_classes = set([i for i in range(7)])
            self.test_eval_indices = {user:items[items % 2 == 0] for user,items in self.rmh_.test_eval_indices.items()}
            self.task_test_rating_matrix = self.rmh_.test_rating_matrix[["username"] + [f"argument_rating_{i}" for i in arg_range]]
            self.task_train_rating_matrix = self.rmh_.final_rating_matrix_w_usernames[["username"] + [f"argument_rating_{i}" for i in arg_range]]
            
        self.numerical_rating_matrix = self.rmh_.final_rating_matrix_w_usernames.drop("username", axis=1).values
    
    def build_lookups(self) -> None:
        """
        Map users and items to numerical values for further indexing.
        """
        self.userid_lookup_ = {username: i for i, username in enumerate(self.rmh_.final_rating_matrix_w_usernames["username"])}
        self.itemid_lookup_ = {item: i-1 for i, item in enumerate(list(self.rmh_.final_rating_matrix_w_usernames.columns))}
        # Reverse the two calculated mappings for bidirectional lookup
        self.username_lookup_ = {user_id: username for username, user_id in self.userid_lookup_.items()}
        self.itemname_lookup_ = {item_id: itemname for itemname, item_id in self.itemid_lookup_.items()}
    
    def compute_bias(self) -> None:
        """
        Compute item as well as user - bias as the deviation from the global average rating as a dictionary.
        """
        if self.is_task_conviction:
            global_average_rating = np.nanmean(self.numerical_rating_matrix[:, 1::2])
            self.user_bias = {user_id: np.nanmean(np.array(self.task_train_rating_matrix[self.rmh_.final_rating_matrix_w_usernames["username"] == username].values[0][2::2],dtype=float)) - global_average_rating for username, user_id in self.userid_lookup_.items()}
        else:
            global_average_rating = np.nanmean(self.numerical_rating_matrix[:, 0::2])

            self.user_bias = {user_id: np.nanmean(np.array(self.task_train_rating_matrix[self.rmh_.final_rating_matrix_w_usernames["username"] == username].values[0][1::2],dtype=float)) - global_average_rating for username, user_id in self.userid_lookup_.items()}
        self.item_bias = {item_id: np.nanmean(self.rmh_.final_rating_matrix_w_usernames[self.itemname_lookup_[item_id]].values) - global_average_rating for item_name, item_id in self.itemid_lookup_.items() if item_name != "username"}
    
    def compute_prior_prob(self) -> None:   
        """
        Compute the prior probability for every item/rating combination and save it into a class dictionary.
        """
        items_without_username = set([self.itemid_lookup_[item] for item in self.itemid_lookup_.keys() if item != "username"])
        # Build dictionary to hold the prior probabilities for all item/class combinations
        self.prior_prob_for_item = {item_id: {} for item_id in items_without_username}
        for item_id in items_without_username:
            for c in self.possibles_classes:
                # Calculate the number of users that rated the item with class c
                class_count = len(self.numerical_rating_matrix[self.numerical_rating_matrix[:,item_id] == c])
                # Calculate all the users that gave a rating for the item
                rated_count = np.sum(~np.isnan(self.numerical_rating_matrix[:,item_id]))
                self.prior_prob_for_item[item_id][c] = (class_count + self.laplacian_smoothing_) / (rated_count + rated_count * self.laplacian_smoothing_) 
    
    def compute_users_that_rated_item_with_class(self) -> None:
        """
        Compute a lookup dictionary that associates a item-id with all possibles classes. 
        These classes itself are associated with all users that rated the associated item with the associated class.
        """
        items_without_username = set([self.itemid_lookup_[item] for item in self.itemid_lookup_.keys() if item != "username"])
        self.users_rated_item_with_class = {i: {c: None} for c in self.possibles_classes for i in items_without_username}
        
        for item_id in items_without_username:
            for c in self.possibles_classes:
                # Calculate the users that rated the item with the class
                users = np.argwhere(self.numerical_rating_matrix[:,item_id] == c).flatten()
                # Set the users in the dictionary
                self.users_rated_item_with_class[item_id][c] = users
                  
    def compute_likelihood(self, _class:int, item:int, user:int, epsilon_shift:float=0.000000001) -> float:
        """
        Compute the likelihood of observed ratings given the class for a provided user-item combination.
        
        Params:
            _class: class label for which the likelihood is computed.
            item: The item - id.
            user: The user -id.
            epsilon_shift: How strongly should the likelihood be shifted if there are no users that gave the same class label to a
            item in question.  
        
        Returns:
            float: The computed likelihood.
        """
        
        # Compute items the user has rated
        items_rated_by_user = np.argwhere(~np.isnan(self.numerical_rating_matrix[user]))
        
        # Get users that rated the provided item with the provided class
        users_rated_item_with_class = self.users_rated_item_with_class[item][_class]
        
        likelihood = 1
        for k in items_rated_by_user:
            # The rating r_u_k given by the provided user to item k
            rating_given_by_user = self.numerical_rating_matrix[user][k]
            # The number of users that rated the provided item with the provided class and that additionaly rated item k with r_u_k
            num_users_identical_rating = np.sum(self.numerical_rating_matrix[[users_rated_item_with_class]][:,k] == rating_given_by_user)
            # The number of users that rated the provided item with the provided class and specified a rating for item k
            all_users = len(self.numerical_rating_matrix[[users_rated_item_with_class]][:,k])
            # Due to assumption of independence of ratings, the likelihood is the result of a product of the single probalities
            ratio = ((num_users_identical_rating + self.laplacian_smoothing_) / (all_users + all_users * self.laplacian_smoothing_))
            if ratio == 0:
                likelihood *= epsilon_shift
            else:
                likelihood *= ratio
            
        return likelihood
    
    def predict(self, user:int, item:int) -> int:
        """
        Calculate the most probable class label given the ratings.

        Params:
            user (int): The user-id for which the prediction is made.
            item (int): The item-id for which the prediction is made.

        Returns:
            int: The class label that maximizes the posterior probability.
        """
        # Get the prior probability for each possible class - label
        prior_probs = {c: self.prior_prob_for_item[item][c] for c in self.possibles_classes}
        # Variable for holding the posterior probability associated with each class label
        posterior_probs = {c: None for c in self.possibles_classes}
        # Calculate the posterior probability for all possible class labels
        for c in self.possibles_classes:
            posterior_probs[c] = self.compute_likelihood(c, item, user) * prior_probs[c]
        # Return the class label that maximizes the posterior probability
        max_posterior = max(posterior_probs, key=posterior_probs.get)
        
        return max_posterior
    
    def evaluate(self) -> float:
        """
        Calculate the performance score of the Naive Bayes model on the test dataset.

        Returns:
            float: The performance score on the test dataset. In the case of the 'Conviction' task it is the mean - accuracy error. In the case of the 'Weight' task it is the RMSE.
        """
        
        trues, preds = [], []
        
        # Only retrieve the test instances of the conviction task (odd-indexed)
        if self.is_task_conviction:
            # Calculate the mean-accuracy for the Prediction of Conviction (PoC) - task 
            mean_acc = 0.0
            # Variable for counting the correct 0/1 prediction
            count_equality = 0
            for username, test_samples in self.test_eval_indices.items():
                # Get the target-user id
                target_user_id = self.userid_lookup_[username[0]]
                for item_id in test_samples:
                # Look up the true value 
                    true_value = self.task_test_rating_matrix[self.task_test_rating_matrix["username"] == username[0]][self.itemname_lookup_[item_id]]
                    prediction = round(self.predict(target_user_id, item_id))
                    #print(inttrue_value)
                    trues.append(int(true_value))
                    preds.append(int(prediction))
                    # If the prediction is correct, increment the counter
                    if  int(true_value) == prediction:
                        count_equality += 1
                # Normalize by the number of test samples for this user
                mean_acc += count_equality / len(test_samples)
                # Set the count equality to 0 for the next user
                count_equality = 0
            # Normalize the error by the number of users in the test-set
            mean_acc /= len(self.test_eval_indices)
            print(f"Accuracy: {mean_acc:.3f}")
            return np.array(trues), np.array(preds)
        
        # Only retrieve the test instances of the weight task (even-indexed)
        else:
            rmse_error, prediction_distance = 0.0, 0.0
            for username, test_samples in self.test_eval_indices.items():
                # Get the target-user-id
                target_user_id = self.userid_lookup_[username[0]]
                for item_id in test_samples:
                    # Look up the true value
                    true_value = int(self.task_test_rating_matrix[self.task_test_rating_matrix["username"] == username[0]][self.itemname_lookup_[item_id]])
                    prediction = int(round(self.predict(target_user_id, item_id)))
                    #if prediction == 0:
                    #    print(username[0], self.itemname_lookup_[item_id], true_value)
                    trues.append(int(true_value))
                    preds.append(int(prediction))
                    prediction_distance += (int(true_value) - prediction)**2
                # Normalize by the number of test samples for this user     
                rmse_error += (prediction_distance / len(test_samples))
                # Set the prediction distance to 0 for the next user
                prediction_distance = 0
            # Normalize the prediction_distance by the number of users in the test-set
            rmse_error /= len(self.test_eval_indices)
            print(f"RMSE: {rmse_error:.3f}")
            return np.array(trues), np.array(preds)


timepoint = "T2_T3"
train_path = f"../../data/{timepoint}/train.csv"
test_path  = f"../../data/{timepoint}/test.csv"
validation_path = f"../../data/{timepoint}/validation.csv"
get_ipython().run_line_magic('run', 'Rating_Matrix_Handler.ipynb')



nb = Naive_Bayes_CF(rmh) if task == "Conviction" else Naive_Bayes_CF(rmh, is_task_conviction=False)
nb.build_lookups()
nb.compute_prior_prob()
nb.compute_users_that_rated_item_with_class()
trues,preds = nb.evaluate()


get_ipython().run_line_magic('run', 'MetricHelper.ipynb')
print("Averaged metrics:\n",mh.compute_average_metrics())

