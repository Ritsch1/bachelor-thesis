#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
import numpy as np
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import subprocess
from abc import ABC, abstractmethod
from typing import Iterable


subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python User_Based_Neighborhood.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


class Neighborhood_Model(ABC):
    """
    Abstract base class for all neighborhood based models. The 'predict', and the 'compute_similarity' - functions need to be implemented by inheriting classes.
    """
    
    def __init__(self, rmh, task:str="Conviction"):
        """
        Params:
            rmh (Rating_Matrix_Handler): A Rating_Matrix_Handler object that provides the relevant rating matrices as well as test indices.
            task (str): The task that the model should predict on. Defaults to "Conviction".
        """
        super().__init__()
        self.rmh_ = rmh
        self.task_ = task
        arg_range = [x for x in range(324,400) if x != 397]
        if self.task_ == "Conviction":
            self.task_train_rating_matrix = self.rmh_.final_rating_matrix_w_usernames[["username"] + [f"statement_attitude_{i}" for i in arg_range]]
            self.task_test_rating_matrix = self.rmh_.test_rating_matrix[["username"] + [f"statement_attitude_{i}" for i in arg_range]]
            self.test_eval_indices = {user:(items[items % 2 == 1] // 2) for user,items in self.rmh_.test_eval_indices.items()}
        else:
            self.task_train_rating_matrix = self.rmh_.final_rating_matrix_w_usernames[["username"] + [f"argument_rating_{i}" for i in arg_range]]
            self.task_test_rating_matrix = self.rmh_.test_rating_matrix[["username"] + [f"argument_rating_{i}" for i in arg_range]]
            self.test_eval_indices = {user:((items[items % 2 == 0] // 2) + 1) for user,items in self.rmh_.test_eval_indices.items()}
    
    def build_lookups(self) -> None:
        """
        Map users and items to numerical values for further indexing.
        """
        self.userid_lookup_ = {username: i for i, username in enumerate(self.task_train_rating_matrix["username"])}
        self.itemid_lookup_ = {item: i for i, item in enumerate(list(self.task_train_rating_matrix.columns))}
        # Reverse the two calculated mappings for bidirectional lookup
        self.username_lookup_ = {user_id: username for username, user_id in self.userid_lookup_.items()}
        self.itemname_lookup_ = {item_id: itemname for itemname, item_id in self.itemid_lookup_.items()}
        
    def calculate_items_rated_by_user(self) -> None:
        """
        Calculate a dictionary containing usernames as keys and a numpy array of rated items as key.
        """
        self.items_rated_by_user = {}
        users = set(self.userid_lookup_.keys())
        for u in users:
            # Calculate the item-indices that are non-na for each user
            self.items_rated_by_user[self.userid_lookup_[u]] = np.argwhere(~pd.isna(self.task_train_rating_matrix[self.task_train_rating_matrix["username"] == u]).values)[:,1][1:]
        for rated_items in self.items_rated_by_user.values():
            # Delete the first entry, as its the username which will not be used for similarity computation
            rated_items = rated_items[1:]
            
    
    def compute_mean_ratings(self, for_users:bool=True) -> None:
        """
        Compute the mean rating for users/items depending on the for_users - flag as a dictionary with users/items as key and the average rating as value.

        Params:
            for_users (bool, optional): If set to True, calculate user-rating means. If set to False, calculate item-rating means. Defaults to True.
        """
        self.user_mean_ratings, self.item_mean_ratings = {}, {}
        
        for username, user_id in self.userid_lookup_.items():
            self.user_mean_ratings[user_id] = np.nanmean(np.array(self.task_train_rating_matrix[self.task_train_rating_matrix["username"] == username].values[0][1:],dtype=float))
    
        # Exclude the username column
        for item in self.task_train_rating_matrix.columns[1:]:
            self.item_mean_ratings[self.itemid_lookup_[item]] = np.nanmean(self.task_train_rating_matrix[item].values)
    
    def compute_mutual_objects(self, iterable1:Iterable, iterable2:Iterable) -> set:
        """
        Computes the mutual objects of two iterables.

        Args:
            iterable1 (Iterable): First iterable object.
            iterable2 (Iterable): Second iterable object.

        Returns:
            set: The mutual objects of the first and second iterable object.
        """
        return set(iterable1).intersection(set(iterable2))

    @abstractmethod
    def predict(self, user:str, item:str) -> int:
        pass
    
    @abstractmethod
    def compute_similarity(self, object1, object2) -> float:
        """
        Compute the similarity between the ratings of both objects. As the similarity is only computed over mutual ratings of both objects, the dimension of both rating vectors must be equal.

        Args:
            ratings1 (np.array): First object.
            ratings2 (np.array): Second object.

        Returns:
            float: Similarity score of both rating vectors.
        """
        pass
    
    @abstractmethod
    def compute_similarity_matrix(self) -> np.array:
        pass

    def evaluate(self, k:int, similarity_threshold:float) -> float:
        pass
        
        


class User_Neighborhood_Pearson_Centered(Neighborhood_Model):
    """
    A user - based neighborhood model that takes into account rating bias by centering the raw data for each user and applying the Pearson Correlation Coefficient for predicting the similarity of user-pairs. 
    """
    def __init__(self, rmh, task='Conviction'):
        super().__init__(rmh, task=task)
        if self.task_ == "Conviction":
            # Subtract the row - mean from all values in that row
            self.mean_centered_train_rating_matrix = self.task_train_rating_matrix.drop("username", axis=1).sub(self.task_train_rating_matrix.drop("username", axis=1).mean(axis=1), axis=0).values
        else:
            # Subtract the row mean from all the values in that row
            self.mean_centered_train_rating_matrix = self.task_train_rating_matrix.drop("username", axis=1).sub(self.task_train_rating_matrix.drop("username", axis=1).mean(axis=1), axis=0).values
    
    def compute_similarity(self, user1:int, user2:int, similarity_function:str="Pearson") -> float:
        """
        Compute the Pearson Correlation Coefficient for the two rating vectors.

        Args:
            user1 (int): First user.
            user2 (int): Second user.
            similarity_function (str): Function that is used to compute the similarity of two users. Can either be RawCosine, RawCosineDiscounted, Pearson. Defaults to Pearson.
        Returns:
            float: Pearson Correlation Coefficient of both rating vectors.
        """

        # Get rated items of both users
        rated_items1 = self.items_rated_by_user[user1]
        rated_items2 = self.items_rated_by_user[user2]
     
        # Get mutual rated items
        mutual_rated_items = self.compute_mutual_objects(rated_items1, rated_items2)
        # If there are no mutual rated items, return 0 for the Pearson Correlation Coefficient
        if len(mutual_rated_items) == 0:
            return 0.0
        
        if similarity_function == "RawCosine":
            ratings = []
            for item in mutual_rated_items:
                r_u1 = int(self.task_train_rating_matrix[self.task_train_rating_matrix["username"] == self.username_lookup_[user1]][self.itemname_lookup_[item]])
                r_u2 = int(self.task_train_rating_matrix[self.task_train_rating_matrix["username"] == self.username_lookup_[user2]][self.itemname_lookup_[item]])
                ratings.append( (r_u1, r_u2))
        
            ratings = np.array(ratings)
            nominator = np.sum(ratings[:,0] * ratings[:,1])
            denominator = np.sqrt(np.sum(ratings[:,0]**2)) * np.sqrt(np.sum(ratings[:,1]**2))
            return nominator / denominator
        
        else:
            # Get mean rating of both users
            mean_rating1 = self.user_mean_ratings[user1]
            mean_rating2 = self.user_mean_ratings[user2]
            
            # Variable holding the difference between actual rating and mean rating for both users, as this value needs to be calculated multiple times
            diffs = []
            for i, item in enumerate(mutual_rated_items):
                # Decrement item id by 1 as the train rating matrix still contains the username
                r_u1 = int(self.task_train_rating_matrix[self.task_train_rating_matrix["username"] == self.username_lookup_[user1]][self.itemname_lookup_[item]])
                r_u2 = int(self.task_train_rating_matrix[self.task_train_rating_matrix["username"] == self.username_lookup_[user2]][self.itemname_lookup_[item]])
                diffs.append( (r_u1 - mean_rating1, r_u2 - mean_rating2) )
            
            # Calculate the nominator and denominator of the Pearson Correlation Coefficient
            # Transform the list into numpy-array for indexing
            diffs = np.array(diffs)
            nominator = np.sum(diffs[:,0] * diffs[:,1])
            denominator = np.sqrt(np.sum(diffs[:,0]**2) * np.sum(diffs[:,1]**2))
            
           # Catch division by zero        
            similarity =  similarity if not np.isnan((similarity := nominator / denominator)) else 0.0 

            return similarity
        
    def compute_similarity_matrix(self):
        """
        Compute a dictionary containing the target-user name as key and a dictionary {user_id:similarity_value} as value for the corresponding similarity value with other users.
        """
        super().compute_similarity_matrix()
        similarity_values = []
        users = self.userid_lookup_.values()
        counter = 1
        for target_user in users:
            print(f"Computation started for user {counter}")
            similarity_values_target_user = []
            for user in users:
                # Set the similarity value of target user with himself to -1 to ensure that his ratings are not used for himself in the prediction
                if target_user == user:
                    similarity_values_target_user.append( (user, -1) ) 
                else:
                    similarity_values_target_user.append( (user, self.compute_similarity(target_user, user)) ) 
        
            similarity_values.append(similarity_values_target_user)
            counter += 1
            
        # Convert the list of lists to a 2d - numpy array for later processing    
        self.pearson_correlation_matrix = np.array(similarity_values)

    def calculate_k_closest_users(self, item:int, target_user:int, k:int, similarity_threshold:float, threshold_decrease_rate:float) -> np.array:
        """
        Calculate the k-closest users to the target-user that rated the same item and whose similarity value is above the similarity threshold.
        
        Params:
            item (int): The item-id for which the k closest users have to be found w.r.t. the target-user.
            target_user (int): The target user-id for which the k-closest users have to be found.
            k (int): The upper bound of the number of users that should be included in the final set. 
            similarity_threshold (float): A similarity threshold to set the minimum degree of similarity that a user has to have in order to be included in the final set.
            threshold_decrease_rate (float): If there are no users with a similarity value above the threshold, decrease the threshold by this rate until there are users with above the new threshold value.
        Returns:
            np.array: A numpy array containing tuples [user_id, similarity_value].
        """
        # Get array of similarities for target user
        user_similarities = self.pearson_correlation_matrix[target_user]
        # Get all users with their rated items
        users_rated_items = tuple(self.items_rated_by_user.items())
        # Filter for all users that have rated the same item
        users_that_rated = np.array([user_id for user_id, rated_items in users_rated_items if item in rated_items])
        # Get all indices of users where pearson similarity >= threshold
        users_sim_bigger_thresh = user_similarities[user_similarities[:,1] >= similarity_threshold]
        # Build intersection of users who rated the item and whose similarity value >= threshold
        user_set = np.intersect1d(users_that_rated, users_sim_bigger_thresh[:,0])
        # Sort depending on the similarity value
        user_similarities = user_similarities[user_similarities[:, 1].argsort()]
        # Depending on the size of the set, return the final [user-ids, similarity-value] tuples
        if len(user_similarities >= k):
            return user_similarities[-k:]
        else:
            return user_similarities
    
    def predict(self, target_user:int, item:int, k:int, similarity_threshold:float, threshold_decrease_rate:float) -> int:
        """
        Params:
            target_user (int): The target user-id for which the k-closest users have to be found.
            item (int): The item-id for which the k closest users have to be found w.r.t. the target-user.
            k (int): The upper bound of the number of users that should be included in the final set.
            similarity_threshold (float): A similarity threshold to set the minimum degree of similarity that a user has to have in order to be included in the final set.
            threshold_decrease_rate (float): If there are no users with a similarity value above the threshold, decrease the threshold by this rate until there are users with above the new threshold value.

        Returns:
            int: A discrete numerical value for this (user,item) - pair.
        """
        super().predict(target_user, item)
        # Get the k -closest [user_id, similarity_value] - tuples for the target-user and item 
        k_closest_users = self.calculate_k_closest_users(item, target_user, k, similarity_threshold, threshold_decrease_rate)
        # Calculate the mean-centered prediction
        nominator = []
        denominator = []
        
        # If there were no users in the similarity set, return the mean of the item
        if len(k_closest_users) == 0:
            return round(self.item_mean_ratings[item])
        
        for user in k_closest_users[:,0]:
            user_int = int(user)
            item_int = int(item)
            pearson_correlation = self.pearson_correlation_matrix[target_user][user_int][1]
            denominator.append(abs(pearson_correlation))
            # Decrement item by 1 as the mean_centered_train_rating_matrix does not contain the username column anymore
            nominator.append(pearson_correlation * self.mean_centered_train_rating_matrix[user_int][item_int-1])
        
        nominator = sum(nominator)
        denominator = sum(denominator)
        pearson_normalized = pearson if not np.isnan((pearson := nominator / denominator)) else 0.0
        
        # Add the mean rating of the target - user
        pearson_normalized += self.user_mean_ratings[target_user]
        return pearson_normalized
    
    def evaluate(self, k:int, similarity_threshold:float, threshold_decrease_rate:float) -> float:
        """
        Evaluate the performance of the model on the test dataset for a specific task.

        Params:
            k (int): The upper bound of the number of users that should be included in the final set.
            similarity_threshold (float): A similarity threshold to set the minimum degree of similarity that a user has to have in order to be included in the final set.
            threshold_decrease_rate (float): If there are no users with a similarity value above the threshold, decrease the threshold by this rate until there are users with above the new threshold value.
            
        Returns:
            float: RMSE if task is "Weight" and mean accuracy if task is "Conviction".
        """
        trues, preds = [], []
        # Filter the evaluation indices based on the task
        if self.task_ == "Conviction":
            # Calculate the mean-accuracy for the Prediction of Conviction (PoC) - task 
            mean_acc = 0.0
            # Variable for counting the correct 0/1 prediction
            count_equality = 0
            for username, test_samples in self.test_eval_indices.items():
                # Get the target-user id
                target_user_id = self.userid_lookup_[username[0]]
                for item_id in test_samples:
                # Look up the true value
                    true_value = self.task_test_rating_matrix[self.task_test_rating_matrix["username"] == username[0]][self.itemname_lookup_[item_id+1]]
                    prediction = round(self.predict(target_user_id, item_id, k, similarity_threshold, threshold_decrease_rate))
                    preds.append(prediction)
                    trues.append(true_value)
                    # If the prediction is correct, increment the counter
                    if int(true_value) == prediction:
                        count_equality += 1
                # Normalize by the number of test samples for this user
                mean_acc += count_equality / len(test_samples)
                # Set the count equality to 0 for the next user
                count_equality = 0
            # Normalize the error by the number of users in the test-set
            mean_acc /= len(self.test_eval_indices)
            print(f"Accuracy: {mean_acc:.3f}")
            
            return np.array(trues), np.array(preds)
        
        else:
            rmse_error, prediction_distance = 0.0, 0.0
            for username, test_samples in self.test_eval_indices.items():
                # Get the target-user-id
                target_user_id = self.userid_lookup_[username[0]]
                for item_id in test_samples:
                    # Look up the true value
                    true_value = self.task_test_rating_matrix[self.task_test_rating_matrix["username"] == username[0]][self.itemname_lookup_[item_id]]
                    prediction = round(self.predict(target_user_id, item_id, k, similarity_threshold, threshold_decrease_rate))
                    preds.append(prediction)
                    trues.append(true_value)
                    prediction_distance += (int(true_value) - prediction)**2
                # Normalize by the number of test samples for this user     
                rmse_error += (prediction_distance / len(test_samples))
                # Set the prediction distance to 0 for the next user
                prediction_distance = 0
            # Normalize the prediction_distance by the number of users in the test-set
            rmse_error /= len(self.test_eval_indices)
            print(f"RMSE: {rmse_error:.3f}")
            
            return np.array(trues), np.array(preds)


unpc = User_Neighborhood_Pearson_Centered(rmh, task)
unpc.build_lookups()
unpc.calculate_items_rated_by_user()
unpc.compute_mean_ratings()
unpc.compute_similarity_matrix()
trues, preds = unpc.evaluate(**user_neighborhood_config)
get_ipython().run_line_magic('run', 'MetricHelper.ipynb')
print(mh.compute_average_metrics())

