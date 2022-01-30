#!/usr/bin/env python
# coding: utf-8

# Imports
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess


# Export notebook as python script to the ../python-code folder
subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python MetricHelper.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


class MetricHelper():
    """
    Class that takes care of the calculation of different metrics.
    """
    def __init__(self, trues:np.array, preds:np.array, task:str="Conviction", *metrics:str):
        """
        Params:
            trues (np.array): True values
            preds (np.array): predictions.
            task (str, optional): The task for which the metrics per class need to be computed. Defaults to "Conviction".
            *metrics: Metrics to compute. Can be one of the following: 'precision', 'recall', 'f1', 'gmean'.
        """
        self.trues_ = trues
        self.preds_ = preds
        self.task_ = task
        self.metrics_ = metrics
        self.classes_ = (0, 1) if task == "Conviction" else (0,1,2,3,4,5,6)
        
    def compute_average_metrics(self) -> dict:
        """
        Calculate the metrics' average over all classes

        Returns:
            dict: Dictionary containing the metric name as key and its averaged value as value.
        """
        metrics_per_class = self.compute_metrics_per_class()
        # metrics averages over all classes with equal weight per class
        metrics_averaged = {}
        sum = 0.0
        for m in self.metrics_:
            for c in self.classes_:
                sum += metrics_per_class[c][m]
            sum /= len(self.classes_)
            metrics_averaged[m] = sum
            sum = 0.0
        
        return metrics_averaged
    
    def compute_metrics_per_class(self) -> dict:
        """
        Compute the metrics Recall, Precision, F1-Score, G-Mean per class. 

        Returns:
            dict: Dictionary containing a class label as key and a dictionary with the metric name as key and its value as value.
        """
        # Compute tuples of (metric_name, metric_value)
        metric_values = [(m, getattr(self, f"compute_{m}")(self.trues_, self.preds_)) for m in self.metrics_]
        # Compute dictionary with metric values per class
        metrics_per_class = {c : {m[0] : m[1][c] for m in metric_values} for c in self.classes_}
        
        return metrics_per_class
     
    def compute_accuracy(self, trues:np.array, preds:np.array) -> dict:
        """
        Params:
            trues (np.array): true values.
            preds (np.array): predicitons corresponding to true values. 

        Returns:
            dict: A dictionary containing the classes as key and the accuracy value for this class as value.
        """
        
        combined = np.array(list(zip(trues, preds)))
        accuracy = {c:None for c in self.classes_}
        for c in accuracy:
            filtered = np.logical_and(combined[:,0] == c, combined[:,1] == c)
            acc = np.sum(filtered)
            acc /= len(filtered)
            accuracy[c] = acc
            
        return accuracy
    
    def compute_precision(self, trues, preds) -> dict:
        """
        Calculate the precision metric for the given true and prediction values.

        Params:
            trues (np.array): [description]
            preds (np.array): [description]
            
        Returns:
            dict: A dictionary containing the classes as key and the precision value for this class as value. 
        """
        combined = np.array(list(zip(trues, preds)))
        precision = {c:None for c in self.classes_}
        for c in precision:
            tp = np.sum(np.logical_and(combined[:,0] == c, combined[:,1] == c))
            fp = np.sum(np.logical_and(combined[:,0] != c, combined[:,1] == c))
            result_label =  tp / (tp + fp)
            if np.isnan(result_label):
                precision[c] = 0
                continue
            else:
                precision[c] = result_label
        
        return precision
    
    
    def compute_recall(self, trues, preds) -> dict:
        """
        Calculate the recall metric for the given true and prediction values.

        Params:
            trues (np.array): [description]
            preds (np.array): [description]
            
        Returns:
            dict: A dictionary containing the classes as key and the recall value for this class as value. 
        """
        combined = np.array(list(zip(trues, preds)))
        recall = {c:None for c in self.classes_}
        for c in recall:
            tp = np.sum(np.logical_and(combined[:,0] == c, combined[:,1] == c))
            fn = np.sum(np.logical_and(combined[:,0] == c, combined[:,1] != c)) 
            result_label =  tp / (tp + fn)
            if np.isnan(result_label):
                recall[c] = 0
                continue
            else:
                recall[c] = result_label 
        
        return recall

    def compute_f1(self, trues:np.array, preds:np.array, beta:float=1.0) -> dict:
        """
        Calculate the precision metric for the given true and prediction values.

        Params:
            trues (np.array): [description]
            preds (np.array): [description]
            
        Returns:
            dict: A dictionary containing the classes as key and the f1 value for this class as value. 
        """
        assert beta >= 0, "beta needs to be non-negative."
        recall = self.compute_recall(trues, preds)
        precision = self.compute_precision(trues, preds)
        f1 = {c:None for c in self.classes_}
        for c in f1:
            score = (2 * recall[c] * precision[c]) / (precision[c] + recall[c])
            if np.isnan(score):
                f1[c] = 0
                continue
            else:
                f1[c] = score
        return f1
    
    def compute_gmean(self,trues, preds) -> dict:
        """
        Calculate the precision metric for the given true and prediction values.

        Params:
            trues (np.array): [description]
            preds (np.array): [description]
            
        Returns:
            dict: A dictionary containing the classes as key and the gmean value for this class as value. 
        """
        combined = np.array(list(zip(trues, preds)))
        gmean = {c:None for c in self.classes_}
        for c in gmean:
            tp = np.sum(np.logical_and(combined[:,0] == c, combined[:,1] == c))
            fn = np.sum(np.logical_and(combined[:,0] == c, combined[:,1] != c)) 
            fp = np.sum(np.logical_and(combined[:,0] != c, combined[:,1] == c)) 
            tn = np.sum(np.logical_and(combined[:,0] != c, combined[:,1] != c))
            result_label = np.sqrt((tp / (tp + fn)) * (tn/(tn+fp)))
            if np.isnan(result_label):
                gmean[c] = 0
                continue
            gmean[c] = np.sqrt((tp / (tp + fn)) * (tn/(tn+fp)))
            
        return gmean


mh = MetricHelper(trues, preds, task, *metrics)

