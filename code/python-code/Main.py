#!/usr/bin/env python
# coding: utf-8

# imports
import ipywidgets as widgets
from ipywidgets.widgets import interact
import subprocess
import json


@interact(dataset=[("T1_T2", "T1_T2"), ("T2_T3", "T2_T3")])
def set_prediction_dataset(dataset:str="T1_T2") -> str:
    """
    Setting the dataset to be trained and evaluated on with a dropdown.

    Parmas:
        dataset (str, optional): Dataset to be trained and evaluated on. Defaults to "T1_T2".

    Returns:
        str: Dataset to be trained and evaluated on.
    """
    return dataset


@interact(task=[("Conviction", "Conviction"), ("Weight", "Weight")])
def set_task(task:str="Conviction") -> str:
    """
    Setting the task to be trained and evaluated on with a dropdown.

    Parmas:
        task (str, optional): task to be trained and evaluated on. Defaults to "Conviction".

    Returns:
        str: task to be trained and evaluated on.
    """
    return task


prediction_goal = "T2_T3"
task = "Weight"
# Metrics that should be evaluated on
metrics = ["precision", "recall", "f1", "gmean"]


subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python Main.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


with open("config.json", "r") as c:
    config = json.loads(c.read())


# Parameters for executing the Rating-Matrix-Handler notebook
train_path = f"../../data/{prediction_goal}/train.csv"
test_path  = f"../../data/{prediction_goal}/test.csv"
validation_path = f"../../data/{prediction_goal}/validation.csv"
get_ipython().run_line_magic('run', 'Rating_Matrix_Handler.ipynb')


get_ipython().run_line_magic('run', 'Graphics.ipynb')


wtmf_config = config["hyperparameters"]["WTMF"]
get_ipython().run_line_magic('run', 'WTMF.ipynb')


similarity_matrix = wtmf.similarity_matrix
tlmf_config = config["hyperparameters"]["TLMF"][prediction_goal][task]
get_ipython().run_line_magic('run', 'TLMF.ipynb')


get_ipython().run_line_magic('run', 'BERT.ipynb')
similarity_matrix = bert.similarity_matrix
get_ipython().run_line_magic('run', 'TLMF.ipynb')


i_autorec_config = config["hyperparameters"]["AutoRec"][prediction_goal][task]
i_autorec_config["rmh"] = rmh
get_ipython().run_line_magic('run', 'AutoRec.ipynb')


get_ipython().run_line_magic('run', 'Naive_Bayes.ipynb')


user_neighborhood_config = config["hyperparameters"]["User_Neighborhood"][prediction_goal][task]
get_ipython().run_line_magic('run', 'User_Based_Neighborhood.ipynb')


get_ipython().run_line_magic('run', 'Majority.ipynb')


data_parameters = config["hyperparameters"]["NN_Baseline"][prediction_goal][task]["data_parameters"]
get_ipython().run_line_magic('run', 'NN_baseline.ipynb')

