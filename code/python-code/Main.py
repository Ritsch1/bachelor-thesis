#!/usr/bin/env python
# coding: utf-8

# imports
import papermill as pm
import ipywidgets as widgets
from ipywidgets.widgets import interact
import subprocess


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


prediction_goal = set_prediction_dataset()


subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python Main.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True")


# Parameters for executing the WTMF algorithm
params = {"k":50, "gamma":0.05, "weight":0.05, "training_iterations":50, "random_seed":1, "print_frequency":1}
# Execute a parametrized version of the WTMF notebook
pm.execute_notebook("WTMF.ipynb", "WTMF.ipynb", params)


# Parameters for executing the Rating-Matrix-Handler notebook
params = {"train_path": "C:\\Users\\Rico\\Desktop\\Diverses\\bachelorarbeit\\bachelor-thesis\\data\\T1_T2\\train.csv",
          "test_path" : "C:\\Users\\Rico\\Desktop\\Diverses\\bachelorarbeit\\bachelor-thesis\\data\\T1_T2\\test.csv"}
# Execute a parametrized version of the Rating-Matrix-Handler notebook
pm.execute_notebook("Rating_Matrix_Handler.ipynb", "Rating_Matrix_Handler.ipynb", params, log_output=False, report_mode=False)


# Parameters for executing the Rating-Matrix-Handler notebook
params = {"wtmf":wtmf, "rmh":rmh, "d":10, "training_iterations":50, "random_seed":1, "print_frequency":1, "r":0.05, "l":0.01, "alpha":0.2, "n":10}
# Execute a parametrized version of the Rating-Matrix-Handler notebook
pm.execute_notebook("TLMF.ipynb", "TLMF.ipynb", params)

