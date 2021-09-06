#!/usr/bin/env python
# coding: utf-8

# imports
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


subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python Main.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


k=1
training_iterations=2
weight=0.05
gamma=0.05
random_seed=1
print_frequency=1
get_ipython().run_line_magic('run', 'WTMF.ipynb')


# Parameters for executing the Rating-Matrix-Handler notebook
train_path = f"C:\\Users\\Rico\\Desktop\\Diverses\\bachelorarbeit\\bachelor-thesis\\data\\{prediction_goal}\\train.csv"
test_path  = f"C:\\Users\\Rico\\Desktop\\Diverses\\bachelorarbeit\\bachelor-thesis\\data\\{prediction_goal}\\test.csv"
get_ipython().run_line_magic('run', 'Rating_Matrix_Handler.ipynb')


# Parameters for executing the Rating-Matrix-Handler notebook
wtmf=wtmf
rmh=rmh
d=1
training_iterations=1
random_seed=1 
print_frequency=1
r=0.05 
l=0.01 
alpha=0.2
n=1
# Execute a parametrized version of the Rating-Matrix-Handler notebook
get_ipython().run_line_magic('run', 'TLMF.ipynb')

