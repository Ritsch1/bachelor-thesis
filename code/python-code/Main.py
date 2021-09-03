#!/usr/bin/env python
# coding: utf-8

# imports
import papermill as pm


get_ipython().system('jupyter nbconvert --output-dir="../python-code" --to python Main.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True')


# Parameters for executing the WTMF algorithm
params = {"k":50, "gamma":0.05, "weight":0.05, "training_iterations":50, "random_seed":1, "print_frequency":1}
# Execute a parametrized version of the WTMF notebook
pm.execute_notebook("WTMF.ipynb", "WTMF.ipynb", params)


# Parameters for executing the Rating-Matrix-Handler notebook
params = {"train_path": "C:\\Users\\Rico\\Desktop\\Diverses\\bachelorarbeit\\bachelor-thesis\\data\\T1_T2\\train.csv",
          "test_path" : "C:\\Users\\Rico\\Desktop\\Diverses\\bachelorarbeit\\bachelor-thesis\\data\\T1_T2\\test.csv"}
# Execute a parametrized version of the Rating-Matrix-Handler notebook
pm.execute_notebook("Rating_Matrix_Handler.ipynb", "Rating_Matrix_Handler.ipynb", params)

