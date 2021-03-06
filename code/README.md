# Code Distribution

The code can be run in an interactive way using jupyter notebooks from the `notebooks` folder.
All the code of the jupyter notebooks is mirrored into the corresponding python files that are located in the `python-code` folder. 

# Setup

Please make sure to execute the following commands from the root folder and using a python version >= 3.8. 

`python -m venv bachelor_arbeit`

Now a folder `bachelor_arbeit` should have been created.

If you are using Windows, execute the following command to activate the environment:

`./bachelor_arbeit/Scripts/activate`

If you are using Linux / Mac, execute the following command to activate the environment:

`source ./bachelor_arbeit/bin/activate`

From within the activated environment:

`python -m pip install --upgrade pip`

`pip install -r requirements.txt`

# Execution of Main notebook

To execute the main notebook, start a jupyter notebook instance from the root folder within the venv environment by using the following command:

`jupyter notebook`

Once the jupyter instance has started, select the ``code/notebooks/Main.ipynb`` file and run all cells to execute training and evaluation for all models.
The specific task - / dataset - combination can be selected using dropdown menus right at the top of the notebook. If you change the dropdown values please mind that only the 
cells below the dropdown menus should be re - run in order to evaluate for the specified task - / dataset - combination.