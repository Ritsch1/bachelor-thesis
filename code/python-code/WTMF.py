#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


get_ipython().system('jupyter nbconvert --output-dir="../python-code" --to python WTMF.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True')


# Read in argument-data
args = pd.read_csv("../../data/arguments.csv", sep=",", usecols=["statement_id", "text_en"])
# Convert to list for processing it further
args = list(args["text_en"])


# Exclude stop words while vectorizing the sentences
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(args)

