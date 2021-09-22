#!/usr/bin/env python
# coding: utf-8

# imports
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
import spacy


subprocess.run("jupyter nbconvert --output-dir='../python-code' --to python BERT.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True", shell=True)


# Read in argument-data
args = pd.read_csv("../../data/arguments.csv", sep=",", usecols=["statement_id", "text_en"])
# Only filter for relevant arguments
relevant_args = set([i for i in range(324, 400)])
args = args[args.statement_id.isin(relevant_args)]
# Convert to list of tuples for processing it further
args = list(zip(args["text_en"], args["statement_id"]))


class BERT():
    
    def __init__(self, args:[str], model_name:str="bert-base-nli-mean-tokens"):
        """
        Params:
            args ([type]): [description]
            model_name (str, optional): [description]. Defaults to "bert-base-nli-mean-tokens".
        """
        self.args = [t[0] for t in args]
        self.model_name = model_name
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
        self.model = SentenceTransformer(model_name, device=self.device)
            
    def calculate_similarity_matrix(self, as_torch_tensor:bool=True) -> None:
        """
        Calculate the similarity matrix based on the cosine-similarity metric. The cosine similarity is applied to normalized
        argument embeddings such that the cosine similarity is equivalent to the dot-product in this case.

        Params:
            as_torch_tensor (bool, optional): Return the similarity matrix as pytorch tensor. If False, return as numpy-array. Defaults to True.
        """        
        # Calculate argument embeddings
        arg_embeddings = self.model.encode(self.args)
        # Calculate the cosine-similarity of the normalize vectors (=dot-product)
        similarity_matrix = cosine_similarity(arg_embeddings)
        # Set all values on diagonal to zero as the similarity of an argument with itself should not be taken into account
        np.fill_diagonal(similarity_matrix, 0)
        if as_torch_tensor:
            self.similarity_matrix =  torch.from_numpy(similarity_matrix).float().to(self.device)
        else:
            self.similarity_matrix = similarity_matrix

