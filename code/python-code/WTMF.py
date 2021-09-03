#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.core.debugger import set_trace
import torch
import spacy
import matplotlib.pyplot as plt


get_ipython().system('jupyter nbconvert --output-dir="../python-code" --to python WTMF.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True')


# Read in argument-data
args = pd.read_csv("../../data/arguments.csv", sep=",", usecols=["statement_id", "text_en"])
# Only filter for relevant arguments
relevant_args = set([i for i in range(325, 400)])
args = args[args.statement_id.isin(relevant_args)]
# Convert to list of tuples for processing it further
args = list(zip(args["text_en"], args["statement_id"]))


class WTMF():
    """
    A class that represents the Weighted Textual Matrix Factorization.
    """
    def __init__(self, args:list):
        """
        Params:
            args (list): A list of (argument, id) - tuples. 
        """
        self.args = [t[0] for t in args]
        self.args_ids = [t[1] for t in args]
        # Initialize GPU for computation if available            
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(machine)
    
    def create_tfidf_matrix(self, exclude_stopwords:bool=True) -> None:
        """
        Create a tfidf - matrix out of the arguments where the rows are words and the columns are sentences.
        
        Params:
            exclude_stopwords (bool): A boolean flag that indicates whether stopwords are kept in the vocabulary or not. Default value is True.
        """
        # Convert all words to lowercase
        self.args = list(map(lambda s : s.lower(), self.args))        
        # Lemmatize the sentences
        nlp = spacy.load("en_core_web_sm")
        self.args = list(map(lambda s : " ".join(token.lemma_ for token in nlp(s)), self.args))
        # Filter out the "-PRON-" - insertion from spacy 
        self.args = list(map(lambda s: s.replace("-PRON-",""), self.args))
        # Exclude stop words while vectorizing the sentences
        if exclude_stopwords:
            vectorizer = TfidfVectorizer(stop_words="english")
        else:
            vectorizer = TfidfVectorizer()
        self.X = vectorizer.fit_transform(self.args)
        # Transform the sparse matrix into a dense matrix and transpose the matrix to represent the words as rows and sentences as columns
        self.X = torch.from_numpy(self.X.toarray().transpose()).float().to(self.device)
        
    def train(self, k:int=50, gamma:float=0.05, weight:float=0.05, training_iterations:int=50, random_seed:int=1, print_frequency:int=1) -> [float]:
        """
        Use stochastic gradient descent to find the two latent factor matrices A (words), B (sentences) 
        that minimize the error of the objective function. 

        Params:
            vector_dimension(int, optional): Dimension of the latent vector space the users and items are mapped to. Defaults to 20.
            gamma (float, optional): Regularization factor to control the overfitting. Defaults to 0.05.
            weight (float, optional): Weight to control the influence of non-present words in a sentence. Defaults to 0.05.
            training_iterations (int, optional): Number of training iterations to take. Defaults to 50.
            random_seed (int, optional): Random seed that is used to intialize the latent factor matrices. Defaults to 1.
            print_frequency (int, optional): The epoch-frequency with which the error is printed to the console. Default to 1.
        
        Returns:
            [float]: A list containing the error values for every iteration.
        """
        
        # Set random seed for reproducability
        torch.manual_seed(random_seed)
        # Randomly initialize the latent factor matrices
        self.A = torch.rand([k, self.X.shape[0]]).to(self.device)
        self.B = torch.rand([k, self.X.shape[1]]).to(self.device)
        # Identity matrix
        I = torch.eye(k).to(self.device)
        
        # Create the weight matrix. Set value to one if value of X is != 0, else set it to the weights' value
        W = torch.ones_like(self.X).to(self.device)
        W[self.X == 0] = weight
        
        # Matrix for updating the latent matrices in optimization
        I_scaled = (gamma * I).to(self.device)
        gamma_half = torch.tensor(gamma / 2).to(self.device)
        
        # Error - variable keep track of it for later visualization 
        error = []
        error_cur = 0.0
        frobenius_norm = torch.linalg.matrix_norm
        inverse = torch.inverse
        for iteration in range(training_iterations):
            
            # Iterate over all words
            for i in range(self.X.shape[0]):
                # Iterate over all sentences
                for j in range(self.X.shape[1]):
                    # Compute error
                    A_T = torch.transpose(self.A, 0, 1).to(self.device)
                    error_cur += ((W[i][j] * ((torch.matmul(A_T[i], self.B[:,j]) - self.X[i][j])**2)) + (gamma_half * ((frobenius_norm(self.A)) + frobenius_norm(self.B))))
                    # Update latent factor matrices
                    W_diag_i = torch.diag(W[i]).to(self.device)
                    W_diag_j = torch.diag(W[:,j]).to(self.device)
                    temp_mat1 = torch.matmul(self.B, W_diag_i).to(self.device)
                    temp_mat2 = torch.matmul(self.A, W_diag_j).to(self.device)
                    # Update latent word vector
                    self.A[:,i] = torch.matmul(inverse(torch.mm(temp_mat1, torch.transpose(self.B, 0, 1)) + (I_scaled)) , torch.matmul(temp_mat1, torch.transpose(self.X[i], 0, 0))).to(self.device)            
                    # Update latent sentence vector
                    self.B[:,j] = torch.matmul(inverse(torch.mm(temp_mat2, A_T) + (I_scaled)) , torch.matmul(temp_mat2, torch.transpose(self.X[:,j], 0, 0))).to(self.device)
                    
            error.append(error_cur)
            error_cur = 0
            # Print out error w.r.t print-frequency
            if iteration % print_frequency == 0:
                print(f"Error:{error[iteration]:.2f}\tCurrent Iteration: {iteration+1}\\{training_iterations}")

        return error
    
    def compute_argument_similarity_matrix(self) -> None:
        """
        Compute the semantic argument similarity between the latent argument - vectors that were optimized within the argument(sentence) matrix B in the WTMF algorithm.
        """
        # Normalize all column - vectors in matrix B, so we can use the dot-product on normalized vectors which is equivalent to the cosine-similarity
        self.B /= torch.norm(self.B, dim=0).to(self.device)
        # Compute pairwise dot-product of all column vectors
        self.similarity_matrix = self.B.T.matmul(self.B).to(self.device)
        # Perform min-max scaling to map the dot-product results from the range [-1,1] to [0,1]
        min_value = torch.min(self.similarity_matrix)
        max_value = torch.max(self.similarity_matrix)
        self.similarity_matrix -= min_value
        self.similarity_matrix /= (max_value - min_value)
        # The diagonal will have the value zero, as the similarity of the argument with itself should not be taken into account as it will always be 1.
        self.similarity_matrix = self.similarity_matrix.fill_diagonal_(0).to(self.device)
    
    def plot_training_error(self, error:[float], **kwargs) -> None:
        """
        Plots the training error for every training iteration.
        
        Params:
            error (list): A list of error - values that correspond to each training iteration of the WTMF - algorithm.    
            **kwargs: Arbitrary many keyword arguments to customize the plot. E.g. color, linewidth or title.
        """        
        plt.plot([i for i in range(1, len(error)+1)], error)
        for k in kwargs.keys():
            # Invoke the function k of the plt - module to customize the plot
            getattr(plt, k) (kwargs[k])
        
        plt.show()


wtmf = WTMF(args)
wtmf.create_tfidf_matrix()
error = wtmf.train()
wtmf.compute_argument_similarity_matrix()
wtmf.plot_training_error(error, title="WTMF Objective function error", xlabel="Iterations", ylabel="Error")

