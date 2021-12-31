# Bachelor Thesis
## Classification Models for Argument Recommender Systems

In the era of Big Data it becomes increasingly difficult to find information that is relevant to oneself. 
Recommender systems are used to cope with this flood of information: 
they do this by providing personalized subsets of information to the user. 
Today, they are used in a wide variety of domains with great success. For instance, 80% of the
streaming time on Netflix is influenced by the recommender system Netflix operates. 


Online-argumentation is a domain where the user's opinion is influenced strongly by the
specific arguments he faces. Problems such as filter bubbles or arguments that
are not relevant to a specific user can be mitigated by using a recommender system
which provides suitable arguments to the user, depending on the objective of the recommender system.\\
In the original paper, data from over 600 individuals on 900 arguments at different points in time was collected.
The goal was to provide a dataset that can be used to evaluate algorithms that predict how persuasive 
an argument is to a specific user.
Three tasks that use the known user-argument interaction data were presented to predict ratings of arguments 
by users that were collected at later points in time:

* Predicting the users' conviction by an argument (binary classification)
* Predicting the strength of the conviction for an argument (multiclass classification in the range [0,6])
* Predicting three convincing statements for a specific user

In this thesis I will focus on the first two tasks.
In order to obtain reference performances for such an algorithm, two baseline algorithms
were presented: a simple majority voter and a more sophisticated nearest-neighbor-algorithm.
The goal of this thesis is to implement an algorithm that exceeds the performance of these two baseline algorithms on the provided dataset.