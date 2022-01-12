# Bachelor Thesis
## Classification Models for Argument Recommender Systems

Argumentation is essential to forming opinions and making decisions. Considering as many arguments as possible that are relevant to oneself is crucial to be able to make a rational decision. An argument recommender system provides arguments about a topic that are relevant for a specific user. In order to do that, it needs to be able to predict the rating of that user for unseen arguments. Thus, argument recommender systems could be a useful tool for decision making. By presenting arguments that run counter to one's own convictions, the widespread problem of confirmation bias could be addressed. 

In this bachelor thesis several recommender algorithms are implemented and evaluated on an argument rating data set provided by Heinrich Heine Universities' deliberate application. These algorithms are:
* Two - level matrix factorization that utilizes linguistic information of arguments
* Matrix factorization using linguistic similarity scores produced by the \acrfull{bert} model
* Autoencoder neural network
* Probabilistic Naive Bayes approach
* User - neighborhood model

The models are evaluated on two tasks:
* Predicting a users' conviction by an argument (binary classification) - Prediction of Conviction (PoC)
* Predicting the strength of the conviction for an argument (multiclass classification in the range [0,6]) - Prediction of Weight (PoW)

The goal is to improve upon the performance of two baseline recommender algorithms that are provided along with the dataset.
The provided baseline metrics accuracy and rmse are discussed in terms of suitability for measuring performance of classification models on imbalanced data sets.
The implemented algorithms are trained on the provided training data set and optimized upon the validation data set. Both, the proposed algorithms as well as the baseline algorithms are evaluated on the baseline metrics as well as on four proposed metrics that take into account the class distribution within the rating data set. These four metrics are the F - Score, G - Mean, Precision \& Recall. 

While using the baseline metrics the baseline algorithms delivered the best performance for some data set / task combinations. This was no longer the case when any of the four proposed metrics were used, indicating that the baseline metrics are not reasonable to use for the provided data set as the ratings within single items are imbalanced. Although it can be observed that the proposed models outperform the baseline models in all evaluations, there is no model that outperforms every other model on the given data set / task combinations. Worth mentioning is that the probabilistic Naive Bayes model achieves the best performance in 62.5% of the evaluated data set / task combinations. 

As the best performing model changes regarding the tasks, there is no silver bullet for choosing a model solely regarding its' performance. 
Since the Naive Bayes model is the least computational complex model of all the models that are presented, it shows that complexity does not necessarily correlate positively with performance improvements.
The results also show that the thorough assessment of a metric regarding its' capability to evaluate the performance of a model is crucial to producing meaningful results.