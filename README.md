Amazon-Food-Reviews-Analysis-and-Modelling Using Various Machine Learning Models
Performed Exploratory Data Analysis, Data Cleaning, Data Visualization and Text Featurization(BOW, tfidf, Word2Vec). Build several ML models like KNN, Naive Bayes, Logistic Regression, SVM, Random Forest, GBDT, LSTM(RNNs) etc.
Objective:
Given a text review, determine the sentiment of the review whether its positive or negative.

Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews

About Dataset
The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.

Number of reviews: 568,454
Number of users: 256,059
Number of products: 74,258
Timespan: Oct 1999 - Oct 2012
Number of Attributes/Columns in data: 10

Attribute Information:

Id
ProductId - unique identifier for the product
UserId - unqiue identifier for the user
ProfileName
HelpfulnessNumerator - number of users who found the review helpful
HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
Score - rating between 1 and 5
Time - timestamp for the review
Summary - brief summary of the review
Text - text of the review
1 Amazon Food Reviews EDA, NLP, Text Preprocessing and Visualization using TSNE
Defined Problem Statement
Performed Exploratory Data Analysis(EDA) on Amazon Fine Food Reviews Dataset plotted Word Clouds, Distplots, Histograms, etc.
Performed Data Cleaning & Data Preprocessing by removing unneccesary and duplicates rows and for text reviews removed html tags, punctuations, Stopwords and Stemmed the words using Porter Stemmer
Documented the concepts clearly
Plotted TSNE plots for Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
2 KNN
Applied K-Nearest Neighbour on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
Used both brute & kd-tree implementation of KNN
Evaluated the test data on various performance metrics like accuracy also plotted Confusion matrix using seaborne
Conclusions:
KNN is a very slow Algorithm takes very long time to train.
Best Accuracy is achieved by Avg Word2Vec Featurization which is of 89.38%.
Both kd-tree and brute algorithms of KNN gives comparatively similar results.
Overall KNN was not that good for this dataset.
3 Naive Bayes
Applied Naive Bayes using Bernoulli NB and Multinomial NB on Different Featurization of Data viz. BOW(uni-gram), tfidf.
Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
Printed Top 25 Important Features for both Negative and Positive Reviews
Conclusions:
Naive Bayes is much faster algorithm than KNN
The performance of bernoulli naive bayes is way much more better than multinomial naive bayes.
Best F1 score is acheived by BOW featurization which is 0.9342
4 Logistic Regression
Applied Logistic Regression on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
Used both Grid Search & Randomized Search Cross Validation
Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
Showed How Sparsity increases as we increase lambda or decrease C when L1 Regularizer is used for each featurization
Did pertubation test to check whether the features are multi-collinear or not
Conclusions:
Sparsity increases as we decrease C (increase lambda) when we use L1 Regularizer for regularization.
TF_IDF Featurization performs best with F1_score of 0.967 and Accuracy of 91.39.
Features are multi-collinear with different featurization.
Logistic Regression is faster algorithm.
5 SVM
Applied SVM with rbf(radial basis function) kernel on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
Used both Grid Search & Randomized Search Cross Validation
Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
Evaluated SGDClassifier on the best resulting featurization
Conclusions:
BOW Featurization with linear kernel with grid search gave the best results with F1-score of 0.9201.
Using SGDClasiifier takes very less time to train.
6 Decision Trees
Applied Decision Trees on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
Used both Grid Search with random 30 points for getting the best max_depth
Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
Plotted feature importance recieved from the decision tree classifier
Conclusions:
BOW Featurization(max_depth=8) gave the best results with accuracy of 85.8% and F1-score of 0.858.
Decision Trees on BOW and tfidf would have taken forever if had taken all the dimensions as it had huge dimension and hence tried with max 8 as max_depth
6 Ensembles(RF&GBDT)
Applied Random Forest on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
Used both Grid Search with random 30 points for getting the best max_depth, learning rate and n_estimators.
Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
Plotted world cloud of feature importance recieved from the RF and GBDT classifier
Conclusions:
TFIDF Featurization in Random Forest (BASE-LEARNERS=10) with random search gave the best results with F1-score of 0.857.
TFIDF Featurization in GBDT (BASE-LEARNERS=275, DEPTH=10) gave the best results with F1-score of 0.8708.
