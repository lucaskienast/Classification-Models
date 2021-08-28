# Classification-Models
Overview of theory and common techniques for classification problems.

- Binary Classification
- Multi-Class Classification
- Multi-Label Classification
- Imbalanced Classification

## Binary Classification
Binary classification refers to those classification tasks that have two class labels, such as email spam detection (spam or not). Typically, binary classification tasks involve one class that is the normal state and another class that is the abnormal state. For example “not spam” is the normal state and “spam” is the abnormal state. The class for the normal state is assigned the class label 0 and the class with the abnormal state is assigned the class label 1. It is common to model a binary classification task with a model that predicts a Bernoulli probability distribution for each example. The Bernoulli distribution is a discrete probability distribution that covers a case where an event will have a binary outcome as either a 0 or 1. For classification, this means that the model predicts a probability of an example belonging to class 1, or the abnormal state. Some algorithms are specifically designed for binary classification and do not natively support more than two classes; examples include Logistic Regression and Support Vector Machines.

Popular algorithms that can be used for binary classification include:

- Logistic Regression
- k-Nearest Neighbors
- Decision Trees
- Support Vector Machine
- Naive Bayes

## Multi-Class Classification
Multi-class classification refers to those classification tasks that have more than two class labels, such as face classification. Unlike binary classification, multi-class classification does not have the notion of normal and abnormal outcomes. Instead, examples are classified as belonging to one among a range of known classes. The number of class labels may be very large on some problems. For example, a model may predict a photo as belonging to one among thousands or tens of thousands of faces in a face recognition system. It is common to model a multi-class classification task with a model that predicts a Multinoulli probability distribution for each example. The Multinoulli distribution is a discrete probability distribution that covers a case where an event will have a categorical outcome, e.g. K in {1, 2, 3, …, K}. For classification, this means that the model predicts the probability of an example belonging to each class label.

Popular algorithms that can be used for multi-class classification include:

- k-Nearest Neighbors
- Decision Trees
- Naive Bayes
- Random Forest
- Gradient Boosting

Algorithms that are designed for binary classification can be adapted for use for multi-class problems. This involves using a strategy of fitting multiple binary classification models for each class vs. all other classes (called one-vs-rest) or one model for each pair of classes (called one-vs-one).

- One-vs-Rest: Fit one binary classification model for each class vs. all other classes
- One-vs-One: Fit one binary classification model for each pair of classes

Binary classification algorithms that can use these strategies for multi-class classification include:

- Logistic Regression
- Support Vector Machine

## Multi-Label Classification
Multi-label classification refers to those classification tasks that have two or more class labels, where one or more class labels may be predicted for each example. Consider the example of photo classification, where a given photo may have multiple objects in the scene and a model may predict the presence of multiple known objects in the photo, such as “bicycle,” “apple,” “person,” etc. This is unlike binary classification and multi-class classification, where a single class label is predicted for each example. It is common to model multi-label classification tasks with a model that predicts multiple outputs, with each output taking predicted as a Bernoulli probability distribution. This is essentially a model that makes multiple binary classification predictions for each example. Classification algorithms used for binary or multi-class classification cannot be used directly for multi-label classification. 

Specialized versions of standard classification algorithms can be used, so-called multi-label versions of the algorithms, including:

- Multi-label Decision Trees
- Multi-label Random Forests
- Multi-label Gradient Boosting

Another approach is to use a separate classification algorithm to predict the labels for each class.

## Imbalanced Classification
Imbalanced classification refers to classification tasks where the number of examples in each class is unequally distributed. Typically, imbalanced classification tasks are binary classification tasks where the majority of examples in the training dataset belong to the normal class and a minority of examples belong to the abnormal class. These problems are modeled as binary classification tasks, although may require specialized techniques. Example applications include:

- Fraud detection
- Outlier detection
- Medical diagnostic tests

Specialized techniques may be used to change the composition of samples in the training dataset by undersampling the majority class or oversampling the minority class. Examples include:

- Random Undersampling
- SMOTE Oversampling

Specialized modeling algorithms may be used that pay more attention to the minority class when fitting the model on the training dataset, such as cost-sensitive machine learning algorithms. Examples include:

- Cost-sensitive Logistic Regression
- Cost-sensitive Decision Trees
- Cost-sensitive Support Vector Machines

## Logistic Regression
Logistic Regression is a classification model that is used when the dependent variable (output) is in the binary format such as 0 (False) or 1 (True). Examples include such as predicting if there is a tumor (1) or not (0) and if an email is a spam (1) or not (0). The logistic function, also called as sigmoid function was initially used by statisticians to describe properties of population growth in ecology. The sigmoid function is a mathematical function used to map the predicted values to probabilities. Logistic Regression has an S-shaped curve and can take values between 0 and 1 but never exactly at those limits. It has the formula of 1 / (1 + e^-value). The Assumptions of the model are the same as linear regression except the assumption of linearity.

## K-Nearest Neighbors (KNN) Classification
The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other. K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). Suppose P1 is the point, for which a label needs to be predicted. First, you find the k closest points to P1 and then classify points by majority vote of its k neighbors. Each object votes for their class and the class with the most votes is taken as the prediction. For finding closest similar points, you find the distance between points using distance measures such as Euclidean distance, Hamming distance, Manhattan distance and Minkowski distance.

KNN is a non-parametric and lazy learning algorithm. Non-parametric means there is no assumption for underlying data distribution. In other words, the model structure determined from the dataset. This will be very helpful in practice where most of the real world datasets do not follow mathematical theoretical assumptions. Lazy algorithm means it does not need any training data points for model generation. All training data used in the testing phase. This makes training faster and testing phase slower and costlier. Costly testing phase means time and memory. In the worst case, KNN needs more time to scan all data points and scanning all data points will require more memory for storing training data. 

## Support Vector Machine (SVM) Classification
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points. To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence. 

Hyperplanes are decision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features. If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane. Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. These are the points that help us build our SVM.

## Naive Bayes Classification
A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem. Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence it is called naive.  For two independent events, P(A,B) = P(A)P(B). This assumption of Bayes Theorem is probably never encountered in practice. Bayes’ Theorem is stated as: P(a|b) = (P(b|a) * P(a)) / P(b). Where P(a|b) is the probability of a given b.

## Decision Tree Classification
A decision tree is a flowchart-like tree structure where an internal node represents feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome. The topmost node in a decision tree is known as the root node. It learns to partition on the basis of the attribute value. It partitions the tree in recursively manner call recursive partitioning. This flowchart-like structure helps you in decision making. It's visualization like a flowchart diagram which easily mimics the human level thinking. That is why decision trees are easy to understand and interpret. Decision Tree is a white box type of ML algorithm. It shares internal decision-making logic, which is not available in the black box type of algorithms such as Neural Network. Its training time is faster compared to the neural network algorithm. The time complexity of decision trees is a function of the number of records and number of attributes in the given data. The decision tree is a distribution-free or non-parametric method, which does not depend upon probability distribution assumptions. Decision trees can handle high dimensional data with good accuracy.

## Random Forest Classification
The random forest is an ensemble classification algorithm consisting of many decisions trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction. The low correlation between models is the key. Just like how investments with low correlations (like stocks and bonds) come together to form a portfolio that is greater than the sum of its parts, uncorrelated models can produce ensemble predictions that are more accurate than any of the individual predictions. The reason for this wonderful effect is that the trees protect each other from their individual errors (as long as they don’t constantly all err in the same direction). While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction. 

## References

Aznar, P. (2020) Decision Trees: Gini vs Entropy. Available at: https://quantdare.com/decision-trees-gini-vs-entropy/ (Accessed: 28 August 2021)

Bose, A. (2019) Handwritten Digit Recognition Using PyTorch - Intro to Neural Networks. Available at: https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627 (Accessed: 28 August 2021)

Brownlee, J. (2020) 4 Types of Classification Tasks in Machine Learning. Available at: https://machinelearningmastery.com/types-of-classification-in-machine-learning/ (Accessed: 27 August 2021)

Chatterjee, D. (2019) All the Annoying Assumptions. Available at: https://towardsdatascience.com/all-the-annoying-assumptions-31b55df246c3 (Accessed: 27 August 2021)

Gandhi, R. (2018) Naive Bayes Classifier. Available at: https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c (Accessed: 28 August 2021)

Gandhi, R. (2018) Support Vector Machine - Introduction to Machine Learning Algorithms. Available at: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47 (Accessed: 27 August 2021)

Gaurav, H. (2021) 5 Classification Algorithms you should know - introductory guide. Available at: https://www.analyticsvidhya.com/blog/2021/05/5-classification-algorithms-you-should-know-introductory-guide/ (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: K-Nearest Neighbors Classification. Available at: https://towardsdatascience.com/machine-learning-basics-k-nearest-neighbors-classification-6c1e0b209542 (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: Logistic Regression. Available at: https://towardsdatascience.com/machine-learning-basics-logistic-regression-890ef5e3a272 (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: Naive Bayes Classification. Available at: https://towardsdatascience.com/machine-learning-basics-naive-bayes-classification-964af6f2a965 (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: Random Forest Classification. Available at: https://towardsdatascience.com/machine-learning-basics-random-forest-classification-499279bac51e (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: Support Vector Machine (SVM) Classification. Available at: https://towardsdatascience.com/machine-learning-basics-support-vector-machine-svm-classification-205ecd28a09d (Accessed: 27 August 2021)

Harrison, O. (2018) Machine Learning Basics with K-Nearest Neighbors Algorithm. Available at: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761 (Accessed: 27 August 2021)

Navlani, A. (2018) Decision Tree Classification in Python. Available at: https://www.datacamp.com/community/tutorials/decision-tree-classification-python (Accessed: 28 August 2021)

Navlani, A. (2018) KNN Classification using Scikit-learn. Available at: https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn (Accessed: 27 August 2021)

Sayad, S. (2021) K Nearest Neighbors - Classification. Available at: https://www.saedsayad.com/k_nearest_neighbors.htm (Accessed: 27 August 2021)

Sharma, N. (2019) Importance of Distance Metrics in Machine Learning Modelling. Available at: https://towardsdatascience.com/importance-of-distance-metrics-in-machine-learning-modelling-e51395ffe60d (Accessed: 27 August 2021)

Udacity (2021) Artificial Intelligence for Trading. Available at: https://www.udacity.com/course/ai-for-trading--nd880 (Accessed: 27 August 2021)

Udacity (2021) deep-learning-v2-pytorch. Available at: https://github.com/udacity/deep-learning-v2-pytorch (Accessed: 28 August 2021)

Verma, S. (2021) Logistic Regression From Scratch in Python. Available at: https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2 (Accessed: 27 August 2021)

Verma, S. (2021) Softmax Regression in Python: Multi-Class Classification. Available at: https://towardsdatascience.com/softmax-regression-in-python-multi-class-classification-3cb560d90cb2 (Accessed: 27 August 2021)

Yiu, T. (2019) Understanding Random Forest. Available at: https://towardsdatascience.com/understanding-random-forest-58381e0602d2 (Accessed: 28 August 2021)

365 Careers (2021) The Data Science Course 2021: Complete Data Science Bootcamp. Available at: https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/ (Accessed: 24 August 2021)
