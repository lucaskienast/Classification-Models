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

## References

Brownlee, J. (2020) 4 Types of Classification Tasks in Machine Learning. Available at: https://machinelearningmastery.com/types-of-classification-in-machine-learning/ (Accessed: 27 August 2021)

Chatterjee, D. (2019) All the Annoying Assumptions. Available at: https://towardsdatascience.com/all-the-annoying-assumptions-31b55df246c3 (Accessed: 27 August 2021)

Gaurav, H. (2021) 5 Classification Algorithms you should know - introductory guide. Available at: https://www.analyticsvidhya.com/blog/2021/05/5-classification-algorithms-you-should-know-introductory-guide/ (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: K-Nearest Neighbors Classification. Available at: https://towardsdatascience.com/machine-learning-basics-k-nearest-neighbors-classification-6c1e0b209542 (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: Logistic Regression. Available at: https://towardsdatascience.com/machine-learning-basics-logistic-regression-890ef5e3a272 (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: Naive Bayes Classification. Available at: https://towardsdatascience.com/machine-learning-basics-naive-bayes-classification-964af6f2a965 (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: Random Forest Classification. Available at: https://towardsdatascience.com/machine-learning-basics-random-forest-classification-499279bac51e (Accessed: 27 August 2021)

Gurucharan, M. (2020) Machine Learning Basics: Support Vector Machine (SVM) Classification. Available at: https://towardsdatascience.com/machine-learning-basics-support-vector-machine-svm-classification-205ecd28a09d (Accessed: 27 August 2021)

Verma, S. (2021) Softmax Regression in Python: Multi-Class Classification. Available at: https://towardsdatascience.com/softmax-regression-in-python-multi-class-classification-3cb560d90cb2 (Accessed: 27 August 2021)
