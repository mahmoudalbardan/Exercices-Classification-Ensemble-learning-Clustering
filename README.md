# TD-2-3

**NOTE: FROM NOW ON, YOU HAVE TO SAVE YOUR CODE. YOU WILL USE IT LATER IN OTHER SESSIONS.**

The challenge
-------------

In this exercise we'll implement logistic regression and apply it to a classification task. 
In the first part of this exercise, we'll build a logistic regression model to predict whether an administration employee will get promoted or not. You have to determine each employee's chance of promotion based on their results on two domain-related exams, age and sex. You have historical data from previous applicants that you can use as a training set for logistic regression. To accomplish this, we're going to build a classification model that estimates the probability of admission based on the exam scores.


- Load the data from [here](https://drive.google.com/file/d/1vIqw8_A2Zx_Qw3QNAmIApFFh-7MPnkdB/view?usp=sharing), examin it using pandas methods, check your variable types and convert categorical variables to numerical 
- Scale your dataset using `MinMaxScaler` from `sklearn.preprocessing`
- Scatter plot of the two scores and use color coding to visualize if the example is positive (promotted) or negative (not promotted)
- Scatter plot of the two scores and use color coding to visualize if the example is male or female 
- Implement the sigmoid function and test it (by plotting it ) using generated data
- Write the cost function that takes (X,y,theta) as entries
- Extract your dataset (to be used in ML) and labels from the dataframe and name them (X, y).
- Initialize your parameter vector theta as a zero np.array of size 5 (=dimension of your data + 1 (bias))
- Compute the initial cost (with initial values of theta) 
- Copy the function below in your code. It computes the gradient during the optimization process
```
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    return grad
```
 - Call the function gradient with the initial values of `theta`
 - Use the function `fmin_tnc` from `scipy.optimize` to find the optimal parameters `theta_opt`
 - Write a prediction function that takes an input vector `**x**` and outputs its class `y`
 - Write a function that estimate the accuracy of your model by a Kfold cross validation
 - Compute the confusion matrix of your model, precision, recall and F-score
 - Use the built in function `sklearn.linear_model.LogisticRegression`, `sklearn.tree.DecisionTreeClassifier` and `sklearn.naive_bayes.GaussianNB`
 and estimate the accuracy of your models using LOOCV (leave one out cross validation)
 - Use a majority voting to aggregate the predictions **(LogReg+ DT + NB)** and estimate the accuracy of the new meta classifier 
 - Build a Bagging algorithm with `decision trees` as base estimator with 50 estimators, Does the accuracy changes?
 - Build a Random subspaces algorithm with `decision trees` as base estimator with 10 estimator, does the accuracy changes?
 - Build a stacking algorithm using **(LogReg+ DT + NB)** ( 50% of the data for D_train, 25% to construct the D_valid and 25 for D_test)
 - Use the built in functon `sklearn.ensemble.AdaBoostClassifier` and evalute the accracy of this model (use 50 estimators of Decision Trees) 







COURS
-----
- [Lecture notes for classification algorithms](https://drive.google.com/file/d/1oGU6CuWIe4UZIFJQti2TFYjH-7kBws8O/view?usp=sharing)
- [John Klein web page](https://john-klein.github.io/)
- Very good tutorials in [Hugo LaRochelle youtube page](https://www.youtube.com/user/hugolarochelle)

For more details, you can check the first chapter of my [PhD thesis](https://drive.google.com/file/d/1QMci-0gAHPBeMn9L-2pRpbLeXROw5A8I/view?usp=sharing)

