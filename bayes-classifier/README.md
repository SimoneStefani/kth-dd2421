*Machine Learning (DD2421) - Royal Institute of Technology KTH*
# Bayesian Learning and Boosting


### 4. Bayesian Learning
Example of a Bayesian classifier using the maximum likelihood method (ML) on Gaussian test data:
<p align="center"><img src="https://github.com/SimoneStefani/kth-dd2421/blob/master/bayes-classifier/assets/categories.png"></p>

**Iris** dataset classified with Naive Bayesian Classifier:

`Final mean classification accuracy  89 with standard deviation 4.16`
<p align="center"><img src="https://github.com/SimoneStefani/kth-dd2421/blob/master/bayes-classifier/assets/iris.png"></p>

**Vowels** dataset classified with Naive Bayesian Classifier:

`Final mean classification accuracy  64.7 with standard deviation 4.03`
<p align="center"><img src="https://github.com/SimoneStefani/kth-dd2421/blob/master/bayes-classifier/assets/vowels.png"></p>

> When can a feature independence assumption be reasonable and when not?

Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. In reality (example in NLP) features tend to be dependent but we still assume them as independent hence the name "Naive". Surprisingly NB models perform well despite the conditional independence assumption. In the case of the iris dataset, for example, we can suppose that `sepal length` and `sepal width` are positively correlated as well as `petal length` and `petal width`.

> How does the decision boundary look for the Iris dataset? How could one improve
the classification results for this scenario by changing classifier or, alternatively,
manipulating the data?

The decision boundary between class 0 and 1 is well defined as expectable for easily separable classes. However classes 1 and 2 are more noisy and often overlap. This leads to an unclear and counterintuitive boundary leaning on the left. This is clearly a result of the weak nature of the classifier. Probably SVM with slack veriables or random forests would have worked better as classifiers. Non-linear transformations to the dataset may also be a solution.


### 5. Boosting

**Iris** dataset classified with Naive Bayesian Classifier and **adaboost**:

`Final mean classification accuracy  94.1 with standard deviation 6.72`
<p align="center"><img src="https://github.com/SimoneStefani/kth-dd2421/blob/master/bayes-classifier/assets/iris_boost.png"></p>

**Vowels** dataset classified with Naive Bayesian Classifier and **adaboost**:

`Final mean classification accuracy  80.2 with standard deviation 3.52`
<p align="center"><img src="https://github.com/SimoneStefani/kth-dd2421/blob/master/bayes-classifier/assets/vowels_boost.png"></p>

> Is there any improvement in classification accuracy? Why/why not?

Yes, there is an improvement of the classification in both the *iris* and *vowel* datasets. This is expectable because the boosting technique allows to concentrate on the missclassified samples and build a more accurate model increasing the variance (NB generally have high bias and low variance).

> Plot the decision boundary of the boosted classifier on *iris* and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?

The boundary is now more complex and fits better the underlying data (see fig above). The tendency to lean on the left has been removed and because boosting allowed the classification algorithm to focus on the missclassified  points.

> Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?

It is indeed possible to increase the accuracy of a weak classifier by means of boosting. However we should be able to run the weak classifier on partitions of the datasets with different distributions.


### 6. Boosting of Decision Tree Classifier
Decision Tree on *Iris* dataset without and with boosting:
`Final mean classification accuracy  92.4 with standard deviation 3.71`
`Final mean classification accuracy  94.6 with standard deviation 3.65`

Decision Tree on *Vowels* dataset without and with boosting:
`Final mean classification accuracy  64.1 with standard deviation 4`
`Final mean classification accuracy  86.6 with standard deviation 3.02`

> Is there any improvement in classification accuracy? Why/why not?

Yes, the boosted version of the algorithm yield better results than the normal decision trees. A decision tree is a weak classifier and we get the biggest increase in accurancy on the Vowels dataset where data points are more mixed (DT generally have low bias and high variance).

> Plot the decision boundary of the boosted classifier on *iris* and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?

The boundary is indeed more "edgy" in the boosted version.

> Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?

Yes, in certain cases (see above).

### 7. Which classifier

* **Outliers:** Naïve Bayes without boosting. Decision trees would tend to overfit the data and also a boosted Bayes classifier would give too much weight to the outliers.

* **Irrelevant inputs: part of the feature space is irrelevant:** Decision Trees. They would tend to split ignore the irrelevant part of the feature space concentrating only on attributes with high information gain.

* **Predictive power:** Naïve Bayes with boosting. It tends to yield the best performance on prediction.

* **Mixed types of data: binary, categorical or continuous features, etc.:** Decision Trees are more flexible and work well both with quantitative and qualitative data while Bayes works better with continuous data. Probably using boosting would increase the accurancy.

* **Scalability: the dimension of the data, D, is large or the number of instances, N, is large, or both:** Decision Trees. Bayes works well even with small datasets while decision trees gain in performance when the dataset is large.


### 8. Olivetti Dataset

Classification of a point from the Olivetti dataset with **Boosted decision tree**:

`Final mean classification accuracy  70.1 with standard deviation 7.05`


Classification of a point from the Olivetti dataset with **Bayes classifier**:

`Final mean classification accuracy  87.7 with standard deviation 3.03`

<p align="center"><img src="https://github.com/SimoneStefani/kth-dd2421/blob/master/bayes-classifier/assets/olivetti_dectree.png"></p>

