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
