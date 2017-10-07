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

Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. In reality (example in NLP) features tend to be dependent but we still assume them as independent hence the name "Naive". Surprisingly NB models perform well despite the conditional independence assumption.

> How does the decision boundary look for the Iris dataset? How could one improve
the classification results for this scenario by changing classifier or, alternatively,
manipulating the data?

The decision boundary between class 0 and 1 is well defined as expectable for easily separable classes. However classes 1 and 2 are more noisy and often overlap. This leads to an unclear and counterintuitive boundary leaning on the left. This is clearly a result of the weak nature of the classifier. Probably SVM with slack veriables or random forests would have worked better as classifiers. 
