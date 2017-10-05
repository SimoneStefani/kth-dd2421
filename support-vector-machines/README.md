*Machine Learning (DD2421) - Royal Institute of Technology KTH*
# Support Vector Machines


### 6. Running and Reporting
#### Assignments
> 1. Move the clusters around to make it easier or harder for the classifier to find a decent boundary. Pay attention to when the `qt` function prints an error message that it can not find a solution.
> 2. Implement some of the non-linear kernels. you should be able to classify very hard datasets.
> 3. The non-linear kernels have parameters; explore how they influence the decision boundary. Reason about this in terms of the bias-variance trade-off.


#### 1. Provided Data
The provided data, generated from the code in the text, is not linearly separable. In fact the quadratic optimization fails with the message `Terminated (singular KKT matrix)`. Polynomial kernels of several degree and radial kernels can easily classify the data.
<p align="center"><img src="https://github.com/SimoneStefani/kth-dd2421/blob/master/support-vector-machines/assets/basic_dt.png"></p>

#### 2. Linearly Separable Data
By tweaking the dataset generator it is possible to obtain data in two linearly separable classes. Thi allow to use a linear kernel for the SVM even though such techique is not superior to other simpler classification methods.
<p align="center"><img src="https://github.com/SimoneStefani/kth-dd2421/blob/master/support-vector-machines/assets/line_sep_dt.png"></p>

### 7. Slack Implementation
#### Assignments
> 1. Explore the role of the parameter `C`. What happens for very large/small values?
> 2. Imagine that you are given data that is not easily separable. When should you opt for more slack rather than going for a more complex model and vice versa?

The variable `C` represent a sort of tolerance for data points which are not correctly classified. If the value of `C` is high it means that we want to minimise the effect of Slack Variables while if the value is low (min 0) it means that the model allows for a great number of miss-classified points.

Slack Variables are useful when a small number of points, often considered noise, prevents the classification of simple underlying model. SVM, in fact, are not made to take in consider classification in a statistical way, where a model can reach a certain accurancy depending on how many points are correctly classified. They are only able to neatly divide two datasets optimizing for the "largest street". Hence the importance of Slack variables to make the technique more usable. It is important still to remebmber that Slack vairables are a trade-off on the accurancy of the model.

Here is an example of a dataset that benefits from Slack variables:
<p align="center"><img src="https://github.com/SimoneStefani/kth-dd2421/blob/master/support-vector-machines/assets/moons_dt.png"></p>
