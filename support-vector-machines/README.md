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
