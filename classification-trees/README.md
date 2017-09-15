*Machine Learning (DD2421) - Royal Institute of Technology KTH*
# Decision Trees


### 2. MONK Datasets
#### Assignment 0:
> Each one of the datasets has properties which makes them hard to learn. Motivate which of the three problems is most difficult for a decision tree algorithm to learn.

* Dataset 1
* Dataset 2 is hard to learn because the true concept behind it involves the value of an attribute in respect to the value of another attribute. Hence the space cannot be split based on the value of a single attribute.
* Dataset 3 contains a 5% noise in the training set

---

### 3. Entropy
#### Assignment 1:
> The file `dtree.py` defines a function entropy which calculates the entropy of a dataset. Import this file along with the monks datasets and use it to calculate the entropy of the _training_ datasets.

| Dataset | Entropy            |
|---------|--------------------|
| MONK-1  | 1.0                |
| MONK-2  | 0.957117428264771  |
| MONK-3  | 0.9998061328047111 |

#### Assignment 2: 
> Explain entropy for a uniform distribution and a non-uniform distribution, present some example distributions with high and low entropy.

---

### 4. Information Gain
#### Assignment 3:
> Use the function `averageGain` (defined in `dtree.py`) to calculate the expected information gain corresponding to each of the six attributes. Note that the attributes are represented as instances of the class Attribute (defined in `monkdata.py`) which you can access via `m.attributes[0]`, ..., `m.attributes[5]`. Based on the results, which attribute should be used for splitting the examples at the root node?
