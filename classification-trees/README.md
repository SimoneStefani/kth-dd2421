*Machine Learning (DD2421) - Royal Institute of Technology KTH*
# Decision Trees


### 2. MONK Datasets
#### Assignment 0:
> Each one of the datasets has properties which makes them hard to learn. Motivate which of the three problems is most difficult for a decision tree algorithm to learn.

* Dataset 1 the concept behind involves *a1* and *a2* being related and thus it is hard to split on one of these attributes.
* Dataset 2 is hard to learn because the true concept behind it involves the value of an attribute in respect to the value of another attribute. Hence the space cannot be split based on the value of a single attribute.
* Dataset 3 contains a 5% noise in the training set and it has the smallest amount of training data.
In general all the datasets have a small amount of training samples compared with test samples.

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

Entropy is the measure of randomness of a process/variable.
* The entropy in a **uniform distribution** (outcomes are equally probable) grows logaritmic in relation with the number of outcomes. Examples of this phenomenon are a perfect die or a fair coin.
* The entropy in a **non-uniform distribution** (one of the values is more probable to occur than the others) is always less than or equal to log<sub>2</sub>(*n*). In fact if one of the values is more probable to occur than the others, an observation that this value occurs is less informative than if some less common outcome had occurred. Conversely, rarer events provide more information when observed.
Examples of this phenomenon are an unbalanced die or an unfair coin.
---

### 4. Information Gain
#### Assignment 3:
> Use the function `averageGain` (defined in `dtree.py`) to calculate the expected information gain corresponding to each of the six attributes. Note that the attributes are represented as instances of the class Attribute (defined in `monkdata.py`) which you can access via `m.attributes[0]`, ..., `m.attributes[5]`. Based on the results, which attribute should be used for splitting the examples at the root node?

| Dataset |     a1     |     a2     |     a3     |     a4     |     a5     |     a6     |
|---------|------------|------------|------------|------------|------------|------------|
|  MONK-1 | 0.07527256 | 0.00583843 | 0.00470757 | 0.0263117  | **0.28703075** | 0.00075786 |
|  MONK-2 | 0.00375618 | 0.0024585  | 0.00105615 | 0.01566425 | **0.01727718** | 0.00624762 |
|  MONK-3 | 0.00712087 | **0.29373617** | 0.00083111 | 0.00289182 | 0.25591172 | 0.00707703 |

The attribute **a5** seems to perform very well both with training sets MONK-1 and MONK-2 while **a2** is better for MONK-3.

#### Assignment 4:
> For splitting we choose the attribute that maximizes the information gain, Eq.3. Looking at Eq.3 how does the entropy of
the subsets, *S<sub>k</sub>*, look like when the information gain is maximized? How can we motivate using the information gain as a heuristic for picking an attribute for splitting? Think about reduction in entropy after the split and what the entropy implies.

The entropy for a subset *S<sub>k</sub>* is lower when the information gain is maximized. This is true because the uncertainty (entropy) in the subset decreses the more information we obtain.

By using the highest *information gain* we can select an attribute to operate a split so that the uncertainty in the subsets decreases the most. This means that the newly create subsets are more "pure" and have a lower entropy. In this way we can move towards a limited number of subsets that contained better classified samples.

---

### 5. Building Decision Trees
#### Assignment 5:
> Build the full decision trees for all three Monk datasets using `buildTree`. Then, use the function `check` to measure the performance of the decision tree on both the training and test datasets. <br> For example to built a tree for `monk1` and compute the performance on the test data you could use
```python
import monkdata as m
import dtree as d
t = d.buildTree(m.monk1, m.attributes)
print(d.check(t, m.monk1test))
```
> Compute the train and test set errors for the three Monk datasets for the full trees. Were your assumptions about the datasets correct? Explain the results you get for the training and test datasets.
