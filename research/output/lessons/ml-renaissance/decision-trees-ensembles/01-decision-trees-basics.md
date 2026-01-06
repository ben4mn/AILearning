# Decision Trees: Learning to Ask Questions

## Introduction

Sometimes the simplest ideas are the most powerful. A decision tree asks a series of yes/no questions about your data, each answer narrowing down the possibilities until it reaches a conclusion. It's the same logic a doctor uses for diagnosis: "Do you have a fever? Is it above 102°F? Do you have a cough?"

Decision trees emerged in the 1980s and became one of the most practical machine learning algorithms. They're easy to understand, fast to train, handle mixed data types naturally, and require minimal preprocessing. While they'd later be overshadowed by ensemble methods that combine many trees, the single decision tree remains a cornerstone of interpretable machine learning.

This lesson introduces how decision trees work, how they learn from data, and why their simplicity is both their strength and their limitation.

## The Decision Tree Model

A decision tree is a flowchart-like structure:
- **Internal nodes** test a feature (e.g., "Is income > $50k?")
- **Branches** represent the test outcomes (yes/no)
- **Leaf nodes** contain predictions (class labels or values)

```
                    [Age > 35?]
                    /          \
                  Yes           No
                  /               \
           [Income > 50k?]    [Student?]
           /        \          /       \
         Yes        No       Yes        No
          |          |        |          |
       [Buy]     [Maybe]   [Don't]    [Maybe]
```

To classify a new example, start at the root and follow branches based on feature values until reaching a leaf. The leaf's label is the prediction.

```python
class DecisionTreeNode:
    def __init__(self):
        self.feature = None        # Which feature to split on
        self.threshold = None      # For numeric: value to compare
        self.left = None           # Left child (feature <= threshold)
        self.right = None          # Right child (feature > threshold)
        self.label = None          # For leaves: the predicted class

def predict(node, example):
    """Traverse tree to make a prediction."""
    if node.label is not None:
        return node.label  # Reached a leaf

    if example[node.feature] <= node.threshold:
        return predict(node.left, example)
    else:
        return predict(node.right, example)
```

## Learning a Decision Tree

How do we construct a tree from training data? The algorithm is elegantly recursive:

1. If all examples have the same class, create a leaf with that class
2. Otherwise, find the best feature and threshold to split on
3. Partition the data by that split
4. Recursively build trees for each partition

```python
def build_tree(X, y, max_depth=None, depth=0):
    """Build a decision tree using recursive splitting."""
    # Base case: all same class
    if len(set(y)) == 1:
        return LeafNode(label=y[0])

    # Base case: maximum depth reached
    if max_depth and depth >= max_depth:
        return LeafNode(label=most_common(y))

    # Find best split
    best_feature, best_threshold = find_best_split(X, y)

    if best_feature is None:
        return LeafNode(label=most_common(y))

    # Partition data
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    # Recursive construction
    node = DecisionTreeNode()
    node.feature = best_feature
    node.threshold = best_threshold
    node.left = build_tree(X[left_mask], y[left_mask], max_depth, depth+1)
    node.right = build_tree(X[right_mask], y[right_mask], max_depth, depth+1)

    return node
```

The key question: how do we find the "best" split?

## Measuring Split Quality

A good split separates the classes cleanly. If one child gets all the "yes" examples and the other gets all the "no" examples, that's perfect. If both children have mixed classes, the split wasn't very informative.

### Information Gain and Entropy

**Entropy** measures the impurity of a set—how mixed the classes are:

**Entropy(S) = -Σᵢ pᵢ log₂(pᵢ)**

Where pᵢ is the proportion of class i.
- Pure set (all one class): Entropy = 0
- Maximally mixed (50/50 for binary): Entropy = 1

```python
import numpy as np

def entropy(y):
    """Calculate entropy of a label array."""
    if len(y) == 0:
        return 0
    proportions = np.bincount(y) / len(y)
    proportions = proportions[proportions > 0]  # Remove zeros
    return -np.sum(proportions * np.log2(proportions))

# Examples:
# entropy([0, 0, 0, 0]) = 0        (pure)
# entropy([0, 0, 1, 1]) = 1        (maximally mixed)
# entropy([0, 0, 0, 1]) = 0.81     (somewhat mixed)
```

**Information gain** measures how much a split reduces entropy:

**Gain(S, A) = Entropy(S) - Σᵥ (|Sᵥ|/|S|) × Entropy(Sᵥ)**

Where Sᵥ are the subsets created by splitting on feature A.

```python
def information_gain(X_feature, y, threshold):
    """Calculate information gain for a split."""
    parent_entropy = entropy(y)

    # Split the data
    left_mask = X_feature <= threshold
    right_mask = ~left_mask

    if sum(left_mask) == 0 or sum(right_mask) == 0:
        return 0

    # Weighted average of child entropies
    n = len(y)
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    child_entropy = (sum(left_mask)/n * left_entropy +
                    sum(right_mask)/n * right_entropy)

    return parent_entropy - child_entropy
```

### Gini Impurity

An alternative to entropy, **Gini impurity** is slightly faster to compute and often gives similar results:

**Gini(S) = 1 - Σᵢ pᵢ²**

```python
def gini(y):
    """Calculate Gini impurity of a label array."""
    if len(y) == 0:
        return 0
    proportions = np.bincount(y) / len(y)
    return 1 - np.sum(proportions ** 2)

# Examples:
# gini([0, 0, 0, 0]) = 0          (pure)
# gini([0, 0, 1, 1]) = 0.5        (maximally mixed)
# gini([0, 0, 0, 1]) = 0.375      (somewhat mixed)
```

The **CART** algorithm (Classification and Regression Trees), developed by Breiman et al. in 1984, used Gini impurity and became the most influential decision tree method.

## Finding the Best Split

To find the best split, we consider all features and all possible thresholds:

```python
def find_best_split(X, y):
    """Find the best feature and threshold to split on."""
    best_gain = 0
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    for feature in range(n_features):
        # Get unique values as potential thresholds
        thresholds = np.unique(X[:, feature])

        for threshold in thresholds:
            gain = information_gain(X[:, feature], y, threshold)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold
```

For continuous features, we typically test thresholds at the midpoints between consecutive sorted values. For categorical features, we can test subsets (though this is more complex).

## Overfitting and Pruning

Decision trees are prone to **overfitting**. Given enough depth, a tree can perfectly classify training data by creating one leaf per training example. This tree memorizes rather than generalizes.

```python
# An overfit tree on noisy data:
#                    [Feature_42 > 3.14159?]
#                    /                      \
#            [Feature_17 > 2.718?]          ...
#            /                    \
#      [label: 0]           [Feature_99 > 1.414?]
#                            /                   \
#                       [label: 1]           [label: 0]
#
# Each leaf contains exactly one training point!
# This tree won't generalize at all.
```

Two strategies prevent overfitting:

### Pre-pruning (Early Stopping)

Stop growing the tree before it's fully grown:
- **Maximum depth**: Don't go deeper than k levels
- **Minimum samples per leaf**: Require at least n examples to create a leaf
- **Minimum samples per split**: Require at least n examples to attempt a split
- **Minimum information gain**: Don't split unless gain exceeds threshold

```python
from sklearn.tree import DecisionTreeClassifier

# Controlled tree with pre-pruning
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    min_samples_split=20,
    min_impurity_decrease=0.01
)
```

### Post-pruning

Grow a full tree, then remove branches that don't improve generalization. **Cost-complexity pruning** (used in CART) balances tree size against training error:

**Cost = Training Error + α × (Number of Leaves)**

Increasing α produces simpler trees. Cross-validation finds the optimal α.

```python
# Cost-complexity pruning with cross-validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Find effective alphas
tree = DecisionTreeClassifier()
path = tree.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

# Cross-validate to find best alpha
best_alpha = 0
best_score = 0
for alpha in alphas:
    tree = DecisionTreeClassifier(ccp_alpha=alpha)
    scores = cross_val_score(tree, X_train, y_train, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_alpha = alpha

final_tree = DecisionTreeClassifier(ccp_alpha=best_alpha)
final_tree.fit(X_train, y_train)
```

## Regression Trees

Decision trees naturally extend to regression by predicting the **mean value** of training examples in each leaf:

```python
def predict_regression(node, example):
    """Predict a continuous value using regression tree."""
    if node.is_leaf:
        return node.mean_value  # Average of training examples in leaf
    if example[node.feature] <= node.threshold:
        return predict_regression(node.left, example)
    else:
        return predict_regression(node.right, example)
```

For finding splits, we minimize **variance** (or equivalently, mean squared error) instead of Gini/entropy:

```python
def variance_reduction(y, left_mask):
    """Calculate variance reduction for a split."""
    right_mask = ~left_mask

    parent_var = np.var(y) * len(y)
    left_var = np.var(y[left_mask]) * sum(left_mask)
    right_var = np.var(y[right_mask]) * sum(right_mask)

    return parent_var - left_var - right_var
```

## Strengths and Limitations

### Strengths
- **Interpretable**: Trees can be visualized and explained
- **Fast**: Training and prediction are O(n log n)
- **Mixed types**: Handle numeric and categorical features
- **Nonlinear**: Capture complex decision boundaries
- **Feature importance**: Splits reveal important features

### Limitations
- **High variance**: Small data changes can produce very different trees
- **Overfitting**: Deep trees memorize rather than generalize
- **Axis-aligned splits**: Can't efficiently represent diagonal boundaries
- **Suboptimal**: Greedy construction doesn't guarantee the best tree

The high variance problem—different training samples producing very different trees—would lead to ensemble methods that combine many trees. But first, we needed to understand why combining predictions could help.

## Key Takeaways

- Decision trees classify by asking a sequence of feature-based questions
- Splits are chosen to maximize information gain or minimize Gini impurity
- The CART algorithm (1984) established the modern framework for decision trees
- Overfitting is controlled through pre-pruning (stopping early) or post-pruning (removing branches)
- Regression trees predict the mean value in each leaf region
- Trees are interpretable but have high variance and can overfit easily

## Further Reading

- Breiman et al. *Classification and Regression Trees* (1984) - The CART book
- Quinlan, J. Ross. *C4.5: Programs for Machine Learning* (1993) - Alternative approach with information gain
- Hastie, Tibshirani, and Friedman. *The Elements of Statistical Learning*, Chapter 9 - Mathematical treatment
- scikit-learn documentation on Decision Trees - Practical guide

---
*Estimated reading time: 10 minutes*
