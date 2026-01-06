# Bagging and Random Forests: The Power of the Crowd

## Introduction

In 1996, Leo Breiman made a surprising observation: if you train multiple decision trees on slightly different versions of your data and average their predictions, the result is dramatically more accurate than any single tree. This technique—**bootstrap aggregating**, or "bagging"—represented a fundamental shift in machine learning philosophy.

Instead of searching for one perfect model, why not combine many imperfect ones? The mathematics showed that combining diverse, noisy predictors could produce a stable, accurate ensemble. Random Forests, introduced by Breiman in 2001, extended this idea to become one of the most successful and widely used algorithms in machine learning history.

This lesson explores how bagging works, why diversity matters, and how Random Forests revolutionized practical machine learning.

## The Variance Problem

Decision trees have a fundamental weakness: **high variance**. Small changes in training data can produce completely different trees.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Train on slightly different subsamples
np.random.seed(42)
tree1 = DecisionTreeClassifier(max_depth=10)
tree1.fit(X_train[:900], y_train[:900])

tree2 = DecisionTreeClassifier(max_depth=10)
tree2.fit(X_train[100:], y_train[100:])

# These trees can have completely different structures!
# Different splits, different depths, different predictions
```

This instability means a single tree is unreliable. It might perform well on one test set and poorly on another. How can we get the benefits of decision trees (nonlinearity, interpretability, fast training) without the variance problem?

## Bias-Variance Tradeoff

To understand bagging, we need the **bias-variance decomposition**. The expected error of a model has three components:

**Expected Error = Bias² + Variance + Irreducible Noise**

- **Bias**: Error from wrong assumptions (too simple a model)
- **Variance**: Error from sensitivity to training data (too complex a model)
- **Noise**: Unavoidable randomness in the data

Decision trees with full depth have:
- **Low bias**: They can fit complex patterns
- **High variance**: They're very sensitive to training data

The goal of bagging is to **reduce variance without increasing bias**.

## Bootstrap Aggregating (Bagging)

Breiman's bagging algorithm (1996):

1. Create B **bootstrap samples** from the training data (sample with replacement)
2. Train a decision tree on each bootstrap sample
3. Average predictions (regression) or vote (classification)

```python
from sklearn.utils import resample

def bagging_train(X, y, n_estimators=100, max_depth=None):
    """Train a bagged ensemble of decision trees."""
    models = []

    for _ in range(n_estimators):
        # Bootstrap sample: sample n points with replacement
        X_boot, y_boot = resample(X, y, n_samples=len(X))

        # Train a deep tree (low bias, high variance)
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X_boot, y_boot)
        models.append(tree)

    return models

def bagging_predict(models, X):
    """Predict by majority voting."""
    predictions = np.array([model.predict(X) for model in models])
    # Return most common prediction for each example
    return np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=predictions
    )
```

### Why Does Averaging Reduce Variance?

Consider B independent predictors, each with variance σ². The variance of their average is:

**Var(average) = σ²/B**

More predictors → lower variance! Of course, our trees aren't truly independent (they're trained on overlapping data), but the bootstrap sampling introduces enough diversity that variance still drops substantially.

```python
# Demonstration: averaging reduces variance
true_value = 10
noise_std = 5

# Single estimator
single_predictions = [true_value + np.random.normal(0, noise_std)
                      for _ in range(1000)]
print(f"Single predictor std: {np.std(single_predictions):.2f}")  # ~5.0

# Average of 100 estimators
ensemble_predictions = [
    np.mean([true_value + np.random.normal(0, noise_std) for _ in range(100)])
    for _ in range(1000)
]
print(f"Ensemble std: {np.std(ensemble_predictions):.2f}")  # ~0.5
```

### Out-of-Bag Estimation

A clever bonus of bootstrap sampling: each bootstrap sample uses only about 63% of the training data (due to sampling with replacement). The remaining ~37% can serve as a validation set for that particular tree.

**Out-of-bag (OOB) error**: For each training example, predict using only trees whose bootstrap sample didn't include that example.

```python
def compute_oob_score(X, y, models, bootstrap_indices):
    """Compute out-of-bag accuracy."""
    n_samples = len(y)
    oob_predictions = np.zeros((n_samples, len(np.unique(y))))

    for i, (model, indices) in enumerate(zip(models, bootstrap_indices)):
        # Find samples NOT in this bootstrap
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False

        if oob_mask.sum() > 0:
            probs = model.predict_proba(X[oob_mask])
            oob_predictions[oob_mask] += probs

    # Predict class with highest average probability
    oob_pred = np.argmax(oob_predictions, axis=1)

    # Accuracy on samples that had at least one OOB prediction
    valid = oob_predictions.sum(axis=1) > 0
    return np.mean(oob_pred[valid] == y[valid])
```

OOB error provides a free estimate of test error without needing a separate validation set!

## Random Forests: More Randomness, Less Correlation

Bagging helped, but the trees were still correlated—they tended to make the same splits on the same features. Leo Breiman's **Random Forest** algorithm (2001) added another layer of randomness: at each split, consider only a random subset of features.

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest with 100 trees
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # Consider sqrt(n_features) at each split
    bootstrap=True,
    oob_score=True
)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.3f}")
```

### Why Restrict Features?

Imagine a dataset where feature 1 is the strongest predictor. In bagging, most trees will split on feature 1 first. The trees are highly correlated—they make similar predictions and similar errors.

By forcing each split to choose from a random feature subset:
- Some trees must find alternative splitting strategies
- Trees become more **diverse**
- Diverse trees make **uncorrelated errors**
- Averaging uncorrelated errors reduces variance more effectively

```python
def random_forest_split(X, y, n_features_try):
    """Find best split considering only a random feature subset."""
    n_features = X.shape[1]

    # Randomly select features to consider
    feature_subset = np.random.choice(
        n_features,
        size=min(n_features_try, n_features),
        replace=False
    )

    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in feature_subset:  # Only these features!
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            gain = information_gain(X[:, feature], y, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold
```

### Hyperparameters

Random Forests have surprisingly few hyperparameters that matter:

**n_estimators**: Number of trees. More is generally better, with diminishing returns.

```python
# Error typically plateaus after 100-500 trees
# More trees = more computation but rarely hurts accuracy
rf = RandomForestClassifier(n_estimators=500)
```

**max_features**: Features to consider at each split.
- Classification: sqrt(n_features) works well
- Regression: n_features/3 works well

```python
# Options for max_features
rf = RandomForestClassifier(max_features='sqrt')  # Classification default
rf = RandomForestClassifier(max_features=0.33)    # 33% of features
rf = RandomForestClassifier(max_features=10)      # Exactly 10 features
```

**max_depth**, **min_samples_leaf**: Usually left at defaults (full trees). Random Forests resist overfitting due to averaging.

## Feature Importance

Random Forests provide natural measures of feature importance:

### Mean Decrease in Impurity (MDI)

Sum the Gini reductions from all splits on each feature, weighted by samples reaching that split:

```python
# Built into scikit-learn
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# Plot feature importances
import matplotlib.pyplot as plt
sorted_idx = np.argsort(importances)[::-1]
plt.barh(range(10), importances[sorted_idx[:10]])
plt.yticks(range(10), [feature_names[i] for i in sorted_idx[:10]])
plt.xlabel('Mean Decrease in Impurity')
```

### Permutation Importance

Measure how much accuracy drops when a feature's values are randomly shuffled:

```python
from sklearn.inspection import permutation_importance

# Compute permutation importance on test set
result = permutation_importance(rf, X_test, y_test, n_repeats=10)

# This measures how much each feature actually helps predictions
# MDI can overestimate importance of high-cardinality features
```

## Parallelization

Random Forests are embarrassingly parallel—each tree is independent:

```python
from sklearn.ensemble import RandomForestClassifier

# Use all CPU cores
rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rf.fit(X_train, y_train)  # Trees train in parallel
```

This made Random Forests practical for large datasets long before GPU acceleration was common.

## Why Random Forests Work So Well

Random Forests succeeded for multiple reasons:

1. **Bias-variance balance**: Deep trees = low bias; averaging = low variance
2. **Robustness**: Works well with default parameters
3. **Feature handling**: No preprocessing needed for mixed types
4. **Missing values**: Can be handled with imputation or surrogate splits
5. **Feature importance**: Built-in interpretability
6. **Speed**: Parallel training and fast prediction
7. **Resistance to overfitting**: More trees rarely hurts

By 2010, Random Forests were the go-to algorithm for tabular data, winning Kaggle competitions and powering production systems at scale.

## Key Takeaways

- Bagging reduces variance by averaging predictions from models trained on bootstrap samples
- Bootstrap samples (sampling with replacement) introduce diversity among trees
- Out-of-bag error provides free validation without a separate test set
- Random Forests add feature subsampling at each split, further decorrelating trees
- Diversity is key: uncorrelated errors cancel when averaged
- Random Forests require minimal tuning and work well out of the box
- Feature importance measures reveal which features drive predictions

## Further Reading

- Breiman, Leo. "Bagging Predictors" (1996) - The original bagging paper
- Breiman, Leo. "Random Forests" (2001) - The definitive Random Forest paper
- Hastie et al. *The Elements of Statistical Learning*, Chapter 15 - Thorough mathematical treatment
- Louppe, Gilles. "Understanding Random Forests" (2014) - Excellent PhD thesis

---
*Estimated reading time: 10 minutes*
