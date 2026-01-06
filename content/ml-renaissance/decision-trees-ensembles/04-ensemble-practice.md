# Ensemble Methods in Practice

## Introduction

By the 2010s, ensemble methods—particularly Random Forests and Gradient Boosting—had become the workhorses of practical machine learning. They powered fraud detection at banks, recommendation systems at tech companies, and medical diagnosis tools in hospitals. When Kaggle competitions crowned winners, ensembles almost always sat at the top.

This lesson bridges theory and practice. We'll explore how to choose between ensemble methods, tune them effectively, handle common challenges, and understand when simpler approaches might be better. These insights represent hard-won knowledge from years of applying these algorithms to real problems.

## Choosing Your Algorithm

### When to Use Random Forests

Random Forests excel when:
- You need a quick, reliable baseline
- The data is noisy or has outliers
- You want minimal hyperparameter tuning
- Training time matters more than maximum accuracy
- Interpretability (feature importance) is important

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest: works well out of the box
rf = RandomForestClassifier(
    n_estimators=500,      # More trees rarely hurts
    max_features='sqrt',   # Works well for classification
    n_jobs=-1,             # Parallelize
    random_state=42
)
rf.fit(X_train, y_train)
# Usually achieves 90%+ of potential accuracy with no tuning
```

### When to Use Gradient Boosting

Gradient Boosting excels when:
- Maximum accuracy is the goal
- You have time for hyperparameter tuning
- The data is relatively clean
- You want to minimize specific loss functions
- Prediction time can be managed

```python
import xgboost as xgb

# XGBoost: higher ceiling but needs tuning
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    early_stopping_rounds=50,
    random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)
```

### Decision Matrix

| Scenario | Recommendation |
|----------|----------------|
| First model on new problem | Random Forest |
| Production system, accuracy critical | Gradient Boosting (tuned) |
| Very noisy labels | Random Forest |
| Highly imbalanced classes | Gradient Boosting with scale_pos_weight |
| Need probability calibration | Random Forest (naturally calibrated) |
| Massive dataset (10M+ rows) | LightGBM |
| Many categorical features | CatBoost |

## Hyperparameter Tuning

### Random Forest Tuning

Random Forests are forgiving. These parameters matter most:

```python
from sklearn.model_selection import RandomizedSearchCV

rf_params = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.3]
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(),
    rf_params,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
rf_search.fit(X_train, y_train)
print(f"Best params: {rf_search.best_params_}")
```

**Practical tips:**
- n_estimators: Start with 500, increase if OOB score still improving
- max_depth: Try None (full trees) first, then restrict if overfitting
- min_samples_leaf: Increase for noisy data or small datasets

### Gradient Boosting Tuning

More sensitive to hyperparameters. Use a staged approach:

```python
# Stage 1: Fix learning_rate=0.1, tune tree parameters
stage1_params = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Stage 2: Tune regularization
stage2_params = {
    'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
    'reg_lambda': [1, 2, 5, 10],
    'gamma': [0, 0.1, 0.5, 1]
}

# Stage 3: Lower learning_rate, increase n_estimators
# learning_rate=0.01-0.05 with n_estimators=1000-5000
```

**Practical tips:**
- Always use early stopping
- Lower learning_rate is usually better (with more trees)
- subsample and colsample_bytree around 0.8 often helps
- max_depth: 4-8 for most problems

## Handling Imbalanced Data

Real-world classification often has severe class imbalance (fraud: 0.1%, spam: 10%, etc.).

### Resampling Approaches

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Combine oversampling minority + undersampling majority
resampler = Pipeline([
    ('over', SMOTE(sampling_strategy=0.5)),
    ('under', RandomUnderSampler(sampling_strategy=0.8))
])

X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
```

### Class Weights

Simpler and often effective:

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Compute class weights
class_weights = dict(zip(
    np.unique(y_train),
    len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
))

rf = RandomForestClassifier(class_weight=class_weights)

# For XGBoost
ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(scale_pos_weight=ratio)
```

### Threshold Tuning

Rather than using 0.5 as the classification threshold:

```python
from sklearn.metrics import precision_recall_curve

# Get probabilities
probs = model.predict_proba(X_valid)[:, 1]

# Find threshold for desired precision/recall
precision, recall, thresholds = precision_recall_curve(y_valid, probs)

# Example: Find threshold for 90% precision
idx = np.argmax(precision >= 0.90)
optimal_threshold = thresholds[idx]
print(f"Threshold for 90% precision: {optimal_threshold:.3f}")
print(f"Recall at this threshold: {recall[idx]:.3f}")
```

## Feature Engineering for Trees

Tree ensembles handle many feature types naturally, but engineering can still help.

### What Trees Handle Well

- Missing values (XGBoost/LightGBM handle natively)
- Mixed numeric/categorical
- Nonlinear relationships
- Feature interactions (within tree depth)

```python
# No scaling needed for trees
# This is fine:
X = np.column_stack([
    income,           # Range: 0 - 1,000,000
    age,              # Range: 18 - 100
    is_subscriber,    # Binary: 0 or 1
    category_encoded  # Integer: 0 - 50
])
```

### What to Engineer

**Interaction features** for deep interactions:
```python
# If interaction is deeper than max_depth, create explicitly
X['feature_a_times_b'] = X['feature_a'] * X['feature_b']
X['feature_a_ratio_b'] = X['feature_a'] / (X['feature_b'] + 1)
```

**Aggregation features** for grouped data:
```python
# For user behavior data, aggregate by user
user_stats = df.groupby('user_id').agg({
    'purchase_amount': ['mean', 'std', 'max', 'count'],
    'days_since_last_purchase': 'min'
})
```

**Time features** from timestamps:
```python
X['hour'] = X['timestamp'].dt.hour
X['day_of_week'] = X['timestamp'].dt.dayofweek
X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
```

## Model Interpretation

### Feature Importance

```python
# Random Forest: Mean Decrease Impurity
importances = rf.feature_importances_

# Permutation importance (more reliable)
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10)

# SHAP values (best for explanation)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### Partial Dependence

See how predictions change with one feature:

```python
from sklearn.inspection import PartialDependenceDisplay

# Show effect of top features
features = [0, 1, (0, 1)]  # Individual + interaction
PartialDependenceDisplay.from_estimator(
    model, X_train, features,
    kind='both'  # Show individual + average
)
```

## Ensemble of Ensembles

Top competition solutions often combine multiple models:

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Simple averaging
voting = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=500)),
    ('xgb', xgb.XGBClassifier(n_estimators=500)),
    ('lgb', LGBMClassifier(n_estimators=500))
], voting='soft')  # Average probabilities

# Stacking: meta-model learns to combine
stacking = StackingClassifier([
    ('rf', RandomForestClassifier(n_estimators=500)),
    ('xgb', xgb.XGBClassifier(n_estimators=500)),
    ('lgb', LGBMClassifier(n_estimators=500))
], final_estimator=LogisticRegression())
```

**Warning**: Stacking adds complexity and training time. Use only when every bit of accuracy matters (competitions, high-stakes production).

## Common Pitfalls

### Data Leakage

The most dangerous mistake:
```python
# WRONG: Scaling before split leaks test info
scaler.fit_transform(X)  # Uses all data statistics
X_train, X_test = split(X)

# RIGHT: Scale only on training data
X_train, X_test = split(X)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use train statistics
```

### Overfitting to Validation

```python
# Repeated hyperparameter tuning on same validation set
# leads to overfitting that validation set
# Solution: Use nested cross-validation or holdout test set
```

### Ignoring Calibration

```python
# Tree probabilities aren't always well-calibrated
from sklearn.calibration import CalibratedClassifierCV

# Calibrate using Platt scaling or isotonic regression
calibrated = CalibratedClassifierCV(model, method='isotonic')
calibrated.fit(X_train, y_train)
# Now predict_proba gives calibrated probabilities
```

## When Not to Use Ensembles

Sometimes simpler is better:

- **Small data (<1000 samples)**: Regularized linear models may generalize better
- **Need real-time predictions (<1ms)**: Single tree or linear model
- **Strict interpretability requirements**: Explainable AI regulations may require simpler models
- **Image/text/sequential data**: Deep learning usually wins
- **Online learning needed**: Ensembles typically require batch retraining

## Key Takeaways

- Random Forests: robust, minimal tuning, good baseline
- Gradient Boosting: higher accuracy ceiling, requires tuning, use early stopping
- XGBoost/LightGBM/CatBoost: production-ready implementations with regularization
- Handle imbalanced data with class weights, resampling, or threshold tuning
- Feature engineering helps even for trees: interactions, aggregations, time features
- Use SHAP values for reliable feature importance and model interpretation
- Combine models carefully—stacking adds complexity

## Further Reading

- "A Practical Guide to Tree Based Learning Algorithms" - scikit-learn documentation
- "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016) - Implementation insights
- "Practical Lessons from Predicting Clicks on Ads at Facebook" (2014) - Industry experience with GBDTs
- Kaggle competition solutions - Real-world ensemble strategies

---
*Estimated reading time: 11 minutes*
