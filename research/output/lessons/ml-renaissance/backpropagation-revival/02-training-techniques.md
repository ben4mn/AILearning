# Practical Training Techniques: Making Backpropagation Work

## Introduction

The 1986 paper showed that backpropagation could train neural networks in principle. Making it work in practice was another matter. Early practitioners discovered that naive implementations often failed: networks got stuck, training took forever, or results didn't generalize to new data.

The late 1980s and 1990s saw intensive research into the engineering of neural network training. Researchers developed initialization schemes, learning rate strategies, regularization methods, and architectural insights. These techniques transformed backpropagation from a promising idea into a practical tool. Many of these insights remain relevant in today's deep learning systems.

This lesson covers the practical advances that made neural network training reliable.

## Weight Initialization

A network's starting point profoundly affects its training. Initialize weights too large, and activations saturate (sigmoid outputs near 0 or 1 everywhere, with near-zero gradients). Initialize too small, and signals vanish as they propagate through layers.

```python
import numpy as np

# Bad: Random weights with high variance
W = np.random.randn(1000, 1000) * 2.0
# Activations after a few layers: all 0 or 1 (saturated)

# Bad: Random weights too small
W = np.random.randn(1000, 1000) * 0.001
# Activations after a few layers: all ~0.5 (no signal)

# Better: Scale by fan-in
# Weights ~ N(0, 1/n_in)
n_in = 1000
W = np.random.randn(1000, 1000) / np.sqrt(n_in)
```

**Xavier/Glorot initialization** (2010) formalized this insight, recommending weights drawn from a distribution with variance 2/(n_in + n_out). But practitioners in the 1990s had already discovered similar rules of thumb.

### Breaking Symmetry

Another initialization concern: if all weights start identical, all hidden units compute the same thing and receive the same gradients. They remain copies of each other forever. Random initialization breaks this symmetry.

```python
# Bad: All weights the same
W = np.ones((100, 100)) * 0.1
# All hidden units are identical, stay identical forever

# Good: Random initialization
W = np.random.randn(100, 100) * 0.1
# Each hidden unit starts different, learns different features
```

## Learning Rate Selection

The learning rate η controls step size in gradient descent:

**w ← w - η × ∂Loss/∂w**

Too large, and the network overshoots minima, oscillating or diverging. Too small, and training takes impractically long.

```python
# Learning rate too high
# Loss: 10.0 → 15.0 → 22.0 → diverges!

# Learning rate too low
# Loss: 10.0 → 9.99 → 9.98 → 9.97 → ... → (still training after hours)

# Learning rate just right
# Loss: 10.0 → 8.5 → 7.2 → 5.8 → 4.1 → 2.5 → 1.2 → ...
```

### Learning Rate Schedules

Practitioners discovered that starting with a larger learning rate and decreasing it over time worked well:

```python
def step_decay(epoch, initial_lr=0.1):
    """Reduce LR by factor of 10 every 30 epochs."""
    drop = 0.1
    epochs_drop = 30
    return initial_lr * (drop ** (epoch // epochs_drop))

def exponential_decay(epoch, initial_lr=0.1, decay_rate=0.95):
    """Exponential decay each epoch."""
    return initial_lr * (decay_rate ** epoch)

# More aggressive early, more careful later
def warmup_then_decay(epoch, initial_lr=0.1, warmup_epochs=5):
    """Warmup then cosine decay."""
    if epoch < warmup_epochs:
        return initial_lr * (epoch / warmup_epochs)
    else:
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / 100))
```

The intuition: early in training, large steps help explore the loss landscape. Later, smaller steps help settle into a good minimum without overshooting.

## Momentum

Standard gradient descent can oscillate in ravines—narrow valleys where the gradient bounces back and forth across the valley floor while making slow progress along it.

**Momentum** adds a "velocity" term that accumulates gradient direction:

```python
# Standard gradient descent
w = w - learning_rate * gradient

# Gradient descent with momentum
velocity = momentum * velocity - learning_rate * gradient
w = w + velocity
```

Momentum smooths the trajectory, damping oscillations and accelerating progress along consistent gradient directions.

```python
def sgd_with_momentum(weights, gradients, velocity, lr=0.01, momentum=0.9):
    """Update weights using momentum."""
    velocity = momentum * velocity - lr * gradients
    weights = weights + velocity
    return weights, velocity

# Without momentum: zigzag path, slow convergence
# With momentum: smooth path, faster convergence
```

Typical momentum values: 0.9 to 0.99.

### Nesterov Momentum

A refinement: look ahead to where momentum will take you, then compute the gradient there:

```python
def nesterov_momentum(weights, gradients_fn, velocity, lr=0.01, momentum=0.9):
    """Nesterov accelerated gradient."""
    # Look ahead
    weights_ahead = weights + momentum * velocity

    # Compute gradient at look-ahead position
    gradients = gradients_fn(weights_ahead)

    # Update
    velocity = momentum * velocity - lr * gradients
    weights = weights + velocity
    return weights, velocity
```

Nesterov momentum often converges faster because it corrects for the "overshoot" before it happens.

## Batch vs. Stochastic Gradient Descent

Computing the gradient over the entire dataset (batch gradient descent) is computationally expensive and provides no additional gradient information during an epoch.

**Stochastic gradient descent (SGD)** computes gradients from single examples:

```python
# Batch gradient descent
for epoch in range(epochs):
    gradient = compute_gradient(entire_dataset)  # Expensive!
    weights -= learning_rate * gradient

# Stochastic gradient descent
for epoch in range(epochs):
    for example in shuffle(dataset):
        gradient = compute_gradient(example)  # Fast, noisy
        weights -= learning_rate * gradient
```

SGD is noisier (each gradient is a noisy estimate of the true gradient) but much faster per update and provides a form of implicit regularization.

### Mini-batch SGD

The practical compromise: compute gradients over small batches of examples:

```python
# Mini-batch gradient descent
batch_size = 32
for epoch in range(epochs):
    for batch in get_batches(dataset, batch_size):
        gradient = compute_gradient(batch)  # Good balance
        weights -= learning_rate * gradient
```

Mini-batches:
- Are more computationally efficient than single examples (GPU parallelism)
- Provide more stable gradients than single examples
- Still allow many updates per epoch

Typical batch sizes: 32, 64, 128, 256.

## Regularization: Fighting Overfitting

Neural networks with many parameters can easily memorize training data without generalizing. Regularization techniques prevent overfitting.

### Weight Decay (L2 Regularization)

Add a penalty for large weights to the loss function:

**Loss_total = Loss_data + λ × Σ w²**

```python
def compute_loss_with_regularization(predictions, targets, weights, lambda_reg=0.001):
    """Loss with L2 regularization."""
    data_loss = np.mean((predictions - targets) ** 2)
    reg_loss = lambda_reg * np.sum(weights ** 2)
    return data_loss + reg_loss

# Gradient includes regularization term
# ∂Loss/∂w = ∂Loss_data/∂w + 2λw
```

Weight decay discourages large weights, preferring smoother functions that generalize better.

### Early Stopping

The simplest and most effective regularization: stop training when validation error starts increasing.

```python
best_val_loss = float('inf')
patience = 10
no_improvement_count = 0

for epoch in range(1000):
    train(model)
    val_loss = evaluate(model, validation_set)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model)  # Save best
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= patience:
        print("Early stopping!")
        break

load_model("best")  # Use best model, not final model
```

Early stopping uses the validation set as an implicit regularizer, preventing the network from overfitting to training data.

### Dropout (Later Innovation)

Introduced by Hinton et al. in 2012, but foreshadowed by earlier work. Randomly drop hidden units during training:

```python
def forward_with_dropout(x, W, dropout_rate=0.5, training=True):
    """Forward pass with dropout."""
    h = sigmoid(x @ W)

    if training:
        # Randomly zero out neurons
        mask = np.random.binomial(1, 1 - dropout_rate, h.shape)
        h = h * mask / (1 - dropout_rate)  # Scale to maintain expected value

    return h
```

Dropout prevents hidden units from co-adapting—each unit must be useful on its own.

## The Importance of Data

Perhaps the biggest practical lesson: more training data helps more than any algorithmic trick.

```python
# Rough relationship (before deep learning era):
# 100 examples: severe overfitting, poor results
# 1,000 examples: moderate overfitting, decent results
# 10,000 examples: mild overfitting, good results
# 100,000+ examples: minimal overfitting, great results
```

This drove interest in data augmentation—creating additional training examples through transformations:

```python
def augment_image(image):
    """Generate augmented versions of an image."""
    augmented = []
    augmented.append(image)
    augmented.append(np.fliplr(image))  # Horizontal flip
    augmented.append(rotate(image, angle=5))  # Small rotation
    augmented.append(crop_and_scale(image))  # Random crop
    augmented.append(image + noise)  # Add noise
    return augmented
```

## Validation and Cross-Validation

Holding out test data isn't enough—you also need validation data to tune hyperparameters without contaminating the test set:

```python
# Split data
train, validation, test = split_data(data, [0.7, 0.15, 0.15])

# Use validation for:
# - Early stopping
# - Learning rate selection
# - Architecture selection
# - Regularization tuning

# Use test only for:
# - Final evaluation (once!)
```

Cross-validation provided more reliable estimates with limited data:

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Key Takeaways

- Weight initialization must balance activation ranges across layers
- Learning rate selection is critical; schedules help (start high, decay)
- Momentum accelerates training and stabilizes gradients
- Mini-batch SGD provides the best tradeoff between speed and gradient quality
- Regularization (weight decay, early stopping) prevents overfitting
- More training data is the most reliable path to better generalization

## Further Reading

- LeCun et al. "Efficient BackProp" (1998) - Comprehensive practical guide
- Glorot and Bengio. "Understanding the difficulty of training deep feedforward neural networks" (2010) - Initialization
- Bottou, Léon. "Large-Scale Machine Learning with Stochastic Gradient Descent" (2010) - SGD theory
- Ruder, Sebastian. "An overview of gradient descent optimization algorithms" (2016) - Modern review

---
*Estimated reading time: 11 minutes*
