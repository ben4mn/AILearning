# Convolutional Networks Explained

## Introduction

When humans look at a photograph, we don't process each pixel independently. We perceive edges, textures, shapes, and ultimately objects. Our visual cortex is organized hierarchically—early areas detect simple features like oriented edges, while later areas respond to complex patterns like faces. Convolutional Neural Networks (CNNs) are designed with similar principles: they learn hierarchical feature representations through layers of local processing, building from simple patterns to complex concepts.

In this lesson, we'll explore the fundamental mechanics of CNNs. We'll understand why convolution is such a powerful operation for processing images, how pooling creates translation invariance, and how stacking these layers creates rich feature hierarchies. By the end, you'll understand not just how CNNs work, but why they work so well for visual recognition tasks.

## The Problem with Fully Connected Networks

Before we appreciate convolution, let's understand why we need it. Consider processing a modest 224x224 color image with a standard fully connected (dense) layer.

```python
# The scale problem with fully connected layers
input_size = 224 * 224 * 3  # 150,528 pixels
hidden_units = 1000

# Number of parameters for ONE dense layer
parameters = input_size * hidden_units  # 150,528,000 parameters!
```

Over 150 million parameters for a single layer! This creates two problems:

1. **Memory and compute**: Training requires astronomical resources
2. **Overfitting**: With so many parameters and limited data, the network memorizes instead of learning

But the deeper problem is conceptual. In a fully connected layer, every output neuron connects to every input pixel. But vision is fundamentally local—whether a pixel is an edge depends on its neighbors, not on pixels across the image. A fully connected layer ignores this locality, treating each pixel independently.

## The Convolution Operation

Convolution exploits the locality of visual features. Instead of connecting to all pixels, each output neuron connects to a small local patch and applies the same operation across the entire image.

```python
import torch
import torch.nn.functional as F

# A 3x3 filter that detects vertical edges
vertical_edge_filter = torch.tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Apply to image: slide filter across and compute dot product at each position
def convolve(image, filter):
    """
    Slide filter across image, computing dot product at each position
    Output shows where the pattern (vertical edges) appears
    """
    return F.conv2d(image, filter)
```

Here's what happens at each position:

1. Place the filter over a small patch (e.g., 3x3 pixels)
2. Multiply corresponding values and sum them (dot product)
3. This scalar becomes one pixel in the output
4. Slide the filter one position and repeat

The same filter weights are used at every position—this is **weight sharing**. A 3x3 filter has only 9 parameters, regardless of image size. A 224x224 image processed by this filter still uses just those 9 parameters.

```python
# Parameter comparison
image_size = 224 * 224 * 3
filter_size = 3 * 3  # Same filter used everywhere

# Fully connected: 150 million parameters
# Convolution: 9 parameters (plus bias)
```

## Why Weight Sharing Works

Weight sharing isn't just a computational trick—it embodies an assumption about visual data called **translation equivariance**: a feature that appears in one location should be detected by the same operation in another location. An edge is an edge whether it's in the top-left or bottom-right.

```python
# Translation equivariance illustrated
original_image = load_image("cat.jpg")
shifted_image = shift_right(original_image, 50)  # Move cat 50 pixels right

# The feature maps also shift by 50 pixels
original_features = conv_layer(original_image)
shifted_features = conv_layer(shifted_image)

# shifted_features ≈ shift_right(original_features, 50)
```

This is exactly the inductive bias we want for vision. A cat detector shouldn't need to learn separate "cat in top-left" and "cat in center" patterns—one learned filter should work everywhere.

## Multiple Filters Learn Different Features

A single filter detects one pattern (say, vertical edges). To detect different patterns, we use multiple filters. Each filter produces one **feature map**, showing where its pattern appears in the input.

```python
import torch.nn as nn

# A convolutional layer with multiple filters
conv_layer = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=64,    # 64 different filters
    kernel_size=3,      # Each filter is 3x3
    padding=1           # Preserve spatial dimensions
)

# Parameters: 64 filters × 3 input channels × 3 × 3 = 1,728 parameters
# Compare to fully connected: millions of parameters
```

The 64 filters learn to detect different features:
- Some detect horizontal edges
- Some detect diagonal edges
- Some detect colors
- Some detect textures

The network learns which features are useful through backpropagation—we don't hand-design them.

## Pooling: Creating Invariance

Convolution provides translation equivariance (features shift with the input), but sometimes we want translation **invariance** (same output regardless of small shifts). Pooling layers provide this.

**Max pooling** takes the maximum value in each local region:

```python
# Max pooling example
input_patch = torch.tensor([
    [1.0, 0.5],
    [0.3, 0.8]
])
max_pooled = torch.max(input_patch)  # = 1.0

# 2x2 max pooling with stride 2
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Input: 224x224 → Output: 112x112 (halved in each dimension)
```

Max pooling achieves several things:

1. **Translation invariance**: Small shifts don't change the maximum
2. **Dimensionality reduction**: Each 2x2 region becomes one value
3. **Feature selection**: Only the strongest activation in each region survives

**Average pooling** takes the mean instead of the maximum, providing smoothing rather than selection.

## Building Feature Hierarchies

The magic of CNNs emerges when we stack convolutional layers. Each layer builds on the previous, creating increasingly abstract representations:

```python
class SimpleHierarchicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Detect edges and simple patterns
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Combine edges into textures and shapes
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: Combine shapes into object parts
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Layer 4: Combine parts into objects
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # Edges
        x = self.pool1(x)
        x = F.relu(self.conv2(x))    # Textures
        x = self.pool2(x)
        x = F.relu(self.conv3(x))    # Parts
        x = self.pool3(x)
        x = F.relu(self.conv4(x))    # Objects
        return x
```

This hierarchical composition is key:
- **Layer 1**: Edges and color gradients
- **Layer 2**: Corners, textures, simple shapes
- **Layer 3**: Object parts (eyes, wheels, windows)
- **Layer 4**: Whole objects (faces, cars, buildings)

Researchers have visualized what each layer learns. Early layers show edge detectors similar to those found in the visual cortex. Later layers show increasingly complex patterns—faces, wheels, architectural elements.

## The Receptive Field

As features propagate through layers, each neuron "sees" an increasingly large region of the original image. This is called the **receptive field**.

```python
# Receptive field growth
# 3x3 conv: Each output neuron sees 3x3 input pixels
# Stack two 3x3 convs: Each output neuron sees 5x5 input pixels
# After pooling and more convs: Receptive field grows further

# Formula (simplified, with stride 1 and no padding):
# Layer 1: RF = 3
# Layer 2: RF = 5
# Layer 3: RF = 7
# With pooling: RF grows even faster
```

By the final layers, each neuron's receptive field might span the entire image. This is how a network can classify an image based on global context—the final neurons integrate information from everywhere, but through a hierarchy of local operations.

## Practical Considerations

Several practical choices affect CNN performance:

**Padding** controls how borders are handled. Without padding, each convolution shrinks the spatial dimensions. With same-padding (e.g., padding=1 for 3x3 kernels), dimensions are preserved:

```python
# No padding: output shrinks
conv_no_pad = nn.Conv2d(3, 64, kernel_size=3, padding=0)
# 224x224 input → 222x222 output

# Same padding: output size preserved
conv_same_pad = nn.Conv2d(3, 64, kernel_size=3, padding=1)
# 224x224 input → 224x224 output
```

**Stride** controls how far the filter moves between positions. Stride > 1 reduces output dimensions, sometimes replacing pooling:

```python
# Stride 2: skip every other position
conv_stride = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
# 224x224 input → 112x112 output
```

**Kernel size** affects the local region considered. 3x3 is most common—two 3x3 layers see the same region as one 5x5 layer but with fewer parameters and more nonlinearity.

## From Features to Classification

CNNs for classification typically end with:
1. Global average pooling (or flattening)
2. One or more fully connected layers
3. Softmax for class probabilities

```python
class ClassificationCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Convolutional backbone (as above)
        self.features = nn.Sequential(
            # ... convolutional layers
        )

        # Global average pooling: one value per channel
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)           # [batch, 256, H, W]
        x = self.gap(x)                # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)      # [batch, 256]
        x = self.classifier(x)         # [batch, num_classes]
        return x
```

Global average pooling is elegant: instead of flattening potentially thousands of spatial values, we average each feature map into a single number. This reduces parameters and provides spatial invariance.

## Why CNNs Work So Well for Vision

The success of CNNs stems from matching the structure of visual data:

1. **Locality**: Visual features are local, and convolution is local
2. **Translation invariance**: Objects can appear anywhere, and weight sharing/pooling handle this
3. **Hierarchy**: Complex patterns compose from simple ones, and stacked layers implement this
4. **Efficiency**: Weight sharing dramatically reduces parameters

These aren't arbitrary design choices—they're inductive biases that match how images work. A network that had to learn these principles from scratch would need far more data.

## Key Takeaways

- Convolution applies the same small filter across the entire image, detecting local patterns while sharing weights (huge parameter reduction)
- Translation equivariance means features shift with the input—a desirable property for vision
- Pooling provides translation invariance and dimensionality reduction, keeping only the strongest features
- Stacking convolutional layers creates feature hierarchies: edges to textures to parts to objects
- The receptive field grows with depth, allowing final layers to integrate global information
- CNNs encode inductive biases matching the structure of visual data, making them remarkably effective for image tasks

## Further Reading

- LeCun, Y., et al. (1998). "Gradient-Based Learning Applied to Document Recognition"
- Zeiler, M., & Fergus, R. (2014). "Visualizing and Understanding Convolutional Networks"
- CS231n: Convolutional Neural Networks for Visual Recognition (Stanford course notes)

---
*Estimated reading time: 12 minutes*
