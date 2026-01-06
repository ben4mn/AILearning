# Architecture Evolution: From VGG to ResNet

## Introduction

AlexNet proved that deep convolutional networks could achieve breakthrough results. But was 8 layers optimal? Were 11x11 filters the right size? In the years following 2012, researchers systematically explored the design space of CNN architectures, pushing deeper, trying different filter sizes, and ultimately discovering fundamental principles about how to build very deep networks.

In this lesson, we'll trace the evolution of CNN architectures from AlexNet through VGG, GoogLeNet, and ResNet. Each architecture taught the field something new about what makes deep networks work. By the end, we'll understand how the problem of building deep networks was progressively solved, culminating in architectures that can reliably train with hundreds of layers.

## VGG: The Power of Simplicity and Depth

In 2014, the Visual Geometry Group at Oxford published VGGNet, an architecture that asked a simple question: what if we just made the network deeper and simpler?

AlexNet used a variety of filter sizes: 11x11, 5x5, 3x3. VGG standardized on a single choice: **3x3 filters everywhere**. The insight was that two 3x3 layers have the same receptive field as one 5x5 layer, but with fewer parameters and more nonlinearity:

```python
# Two 3x3 convolutions vs one 5x5
# Receptive field: same (5x5)
# Parameters (per output channel):
#   5x5: 25 parameters
#   3x3 + 3x3: 9 + 9 = 18 parameters
# Nonlinearities:
#   5x5: 1 ReLU
#   3x3 + 3x3: 2 ReLUs

# VGG block: repeated 3x3 convolutions
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels, kernel_size=3, padding=1
            ))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
```

VGG came in several variants:
- VGG-11: 11 layers (8 conv + 3 FC)
- VGG-16: 16 layers (13 conv + 3 FC)
- VGG-19: 19 layers (16 conv + 3 FC)

VGG-16 achieved 7.3% top-5 error on ImageNet, substantially improving on AlexNet's 15.3%. The architecture was elegant and easy to understand—just stack 3x3 convolutions with occasional pooling.

```python
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64 filters
            *self._make_block(3, 64, 2),
            # Block 2: 128 filters
            *self._make_block(64, 128, 2),
            # Block 3: 256 filters
            *self._make_block(128, 256, 3),
            # Block 4: 512 filters
            *self._make_block(256, 512, 3),
            # Block 5: 512 filters
            *self._make_block(512, 512, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
```

VGG's legacy includes:
- Demonstrating that depth matters more than filter complexity
- Providing a simple, reusable architecture pattern
- Becoming a standard pretrained feature extractor for transfer learning

The downside: VGG was expensive. VGG-16 has 138 million parameters, most in the fully connected layers, making it memory-hungry and slow.

## GoogLeNet: Efficiency Through Inception

While VGG pushed depth, Google's GoogLeNet (also 2014) asked: how can we build deep networks more efficiently?

The key innovation was the **Inception module**, which applied multiple filter sizes in parallel and concatenated the results:

```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3,
                 ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        # Path 1: 1x1 convolution
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # Path 2: 1x1 then 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )

        # Path 3: 1x1 then 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)
        )

        # Path 4: max pool then 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)  # Concatenate along channel dimension
```

The intuition: different image regions benefit from different receptive field sizes. Rather than choosing one filter size, let the network have all of them and learn which to use.

Key efficiency tricks:
- **1x1 convolutions for dimensionality reduction**: Before expensive 3x3 or 5x5 filters, a 1x1 conv reduces channel count
- **No fully connected layers at the end**: Global average pooling replaced the heavy FC layers
- **Auxiliary classifiers**: Training signals injected at intermediate layers to combat vanishing gradients

GoogLeNet achieved 6.7% top-5 error with only 5 million parameters—30x smaller than VGG while being more accurate.

## The Depth Problem Resurfaces

VGG and GoogLeNet pushed to 19 and 22 layers respectively. What about 50 layers? 100 layers?

Researchers tried, and encountered a puzzling problem. Beyond about 20 layers, adding more layers made accuracy worse—not from overfitting, but from **degradation**. Training error increased with depth.

```
# Observation circa 2014-2015
Depth | Training Error | Test Error
20    | 3.5%          | 6.0%
30    | 4.0%          | 6.5%
50    | 6.0%          | 8.0%

# Adding layers made training worse!
```

This couldn't be explained by vanishing gradients alone (batch normalization helped with that). Something else was limiting depth.

The problem: it's hard to learn an identity mapping. If the optimal function for some layer is to do nothing (pass inputs through unchanged), a standard layer must learn weights that approximate identity. This is difficult and introduces unnecessary optimization challenges.

## ResNet: Skip Connections to the Rescue

In 2015, Microsoft Research's Kaiming He and colleagues introduced **Residual Networks (ResNets)**, solving the depth problem with a elegantly simple idea: skip connections.

Instead of learning the underlying mapping H(x), each layer learns the residual F(x) = H(x) - x. The layer's output is F(x) + x:

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x  # The skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # Add the skip connection
        out = F.relu(out)

        return out
```

Why does this work?

1. **Identity is easy**: If the optimal transformation is identity, the layer just needs to learn F(x) = 0, which is easy (weights → 0)

2. **Gradient flow**: The skip connection provides a direct path for gradients. Even if the layer's contribution is small, gradients can flow through.

3. **Ensemble interpretation**: A ResNet can be viewed as an ensemble of shallower networks (paths that skip various layers)

```python
# ResNet gradient flow (simplified)
# With skip connection: gradient = d_loss/d_output * (1 + d_F/d_x)
# The "1" from the skip connection prevents vanishing

# Without skip connection: gradient = d_loss/d_output * d_F/d_x
# If d_F/d_x < 1, gradient vanishes exponentially with depth
```

## ResNet Results and Variants

ResNet achieved stunning results:

| Model | Layers | Top-5 Error | Parameters |
|-------|--------|-------------|------------|
| ResNet-34 | 34 | 5.71% | 21M |
| ResNet-50 | 50 | 5.25% | 25M |
| ResNet-101 | 101 | 4.60% | 44M |
| ResNet-152 | 152 | 4.49% | 60M |

For the first time, networks could be trained with 100+ layers, and more depth consistently improved results.

The **bottleneck design** made deeper networks practical:

```python
class BottleneckBlock(nn.Module):
    """
    For deeper networks: reduce dimension, convolve, restore dimension
    Reduces computation in the expensive 3x3 convolution
    """
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection (projection if dimensions change)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                    if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += residual
        return F.relu(out)
```

## After ResNet: Continued Evolution

ResNet opened the floodgates to very deep architectures:

**DenseNet (2017)**: Instead of adding skip connections, concatenate all previous layer outputs. Every layer receives input from all preceding layers.

```python
# DenseNet: dense connections
# Layer n receives: [layer_0, layer_1, ..., layer_{n-1}]
# Feature reuse and gradient flow maximized
```

**ResNeXt (2017)**: ResNet meets Inception—use grouped convolutions to increase width efficiently.

**EfficientNet (2019)**: Systematically scale depth, width, and resolution together using neural architecture search.

**Vision Transformers (2020s)**: Transformers from NLP applied to vision, challenging CNNs' dominance (we'll cover these later).

## Design Principles Learned

The architecture evolution taught fundamental principles:

1. **Depth enables abstraction**: Deeper networks learn richer hierarchies
2. **Skip connections enable depth**: Direct paths for gradients are essential
3. **Batch normalization stabilizes training**: Nearly universal in modern networks
4. **1x1 convolutions are powerful**: Cheap way to change channel dimensions
5. **Global average pooling beats FC**: Fewer parameters, built-in spatial invariance
6. **Wider and deeper both help**: But depth is often more parameter-efficient

```python
# Modern CNN recipe (simplified)
def modern_cnn_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        # ... possibly more conv/bn/relu
        # Skip connection around the block
    )
```

## Key Takeaways

- VGG (2014) showed that uniform 3x3 filters and greater depth outperform complex architectures—simplicity and depth beat sophistication
- GoogLeNet/Inception (2014) introduced parallel filter branches and 1x1 convolutions for efficiency, achieving better accuracy with far fewer parameters
- ResNet (2015) solved the degradation problem with skip connections, enabling training of 100+ layer networks
- The residual formulation makes identity mappings easy to learn and enables unimpeded gradient flow
- These architectures established design principles still used today: batch normalization, skip connections, and global average pooling

## Further Reading

- Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG)
- Szegedy, C., et al. (2015). "Going Deeper with Convolutions" (GoogLeNet)
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition" (ResNet)
- He, K., et al. (2016). "Identity Mappings in Deep Residual Networks" (ResNet improvements)

---
*Estimated reading time: 12 minutes*
