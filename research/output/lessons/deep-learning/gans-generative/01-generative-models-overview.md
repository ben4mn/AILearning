# Generative Models Overview

## Introduction

Most of the deep learning we've explored so far has been discriminative: given an input, predict an output. Classification asks "is this a cat or a dog?" Regression asks "what's the house price?" But there's another way to think about learning: instead of classifying data, learn to generate it. What distribution produced these images? Can we sample new images from that distribution?

Generative models aim to capture the underlying probability distribution of data. Once learned, this distribution can be sampled to create new examples: new faces, new music, new text. This capability has profound implications—from artistic creation to data augmentation to understanding the structure of data itself.

In this lesson, we'll explore the landscape of generative models, understand what makes them different from discriminative models, and see why this distinction matters for both theory and applications.

## Discriminative vs Generative

The distinction is fundamental:

```python
# Discriminative model: P(y|x)
# "Given this image x, what's the probability it's class y?"
class DiscriminativeClassifier:
    def forward(self, x):
        return P(y | x)  # Direct mapping to class probabilities

# Generative model: P(x) or P(x|z)
# "What's the probability of this data point?" or
# "Given latent code z, generate data x"
class GenerativeModel:
    def forward(self, z):
        return x  # Generate a new sample
```

Discriminative models draw decision boundaries. Generative models model the data distribution itself.

Consider handwritten digits:
- **Discriminative**: "This image is probably a 7" (classification)
- **Generative**: "This is what a typical 7 looks like" (generation)

The generative approach is harder—you need to model all the variation in the data, not just what separates classes.

## Why Generate?

Generative models enable capabilities beyond classification:

**Data Augmentation**: Generate synthetic training data when real data is scarce or expensive.

```python
# Augment medical imaging dataset
real_images = load_medical_scans()  # Only 100 samples
synthetic_images = generative_model.sample(1000)  # Generate 1000 more
training_data = real_images + synthetic_images
```

**Creativity and Art**: Generate novel images, music, or text that humans find interesting.

**Anomaly Detection**: Learn what "normal" looks like; detect deviations.

```python
def is_anomaly(x, generative_model):
    probability = generative_model.log_prob(x)
    return probability < threshold  # Low probability = anomaly
```

**Representation Learning**: The latent space of generative models often captures meaningful features.

**Simulation and Prediction**: Generate possible futures for planning and decision-making.

## Types of Generative Models

Several approaches to generative modeling emerged:

### Explicit Density Models

Model P(x) directly with a tractable density function.

**Autoregressive Models**: Factor P(x) into a product of conditionals.

```python
# P(x) = P(x_1) * P(x_2|x_1) * P(x_3|x_1,x_2) * ...
# Used in: PixelRNN, PixelCNN, GPT

def autoregressive_likelihood(x, model):
    log_prob = 0
    for i in range(len(x)):
        log_prob += log(model.predict(x[:i], x[i]))
    return log_prob
```

Generation is sequential: generate x_1, then x_2 given x_1, etc.

**Flow-based Models**: Learn invertible transformations from simple distributions.

```python
# z ~ N(0, I)  (simple distribution)
# x = f(z)     (invertible transformation)
# P(x) = P(z) * |det(df^{-1}/dx)|

class NormalizingFlow:
    def __init__(self):
        self.transforms = [InvertibleLayer() for _ in range(n)]

    def forward(self, z):
        x = z
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse(self, x):
        z = x
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z
```

Exact likelihood computation but limited expressiveness per layer.

### Implicit Density Models

Don't compute P(x) directly—just learn to sample from it.

**Generative Adversarial Networks (GANs)**: Learn through adversarial training.

```python
# Generator: z → x
# Discriminator: x → [real or fake?]
# No explicit density, but can sample realistic x

class GAN:
    def sample(self, n):
        z = torch.randn(n, latent_dim)
        x = self.generator(z)
        return x

    # No log_prob method—density is implicit
```

GANs produce high-quality samples but can't evaluate likelihood.

### Approximate Density Models

Compute approximate or lower-bound likelihoods.

**Variational Autoencoders (VAEs)**: Encode to latent space, decode with reconstruction loss.

```python
class VAE:
    def encode(self, x):
        # Return mean and variance of approximate posterior
        return mu, log_var

    def decode(self, z):
        return reconstructed_x

    def sample(self):
        z = torch.randn(latent_dim)
        return self.decode(z)

    def elbo(self, x):
        # Evidence lower bound on log P(x)
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        recon = self.decode(z)

        recon_loss = reconstruction_error(x, recon)
        kl_loss = kl_divergence(mu, log_var)

        return -recon_loss - kl_loss
```

VAEs provide both sampling and (approximate) density estimation.

## Trade-offs Among Approaches

Each approach has strengths and weaknesses:

| Model Type | Sample Quality | Density Estimation | Training | Diversity |
|------------|---------------|-------------------|----------|-----------|
| Autoregressive | Good | Exact | Slow | High |
| Flow-based | Medium | Exact | Fast | Medium |
| VAE | Medium | Approximate | Fast | High |
| GAN | Excellent | None | Tricky | Mode collapse risk |

GANs historically produced the sharpest images but lacked density estimation. Autoregressive models (like GPT) produce high-quality text but generate slowly. VAEs train reliably but produce blurry images. Research continues to address these trade-offs.

## The Latent Space

Most generative models involve a latent space—a lower-dimensional representation from which data is generated:

```python
# Latent space operations
def interpolate(z1, z2, alpha):
    """Smoothly interpolate between two latent codes"""
    return (1 - alpha) * z1 + alpha * z2

def arithmetic(z_man, z_woman, z_king):
    """Vector arithmetic in latent space"""
    z_queen = z_king - z_man + z_woman
    return z_queen

# Generate faces along a path
z1 = encode_face(image1)
z2 = encode_face(image2)
for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    z_interp = interpolate(z1, z2, alpha)
    face = decode(z_interp)
    display(face)
```

A well-learned latent space is:
- **Smooth**: Nearby points generate similar outputs
- **Disentangled**: Different dimensions capture different factors (pose, color, expression)
- **Complete**: All points in the space map to valid outputs

## The Challenge of Evaluation

Evaluating generative models is notoriously difficult:

```python
# For classification: accuracy, precision, recall (clear metrics)
# For generation: ???

evaluation_challenges = {
    'no_ground_truth': "What's the 'right' generated image?",
    'subjective_quality': "Is this face realistic? (asks humans)",
    'diversity_vs_quality': "Many mediocre samples or few great ones?",
    'likelihood_vs_quality': "High likelihood ≠ good samples",
}
```

Common metrics emerged:
- **Inception Score (IS)**: Quality and diversity via pretrained classifier
- **Frechet Inception Distance (FID)**: Compare real and generated feature distributions
- **Human evaluation**: Ask people to judge realism

None are perfect. A model can have good FID but produce unrealistic samples in edge cases.

## Historical Context

Generative modeling has a long history:

- **1980s**: Boltzmann machines (Hinton)
- **2006**: Deep Belief Networks, layer-wise pretraining (Hinton)
- **2013**: Variational Autoencoders (Kingma & Welling)
- **2014**: GANs (Goodfellow et al.) - our next focus
- **2015+**: DCGAN, StyleGAN, and rapid progress
- **2020+**: Diffusion models (DALL-E 2, Stable Diffusion)

The 2010s saw an explosion in generative model quality, with GANs leading the way for images.

## Key Takeaways

- Discriminative models learn P(y|x) to classify; generative models learn P(x) to generate new samples from the data distribution
- Generative capabilities enable data augmentation, creative applications, anomaly detection, and representation learning
- Major approaches include autoregressive models (exact density, slow sampling), flows (exact density, invertible), VAEs (approximate density, fast), and GANs (no density, high quality)
- The latent space in generative models encodes meaningful structure—interpolation and arithmetic often work
- Evaluating generative models is challenging; metrics like FID help but don't capture everything humans care about

## Further Reading

- Goodfellow, I. (2016). "NIPS Tutorial: Generative Adversarial Networks"
- Kingma, D., & Welling, M. (2014). "Auto-Encoding Variational Bayes" (VAE)
- Oord, A. et al. (2016). "Pixel Recurrent Neural Networks"
- Dinh, L., et al. (2017). "Density estimation using Real-NVP" (Flows)

---
*Estimated reading time: 10 minutes*
