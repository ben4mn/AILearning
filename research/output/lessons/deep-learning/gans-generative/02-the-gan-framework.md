# The GAN Framework: Generator vs Discriminator

## Introduction

In 2014, Ian Goodfellow introduced an idea that would reshape generative modeling. Instead of explicitly modeling the data distribution, what if two neural networks played a game? One network tries to generate fake data that looks real; the other tries to distinguish real data from fake. As they compete, both get better—and the generator eventually produces samples indistinguishable from real data.

This adversarial framework, called the Generative Adversarial Network (GAN), produced images of unprecedented quality. It also introduced training challenges and instabilities that would occupy researchers for years. In this lesson, we'll understand the GAN framework, its mathematical foundations, and why the adversarial approach proved so powerful.

## The Adversarial Game

A GAN consists of two networks:

**Generator (G)**: Takes random noise z and produces a sample G(z).
- Goal: Generate samples that fool the discriminator
- Input: Random vector z ~ N(0, I)
- Output: Fake data x_fake = G(z)

**Discriminator (D)**: Takes a sample and outputs the probability it's real.
- Goal: Correctly classify real vs. fake samples
- Input: Data sample x
- Output: D(x) = probability x is real

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        return self.net(x)
```

## The Training Objective

The GAN objective is a minimax game:

```
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

In English:
- **D wants to maximize**: Correctly identify real samples (high D(x)) and fake samples (low D(G(z)))
- **G wants to minimize**: Make D(G(z)) close to 1 (fool the discriminator)

```python
def train_step(generator, discriminator, real_data, latent_dim, d_optimizer, g_optimizer):
    batch_size = real_data.size(0)

    # === Train Discriminator ===
    d_optimizer.zero_grad()

    # Real data
    real_labels = torch.ones(batch_size, 1)
    real_output = discriminator(real_data)
    d_loss_real = F.binary_cross_entropy(real_output, real_labels)

    # Fake data
    z = torch.randn(batch_size, latent_dim)
    fake_data = generator(z).detach()  # Don't update G yet
    fake_labels = torch.zeros(batch_size, 1)
    fake_output = discriminator(fake_data)
    d_loss_fake = F.binary_cross_entropy(fake_output, fake_labels)

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    # === Train Generator ===
    g_optimizer.zero_grad()

    z = torch.randn(batch_size, latent_dim)
    fake_data = generator(z)
    fake_output = discriminator(fake_data)

    # Generator wants discriminator to think fake is real
    g_loss = F.binary_cross_entropy(fake_output, real_labels)  # Note: real_labels!
    g_loss.backward()
    g_optimizer.step()

    return d_loss.item(), g_loss.item()
```

## The Non-Saturating Generator Loss

The original generator objective has a problem:

```python
# Original: G minimizes log(1 - D(G(z)))
# When D is good at rejecting fakes, D(G(z)) ≈ 0
# log(1 - 0) = log(1) = 0 → no gradient!
```

In early training, D easily rejects poor fakes, giving G almost no learning signal.

The fix: instead of minimizing log(1 - D(G(z))), maximize log(D(G(z))):

```python
# Non-saturating objective
# G maximizes log(D(G(z)))
# When D(G(z)) is small, log(D(G(z))) is very negative → strong gradient
g_loss = -torch.log(discriminator(fake_data)).mean()
```

This is mathematically different but provides the same gradient direction with stronger signal.

## Nash Equilibrium

At the optimal solution, the generator produces samples from the true data distribution:

```
G* produces samples from p_data
D* outputs 1/2 for all samples (can't distinguish real from fake)
```

This is a Nash equilibrium: neither player can improve by changing strategy unilaterally.

In practice, reaching this equilibrium is difficult. Training often oscillates, diverges, or gets stuck.

## Why GANs Work

The adversarial setup provides several advantages:

**1. Implicit Distribution**: No need to define an explicit density function. The generator implicitly represents the distribution through its samples.

**2. Powerful Critic**: The discriminator provides a rich loss signal—not just "is this image good?" but a learned, adaptive quality measure.

**3. Direct Optimization for Perceptual Quality**: Unlike VAE reconstruction loss (pixel-wise), GANs optimize for discriminator fooling, which correlates with human perception.

**4. No Likelihood Computation**: For complex data like images, computing exact likelihoods is intractable. GANs sidestep this entirely.

## The Mode Collapse Problem

The most notorious GAN failure mode is **mode collapse**: the generator finds a few samples that fool the discriminator and produces only those.

```
Training data: Faces of many ages, ethnicities, expressions
Mode-collapsed GAN: Generates only young white female faces

Why? These fooled the discriminator at some point, and G got "stuck"
```

```python
# Detection: Generate many samples, check diversity
samples = [generator(torch.randn(latent_dim)) for _ in range(1000)]
unique_samples = cluster_and_count(samples)

if unique_samples < expected:
    print("Warning: Possible mode collapse")
```

Mode collapse happens because:
- G has no incentive for diversity—just for fooling D
- If G finds one good "answer," it exploits it
- D can't push back effectively once overwhelmed

## Training Instabilities

GANs are notoriously hard to train:

```python
training_failures = {
    'mode_collapse': "G produces limited variety",
    'oscillation': "D and G quality fluctuate wildly",
    'vanishing_gradients': "D becomes too good, G gets no signal",
    'divergence': "One network dominates, training fails",
    'non_convergence': "Metrics oscillate, never stabilize",
}
```

Practitioners developed heuristics:
- **Balance D and G training**: Sometimes train D more steps per G step
- **Label smoothing**: Use 0.9 instead of 1.0 for real labels
- **Instance noise**: Add noise to inputs to prevent D from overfitting
- **Spectral normalization**: Constrain D's weights

## Alternative Loss Functions

The original GAN loss led to various improvements:

**Wasserstein GAN (WGAN)**:
Replace classification loss with Wasserstein distance.

```python
# WGAN critic (not discriminator—no sigmoid)
def wgan_d_loss(real_output, fake_output):
    return fake_output.mean() - real_output.mean()

def wgan_g_loss(fake_output):
    return -fake_output.mean()

# Requires weight clipping or gradient penalty
```

WGAN provides smoother gradients and more meaningful loss curves.

**Least Squares GAN (LSGAN)**:
Use squared error instead of cross-entropy.

```python
def lsgan_d_loss(real_output, fake_output):
    return ((real_output - 1)**2).mean() + (fake_output**2).mean()

def lsgan_g_loss(fake_output):
    return ((fake_output - 1)**2).mean()
```

More stable training, reduced mode collapse.

**Hinge Loss**:

```python
def hinge_d_loss(real_output, fake_output):
    return F.relu(1 - real_output).mean() + F.relu(1 + fake_output).mean()

def hinge_g_loss(fake_output):
    return -fake_output.mean()
```

Used in many modern architectures (BigGAN, StyleGAN).

## Evaluating GANs

Common evaluation metrics:

**Inception Score (IS)**:

```python
# Higher is better
# Measures: (1) Each sample looks like a clear class
#           (2) Overall diversity across classes
IS = exp(E[KL(p(y|x) || p(y))])
```

**Frechet Inception Distance (FID)**:

```python
# Lower is better
# Compare feature distributions of real and generated samples
def fid(real_features, fake_features):
    mu_real, sigma_real = mean_and_cov(real_features)
    mu_fake, sigma_fake = mean_and_cov(fake_features)

    diff = mu_real - mu_fake
    return diff @ diff + trace(sigma_real + sigma_fake - 2*sqrt(sigma_real @ sigma_fake))
```

FID became the standard for image quality evaluation.

## The GAN Impact

GANs transformed generative modeling:

**Before GANs (2014)**:
- Blurry generated images
- Visible artifacts
- Clearly distinguishable from real

**After GANs matured (2019+)**:
- Photorealistic faces
- Coherent, detailed images
- Often indistinguishable from real (by humans and classifiers)

This quality leap had both exciting and concerning implications.

## Key Takeaways

- GANs consist of a generator (produces samples) and discriminator (classifies real vs fake) that train adversarially
- The minimax objective pushes the generator toward producing samples from the true data distribution
- The non-saturating generator loss (maximizing log(D(G(z)))) provides better gradients than the original formulation
- Mode collapse—generating limited variety—is the most common failure mode
- Training instabilities led to many techniques: WGAN, spectral normalization, careful hyperparameter tuning
- FID became the standard evaluation metric, comparing feature distributions of real and generated samples

## Further Reading

- Goodfellow, I., et al. (2014). "Generative Adversarial Nets"
- Arjovsky, M., et al. (2017). "Wasserstein GAN"
- Miyato, T., et al. (2018). "Spectral Normalization for Generative Adversarial Networks"
- Lucic, M., et al. (2018). "Are GANs Created Equal? A Large-Scale Study"

---
*Estimated reading time: 11 minutes*
