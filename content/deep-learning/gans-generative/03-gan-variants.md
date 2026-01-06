# GAN Variants: From DCGAN to StyleGAN

## Introduction

The original GAN paper used simple fully connected networks and generated blurry, low-resolution images. Within a few years, researchers developed architectural innovations that pushed GANs from curiosity to production-ready technology. DCGAN introduced convolutional architectures. Progressive GAN trained at increasing resolutions. StyleGAN revolutionized control over generated outputs.

In this lesson, we'll trace the evolution of GAN architectures, understanding the key innovations that enabled photorealistic image generation. Each advance addressed specific limitations of previous approaches, cumulatively building toward the stunning results we see today.

## DCGAN: Deep Convolutional GANs (2015)

Radford, Metz, and Chintala's DCGAN paper established architectural guidelines that became standard:

```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, feature_maps=64):
        super().__init__()
        # Start from latent vector, progressively upsample
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # 4x4

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 8x8

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 16x16

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 32x32

            nn.ConvTranspose2d(feature_maps, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 64x64 output
        )

    def forward(self, z):
        return self.main(z.view(z.size(0), -1, 1, 1))


class DCGANDiscriminator(nn.Module):
    def __init__(self, feature_maps=64):
        super().__init__()
        # Mirror of generator: downsample to scalar
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # ... continue downsampling

            # Final: 4x4 -> 1x1
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)
```

DCGAN guidelines:
- **No pooling**: Use strided convolutions for up/downsampling
- **Batch normalization**: In both G and D (except G output, D input)
- **No fully connected layers**: Except maybe at start/end
- **ReLU in G**: LeakyReLU in D
- **Tanh output in G**: Normalize images to [-1, 1]

These guidelines made training more stable and results more consistent.

## Conditional GANs (cGAN)

Standard GANs generate random samples. Conditional GANs add control:

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Concatenate latent and label embedding
        self.main = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),
            # ... rest of generator
        )

    def forward(self, z, labels):
        label_embed = self.label_embedding(labels)
        combined = torch.cat([z, label_embed], dim=1)
        return self.main(combined)

# Generate a specific digit
z = torch.randn(1, latent_dim)
label = torch.tensor([7])  # Generate a "7"
image = generator(z, label)
```

Applications:
- **Image-to-image translation**: pix2pix (sketches → photos)
- **Super-resolution**: SRGAN (low-res → high-res)
- **Class-conditional**: Generate specific categories

## pix2pix: Image Translation (2016)

Isola et al.'s pix2pix learns mappings between image domains:

```
Input: Sketch of a shoe
Output: Photorealistic shoe image

Input: Satellite map
Output: Street map

Input: Edge detection
Output: Original photo
```

Key innovations:
- **Paired training data**: Need corresponding input-output pairs
- **U-Net generator**: Skip connections preserve spatial details
- **PatchGAN discriminator**: Classify NxN patches, not whole image

```python
# PatchGAN: Output is grid of real/fake probabilities
# Each patch classifies whether that region looks real
class PatchDiscriminator(nn.Module):
    def forward(self, x):
        # Returns (batch, 1, H/16, W/16) or similar
        # Each spatial location is a real/fake prediction
        pass
```

## CycleGAN: Unpaired Translation (2017)

What if you don't have paired training data?

```
I want to convert horses → zebras
But I don't have matching horse-zebra image pairs
```

CycleGAN uses **cycle consistency**:

```python
# Two generators: G_AB (horse → zebra), G_BA (zebra → horse)
# Two discriminators: D_A (is it a real horse?), D_B (is it a real zebra?)

def cycle_consistency_loss(G_AB, G_BA, real_A, real_B):
    # A → B → A should reconstruct A
    fake_B = G_AB(real_A)
    reconstructed_A = G_BA(fake_B)
    loss_A = L1(real_A, reconstructed_A)

    # B → A → B should reconstruct B
    fake_A = G_BA(real_B)
    reconstructed_B = G_AB(fake_A)
    loss_B = L1(real_B, reconstructed_B)

    return loss_A + loss_B
```

The insight: if G_AB and G_BA are inverses, the mapping is meaningful.

Applications: style transfer, season change, art style conversion.

## Progressive GAN (2017)

Karras et al. at NVIDIA addressed high-resolution generation:

```
Problem: Training 1024x1024 directly is very unstable
Solution: Start small, progressively add layers
```

```python
# Training phases:
# Phase 1: Train at 4x4
# Phase 2: Add layers for 8x8, blend smoothly
# Phase 3: Add layers for 16x16, blend smoothly
# ... continue to 1024x1024

def progressive_training():
    for resolution in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        # Add new layers for higher resolution
        generator.grow()
        discriminator.grow()

        # Smoothly blend in new layers
        for alpha in range(0, 1, step=0.001):
            # alpha=0: use only old layers
            # alpha=1: use only new layers
            output = alpha * new_output + (1-alpha) * upsampled_old_output
```

Progressive GAN was the first to generate photorealistic 1024x1024 faces.

## StyleGAN: Style-Based Generation (2018)

Karras et al.'s StyleGAN revolutionized control over generation:

**Key innovation**: Inject style at multiple scales through "adaptive instance normalization"

```python
class StyleGenerator(nn.Module):
    def __init__(self):
        # Mapping network: z → w (intermediate latent space)
        self.mapping = MappingNetwork(latent_dim=512, layers=8)

        # Synthesis network with style injection
        self.synthesis = SynthesisNetwork()

    def forward(self, z):
        # Map to intermediate space
        w = self.mapping(z)

        # Inject style at each layer
        # w controls different attributes at different resolutions
        # Early layers: pose, face shape
        # Middle layers: features, hairstyle
        # Late layers: colors, fine details
        return self.synthesis(w)
```

The **W space** (intermediate latent space) is more disentangled:
- Moving in Z space changes many attributes unpredictably
- Moving in W space changes attributes more independently

**Style mixing**: Use different w vectors at different layers:

```python
# Create hybrid: pose from face1, details from face2
w1 = mapping(z1)
w2 = mapping(z2)

# Early layers use w1 (controls coarse features)
# Late layers use w2 (controls fine features)
hybrid = synthesis(early=w1, late=w2)
```

## StyleGAN2 and StyleGAN3 (2019, 2021)

Continued refinements:
- **StyleGAN2**: Removed artifacts (water droplet effect), improved quality
- **StyleGAN3**: Alias-free generation, better video synthesis

```python
# StyleGAN3: Equivariant to translation and rotation
# Features in intermediate layers maintain consistent relationship to output
# Enables smooth video generation without "texture sticking"
```

## BigGAN: Scaling Up (2018)

Brock et al. showed that bigger is better:

```python
# BigGAN scaling
model_sizes = {
    'small': {'channels': 64, 'batch': 256},
    'medium': {'channels': 96, 'batch': 512},
    'large': {'channels': 128, 'batch': 2048},  # Requires TPUs
}

# Bigger batch sizes enable:
# - More diverse gradients
# - Better coverage of modes
# - More stable training
```

BigGAN achieved unprecedented diversity and quality on ImageNet-scale generation (1000 classes).

## Architecture Comparison

| Architecture | Year | Resolution | Key Innovation |
|--------------|------|------------|----------------|
| DCGAN | 2015 | 64x64 | Convolutional architecture |
| pix2pix | 2016 | 256x256 | Image-to-image translation |
| CycleGAN | 2017 | 256x256 | Unpaired translation |
| ProGAN | 2017 | 1024x1024 | Progressive growing |
| StyleGAN | 2018 | 1024x1024 | Style-based synthesis |
| BigGAN | 2018 | 512x512 | Large-scale training |
| StyleGAN2 | 2019 | 1024x1024 | Artifact removal |
| StyleGAN3 | 2021 | 1024x1024 | Alias-free synthesis |

## Code Example: Training DCGAN

```python
def train_dcgan(generator, discriminator, dataloader, epochs=100):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)

            # Train discriminator
            d_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Real images
            d_real = discriminator(real_images)
            d_loss_real = criterion(d_real, real_labels)

            # Fake images
            z = torch.randn(batch_size, 100)
            fake_images = generator(z)
            d_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(d_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            d_fake = discriminator(fake_images)
            g_loss = criterion(d_fake, real_labels)  # Fool discriminator
            g_loss.backward()
            g_optimizer.step()
```

## Key Takeaways

- DCGAN established convolutional architecture guidelines that became standard for image GANs
- Conditional GANs add control through label embeddings, enabling class-specific generation
- pix2pix uses paired data for image-to-image translation; CycleGAN uses cycle consistency for unpaired translation
- Progressive GAN grows resolution gradually, enabling stable training at 1024x1024
- StyleGAN injects style at multiple scales, providing fine-grained control through the intermediate W space
- BigGAN demonstrated that scaling batch size and model capacity dramatically improves diversity and quality

## Further Reading

- Radford, A., et al. (2016). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (DCGAN)
- Isola, P., et al. (2017). "Image-to-Image Translation with Conditional Adversarial Networks" (pix2pix)
- Karras, T., et al. (2018). "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
- Karras, T., et al. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks" (StyleGAN)

---
*Estimated reading time: 11 minutes*
