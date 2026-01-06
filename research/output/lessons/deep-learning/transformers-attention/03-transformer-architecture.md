# Transformer Architecture Deep Dive

## Introduction

Now that we understand the motivations and key innovations of the Transformer, let's examine its architecture in detail. The Transformer's elegance lies in its composition of simple, well-understood components: embeddings, attention, feedforward networks, normalization, and residual connections. Each piece serves a clear purpose, and together they form a remarkably powerful sequence-processing machine.

In this lesson, we'll walk through the complete Transformer architecture layer by layer, understanding the role of each component. We'll also explore variations and design choices that emerged as the architecture evolved.

## Architecture Overview

The full Transformer consists of:

```
Input Tokens → Token Embedding + Positional Encoding
      ↓
   Encoder Stack (N layers)
      ↓
   Encoder Output
      ↓
   Decoder Stack (N layers, with cross-attention to encoder)
      ↓
   Linear + Softmax
      ↓
   Output Probabilities
```

The original paper used N=6 encoder layers and N=6 decoder layers. Let's examine each component.

## Token Embeddings

The first step converts discrete tokens to continuous vectors:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens):
        # Scale embeddings by sqrt(d_model) as in original paper
        return self.embedding(tokens) * math.sqrt(self.embed_dim)
```

The scaling factor (sqrt(d_model)) ensures embeddings start at a reasonable magnitude relative to positional encodings.

## Positional Encoding Details

The sinusoidal encoding creates unique, smooth position representations:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        self.register_buffer('pe', pe)  # Not a parameter, but saved

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

Properties of sinusoidal encoding:
- Each position gets a unique vector
- Nearby positions have similar encodings
- The encoding at position p + k can be expressed as a linear function of the encoding at position p (enabling relative position learning)

## The Encoder Layer

Each encoder layer has two sub-layers:

```python
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Sub-layer 1: Multi-head self-attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Sub-layer 2: Position-wise feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention with residual and norm
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward with residual and norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
```

The pattern "Sublayer → Dropout → Add → Normalize" is repeated throughout.

## The Feedforward Network

The feedforward network is position-wise (same transformation at each position):

```python
# At each position independently:
# d_model → d_ff → ReLU → d_ff → d_model

# Original paper: d_model=512, d_ff=2048 (4x expansion)
```

Why is the FFN needed? Self-attention is fundamentally a weighted averaging operation. The FFN adds nonlinear transformation capacity. Each position can be independently transformed before the next attention layer.

Think of it as: attention routes information, FFN transforms information.

## The Decoder Layer

Decoder layers have three sub-layers:

```python
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Sub-layer 1: Masked self-attention (can't see future)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Sub-layer 2: Cross-attention to encoder outputs
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Sub-layer 3: Position-wise feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention to encoder
        # Query from decoder, keys/values from encoder
        cross_output = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_output))

        # Feedforward
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x
```

The causal mask ensures position i only attends to positions 0, 1, ..., i-1 during self-attention.

## Layer Normalization

Unlike batch normalization (normalizes across batch), layer normalization normalizes across the embedding dimension:

```python
class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

Layer norm is preferred over batch norm for sequences because:
- Works with variable-length sequences
- No batch statistics needed (works with batch size 1)
- More stable training

## Pre-Norm vs Post-Norm

The original Transformer used post-normalization:

```python
# Post-norm (original)
x = LayerNorm(x + Sublayer(x))

# Pre-norm (often used now)
x = x + Sublayer(LayerNorm(x))
```

Pre-norm has better gradient flow and is easier to train for very deep models. Most modern implementations use pre-norm.

## The Full Model

Putting it all together:

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, ff_dim=2048,
                 max_len=5000, dropout=0.1):
        super().__init__()

        # Embeddings
        self.src_embed = TokenEmbedding(src_vocab, embed_dim)
        self.tgt_embed = TokenEmbedding(tgt_vocab, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(embed_dim, tgt_vocab)

    def encode(self, src, src_mask):
        x = self.pos_encoding(self.src_embed(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.pos_encoding(self.tgt_embed(tgt))
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_proj(decoder_output)
        return logits
```

## Hyperparameters

The original paper defined two model sizes:

| Parameter | Base Model | Big Model |
|-----------|------------|-----------|
| d_model (embed_dim) | 512 | 1024 |
| N (layers) | 6 | 6 |
| d_ff (FFN hidden) | 2048 | 4096 |
| h (heads) | 8 | 16 |
| d_k = d_v (per head) | 64 | 64 |
| Dropout | 0.1 | 0.3 |
| Parameters | 65M | 213M |

Modern models often use more layers and larger dimensions, scaling to billions of parameters.

## Training Details

The original training used:

**Label smoothing**: Instead of one-hot targets, use soft targets (e.g., 0.9 for correct class, 0.1 / (vocab - 1) distributed among others). Improves generalization.

**Learning rate schedule**: Warmup then decay

```python
def transformer_lr_schedule(step, d_model, warmup_steps=4000):
    """
    Learning rate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
```

**Adam optimizer**: With beta1=0.9, beta2=0.98, epsilon=1e-9

**Regularization**: Dropout on attention weights, FFN, and embeddings

## Computational Complexity

The self-attention operation has O(n^2 * d) complexity per layer:
- n^2: Every position attends to every other position
- d: The embedding dimension

This quadratic scaling becomes a problem for very long sequences (>2000 tokens). Various efficient attention variants have been proposed (Longformer, BigBird, Linear attention).

For typical sequence lengths (256-512 tokens), the parallelization benefits outweigh the quadratic cost compared to RNNs.

## Variations and Extensions

Since 2017, many Transformer variations have emerged:

- **Encoder-only** (BERT): Bidirectional, good for understanding
- **Decoder-only** (GPT): Autoregressive, good for generation
- **T5**: Text-to-text framework, encoder-decoder
- **Relative positional encoding**: Better length generalization
- **Rotary embeddings (RoPE)**: Encode relative positions in attention
- **Flash Attention**: Memory-efficient attention computation
- **Mixture of Experts**: Sparse activation for efficiency

## Key Takeaways

- Token embeddings scaled by sqrt(d_model) combined with sinusoidal positional encodings form the input representation
- Encoder layers stack self-attention and feedforward networks with residual connections and layer normalization
- Decoder layers add masked self-attention (for autoregressive generation) and cross-attention (to encoder outputs)
- Layer normalization across the embedding dimension is preferred over batch normalization for sequences
- Pre-norm ordering often works better than the original post-norm for training stability
- Self-attention has O(n^2) complexity, making very long sequences challenging

## Further Reading

- Vaswani, A., et al. (2017). "Attention Is All You Need" (original paper)
- The Annotated Transformer (Harvard NLP): https://nlp.seas.harvard.edu/annotated-transformer/
- On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)

---
*Estimated reading time: 11 minutes*
