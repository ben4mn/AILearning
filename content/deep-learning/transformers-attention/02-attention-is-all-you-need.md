# Attention Is All You Need: The Transformer Paper

## Introduction

In June 2017, a team at Google published a paper with the provocative title "Attention Is All You Need." The claim seemed extreme: they proposed an architecture for sequence modeling that used no recurrence, no convolutions—only attention. This wasn't a minor modification; it was a fundamental rethinking of how to process sequences.

The results validated the audacious claim. On machine translation benchmarks, the Transformer achieved state-of-the-art quality while training far faster than LSTM-based models. But the paper's true impact went far beyond translation. The Transformer architecture would become the foundation for GPT, BERT, and virtually every major language model that followed.

In this lesson, we'll explore the key ideas from this landmark paper: self-attention, multi-head attention, positional encoding, and why removing recurrence was both possible and beneficial.

## The Case Against Recurrence

RNNs have a fundamental limitation: sequential processing.

```python
# RNN: must process positions in order
for t in range(sequence_length):
    hidden = rnn_step(input[t], hidden)  # Can't parallelize across t
```

This sequential dependency means:
- **Training bottleneck**: GPUs excel at parallel operations; sequential processing wastes their power
- **Long paths**: Information from position 0 must pass through all intermediate positions to reach position 100
- **Memory constraints**: Must store all intermediate states for backpropagation

The Transformer team asked: can we build a sequence model from attention alone, eliminating recurrence entirely?

## Self-Attention: Relating Positions Within a Sequence

Standard attention relates two different sequences (encoder and decoder). **Self-attention** relates positions within the same sequence:

```python
def self_attention(x, W_q, W_k, W_v):
    """
    x: input sequence (seq_len, embed_dim)
    Each position attends to all positions (including itself)
    """
    # Every position generates a query, key, and value
    Q = x @ W_q  # Queries: "What am I looking for?"
    K = x @ W_k  # Keys: "What do I contain?"
    V = x @ W_v  # Values: "What information should I pass?"

    # Attention scores: how much does position i attend to position j?
    scores = Q @ K.T / math.sqrt(d_k)

    # Weights: softmax normalizes across positions
    weights = F.softmax(scores, dim=-1)

    # Output: weighted sum of values
    output = weights @ V

    return output
```

At each position, the model can gather information from any other position based on content similarity. No sequential processing required.

## The Parallelization Advantage

Self-attention computes all position interactions simultaneously:

```python
# Self-attention: all positions computed in parallel
Q = X @ W_q  # (seq_len, d_k) - all queries at once
K = X @ W_k  # (seq_len, d_k) - all keys at once
V = X @ W_v  # (seq_len, d_v) - all values at once

# One matrix multiplication for all attention scores
scores = Q @ K.T  # (seq_len, seq_len) - all pairs at once!

# All outputs in parallel
output = softmax(scores) @ V  # (seq_len, d_v)
```

This is a sequence of matrix multiplications—exactly what GPUs do best. A 100-position sequence requires the same number of operations as a 10-position sequence (just larger matrices).

## Multi-Head Attention

A single attention head might focus on one type of relationship (e.g., syntactic). To capture multiple relationship types, the Transformer uses **multi-head attention**:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, head_dim)

        # Scaled dot-product attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(weights, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, -1)

        # Final linear projection
        output = self.W_o(attention_output)

        return output
```

With 8 heads, the model learns 8 different attention patterns:
- One head might track subject-verb relationships
- Another might track positional proximity
- Another might handle named entities

## Positional Encoding

Self-attention has no inherent notion of position. The sentence "dog bites man" produces the same attention patterns as "man bites dog" if we don't somehow encode position.

The Transformer adds **positional encodings** to the input:

```python
def positional_encoding(max_len, embed_dim):
    """
    Sinusoidal positional encoding from the original paper
    """
    pe = torch.zeros(max_len, embed_dim)

    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                         -(math.log(10000.0) / embed_dim))

    pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

    return pe

# Usage: add to embeddings
x = token_embeddings + positional_encoding[:seq_len]
```

Why sinusoids?
- Unique encoding for each position
- Smooth and continuous
- Can theoretically extrapolate to longer sequences
- The model can learn to attend to relative positions

Later work would explore learned positional embeddings and relative position representations.

## The Transformer Block

A Transformer layer combines multi-head attention with feedforward networks:

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head self-attention with residual connection
        attended = self.attention(x)
        x = self.norm1(x + self.dropout(attended))

        # Feedforward with residual connection
        fed_forward = self.ffn(x)
        x = self.norm2(x + self.dropout(fed_forward))

        return x
```

Key components:
- **Residual connections**: Enable deep stacking (like ResNet)
- **Layer normalization**: Stabilize training
- **Feedforward network**: Position-wise, same across all positions

## Encoder-Decoder Structure

For translation, the Transformer uses an encoder-decoder structure:

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = positional_encoding(1000, embed_dim)

        # Encoder: self-attention on source
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])

        # Decoder: masked self-attention + cross-attention to encoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(embed_dim, vocab_size)
```

The decoder has two attention mechanisms:
1. **Masked self-attention**: Can only attend to previous positions (preserves autoregressive property)
2. **Cross-attention**: Attends to encoder outputs (like Bahdanau attention)

## Masking for Autoregressive Generation

During training, the decoder sees the full target sequence but must not peek at future positions:

```python
def causal_mask(seq_len):
    """
    Lower triangular mask: position i can only see positions <= i
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Usage in attention
scores = Q @ K.T
scores = scores + causal_mask(seq_len)  # Future positions → -inf
weights = F.softmax(scores, dim=-1)     # -inf → 0 after softmax
```

## Training and Results

The paper trained Transformers on WMT English-German and English-French translation:

**Results:**
- English-German: 28.4 BLEU (vs 24.6 for best RNN)
- English-French: 41.0 BLEU (vs 41.2 for best ensemble)

**Training efficiency:**
- 12 hours on 8 P100 GPUs for base model
- 3.5 days for big model
- Far faster than comparable LSTM training

## Why Did It Work?

Several factors contributed to the Transformer's success:

1. **Parallelization**: All positions computed simultaneously
2. **Short attention paths**: Any position directly attends to any other (vs. O(n) RNN path)
3. **Rich attention**: Multi-head attention captures diverse relationships
4. **Residual connections**: Enable training of deep models
5. **Scale-friendly**: Larger models brought consistent improvements

## The Broader Impact

The paper's influence extended far beyond translation:

- **BERT (2018)**: Transformer encoder for bidirectional language understanding
- **GPT (2018+)**: Transformer decoder for language generation
- **Vision Transformer (2020)**: Applied to image classification
- **Whisper, DALL-E, and beyond**: Audio, image, multimodal transformers

The Transformer became the dominant architecture for almost all deep learning involving sequences—and increasingly, other data types too.

## Key Takeaways

- The Transformer replaces recurrence with self-attention, enabling parallel processing of all positions simultaneously
- Multi-head attention allows the model to learn multiple types of position relationships (syntax, semantics, proximity)
- Positional encodings inject position information since self-attention has no inherent position awareness
- Residual connections and layer normalization enable training of deep Transformer stacks
- The encoder-decoder structure with masked self-attention maintains autoregressive generation capability
- Training is dramatically faster than RNNs due to parallelization, with better or equal quality

## Further Reading

- Vaswani, A., et al. (2017). "Attention Is All You Need"
- The Illustrated Transformer (Jay Alammar's blog)
- The Annotated Transformer (Harvard NLP)

---
*Estimated reading time: 11 minutes*
