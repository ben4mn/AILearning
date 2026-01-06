# The Attention Mechanism

## Introduction

In 2014, Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio introduced a simple but revolutionary idea to neural machine translation: instead of compressing an entire source sentence into a single fixed-size vector, let the decoder look back at any position in the source sentence when generating each target word. This mechanism—attention—fundamentally changed how neural networks handle sequences.

Attention didn't just improve translation quality. It provided a new way of thinking about neural computation: instead of fixed information pathways, attention allows dynamic, content-based routing. The query "what should I focus on?" could get different answers depending on context. This flexibility would eventually form the foundation of the transformer architecture.

In this lesson, we'll understand how attention works, why it solved the seq2seq bottleneck problem, and how the query-key-value formulation provides a general framework for relating different parts of data.

## The Seq2Seq Bottleneck Revisited

Recall the standard seq2seq architecture:

```
Source: "The cat sat on the mat"
         ↓ ↓ ↓ ↓ ↓ ↓
        [LSTM encoder]
         ↓
        [single hidden vector h]  ← Everything compressed here!
         ↓
        [LSTM decoder]
         ↓ ↓ ↓ ↓ ↓ ↓ ↓
Target: "Le chat était assis sur le tapis"
```

All information about the source sentence must fit in a single vector (typically 256-1024 dimensions). For long sentences, this bottleneck severely limits translation quality.

The evidence was clear: translation quality degraded significantly as source sentence length increased beyond 20-30 words.

## The Attention Solution

Bahdanau attention allows the decoder to "look back" at all encoder states:

```python
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.score_proj = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (seq_len, batch, encoder_dim) - all encoder states
        decoder_hidden: (batch, decoder_dim) - current decoder state
        Returns: context vector and attention weights
        """
        # Project encoder outputs
        encoder_projected = self.encoder_proj(encoder_outputs)
        # Shape: (seq_len, batch, attention_dim)

        # Project decoder hidden state
        decoder_projected = self.decoder_proj(decoder_hidden)
        # Shape: (batch, attention_dim)

        # Combine and compute scores
        combined = torch.tanh(encoder_projected + decoder_projected.unsqueeze(0))
        scores = self.score_proj(combined).squeeze(-1)
        # Shape: (seq_len, batch)

        # Softmax over source positions
        attention_weights = F.softmax(scores, dim=0)

        # Weighted sum of encoder outputs
        context = torch.sum(attention_weights.unsqueeze(-1) * encoder_outputs, dim=0)
        # Shape: (batch, encoder_dim)

        return context, attention_weights
```

At each decoding step:
1. Compare decoder state to each encoder state
2. Compute attention weights (which source positions are relevant?)
3. Take weighted sum of encoder states
4. Use this context vector for prediction

## Visualizing Attention

Attention weights form an interpretable alignment:

```
Translating: "The cat sat on the mat" → "Le chat était assis sur le tapis"

                The   cat   sat   on   the   mat
Le              0.8   0.1   0.05  0.0  0.03  0.02
chat            0.1   0.7   0.1   0.0  0.05  0.05
était           0.05  0.1   0.6   0.05 0.1   0.1
assis           0.0   0.1   0.7   0.1  0.05  0.05
sur             0.0   0.0   0.05  0.8  0.1   0.05
le              0.0   0.0   0.0   0.1  0.7   0.2
tapis           0.0   0.0   0.0   0.0  0.1   0.9
```

When generating "chat" (cat), attention focuses on "cat." When generating "tapis" (mat), attention focuses on "mat." The model learns this alignment from parallel text—no explicit word alignments needed.

## Why Attention Helps Gradients

Beyond the bottleneck, attention provides direct gradient paths:

```
Without attention:
  Error at position 50 → backprop through 50 LSTM steps → vanishing gradient

With attention:
  Error at position 50 → direct connection to relevant source positions
  Gradients flow through attention weights, bypassing many RNN steps
```

This isn't a complete solution to vanishing gradients (the attention computation still involves the decoder state), but it provides shortcuts that help.

## The Query-Key-Value Framework

The attention mechanism can be generalized to query-key-value (QKV) formulation:

```python
def attention_qkv(query, keys, values):
    """
    query: what am I looking for? (decoder state)
    keys: what information is available? (encoder states)
    values: what should I retrieve? (also encoder states)

    In Bahdanau attention, keys == values == encoder outputs
    """
    # Compute attention scores
    scores = similarity(query, keys)  # Various similarity functions possible

    # Normalize to get weights
    weights = softmax(scores)

    # Weighted sum of values
    output = sum(weights * values)

    return output, weights
```

This abstraction reveals attention as information retrieval:
- **Query**: The question ("What should I look at to generate the next word?")
- **Keys**: The index ("These are the source word representations")
- **Values**: The content ("This is what each source position actually contains")

## Different Attention Variants

Several attention mechanisms emerged:

### Dot-Product Attention

Simplest form: similarity is the dot product.

```python
def dot_product_attention(query, keys, values):
    # query: (batch, dim)
    # keys: (seq_len, batch, dim)

    scores = torch.matmul(keys.transpose(0, 1), query.unsqueeze(-1)).squeeze(-1)
    # Shape: (batch, seq_len)

    weights = F.softmax(scores, dim=-1)
    context = torch.bmm(weights.unsqueeze(1), values.transpose(0, 1)).squeeze(1)

    return context, weights
```

Efficient but requires query and keys to have the same dimension.

### Scaled Dot-Product Attention

The version used in transformers:

```python
def scaled_dot_product_attention(query, keys, values):
    d_k = query.size(-1)

    # Scale by sqrt(d_k) to prevent softmax saturation
    scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(d_k)

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, values)

    return output, weights
```

Scaling prevents dot products from growing too large as dimension increases (which would push softmax into saturation).

### Additive (Bahdanau) Attention

Uses a learned comparison:

```python
def additive_attention(query, keys, values, W_q, W_k, v):
    # Project and combine
    scores = v @ tanh(W_q @ query + W_k @ keys)

    weights = softmax(scores)
    context = weights @ values

    return context, weights
```

More expressive (learnable transformation) but slower than dot product.

## Attention Beyond Translation

The attention mechanism quickly spread to other tasks:

### Image Captioning

Attend to different image regions when generating each word:

```python
class ImageCaptioner(nn.Module):
    def __init__(self):
        self.cnn = ResNet()  # Extract spatial features
        self.attention = Attention()
        self.lstm = nn.LSTM(...)

    def forward(self, image):
        # CNN produces spatial feature map
        features = self.cnn(image)  # (batch, 14, 14, 2048)

        # Flatten spatial dimensions
        features = features.view(batch, 196, 2048)  # 196 "positions"

        # Generate caption word by word
        for t in range(max_len):
            # Attend to image regions
            context, weights = self.attention(decoder_hidden, features)
            # weights shows which image regions are relevant

            # Generate next word
            word, decoder_hidden = self.lstm(context, decoder_hidden)
```

When generating "dog," attention focuses on the dog region. When generating "park," attention shifts to the background.

### Question Answering

Attend to passage positions when answering:

```python
# Question: "What color is the car?"
# Passage: "The red car was parked in the garage."

# Attention weights when predicting answer:
#        The   red   car   was  parked  in   the  garage
# Answer: 0.1   0.7   0.15  0.02  0.01  0.01 0.0   0.01

# Model attends to "red" to answer "What color?"
```

### Speech Recognition

Attend to audio frames when generating text:

```python
# Audio (spectrogram frames) → Attention → Text (characters/words)
# Handles variable-length audio and varying speaking speeds
```

## The Implications of Attention

Attention revealed something fundamental about neural network design:

1. **Dynamic computation**: Instead of fixed pathways, route information based on content
2. **Direct connections**: Any-to-any connections between positions, bypassing sequential bottlenecks
3. **Interpretability**: Attention weights show what the model is "looking at"
4. **Parallelization potential**: Attention doesn't require sequential computation

This last point would become crucial. RNNs compute sequentially—you can't process position 10 until you've processed positions 1-9. But attention can compare any pair of positions in parallel.

The transformer architecture would push this insight to its logical conclusion: what if we replaced recurrence entirely with attention?

## Key Takeaways

- Attention solves the seq2seq bottleneck by allowing the decoder to access all encoder states, not just a single compressed vector
- The query-key-value framework generalizes attention: query asks what to look for, keys index available information, values contain retrievable content
- Attention weights form interpretable alignments showing which input positions influence each output
- Scaled dot-product attention prevents softmax saturation and became the standard variant
- Attention enables direct gradient flow and parallel computation—insights that motivated the transformer architecture

## Further Reading

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate"
- Luong, M., Pham, H., & Manning, C. (2015). "Effective Approaches to Attention-based Neural Machine Translation"
- Xu, K., et al. (2015). "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"

---
*Estimated reading time: 11 minutes*
