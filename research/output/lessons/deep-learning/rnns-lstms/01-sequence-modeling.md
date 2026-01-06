# Sequence Modeling and Recurrent Networks

## Introduction

Language, speech, music, time series, DNA—some of the most important data in the world comes in sequences. Unlike images where pixels can be shuffled without losing all meaning, sequences have inherent order: "dog bites man" means something very different from "man bites dog." Processing sequences requires architectures that understand order, context, and variable-length inputs.

Recurrent Neural Networks (RNNs) were designed specifically for sequential data. They maintain a hidden state that evolves as they process each element, carrying information from the past into the future. This simple idea—memory through recurrence—proved remarkably powerful and problematic in equal measure.

In this lesson, we'll explore why sequences require specialized architectures, how vanilla RNNs work, and what they can accomplish. We'll set the stage for understanding why LSTMs were needed to overcome RNNs' fundamental limitations.

## Why Sequences Are Different

Consider feeding a sentence to a standard feedforward neural network:

```python
# Problem: sentences have different lengths
sentence1 = "I like cats"           # 3 words
sentence2 = "The quick brown fox"   # 4 words
sentence3 = "She"                   # 1 word

# Feedforward networks need fixed-size input
# How do we handle variable length?

# Option 1: Pad to maximum length (wasteful, arbitrary cutoff)
# Option 2: Bag of words (loses order completely)
# Option 3: Use an architecture that handles sequences naturally
```

The order problem is even more fundamental. In a feedforward network, each input dimension is independent—position 3 has no special relationship to position 4. But in language, adjacent words interact heavily: "not good" means the opposite of "good."

Convolutional networks partially address this with local receptive fields, but they're designed for spatial relationships, not sequential ones. We need something that processes one element at a time while remembering what came before.

## The Recurrent Idea

RNNs process sequences element by element, maintaining a hidden state that encodes information about previous elements:

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Same weights applied at every timestep
        self.i2h = nn.Linear(input_size, hidden_size)   # Input to hidden
        self.h2h = nn.Linear(hidden_size, hidden_size)  # Hidden to hidden

    def forward(self, x_sequence):
        """
        x_sequence: (seq_len, batch_size, input_size)
        Returns: final hidden state
        """
        batch_size = x_sequence.size(1)
        hidden = torch.zeros(batch_size, self.hidden_size)

        for x_t in x_sequence:
            # New hidden state combines current input and previous hidden state
            hidden = torch.tanh(self.i2h(x_t) + self.h2h(hidden))

        return hidden
```

The key equation:

```
h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
```

At each timestep t:
1. Take the current input x_t
2. Take the previous hidden state h_{t-1}
3. Combine them with learned weights
4. Apply nonlinearity (tanh) to get new hidden state h_t

The same weights (W_ih, W_hh) are used at every timestep—this is **weight sharing across time**, analogous to CNNs sharing weights across space.

## Unrolling Through Time

We can visualize an RNN by "unrolling" it across timesteps:

```
x_0    x_1    x_2    x_3
 ↓      ↓      ↓      ↓
[RNN]→[RNN]→[RNN]→[RNN]→ h_4
 h_0   h_1    h_2    h_3
```

Each box is the same RNN cell with the same weights, but processing different inputs. The arrows show how hidden state flows from one timestep to the next.

This unrolled view is exactly how we compute gradients: **Backpropagation Through Time (BPTT)** treats the unrolled network as a very deep feedforward network and applies standard backpropagation.

```python
def bptt_conceptual(rnn, sequence, target, loss_fn):
    """
    Backpropagation through time (conceptual)
    """
    # Forward pass: store all hidden states
    hidden_states = []
    hidden = initial_hidden()

    for x_t in sequence:
        hidden = rnn.step(x_t, hidden)
        hidden_states.append(hidden)

    # Compute loss at final step (or all steps)
    loss = loss_fn(hidden_states[-1], target)

    # Backward pass: propagate gradients back through time
    for t in reversed(range(len(sequence))):
        # Gradient flows back through hidden state connections
        grad_hidden = compute_gradient(hidden_states, t)
        update_weights(rnn, grad_hidden)
```

## What Can RNNs Do?

RNNs enable several powerful patterns:

**Many-to-One**: Sequence classification (sentiment analysis)
```
[word1]→[word2]→[word3]→[hidden]→[prediction]
"I hate this movie" → negative
```

**One-to-Many**: Sequence generation
```
[seed]→[RNN]→word1→[RNN]→word2→[RNN]→word3
"Once upon a" → "time there was..."
```

**Many-to-Many**: Sequence-to-sequence (translation)
```
[Hello]→[World]→[hidden]→[Bonjour]→[le]→[monde]
```

**Many-to-Many (aligned)**: Sequence labeling (POS tagging)
```
[The]→[quick]→[brown]→[fox]
  ↓       ↓       ↓       ↓
 DET     ADJ     ADJ      N
```

## The Power of Memory

The hidden state acts as memory, accumulating information over the sequence:

```python
# Example: counting characters
# Network learns to count 'a's in a string

text = "banana"
# h_0: encode 'b' → no 'a' seen
# h_1: encode 'a' → one 'a' seen
# h_2: encode 'n' → still one 'a'
# h_3: encode 'a' → two 'a's seen
# h_4: encode 'n' → still two 'a's
# h_5: encode 'a' → three 'a's seen

# Final hidden state encodes "three 'a's in the sequence"
```

This memory enables RNNs to:
- Track subject-verb agreement across words
- Remember the opening theme of a musical piece
- Model dependencies in time series data
- Learn grammar and syntax implicitly

## Bidirectional RNNs

Standard RNNs only see the past. But for many tasks (translation, speech recognition), the future matters too. Bidirectional RNNs process the sequence in both directions:

```python
class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_rnn = nn.RNN(input_size, hidden_size)
        self.backward_rnn = nn.RNN(input_size, hidden_size)

    def forward(self, x):
        # Forward direction: left to right
        forward_out, _ = self.forward_rnn(x)

        # Backward direction: right to left
        x_reversed = torch.flip(x, dims=[0])
        backward_out, _ = self.backward_rnn(x_reversed)
        backward_out = torch.flip(backward_out, dims=[0])

        # Concatenate both directions
        return torch.cat([forward_out, backward_out], dim=-1)
```

At each position, the bidirectional hidden state combines:
- All information from the left (forward RNN)
- All information from the right (backward RNN)

This provides full context for decisions like translation where word meaning depends on the whole sentence.

## Stacking RNN Layers

Like CNNs, RNNs can be stacked for hierarchical representations:

```python
# Multi-layer RNN
class StackedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.RNN(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x
```

Each layer processes the sequence of hidden states from the layer below. Lower layers capture local patterns; higher layers capture more abstract, long-range dependencies.

## Early Successes

Before LSTMs dominated, vanilla RNNs achieved notable successes:

- **Elman networks (1990)**: Learned simple grammars
- **Speech recognition**: Hybrid HMM-RNN systems
- **Handwriting recognition**: Connected cursive text
- **Music generation**: Simple melodies and rhythms

But researchers kept hitting a wall: RNNs struggled with long sequences. A network processing a 100-word sentence would struggle to connect the first word to the last. Something was fundamentally limiting how far information could flow.

## The Preview of Problems

Consider training an RNN on this task:

```
Input:  "The cat, which was sitting on the mat, [MASK]"
Target: "sat" (agreeing with "cat", not "mat")

# The network must remember "cat" across 8 intervening words
# to predict the correct verb agreement
```

Vanilla RNNs struggle with this. The hidden state after 8 words has been overwritten so many times that the signal from "cat" is nearly gone.

We can see this mathematically. At each step, the hidden state is transformed by a matrix W_hh:

```
h_8 = f(W @ h_7) = f(W @ f(W @ h_6)) = ... = f(W^8 @ h_0 + ...)
```

If the largest eigenvalue of W is less than 1, W^8 → 0. If greater than 1, W^8 → ∞. Either the signal vanishes or explodes. This is the famous **vanishing/exploding gradient problem** for RNNs, which we'll explore in depth in the next lesson.

## Key Takeaways

- Sequences require specialized architectures that understand order and handle variable length—feedforward networks can't naturally process sequential data
- RNNs maintain a hidden state that evolves through the sequence, accumulating information from previous elements
- The same weights are applied at every timestep (weight sharing across time), enabling processing of arbitrary-length sequences
- Bidirectional RNNs process sequences in both directions, providing full context at each position
- Vanilla RNNs work well for short-range dependencies but struggle with long sequences due to the vanishing gradient problem

## Further Reading

- Elman, J. (1990). "Finding Structure in Time"
- Rumelhart, D., Hinton, G., & Williams, R. (1986). "Learning representations by back-propagating errors"
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" - Chapter 10: Sequence Modeling

---
*Estimated reading time: 10 minutes*
