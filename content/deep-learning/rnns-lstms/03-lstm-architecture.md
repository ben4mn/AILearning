# LSTM Architecture: Gates, Cells, and Memory

## Introduction

In 1997, Sepp Hochreiter and Jurgen Schmidhuber published a paper that would eventually transform sequence modeling. Their Long Short-Term Memory (LSTM) architecture was designed with one goal: solve the vanishing gradient problem. The solution was elegant—instead of a single hidden state that gets overwritten at each step, maintain a separate cell state that can carry information unchanged across many timesteps, controlled by learned gates that decide what to remember, forget, and output.

LSTMs were ahead of their time. For years, they were considered too complex and computationally expensive. But when computing power caught up in the 2010s, LSTMs became the default architecture for sequence modeling, powering everything from machine translation to speech recognition. In this lesson, we'll understand exactly how LSTMs work and why their design so effectively addresses the vanishing gradient problem.

## The Core Insight: Additive State Updates

The fundamental problem with vanilla RNNs is that the hidden state is completely transformed at each step:

```python
# Vanilla RNN: hidden state is overwritten
h_t = tanh(W @ h_{t-1} + U @ x_t + b)
```

Every element of h_{t-1} goes through a nonlinear transformation. Information degrades with each step.

LSTM's key innovation is the **cell state** c_t, which is updated additively:

```python
# LSTM: cell state is updated additively
c_t = f_t * c_{t-1} + i_t * candidate_t
```

This is a weighted sum: some old information (f_t fraction of c_{t-1}) plus some new information (i_t fraction of candidate). If f_t ≈ 1 and i_t ≈ 0, then c_t ≈ c_{t-1}—information is preserved exactly.

## The Four Gates

LSTMs use four gates to control information flow. Each gate is a layer with sigmoid activation (output between 0 and 1):

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # All gates computed from same inputs (for efficiency)
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)

        # Compute all gates at once
        gates = self.gates(combined)

        # Split into four components
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)

        # Apply activations
        i_t = torch.sigmoid(i_t)  # Input gate
        f_t = torch.sigmoid(f_t)  # Forget gate
        g_t = torch.tanh(g_t)     # Candidate values
        o_t = torch.sigmoid(o_t)  # Output gate

        # Update cell state
        c_t = f_t * c_prev + i_t * g_t

        # Compute hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
```

Let's understand each gate:

### Forget Gate (f_t)

The forget gate decides what to discard from the cell state:

```
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
```

- f_t ≈ 1: Keep this information
- f_t ≈ 0: Forget this information

For example, when ending a paragraph in language modeling, the forget gate might clear out subject/verb agreement information that's no longer relevant.

### Input Gate (i_t)

The input gate decides what new information to store:

```
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
```

- i_t ≈ 1: Store this new information
- i_t ≈ 0: Ignore this input

Combined with the candidate values, this determines what gets written to the cell state.

### Candidate Values (g_t)

The candidate values are potential new information to add:

```
g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)
```

Unlike gates (sigmoid, 0-1), candidates use tanh (-1 to 1), allowing both positive and negative updates to the cell state.

### Output Gate (o_t)

The output gate decides what to expose from the cell state:

```
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
```

The cell state can store information that's not immediately useful. The output gate filters what should influence the current output.

## Information Flow Diagram

Visualizing the LSTM cell:

```
                     c_{t-1} ──────────→ [×] ────────→ [+] ──────────→ c_t
                                          ↑             ↑
                                         f_t          i_t × g_t
                                          ↑             ↑
                  ┌───────────────────────┼─────────────┼────────────┐
                  │                       │             │            │
x_t ─────────────→│ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │            │
                  │ │σ f │ │σ i│ │tanh g│ │σ o│       │            │
h_{t-1} ─────────→│ └────┘ └────┘ └────┘ └────┘       │            │
                  │    │       │     │       │         │            │
                  └────┼───────┼─────┼───────┼─────────┘            │
                       ↓       ↓     ↓       ↓                      ↓
                      f_t     i_t   g_t     o_t ──────→ [×] ←── tanh(c_t)
                                                         │
                                                         ↓
                                                        h_t
```

The cell state c_t flows horizontally with only element-wise operations—no matrix multiplications that could cause gradient issues.

## Why Gradients Don't Vanish

The critical path for gradient flow is through the cell state:

```
c_t = f_t * c_{t-1} + i_t * g_t
```

The gradient of c_t with respect to c_{t-1} is simply f_t (the forget gate value). If f_t ≈ 1, the gradient passes through unchanged!

```python
# Gradient flow through cell state
def gradient_flow(forget_gates):
    """
    Gradient flows through product of forget gates
    """
    gradient = 1.0
    for f_t in forget_gates:
        gradient *= f_t  # If f_t ≈ 1, gradient preserved

    return gradient

# Example: 100 timesteps with forget gate = 0.99
gradient_100_steps = 0.99 ** 100  # ≈ 0.37

# Compare to vanilla RNN: 0.5 ** 100 ≈ 7.8e-31
```

With forget gates near 1, gradients can flow across hundreds of timesteps. The network learns what forget gate values to use—when long-term memory is important, it learns to keep f_t high.

## The Cell State as "Conveyor Belt"

A useful metaphor: the cell state is a conveyor belt running through the network. Information can hop on (via input gate) or hop off (via forget gate), but the belt itself keeps moving without transformation.

```
Timestep:     0    1    2    3    ...   99   100
Cell state:  [A]  [A]  [A]  [A]  ...  [A]  [A]

# Information 'A' from timestep 0 reaches timestep 100
# because forget gates kept f_t ≈ 1 for this information
```

The hidden state h_t is what the network exposes at each step, but the cell state c_t is the memory that persists.

## Initialization Matters

The forget gate bias is often initialized to 1 or higher, so that f_t starts near 1 and information flows by default:

```python
# Forget gate bias initialization
def init_lstm(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            # Set forget gate bias to 1
            n = param.size(0)
            start, end = n//4, n//2  # Forget gate portion
            param.data[start:end].fill_(1.0)
```

This is the "remember by default" principle. The network must learn to forget, rather than learning to remember.

## Peephole Connections (Variant)

Some LSTM variants add "peephole" connections, letting gates see the cell state directly:

```python
# Peephole connections
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + W_cf @ c_{t-1} + b_f)  # Also sees c_{t-1}
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + W_ci @ c_{t-1} + b_i)
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + W_co @ c_t + b_o)      # Sees c_t
```

Peepholes let gates make decisions based on what's stored in the cell, not just the hidden state. Research shows they help on some tasks but aren't always necessary.

## GRU: A Simplified Alternative

The Gated Recurrent Unit (GRU), proposed by Cho et al. in 2014, simplifies the LSTM:

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gates = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        combined = torch.cat([x_t, h_prev], dim=1)

        gates = torch.sigmoid(self.gates(combined))
        z_t, r_t = gates.chunk(2, dim=1)

        # Reset gate applied to previous hidden state
        candidate = torch.tanh(self.candidate(
            torch.cat([x_t, r_t * h_prev], dim=1)
        ))

        # Update: interpolate between previous and candidate
        h_t = (1 - z_t) * h_prev + z_t * candidate

        return h_t
```

GRU differences:
- No separate cell state (just hidden state)
- Two gates instead of three (update z_t, reset r_t)
- Fewer parameters, faster training
- Performance often comparable to LSTM

The update equation `h_t = (1-z) * h_prev + z * candidate` still allows gradient flow when z ≈ 0.

## When to Use LSTM vs GRU

```
LSTM advantages:
- More expressive (separate cell and hidden states)
- Better on very long sequences
- More extensive research and tuning guidelines

GRU advantages:
- Fewer parameters (faster training)
- Works well on smaller datasets
- Simpler to understand and implement

General guidance:
- Try GRU first (faster experiments)
- Use LSTM if GRU underperforms
- For state-of-the-art: often LSTM with attention
```

## Key Takeaways

- LSTMs solve vanishing gradients through additive cell state updates and learned gates
- The forget gate (f_t) controls what information persists; when f_t ≈ 1, gradients flow unchanged
- The input gate (i_t) and candidate (g_t) control what new information is written
- The output gate (o_t) controls what is exposed from the cell state to the hidden state
- The cell state acts as a "conveyor belt" carrying information across timesteps
- GRUs offer a simpler alternative with similar performance for many tasks

## Further Reading

- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
- Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (GRU)
- Greff, K., et al. (2017). "LSTM: A Search Space Odyssey" (comprehensive LSTM analysis)
- Jozefowicz, R., et al. (2015). "An Empirical Exploration of Recurrent Network Architectures"

---
*Estimated reading time: 11 minutes*
