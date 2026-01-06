# Embeddings Deep Dive

## Introduction

Tokenization converts text into token IDs—but those IDs are just arbitrary numbers. The token "cat" might be ID 2368, while "dog" might be 3271. These numbers don't capture that cats and dogs are both animals, that they're more similar to each other than to "democracy" or "photosynthesis."

This is where **embeddings** come in. Embeddings transform discrete token IDs into dense vectors in a high-dimensional space where meaning is encoded in the geometry. Words with similar meanings cluster together. Relationships like "king is to queen as man is to woman" emerge as vector operations. This transformation from discrete symbols to continuous space is one of the foundational ideas in modern NLP.

In this lesson, we'll explore how embeddings work, the different types of embeddings in transformer models, and why this representation is so powerful for language understanding.

## From IDs to Vectors

### The Embedding Layer

Every language model starts with an **embedding layer**—a lookup table that maps token IDs to vectors:

```python
import numpy as np

# Conceptual embedding table
vocab_size = 50000
embedding_dim = 768  # Typical dimension

# The embedding matrix: one row per token
embeddings = np.random.randn(vocab_size, embedding_dim)

# Looking up a token's embedding
token_id = 2368  # ID for "cat"
cat_vector = embeddings[token_id]  # 768-dimensional vector
```

The embedding dimension (768 for BERT, 12288 for GPT-4) determines how much information each vector can carry.

### What's in a Vector?

Initially, embeddings are random—"cat" and "dog" have no special relationship. During training, embeddings adjust to minimize prediction error. Through this process:

- Similar words develop similar vectors
- Related concepts cluster in embedding space
- Semantic and syntactic patterns emerge

```python
# After training, we might see:
similarity(embed("cat"), embed("dog")) = 0.85    # High
similarity(embed("cat"), embed("democracy")) = 0.12  # Low
similarity(embed("run"), embed("running")) = 0.91  # High
similarity(embed("run"), embed("sprint")) = 0.78   # Related

# Where similarity is often cosine similarity:
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

### The Word2Vec Legacy

The idea that words could be meaningfully represented as vectors gained prominence with **Word2Vec** (2013). It demonstrated striking properties:

```python
# Famous word vector arithmetic
king - man + woman ≈ queen
paris - france + japan ≈ tokyo
walked - walk + swim ≈ swam
```

These relationships emerge from training, not explicit programming. Word2Vec showed that distributional patterns—what words appear near what other words—contain rich semantic information.

Modern LLM embeddings are far more sophisticated, but they build on this foundational insight.

## Token Embeddings in Transformers

### Static vs. Contextual Embeddings

Word2Vec gives each word one fixed vector. "Bank" has the same embedding whether it means a financial institution or a river bank. This is called **static embedding**.

Transformer models produce **contextual embeddings**—the same word gets different vectors depending on surrounding context:

```python
# In a transformer model
sentence1 = "I deposited money at the bank"
sentence2 = "I sat by the river bank"

# The token "bank" gets different vectors in each case!
bank_vector_1 = model.embed(sentence1)[6]  # Financial meaning
bank_vector_2 = model.embed(sentence2)[5]  # River meaning

# These vectors differ, capturing different senses
similarity(bank_vector_1, bank_vector_2) < similarity(
    bank_vector_1, model.embed("I withdrew cash from the bank")[6]
)
```

This context-dependence is crucial for understanding language, where meaning depends heavily on context.

### How Context Gets Incorporated

Transformers build contextual embeddings through their attention layers:

```python
# Simplified transformer flow
def transformer_layer(embeddings, attention_mask):
    # Self-attention: each position attends to all others
    context_aware = self_attention(embeddings, attention_mask)

    # Feed-forward: process each position
    output = feed_forward(context_aware)

    return output

# After multiple layers, embeddings are deeply contextual
def full_model(token_ids):
    # Start with static token embeddings
    embeddings = embedding_table[token_ids]

    # Add positional information
    embeddings += positional_embeddings

    # Pass through many transformer layers
    for layer in transformer_layers:
        embeddings = layer(embeddings)

    # Final embeddings are contextual
    return embeddings
```

Each layer adds more context. Early layers capture local syntax; later layers capture deeper semantics and long-range dependencies.

## Positional Embeddings

### The Position Problem

Self-attention is **permutation invariant**—it treats "cat sat mat" the same as "mat sat cat." But word order matters enormously in language! "Dog bites man" and "Man bites dog" have very different meanings.

**Positional embeddings** solve this by encoding position information:

```python
# Each position gets its own embedding
max_positions = 2048
embedding_dim = 768
positional_embeddings = np.random.randn(max_positions, embedding_dim)

def add_positions(token_embeddings):
    seq_length = len(token_embeddings)
    positions = positional_embeddings[:seq_length]
    return token_embeddings + positions
```

### Types of Positional Encodings

**Learned positional embeddings**: A trainable embedding for each position. Simple and effective, used in GPT and BERT.

```python
# Learned positions
position_embedding_table = nn.Embedding(max_positions, embedding_dim)
```

**Sinusoidal encodings**: Fixed mathematical patterns (used in original transformer):

```python
def sinusoidal_encoding(position, dim):
    encodings = []
    for i in range(dim):
        if i % 2 == 0:
            encodings.append(np.sin(position / 10000**(i/dim)))
        else:
            encodings.append(np.cos(position / 10000**((i-1)/dim)))
    return np.array(encodings)
```

**Rotary Position Embeddings (RoPE)**: Rotates embeddings based on position, enabling better extrapolation to longer sequences. Used in LLaMA and many recent models.

**ALiBi**: Adds position-based biases directly to attention scores rather than embeddings. Enables better length generalization.

### Why Position Encoding Matters

Position encoding affects:
- Maximum sequence length the model handles
- How well the model generalizes to longer sequences
- Computational efficiency
- Understanding of structural patterns

Different approaches trade off these considerations differently.

## Specialized Embeddings

### Segment Embeddings

Some models (especially BERT-style) use segment embeddings to distinguish parts of input:

```python
# BERT example with two segments
text = "[CLS] Is this a question? [SEP] Yes it is. [SEP]"
#        ^---- Segment A ----^      ^- Segment B -^

segment_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
segment_embeddings = segment_table[segment_ids]

# Added to token + position embeddings
final = token_emb + position_emb + segment_emb
```

### Type Embeddings

Chat models often use type embeddings to distinguish roles:

```python
# Distinguishing system, user, assistant
types = ["system", "user", "assistant", "user", "assistant"]
type_embeddings = type_table[type_ids]
```

### Layer-Specific Embeddings

Research has shown that different transformer layers capture different information:

- **Early layers**: Surface features, syntax
- **Middle layers**: Semantic meaning, entity types
- **Late layers**: Task-specific features

For some applications, using embeddings from specific layers works better than final-layer embeddings.

## Embedding Spaces

### Geometry of Meaning

High-dimensional embedding spaces have fascinating geometric properties:

```python
# Cosine similarity measures angle between vectors
# Common for comparing embeddings

def semantic_similarity(word1, word2, model):
    e1 = model.embed(word1)
    e2 = model.embed(word2)
    return cosine_similarity(e1, e2)

# Related words have similar embeddings
semantic_similarity("happy", "joyful")   # ~0.8
semantic_similarity("happy", "sad")      # ~0.5 (related but opposite)
semantic_similarity("happy", "hydraulic") # ~0.1 (unrelated)
```

### Clustering and Visualization

We can visualize embedding spaces using dimensionality reduction:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce 768 dimensions to 2 for visualization
words = ["cat", "dog", "fish", "apple", "orange", "banana",
         "run", "walk", "sprint", "france", "germany", "italy"]
embeddings = [model.embed(w) for w in words]

reduced = TSNE(n_components=2).fit_transform(embeddings)

plt.scatter(reduced[:, 0], reduced[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, reduced[i])
# Animals, fruits, actions, and countries cluster separately!
```

### Embedding Arithmetic

Like Word2Vec, modern embeddings sometimes support meaningful arithmetic:

```python
# Relationship encoding (works with varying reliability)
man_to_woman = embed("woman") - embed("man")
king + man_to_woman ≈ queen

# Country to capital
france_to_paris = embed("paris") - embed("france")
embed("germany") + france_to_paris ≈ embed("berlin")
```

This property is interesting but shouldn't be relied upon for real applications—it's not always consistent.

## Practical Applications

### Semantic Search

Embeddings enable search by meaning rather than keywords:

```python
def semantic_search(query, documents, top_k=5):
    # Embed the query
    query_emb = embed(query)

    # Embed all documents (usually pre-computed)
    doc_embs = [embed(doc) for doc in documents]

    # Find most similar
    similarities = [cosine_similarity(query_emb, d) for d in doc_embs]
    top_indices = np.argsort(similarities)[-top_k:]

    return [documents[i] for i in reversed(top_indices)]

# Works even without keyword overlap
results = semantic_search(
    "feeling down",
    ["I'm sad today", "The stock went down", "Happy birthday!"]
)
# Returns "I'm sad today" first despite no word overlap
```

### Sentence and Document Embeddings

While token embeddings represent individual tokens, we often need full sentence or document representations:

```python
# Simple approach: mean pooling
def sentence_embedding(text, model):
    token_embeddings = model.encode(text)  # [seq_len, dim]
    return token_embeddings.mean(axis=0)   # [dim]

# Better: use dedicated sentence embedding models
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("This is a sentence.")
```

Models like Sentence-BERT are specifically trained to produce good sentence-level embeddings.

### Embedding-Based Classification

Embeddings can power efficient classifiers:

```python
from sklearn.linear_model import LogisticRegression

# Create embeddings for training data
X_train = [get_embedding(text) for text in train_texts]
y_train = train_labels

# Train simple classifier on embeddings
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Classify new text
X_new = get_embedding("This movie was fantastic!")
prediction = classifier.predict([X_new])
```

## Key Takeaways

1. **Embeddings transform discrete tokens into continuous vectors** where semantic similarity corresponds to geometric proximity.

2. **Modern embeddings are contextual**—the same word gets different vectors based on surrounding context.

3. **Positional embeddings encode word order**, which is crucial since attention itself is order-invariant.

4. **Different transformer layers capture different information**, from surface syntax to deep semantics.

5. **Embedding geometry enables semantic operations** like similarity search and (sometimes) analogy completion.

6. **Practical applications include semantic search, classification, and clustering** across many domains.

## Further Reading

- "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013) - Word2Vec
- "Attention Is All You Need" (Vaswani et al., 2017) - Positional encodings in transformers
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019)
