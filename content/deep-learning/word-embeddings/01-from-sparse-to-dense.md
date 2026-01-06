# From Sparse to Dense: The Representation Revolution

## Introduction

How do you represent a word to a computer? For decades, the standard approach was brutally simple: assign each word a unique ID and represent it as a one-hot vector—a vector with all zeros except for a single 1 at the word's position. This worked for traditional NLP systems, but it had a fundamental flaw: the representation said nothing about what words meant. "Cat" and "dog" were just as different as "cat" and "democracy."

The shift from sparse one-hot vectors to dense learned embeddings was one of the most important transitions in NLP. Words became points in a continuous space where distance reflected meaning. Similar words clustered together. Relationships between words could be captured as vector operations. This wasn't just a computational convenience—it was a new way of thinking about language.

In this lesson, we'll explore why one-hot representations were problematic, what we want from word representations, and how the path to dense embeddings emerged. Understanding this transition is essential for appreciating modern NLP.

## The One-Hot Problem

In one-hot encoding, each word gets a unique vector:

```python
vocabulary = ["the", "cat", "sat", "on", "mat", "dog", "ran"]
vocab_size = 7

def one_hot(word):
    idx = vocabulary.index(word)
    vector = [0] * vocab_size
    vector[idx] = 1
    return vector

# "cat" = [0, 1, 0, 0, 0, 0, 0]
# "dog" = [0, 0, 0, 0, 0, 1, 0]
# "democracy" would be [0, 0, 0, 0, 0, 0, 0, 1] if added
```

This representation has three major problems:

### 1. Dimensionality

Real vocabularies have tens or hundreds of thousands of words:

```python
# Typical vocabulary sizes
vocab_sizes = {
    'Small model': 10_000,
    'GPT-2': 50_257,
    'BERT': 30_522,
    'Large vocabulary': 100_000+
}

# One-hot vector for 50,000 word vocabulary
# Each word = 50,000 dimensions, 49,999 zeros
```

Operations on 50,000-dimensional sparse vectors are inefficient. Models processing these representations have enormous parameter counts.

### 2. No Similarity Information

The mathematical problem:

```python
import numpy as np

cat = np.array([0, 1, 0, 0, 0, 0, 0])
dog = np.array([0, 0, 0, 0, 0, 1, 0])
democracy = np.array([0, 0, 0, 0, 0, 0, 1])

# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_sim(cat, dog))        # 0.0
print(cosine_sim(cat, democracy))  # 0.0

# Cat is equally dissimilar to dog and democracy!
# But semantically, cat and dog are related (both animals)
```

One-hot vectors are **orthogonal**: every word is maximally different from every other word. There's no notion of similarity encoded in the representation.

### 3. No Generalization

When a model learns something about "cat," it learns nothing about "kitten" or "feline" or "dog." Each word is an island.

```python
# Suppose model learns: "The cat sat on the mat"
# It gets no help with: "The kitten sat on the rug"
# Even though kitten≈cat and rug≈mat semantically

# Every word combination must be learned independently
```

This meant NLP models needed enormous amounts of data to cover all word combinations—or they generalized poorly.

## What We Want: The Distributional Hypothesis

The path forward came from linguistics. In 1957, linguist John Firth wrote: "You shall know a word by the company it keeps." This is the **distributional hypothesis**: words that appear in similar contexts have similar meanings.

Consider these sentences:
```
The dog ran across the park.
The cat ran across the park.
The puppy ran across the yard.
```

"Dog," "cat," and "puppy" appear in similar contexts (before "ran across the..."). They're all things that can run across parks. This contextual similarity reflects semantic similarity—they're all animals.

The distributional hypothesis suggests we can learn word meaning from usage patterns, not from explicit definitions.

## Early Distributional Approaches

Before neural embeddings, researchers built word representations from co-occurrence statistics:

### Term-Document Matrix

```python
# Documents as columns, words as rows
# Cell (i,j) = count of word i in document j

documents = [
    "cat sat mat",
    "dog ran park",
    "cat dog animal"
]

# Term-document matrix (simplified)
#        doc1  doc2  doc3
# cat      1     0     1
# dog      0     1     1
# sat      1     0     0
# ran      0     1     0
# mat      1     0     0
# park     0     1     0
# animal   0     0     1
```

Words with similar document distributions (appearing in similar documents) are likely related.

### Word Co-occurrence Matrix

```python
# Count how often words appear together within a window

text = "the cat sat on the mat the dog ran on the mat"
window_size = 2

# Co-occurrence matrix (simplified)
#        the  cat  sat  on   mat  dog  ran
# the     -    2    1    2    2    1    1
# cat     2    -    1    0    0    1    0
# sat     1    1    -    1    1    0    0
# ...

# Words appearing in similar contexts have similar row vectors
```

### Latent Semantic Analysis (LSA)

Apply SVD (Singular Value Decomposition) to reduce the co-occurrence matrix:

```python
from scipy import linalg

# Co-occurrence matrix: (vocab_size × vocab_size)
cooccurrence_matrix = build_cooccurrence(corpus)

# SVD: decompose into U @ S @ V.T
U, S, Vt = linalg.svd(cooccurrence_matrix, full_matrices=False)

# Keep top k dimensions
k = 300
word_vectors = U[:, :k] @ np.diag(S[:k])

# Now each word is a 300-dimensional dense vector
# Similar words have similar vectors
```

LSA and similar techniques (like Latent Dirichlet Allocation for topics) showed that low-dimensional dense representations could capture semantic similarity. But they had limitations:
- Counting doesn't distinguish important from unimportant co-occurrences
- Linear dimensionality reduction might miss complex patterns
- No clear way to handle out-of-vocabulary words

## The Neural Path: Learning Representations

The breakthrough insight: instead of computing statistics and then reducing dimensions, learn dense word representations directly by training a neural network on a prediction task.

Yoshua Bengio's 2003 paper "A Neural Probabilistic Language Model" showed this was possible:

```python
class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size):
        super().__init__()
        # Learned word embeddings
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # Predict next word from context
        self.hidden = nn.Linear(context_size * embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_words):
        # Look up embeddings
        embeds = self.embeddings(context_words)  # (batch, context, embed)
        embeds = embeds.view(embeds.size(0), -1)  # Flatten context

        # Predict next word
        hidden = torch.tanh(self.hidden(embeds))
        logits = self.output(hidden)
        return logits
```

The embeddings were learned end-to-end by predicting next words. Words that predict similar next words develop similar embeddings.

But training was slow. The output layer over the full vocabulary was expensive. The model was ahead of its time—the full realization would come a decade later with Word2Vec.

## Properties of Good Word Vectors

What should word representations capture?

```python
# Similarity: Related words should be close
cosine_sim(embed("king"), embed("queen"))  # High
cosine_sim(embed("king"), embed("banana"))  # Low

# Analogy: Relationships should be consistent
# "man is to woman as king is to queen"
embed("king") - embed("man") + embed("woman") ≈ embed("queen")

# Clustering: Semantic categories should group together
# Animals: [cat, dog, horse, elephant] cluster together
# Countries: [France, Germany, Japan] cluster together

# Regularity: Similar relationships, similar vector offsets
# country-capital relationship:
embed("France") - embed("Paris") ≈ embed("Japan") - embed("Tokyo")
```

These properties would emerge naturally from the neural approaches we'll explore in the next lesson.

## The Dense Representation Advantage

Dense embeddings transform NLP:

```python
# One-hot: 50,000 dimensions, mostly zeros
one_hot_size = 50_000  # Only 1 non-zero

# Dense embedding: 300 dimensions, all meaningful
embedding_size = 300  # All values contribute

# Parameter savings
# LSTM with one-hot input: 50,000 × hidden_size parameters
# LSTM with embeddings: 300 × hidden_size parameters

# Generalization
# "I love my cat" → positive sentiment
# "I love my kitten" → also recognized as positive
# Because embed("cat") ≈ embed("kitten")
```

Dense representations also enable:
- Efficient nearest neighbor search
- Smooth interpolation between words
- Transfer learning (use embeddings trained on large corpus)
- Visualization (project to 2D, see clusters)

## Key Takeaways

- One-hot vectors represent words as orthogonal, high-dimensional sparse vectors—no similarity information is encoded
- The distributional hypothesis: words in similar contexts have similar meanings—this is the foundation for learned representations
- Early approaches (LSA, co-occurrence matrices) derived dense vectors from statistics, but had limitations
- Neural language models learn embeddings end-to-end by predicting words from context
- Good word vectors capture similarity, analogy, clustering, and regularity—encoding semantic relationships as geometric relationships

## Further Reading

- Firth, J. R. (1957). "A Synopsis of Linguistic Theory 1930-1955"
- Deerwester, S., et al. (1990). "Indexing by Latent Semantic Analysis"
- Bengio, Y., et al. (2003). "A Neural Probabilistic Language Model"
- Turney, P., & Pantel, P. (2010). "From Frequency to Meaning: Vector Space Models of Semantics"

---
*Estimated reading time: 10 minutes*
