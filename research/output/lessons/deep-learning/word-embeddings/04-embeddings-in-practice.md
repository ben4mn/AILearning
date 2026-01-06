# Embeddings in Practice: Applications and Challenges

## Introduction

Word embeddings revolutionized NLP not through any single application, but by becoming universal infrastructure. Whether you were building a sentiment classifier, a search engine, or a chatbot, you probably started with Word2Vec or GloVe. The vectors became a shared language for representing meaning computationally.

But embeddings came with challenges. They captured human biases alongside human semantics. They sometimes encoded relationships that seemed reasonable in training data but were problematic in applications. And as researchers explored their properties, they discovered both remarkable capabilities and troubling limitations.

In this lesson, we'll explore how embeddings were used in practice, the bias issues that emerged, and how embeddings evolved to become the foundation for modern NLP systems.

## Transfer Learning with Embeddings

The most common use of pretrained embeddings was initialization:

```python
import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_size, num_classes):
        super().__init__()
        vocab_size, embed_dim = pretrained_embeddings.shape

        # Initialize embeddings from pretrained vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # Option: freeze embeddings or fine-tune
        self.embedding.weight.requires_grad = True  # Fine-tune

        # Task-specific layers
        self.lstm = nn.LSTM(embed_dim, hidden_size, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        output, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        return self.classifier(hidden)
```

This pattern was ubiquitous:

1. **Load pretrained embeddings** (Word2Vec, GloVe, FastText)
2. **Build task-specific architecture** on top
3. **Fine-tune or freeze** embeddings based on data size

With small datasets, freezing embeddings prevented overfitting. With larger datasets, fine-tuning improved task performance.

## Semantic Search and Retrieval

Embeddings enabled semantic search beyond keyword matching:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors

    def embed_document(self, text):
        """Average word vectors"""
        words = text.lower().split()
        vectors = [self.word_vectors[w] for w in words
                   if w in self.word_vectors]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.word_vectors.vector_size)

    def search(self, query, documents, top_k=5):
        query_vec = self.embed_document(query)
        doc_vecs = [self.embed_document(doc) for doc in documents]

        similarities = cosine_similarity([query_vec], doc_vecs)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(documents[i], similarities[i]) for i in top_indices]

# Example: "affordable laptop" matches "cheap computer"
# even without exact keyword match
```

Applications included:
- **FAQ matching**: Find similar questions in a knowledge base
- **Product search**: "Affordable laptop" → "budget notebook computer"
- **Document clustering**: Group similar documents by content

## Analogical Reasoning

The analogy property enabled creative applications:

```python
def find_analogy(word_vectors, a, b, c, restrict_vocab=50000):
    """
    a is to b as c is to ?
    Returns b - a + c
    """
    # Get vectors
    vec = word_vectors[b] - word_vectors[a] + word_vectors[c]

    # Find nearest word (excluding a, b, c)
    similarities = cosine_similarity([vec], word_vectors.vectors)[0]

    # Remove input words
    for word in [a, b, c]:
        similarities[word_vectors.key_to_index[word]] = -np.inf

    best_idx = np.argmax(similarities)
    return word_vectors.index_to_key[best_idx]

# Applications:
# - "Paris" - "France" + "Germany" = "Berlin" → Capital lookup
# - "walked" - "walk" + "swim" = "swam" → Tense conversion
# - "king" - "man" + "woman" = "queen" → Gender transformation
```

These transformations worked surprisingly well for:
- Morphological transformations (tense, plurals)
- Geographical relationships (country-capital, country-language)
- Some semantic relationships

But they also revealed problems we'll discuss shortly.

## Embeddings for Visualization

Projecting embeddings to 2D revealed semantic structure:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(word_vectors, words, perplexity=30):
    # Get vectors for selected words
    vectors = [word_vectors[w] for w in words if w in word_vectors]
    valid_words = [w for w in words if w in word_vectors]

    # Project to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    projected = tsne.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.6)
    for i, word in enumerate(valid_words):
        plt.annotate(word, (projected[i, 0], projected[i, 1]))
    plt.title("Word Embedding Space")
    plt.show()

# Visualize: countries cluster, animals cluster, verbs cluster
```

These visualizations showed:
- Clear semantic clusters
- Relationships encoded as directions
- The geometric nature of meaning in embedding space

## The Bias Problem

As embeddings became widely deployed, researchers discovered they encoded societal biases:

```python
# Gender stereotypes in embeddings
def profession_gender_bias(word_vectors):
    """
    Measure how professions associate with gender
    """
    # Gender direction: "he" - "she"
    gender_direction = word_vectors['he'] - word_vectors['she']

    professions = ['doctor', 'nurse', 'engineer', 'teacher',
                   'programmer', 'homemaker', 'scientist', 'secretary']

    for profession in professions:
        proj = np.dot(word_vectors[profession], gender_direction)
        direction = "male" if proj > 0 else "female"
        print(f"{profession}: {direction} ({abs(proj):.3f})")

# Typical results:
# engineer: male (0.12)
# nurse: female (0.15)
# programmer: male (0.10)
# homemaker: female (0.18)
# scientist: male (0.08)
# secretary: female (0.14)
```

The famous paper "Man is to Computer Programmer as Woman is to Homemaker" (Bolukbasi et al., 2016) showed:

```python
# Problematic analogies
find_analogy("man", "computer_programmer", "woman")  # → "homemaker"
find_analogy("man", "doctor", "woman")               # → "nurse"
```

These weren't bugs—they reflected patterns in the training data (news articles, web text). But deploying these embeddings in applications could perpetuate stereotypes:
- Resume screening favoring male-associated terms
- Search results reinforcing occupational stereotypes
- Recommendations that limited users' options

### Debiasing Attempts

Researchers developed debiasing techniques:

```python
def debias_word_vectors(word_vectors, gender_pairs):
    """
    Remove gender direction from neutral words
    """
    # Find gender direction from definitional pairs
    # (man, woman), (he, she), (him, her)
    gender_vecs = [word_vectors[m] - word_vectors[f]
                   for m, f in gender_pairs]
    gender_direction = np.mean(gender_vecs, axis=0)
    gender_direction /= np.linalg.norm(gender_direction)

    # For neutral words, project out gender component
    debiased = {}
    for word, vec in word_vectors.items():
        if is_neutral_word(word):
            # Remove gender component
            gender_component = np.dot(vec, gender_direction) * gender_direction
            debiased[word] = vec - gender_component
        else:
            debiased[word] = vec

    return debiased
```

But debiasing proved incomplete:
- Stereotypes can be recovered from "debiased" embeddings
- Different types of bias require different treatments
- The fundamental issue is biased training data

## Compositionality Challenges

Word embeddings don't naturally compose. How do you represent a phrase?

```python
# Naive: average word vectors
def phrase_embedding_average(phrase, word_vectors):
    words = phrase.lower().split()
    vecs = [word_vectors[w] for w in words if w in word_vectors]
    return np.mean(vecs, axis=0)

# Problem: loses word order
# "dog bites man" ≈ "man bites dog" (same average)
# "not good" ≈ "good" (average is close to "good")
```

Better approaches emerged:
- **Weighted averaging** (by importance)
- **Recurrent encoding** (LSTM over word vectors)
- **Sentence embeddings** (dedicated models like InferSent, USE)

But the fundamental limitation remained: word embeddings capture word-level meaning, not compositional semantics.

## The Road to Contextualized Embeddings

Word2Vec/GloVe gave one vector per word. But words have multiple meanings:

```python
# Same word, different meanings
sentences = [
    "I went to the bank to deposit money",  # Financial institution
    "We sat on the river bank",              # Riverbank
    "You can bank on her support",           # Rely upon
]

# Word2Vec: same vector for "bank" in all three
# What we want: different vectors based on context
```

This limitation drove the development of **contextualized embeddings**:
- **ELMo (2018)**: LSTM-based, context-dependent representations
- **BERT (2018)**: Transformer-based, bidirectional context
- **GPT (2018+)**: Autoregressive contextualized representations

These models replaced static embeddings with dynamic, context-sensitive representations—but they built on the foundation that Word2Vec established.

## Word Embeddings as Foundation

Despite being superseded for many tasks, word embeddings remain important:

```python
# Still used for:
use_cases = [
    "Initialization for specialized domains",
    "Lightweight models for edge devices",
    "Interpretability (can examine individual vectors)",
    "Feature engineering for ML pipelines",
    "Cross-lingual transfer (aligned embeddings)",
]

# And conceptually:
# The idea of "meaning as geometry" underlies all modern NLP
# Tokens in GPT have embeddings
# BERT starts with token embeddings
# The transformer architecture operates on embedding space
```

The insight that words can be represented as points in a continuous space where distance reflects meaning—this foundational idea from Word2Vec permeates modern NLP.

## Key Takeaways

- Pretrained embeddings enabled transfer learning for NLP: initialize from large-corpus training, then fine-tune on task-specific data
- Semantic search, analogical reasoning, and visualization became possible through embedding similarity
- Word embeddings encode societal biases from training data, leading to stereotypical associations that can be harmful in applications
- Debiasing techniques help but don't fully solve the problem—biases can often be recovered from "debiased" embeddings
- Static word embeddings don't handle polysemy (multiple meanings) or compositionality (phrase meaning), motivating contextualized embeddings like BERT

## Further Reading

- Bolukbasi, T., et al. (2016). "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings"
- Caliskan, A., Bryson, J., & Narayanan, A. (2017). "Semantics derived automatically from language corpora contain human-like biases"
- Peters, M., et al. (2018). "Deep contextualized word representations" (ELMo)
- Ethayarajh, K. (2019). "How Contextual are Contextualized Word Representations?"

---
*Estimated reading time: 10 minutes*
