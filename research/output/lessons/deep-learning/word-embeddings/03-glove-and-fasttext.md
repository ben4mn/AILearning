# GloVe and FastText: Extending Word Embeddings

## Introduction

Word2Vec's success sparked a wave of research into word embeddings. Two particularly influential extensions emerged: GloVe from Stanford, which unified neural and count-based approaches, and FastText from Facebook, which learned representations for subword units. Each addressed specific limitations of Word2Vec while achieving comparable or better performance.

In this lesson, we'll explore how GloVe's global co-occurrence matrix approach differs from Word2Vec's local context windows, and how FastText's subword embeddings handle rare words and morphological relationships. Understanding these alternatives deepens our appreciation for what word embeddings capture and how to choose among them for different applications.

## GloVe: Global Vectors for Word Representation

Jeffrey Pennington, Richard Socher, and Christopher Manning at Stanford introduced GloVe in 2014. Their insight: Word2Vec implicitly factorizes a word-context co-occurrence matrix. Why not do it explicitly?

### The Co-occurrence Matrix

First, count how often words appear together within a window across the entire corpus:

```python
def build_cooccurrence_matrix(corpus, vocab, window=5):
    """
    X[i,j] = how often word i appears within window of word j
    """
    vocab_size = len(vocab)
    X = np.zeros((vocab_size, vocab_size))

    for sentence in corpus:
        for i, word_i in enumerate(sentence):
            for j in range(max(0, i-window), min(len(sentence), i+window+1)):
                if i != j:
                    word_j = sentence[j]
                    # Distance weighting: closer words count more
                    distance = abs(i - j)
                    X[vocab[word_i], vocab[word_j]] += 1.0 / distance

    return X
```

### The GloVe Objective

GloVe learns embeddings that predict log co-occurrence counts:

```python
def glove_objective(W, b, X):
    """
    Minimize weighted squared error between
    dot(w_i, w_j) + b_i + b_j and log(X_ij)
    """
    vocab_size = W.shape[0]
    loss = 0

    for i in range(vocab_size):
        for j in range(vocab_size):
            if X[i, j] > 0:
                # Weighting function: caps influence of very common pairs
                weight = min((X[i, j] / x_max) ** 0.75, 1.0)

                # Target: log of co-occurrence count
                target = np.log(X[i, j])

                # Prediction: dot product plus biases
                pred = np.dot(W[i], W[j]) + b[i] + b[j]

                loss += weight * (pred - target) ** 2

    return loss
```

The key insight: **log(X_ij) should be proportional to w_i dot w_j**. This is exactly what matrix factorization would give us, but with a learned weighting scheme.

### Why the Weighting Matters

Not all co-occurrences are equally informative:

```python
def weighting_function(x_ij, x_max=100, alpha=0.75):
    """
    - Very common pairs (the, of) → weight capped at 1
    - Rare pairs → lower weight, less influence
    - Prevents frequent pairs from dominating
    """
    if x_ij < x_max:
        return (x_ij / x_max) ** alpha
    else:
        return 1.0
```

"The" and "of" co-occur constantly, but this tells us little about semantics. The weighting caps their influence while still learning from rare, informative co-occurrences.

### GloVe vs Word2Vec

| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| Training | Stochastic (random samples) | Global (all co-occurrences) |
| Objective | Predict context words | Factorize log-counts |
| Efficiency | Online, easy to parallelize | Requires matrix storage |
| Theory | Neural network learning | Matrix factorization + weighting |
| Performance | Similar | Similar |

In practice, GloVe and Word2Vec produce embeddings of comparable quality. GloVe's advantage is clearer theoretical grounding; Word2Vec's advantage is easier incremental training.

### Using GloVe

```python
# Load pretrained GloVe embeddings
def load_glove(filepath):
    embeddings = {}
    with open(filepath, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# GloVe comes pretrained on:
# - Wikipedia + Gigaword (6B tokens)
# - Common Crawl (42B, 840B tokens)
# - Twitter (27B tokens)
```

## FastText: Subword Embeddings

Facebook AI Research (FAIR) released FastText in 2016, addressing a key Word2Vec limitation: handling of unknown and morphologically related words.

### The Subword Idea

Instead of one vector per word, FastText represents words as bags of character n-grams:

```python
def get_subwords(word, min_n=3, max_n=6):
    """
    Extract character n-grams from word
    """
    # Add boundary markers
    word = f"<{word}>"

    subwords = []
    for n in range(min_n, max_n + 1):
        for i in range(len(word) - n + 1):
            subwords.append(word[i:i+n])

    return subwords

# Example
get_subwords("where")
# ['<wh', 'whe', 'her', 'ere', 're>',
#  '<whe', 'wher', 'here', 'ere>',
#  '<wher', 'where', 'here>',
#  '<where', 'where>']
```

The word vector is the sum of its subword vectors:

```python
class FastTextEmbedding(nn.Module):
    def __init__(self, num_subwords, embed_dim):
        super().__init__()
        self.subword_embed = nn.Embedding(num_subwords, embed_dim)

    def forward(self, word):
        # Get subword indices
        subword_ids = get_subword_ids(word)

        # Sum subword vectors
        subword_vecs = self.subword_embed(subword_ids)
        word_vec = subword_vecs.sum(dim=0)

        return word_vec
```

### Benefits of Subwords

**1. Handling Unknown Words**

```python
# Word2Vec: no representation for unknown words
word2vec.get("unfamiliarword123")  # KeyError!

# FastText: build from subwords
fasttext.get("unfamiliarword123")
# Uses subwords: <un, unf, nfa, fam, ami, mil, ili, lia, iar, ...
# Even nonsense words get reasonable vectors
```

**2. Morphological Relationships**

```python
# Words with shared morphemes get related vectors
# "unhappy" contains subwords from "happy"
# "happiness" contains subwords from "happy"
# "unhappiness" contains subwords from both

subwords_happy = get_subwords("happy")      # hap, app, ppy, ...
subwords_unhappy = get_subwords("unhappy")  # unh, nha, hap, app, ppy, ...

# Overlap in subwords → similarity in vectors
```

**3. Rare Words**

Even rare words share subwords with common words:

```python
# "electroencephalograph" is rare
# But shares subwords with:
# - "electro-" (electric, electron, electrode)
# - "-graph" (photograph, telegraph)

# FastText can generalize from common words to rare ones
```

### FastText Training

FastText uses the same Skip-gram training as Word2Vec, just with subword representations:

```python
def fasttext_skipgram_loss(target_word, context_word, negatives):
    # Get word vectors (sum of subword vectors)
    target_vec = embed(target_word)
    context_vec = embed(context_word)

    # Same negative sampling loss as Word2Vec
    loss = -log(sigmoid(dot(target_vec, context_vec)))
    for neg in negatives:
        neg_vec = embed(neg)
        loss -= log(sigmoid(-dot(target_vec, neg_vec)))

    return loss
```

### Hashing for Efficiency

With all possible n-grams, the vocabulary would be enormous. FastText uses hashing:

```python
def hash_subword(subword, bucket_size=2_000_000):
    """
    Hash n-gram to fixed bucket
    Collisions are acceptable (words still differ)
    """
    return hash(subword) % bucket_size

# Storage: 2M vectors for subwords
# Plus vectors for actual words in vocabulary
```

### FastText for Classification

FastText also provides fast text classification:

```python
# FastText classifier: average word vectors, linear classifier
class FastTextClassifier:
    def __init__(self, vocab_size, embed_dim, num_classes):
        self.embed = FastTextEmbedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, text):
        # Average word vectors
        word_vecs = [self.embed(word) for word in text]
        text_vec = torch.mean(torch.stack(word_vecs), dim=0)

        # Classify
        return self.classifier(text_vec)
```

This simple architecture trains in minutes and often matches more complex models on classification tasks.

## Comparing the Approaches

```python
# When to use each:

use_word2vec_when = [
    "Speed is critical",
    "Vocabulary is well-defined",
    "No morphological complexity",
    "English or similar languages",
]

use_glove_when = [
    "Theoretical interpretation matters",
    "Large pretrained models available",
    "Static corpus (not streaming)",
]

use_fasttext_when = [
    "Many rare or unknown words",
    "Morphologically rich languages (Finnish, Turkish)",
    "User-generated content (typos, slang)",
    "Need classification as well as embeddings",
]
```

### Performance Comparison

On standard benchmarks:

```
Word Similarity (Spearman correlation):
  Word2Vec:  ~0.70
  GloVe:     ~0.70
  FastText:  ~0.72

Analogy Accuracy:
  Word2Vec:  ~0.74
  GloVe:     ~0.75
  FastText:  ~0.77

Rare Word Similarity:
  Word2Vec:  ~0.45
  GloVe:     ~0.44
  FastText:  ~0.55  ← Better on rare words
```

FastText's advantage on rare words comes from subword generalization.

## The Embedding Ecosystem

By 2016, embeddings had become infrastructure:

```python
# Pretrained embeddings available for:
languages = ["English", "Spanish", "Chinese", "Arabic", ...]  # 150+ languages
sizes = [50, 100, 200, 300, 1000]  # dimension options
sources = ["Wikipedia", "Common Crawl", "Twitter", "News", ...]

# Standard workflow:
# 1. Download pretrained embeddings
# 2. Initialize embedding layer
# 3. Fine-tune on task (or freeze)
```

This ecosystem lowered the barrier to NLP. Anyone could build on vectors trained on billions of words.

## Limitations Shared by All

Despite their differences, Word2Vec, GloVe, and FastText share limitations:

```python
limitations = {
    'polysemy': 'One vector per word, regardless of meaning',
    'context': 'Same vector regardless of sentence context',
    'composition': 'No principled way to combine word vectors',
    'knowledge': 'No world knowledge beyond distributional patterns',
}

# Example: "bank" has one vector for:
# - financial institution
# - river bank
# - memory bank
# Context-dependent embeddings (BERT, etc.) will address this
```

## Key Takeaways

- GloVe explicitly factorizes a weighted log co-occurrence matrix, providing theoretical grounding for why embeddings capture semantic relationships
- GloVe's weighting function prevents overly common pairs from dominating while preserving signal from rare, informative co-occurrences
- FastText represents words as sums of character n-gram vectors, enabling handling of unknown words and morphological relationships
- FastText excels on morphologically rich languages and user-generated content with typos and rare words
- All three approaches produce comparable embeddings for common words, differing mainly in handling of rare words and theoretical foundations

## Further Reading

- Pennington, J., Socher, R., & Manning, C. (2014). "GloVe: Global Vectors for Word Representation"
- Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information"
- Joulin, A., et al. (2017). "Bag of Tricks for Efficient Text Classification"
- FastText documentation and pretrained models: https://fasttext.cc/

---
*Estimated reading time: 11 minutes*
