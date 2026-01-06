# Word2Vec: The Embedding Revolution

## Introduction

In 2013, Tomas Mikolov and colleagues at Google published a pair of papers that would reshape NLP. Their method, Word2Vec, could learn high-quality word representations from billions of words in just hours—not days or weeks. The resulting embeddings exhibited remarkable properties: mathematical operations on word vectors produced meaningful results. "King - Man + Woman = Queen" wasn't just a demo; it was a window into the structure that language models were learning.

Word2Vec succeeded not through complexity but through simplicity. By stripping neural language models to their essence, Mikolov made them fast enough to train on internet-scale data. In this lesson, we'll understand how Word2Vec works, why it works so well, and what its success revealed about the nature of language.

## The Two Architectures

Word2Vec comes in two flavors, each solving the same core problem differently:

### Skip-gram: Predict Context from Word

Given a target word, predict the surrounding context words:

```python
# Training example for "the cat sat on the mat"
# Window size = 2
# Target: "sat"
# Context: ["the", "cat", "on", "the"]

# For each (target, context) pair:
# ("sat", "the")
# ("sat", "cat")
# ("sat", "on")
# ("sat", "the")

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target, context):
        # Embed target word
        target_vec = self.target_embed(target)  # (batch, embed_dim)

        # Embed context word
        context_vec = self.context_embed(context)  # (batch, embed_dim)

        # Score = dot product
        score = (target_vec * context_vec).sum(dim=1)

        return score
```

The objective: maximize the probability of observing actual context words given the target.

### CBOW: Predict Word from Context

Given surrounding context words, predict the target word:

```python
# Training example for "the cat sat on the mat"
# Window size = 2
# Context: ["the", "cat", "on", "the"]
# Target: "sat"

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_words):
        # Embed and average context
        embeds = self.embed(context_words)  # (batch, context_size, embed_dim)
        context_vec = embeds.mean(dim=1)    # (batch, embed_dim)

        # Predict target word
        logits = self.output(context_vec)   # (batch, vocab_size)
        return logits
```

CBOW averages context word vectors and predicts the center word.

**Trade-offs:**
- Skip-gram: Better for rare words (each word generates multiple examples)
- CBOW: Faster training (one prediction per context window)
- In practice, Skip-gram with negative sampling became most popular

## The Softmax Bottleneck

The naive approach requires computing softmax over the entire vocabulary:

```python
def naive_skip_gram_loss(target_vec, context_vec, all_word_vecs):
    # Score target with context
    positive_score = torch.dot(target_vec, context_vec)

    # Score target with ALL words (expensive!)
    all_scores = torch.matmul(target_vec, all_word_vecs.T)

    # Softmax probability
    log_prob = positive_score - torch.logsumexp(all_scores, dim=0)

    return -log_prob
```

With a vocabulary of 100,000 words, you're doing 100,000 dot products for every training example. This is prohibitively slow.

## Negative Sampling: The Key Innovation

The breakthrough was **negative sampling**: instead of normalizing over all words, sample a few "negative" examples:

```python
def negative_sampling_loss(target_vec, context_vec, negative_vecs, k=5):
    """
    Positive: (target, actual_context) should have high score
    Negatives: (target, random_words) should have low score
    """
    # Positive example: should be 1
    positive_score = torch.sigmoid(torch.dot(target_vec, context_vec))
    positive_loss = -torch.log(positive_score)

    # Negative examples: should be 0
    negative_loss = 0
    for neg_vec in negative_vecs:
        neg_score = torch.sigmoid(-torch.dot(target_vec, neg_vec))
        negative_loss -= torch.log(neg_score)

    return positive_loss + negative_loss
```

Instead of "what's the probability of this context word among all words," we ask "is this a real context word or a randomly sampled word?"

Negative samples are drawn from the **unigram distribution**, raised to the 3/4 power to give rare words slightly more weight:

```python
def sample_negatives(word_freqs, k=5):
    # Raise frequencies to 3/4 power
    adjusted_freqs = word_freqs ** 0.75
    probs = adjusted_freqs / adjusted_freqs.sum()

    # Sample k negative words
    return np.random.choice(vocab_size, size=k, p=probs)
```

With k=5-20 negatives per positive, training becomes ~10,000x faster.

## The Training Process

Word2Vec training is remarkably efficient:

```python
def train_word2vec(corpus, embed_dim=300, window=5, negatives=5, epochs=5):
    model = SkipGram(vocab_size, embed_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025)

    for epoch in range(epochs):
        # Linear learning rate decay
        lr = 0.025 * (1 - epoch / epochs)

        for sentence in corpus:
            for i, target_word in enumerate(sentence):
                # Get context words within window
                context = sentence[max(0,i-window):i] + \
                         sentence[i+1:min(len(sentence),i+window+1)]

                for context_word in context:
                    # Sample negatives
                    negatives = sample_negatives(word_freqs, k=5)

                    # Compute loss and update
                    loss = negative_sampling_loss(
                        model.target_embed(target_word),
                        model.context_embed(context_word),
                        [model.context_embed(n) for n in negatives]
                    )
                    loss.backward()
                    optimizer.step()
```

The original implementation was even more optimized:
- Written in C
- Used hierarchical softmax as an alternative to negative sampling
- Subsampling of frequent words (skip "the" sometimes)
- Multiple threads processing different parts of the corpus

Google trained Word2Vec on 100 billion words from Google News. Training completed in less than a day on a single machine.

## The Magic of Word Analogies

The most famous Word2Vec result: semantic analogies through vector arithmetic.

```python
def analogy(a, b, c, embeddings):
    """
    a is to b as c is to ?
    Example: man is to woman as king is to ?
    """
    # Get vectors
    a_vec = embeddings[a]
    b_vec = embeddings[b]
    c_vec = embeddings[c]

    # Compute target vector
    target = b_vec - a_vec + c_vec

    # Find nearest word (excluding a, b, c)
    similarities = cosine_similarity(target, embeddings)
    return most_similar_word(similarities, exclude=[a, b, c])

# Examples that work:
analogy("man", "woman", "king")     # → "queen"
analogy("paris", "france", "tokyo")  # → "japan"
analogy("slow", "slower", "fast")    # → "faster"
analogy("walk", "walking", "swim")   # → "swimming"
```

These analogies emerge without explicit training. The model learns that:
- Gender is encoded as a consistent direction in the embedding space
- Country-capital relationships form parallel lines
- Tenses follow regular patterns

## Why Does It Work?

The remarkable properties of Word2Vec embeddings arise from the training objective. Words that predict similar contexts get similar vectors.

Consider:
```
"The [king] sat on the throne"
"The [queen] sat on the throne"
"The [monarch] sat on the throne"
```

King, queen, and monarch appear in similar contexts, so they develop similar embeddings. But king and queen differ in contexts involving gender:

```
"The [king] married the princess"
"The [queen] married the prince"
```

This difference is captured in the vector offset, which is similar to other male-female pairs.

Mathematically, the training objective implicitly factorizes a word-context co-occurrence matrix, similar to older count-based methods but with better optimization.

## Practical Considerations

### Hyperparameters

```python
best_practices = {
    'embed_dim': 300,      # Standard; 100-500 common
    'window': 5,           # Context window; 5-10 for syntactic, 2-5 for semantic
    'min_count': 5,        # Ignore rare words
    'negatives': 5,        # Negative samples per positive
    'subsampling': 1e-5,   # Subsample frequent words
    'epochs': 5,           # Usually sufficient
    'learning_rate': 0.025, # With linear decay
}
```

### Using Pretrained Embeddings

Pretrained Word2Vec embeddings became widely used:

```python
# Load Google's pretrained Word2Vec
from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True
)

# Use as fixed features
def embed_sentence(sentence, word_vectors):
    vectors = [word_vectors[word] for word in sentence
               if word in word_vectors]
    return np.mean(vectors, axis=0)

# Or initialize neural network embeddings
embedding_layer = nn.Embedding(vocab_size, 300)
for word, idx in vocabulary.items():
    if word in word_vectors:
        embedding_layer.weight.data[idx] = torch.tensor(word_vectors[word])
```

### Limitations

```python
# Word2Vec limitations:

# 1. One vector per word (no polysemy)
# "bank" (financial) and "bank" (river) have the same vector

# 2. Out-of-vocabulary words get no representation
"unfamiliarwordasdf" in word_vectors  # False

# 3. No subword information
# "unhappy" doesn't obviously relate to "happy"

# 4. Context-independent
# "I love my bank" (positive) vs "I robbed a bank" (negative)
# Same vector for "bank" in both contexts
```

## Impact on NLP

Word2Vec transformed NLP practice:

**Before Word2Vec:**
- Hand-crafted features (POS tags, parse trees, gazettes)
- Sparse representations
- Task-specific feature engineering

**After Word2Vec:**
- Pretrained embeddings as standard input
- Dense representations everywhere
- Transfer learning via embedding initialization

Almost every NLP system from 2013-2018 used Word2Vec or similar embeddings:
- Sentiment analysis: Average word vectors, classify with SVM
- Named entity recognition: Word vectors as features for CRF
- Machine translation: Initialize encoder/decoder embeddings
- Question answering: Similarity search in embedding space

## Key Takeaways

- Word2Vec learns word embeddings by predicting context words (Skip-gram) or target words from context (CBOW)
- Negative sampling makes training tractable: compare positive examples against a few random negatives rather than the entire vocabulary
- Training on billions of words produces embeddings where vector arithmetic captures semantic relationships (king - man + woman = queen)
- Words appearing in similar contexts develop similar vectors—the distributional hypothesis in action
- Word2Vec embeddings became the standard input for NLP systems, enabling transfer learning from large text corpora

## Further Reading

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
- Goldberg, Y., & Levy, O. (2014). "word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method"
- Levy, O., & Goldberg, Y. (2014). "Neural Word Embedding as Implicit Matrix Factorization"

---
*Estimated reading time: 11 minutes*
