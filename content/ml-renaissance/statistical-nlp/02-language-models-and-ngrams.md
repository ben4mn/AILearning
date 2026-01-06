# Language Models and N-grams

## Introduction

At the heart of the statistical NLP revolution lay a deceptively simple idea: predict the next word. If you can estimate the probability of each possible word following a given context, you have a **language model**—and language models, it turned out, were the Swiss Army knife of natural language processing.

From speech recognition to machine translation to spelling correction, language models provided a way to distinguish fluent, probable sequences from awkward, unlikely ones. And the workhorse of 1990s language modeling was the **n-gram**: a model based on counting short sequences of words in large text corpora.

In this lesson, we'll explore how n-gram language models work, why they were so successful, and what their limitations revealed about the nature of language itself.

## What Is a Language Model?

A **language model** assigns probabilities to sequences of words. Given a sentence like "The cat sat on the," a language model estimates the probability of each possible next word: "mat" might be likely, "metaphysics" unlikely, and "asdfgh" essentially impossible.

Formally, a language model estimates:

**P(w₁, w₂, ..., wₙ)** — the probability of a complete sequence

Using the chain rule of probability, we can decompose this as:

**P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁,...,wₙ₋₁)**

This says: the probability of a sentence is the probability of the first word, times the probability of the second word given the first, times the probability of the third given the first two, and so on.

The challenge is estimating these conditional probabilities. In principle, P(wₙ|w₁,...,wₙ₋₁) depends on the entire preceding context—every word that came before. But we never have enough data to reliably estimate probabilities for most long contexts.

```python
# The problem: most word sequences never appear in training data
# Even with a billion words of text, specific sequences are rare

context = "The cat sat on the fluffy purple"
next_word = "unicorn"

# We probably never saw this exact context in training
# So P(unicorn | The cat sat on the fluffy purple) = ???
```

## The N-gram Approximation

The n-gram solution is elegant in its simplicity: **pretend that only the last few words matter**. Instead of conditioning on the entire history, we condition only on the previous n-1 words.

A **bigram** (n=2) model assumes each word depends only on the immediately preceding word:
- P(wₙ|w₁,...,wₙ₋₁) ≈ P(wₙ|wₙ₋₁)

A **trigram** (n=3) model uses two words of context:
- P(wₙ|w₁,...,wₙ₋₁) ≈ P(wₙ|wₙ₋₂,wₙ₋₁)

This is the **Markov assumption**: the future depends only on the recent past, not the distant past.

```python
# Training an n-gram language model
from collections import defaultdict

def train_ngram_model(corpus, n=3):
    """Train an n-gram language model from a corpus."""
    counts = defaultdict(lambda: defaultdict(int))
    context_totals = defaultdict(int)

    for sentence in corpus:
        # Add start and end tokens
        tokens = ['<s>'] * (n-1) + sentence.split() + ['</s>']

        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i:i+n-1])
            word = tokens[i+n-1]
            counts[context][word] += 1
            context_totals[context] += 1

    # Convert counts to probabilities
    probs = {}
    for context in counts:
        probs[context] = {}
        for word in counts[context]:
            probs[context][word] = counts[context][word] / context_totals[context]

    return probs

# Example usage
corpus = ["the cat sat on the mat", "the dog sat on the floor"]
model = train_ngram_model(corpus, n=2)
# model[('the',)]['cat'] ≈ 0.33 (the cat appears 1/3 of times after 'the')
```

To estimate these probabilities, we simply count! We go through a large corpus, count how often each n-gram appears, and normalize:

**P(wₙ|wₙ₋₂,wₙ₋₁) = Count(wₙ₋₂,wₙ₋₁,wₙ) / Count(wₙ₋₂,wₙ₋₁)**

This is **Maximum Likelihood Estimation (MLE)**: the probability is the observed frequency.

## The Sparsity Problem

There's a catch. Language is creative—people constantly produce new word combinations. Even in a billion-word corpus, most possible trigrams never appear. What probability should we assign to an unseen n-gram?

If we use pure MLE, unseen n-grams get probability zero. This is catastrophic: a single unseen trigram makes an entire sentence have zero probability. "The cat sat on the mat" might get probability zero just because "on the mat" never appeared in training.

This is the **sparsity problem**, and solving it consumed much of 1990s NLP research. The solution: **smoothing**—redistributing probability mass from seen events to unseen ones.

### Add-One (Laplace) Smoothing

The simplest approach: pretend every n-gram appeared at least once.

```python
def laplace_smoothed_prob(context, word, counts, vocab_size):
    """Add-one smoothing for n-gram probabilities."""
    count = counts[context][word] + 1
    total = sum(counts[context].values()) + vocab_size
    return count / total
```

This works but is too aggressive—it steals too much probability from common words.

### Good-Turing Smoothing

Good-Turing estimation, developed during WWII by Alan Turing and I.J. Good for codebreaking, estimates how much probability to reserve for unseen events based on how many events appeared exactly once.

The intuition: if many events appeared exactly once, we should expect many more events that didn't appear at all (just by chance). The number of singletons tells us about the "missing mass" of probability.

### Kneser-Ney Smoothing

By the mid-1990s, **Kneser-Ney smoothing** emerged as the gold standard. It combined discounting (subtracting a fixed amount from each count) with a clever backoff scheme that considered not just word frequency, but **word versatility**—how many different contexts a word appeared in.

For example, "Francisco" is a common word, but almost always follows "San." In contrast, "the" appears in many different contexts. Kneser-Ney captures this distinction, making it better at predicting words in new contexts.

```python
# Kneser-Ney uses continuation probability for backoff
# P_continuation(w) = |{v : count(v,w) > 0}| / |{(v',w') : count(v',w') > 0}|

# "Francisco" has low continuation probability (only follows "San")
# "the" has high continuation probability (follows many words)
```

## Evaluation: Perplexity

How do we measure if one language model is better than another? The standard metric is **perplexity**: how surprised the model is by test data.

Perplexity is the inverse probability of the test set, normalized by number of words:

**Perplexity = 2^(-1/N × Σ log₂ P(wᵢ|context))**

Lower perplexity means the model assigns higher probability to the test data—it's less "perplexed" by what it sees. A perplexity of 100 means the model is as uncertain as if choosing uniformly among 100 options at each step.

```python
import math

def perplexity(model, test_sentences, n):
    """Calculate perplexity of n-gram model on test data."""
    log_prob_sum = 0
    word_count = 0

    for sentence in test_sentences:
        tokens = ['<s>'] * (n-1) + sentence.split() + ['</s>']

        for i in range(n-1, len(tokens)):
            context = tuple(tokens[i-n+1:i])
            word = tokens[i]

            prob = model.get(context, {}).get(word, 1e-10)  # Smoothing
            log_prob_sum += math.log2(prob)
            word_count += 1

    return 2 ** (-log_prob_sum / word_count)

# Lower is better: perplexity 50 beats perplexity 100
```

Throughout the 1990s, researchers competed to reduce perplexity on standard benchmarks. Trigram models with Kneser-Ney smoothing achieved perplexities around 100-150 on news text—a remarkable improvement over simpler models.

## What N-grams Captured (and Missed)

N-gram models were surprisingly good at capturing local linguistic patterns:

**What they captured:**
- Word collocations: "New York," "United States"
- Local syntax: "the" followed by nouns, "is" followed by verbs/adjectives
- Common phrases: "in order to," "on the other hand"
- Topic words: "president" makes "election" more likely

**What they missed:**
- Long-distance dependencies: "The dog that the cat that the rat bit chased ran away"
- Semantic coherence: n-grams can't distinguish meaningful from nonsensical
- Global structure: document-level organization, narrative arc

The fundamental limitation is the fixed context window. A trigram model treats "The hungry wolf ate the sheep" and "The friendly wolf protected the sheep" identically after seeing "ate the"—it can't remember the wolf was hungry versus friendly.

## Legacy and Transition

N-gram language models dominated NLP through the 2000s. They powered:
- Speech recognition systems in phones and assistants
- Machine translation systems from Google and others
- Spelling and grammar correction
- Text input prediction on mobile keyboards

But researchers increasingly recognized their limitations. Newer approaches would use continuous representations (word embeddings) and neural networks that could capture longer-range dependencies. The n-gram revolution was itself revolutionized—but the core insight remained: learn from data, measure with probability.

## Key Takeaways

- A language model estimates probabilities of word sequences, enabling systems to distinguish fluent from awkward text
- N-gram models make a Markov assumption: predict the next word based only on the previous n-1 words
- Smoothing techniques like Kneser-Ney are essential to handle unseen n-grams in test data
- Perplexity measures how well a language model predicts held-out test data
- N-grams capture local patterns effectively but miss long-distance dependencies

## Further Reading

- Chen, Stanley and Goodman, Joshua. "An Empirical Study of Smoothing Techniques for Language Modeling" (1999) - Comprehensive comparison
- Jurafsky, Daniel and Martin, James. *Speech and Language Processing*, Chapter 3 - Accessible textbook treatment
- Kneser, Reinhard and Ney, Hermann. "Improved backing-off for m-gram language modeling" (1995) - The smoothing breakthrough
- Shannon, Claude. "Prediction and Entropy of Printed English" (1951) - Early statistical analysis

---
*Estimated reading time: 10 minutes*
