# Tokenization Algorithms: BPE, WordPiece, and Beyond

## Introduction

In the previous lesson, we explored why tokenization matters and what tokens are conceptually. Now we'll dive into *how* tokenizers actually work—the algorithms that decide where to split text and what vocabulary to use. These algorithms are elegant solutions to a challenging problem: how do you break text into pieces that are meaningful, efficient, and handle any input?

The three dominant approaches—Byte Pair Encoding (BPE), WordPiece, and SentencePiece—each take slightly different approaches but share common principles. Understanding these algorithms demystifies the preprocessing that happens before any LLM computation begins.

## The Core Problem

Before exploring solutions, let's understand the problem precisely:

**Given**: A large corpus of training text
**Goal**: Create a vocabulary of tokens and rules for splitting any text into those tokens

**Constraints**:
- Vocabulary should be fixed size (typically 30,000-100,000 tokens)
- Must handle any input, including novel words
- Should produce reasonably short sequences
- Common patterns should be single tokens
- Rare patterns should decompose into common pieces

Character-level fails (sequences too long). Word-level fails (vocabulary too large, can't handle new words). Subword tokenization threads the needle.

## Byte Pair Encoding (BPE)

### Origins

BPE was originally a data compression algorithm from 1994. In 2016, researchers realized it could solve subword tokenization beautifully. Today, it's the foundation of GPT models' tokenizers.

### The Algorithm

BPE builds a vocabulary iteratively by merging the most frequent pairs:

```python
# Simplified BPE algorithm
def train_bpe(corpus, vocab_size):
    # Start with character-level vocabulary
    vocab = set(all_characters_in_corpus)

    # Pre-tokenize into words, track frequencies
    word_freqs = count_word_frequencies(corpus)

    # Iteratively merge most frequent pairs
    while len(vocab) < vocab_size:
        # Count all adjacent pairs
        pair_counts = count_pairs(word_freqs)

        # Find most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)

        # Merge this pair into a new token
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)

        # Update word representations
        word_freqs = merge_pair_in_words(word_freqs, best_pair)

    return vocab, merge_rules
```

### Concrete Example

Let's trace through BPE on a small corpus:

```
Corpus: "low lower lowest lowly"

Step 0: Split into characters
Words: ['l','o','w'], ['l','o','w','e','r'], ['l','o','w','e','s','t'], ['l','o','w','l','y']
Initial vocab: {l, o, w, e, r, s, t, y}

Step 1: Count pairs
('l','o'): 4, ('o','w'): 4, ('w','e'): 2, ('e','r'): 1, ...
Most frequent: ('l','o') or ('o','w') - let's pick ('l','o')
Merge: create token 'lo'
Words become: ['lo','w'], ['lo','w','e','r'], ['lo','w','e','s','t'], ['lo','w','l','y']

Step 2: Count pairs again
('lo','w'): 4, ('w','e'): 2, ('w','l'): 1, ...
Most frequent: ('lo','w')
Merge: create token 'low'
Words become: ['low'], ['low','e','r'], ['low','e','s','t'], ['low','l','y']

Step 3: Count pairs
('low','e'): 2, ('e','r'): 1, ('e','s'): 1, ('low','l'): 1, ...
Most frequent: ('low','e')
Merge: create token 'lowe'
Words become: ['low'], ['lowe','r'], ['lowe','s','t'], ['low','l','y']

Continue until desired vocabulary size...
```

### Key Properties

**Greedy but effective**: BPE always merges the most frequent pair. This greedy approach works remarkably well in practice.

**Deterministic encoding**: Given a trained BPE model, any text encodes the same way:
```python
# Encoding uses learned merge rules in order
text = "lowest"
# Apply merges: l+o→lo, lo+w→low, low+e→lowe
# Result: ["lowe", "st"] or similar
```

**Novel word handling**: Unknown words decompose into known pieces:
```
"ChatGPT" → ["Chat", "G", "PT"]  # Even if never seen together
```

## WordPiece

### Origins and Use

Developed by Google, WordPiece powers BERT, DistilBERT, and related models. It's conceptually similar to BPE but differs in how it selects merges.

### The Algorithm

Instead of merging the most *frequent* pair, WordPiece merges the pair that maximizes likelihood:

```python
# WordPiece scoring (simplified)
def score_pair(pair, word_freqs):
    pair_freq = count_pair_frequency(pair, word_freqs)
    first_freq = count_token_frequency(pair[0])
    second_freq = count_token_frequency(pair[1])

    # Merge pairs where combining is "surprising"
    # (more than just random co-occurrence)
    return pair_freq / (first_freq * second_freq)
```

This likelihood-based scoring tends to merge pairs that occur together more than expected by chance.

### The ## Prefix Convention

WordPiece marks continuation tokens with "##":

```
"unbelievable" → ["un", "##believ", "##able"]

The ## indicates "this continues the previous token"
```

This convention clarifies word boundaries:
```
"un" at start of word ≠ "##un" continuing a word
```

### Comparison with BPE

| Aspect | BPE | WordPiece |
|--------|-----|-----------|
| Merge criterion | Frequency | Likelihood |
| Continuation marker | None (uses spaces) | ## prefix |
| Primary use | GPT models | BERT models |
| Encoding | Deterministic | Deterministic |

In practice, both work well. The choice often follows the model architecture tradition.

## SentencePiece

### The Problem It Solves

BPE and WordPiece typically require pre-tokenization—splitting text into words first. But what counts as a "word" varies by language:

- **English**: Spaces separate words
- **Chinese/Japanese**: No spaces between words
- **German**: Compound words like "Handschuh" (glove, literally "hand-shoe")

**SentencePiece** treats text as a raw byte stream, eliminating pre-tokenization:

```python
# Traditional approach
text = "Hello world" → ["Hello", "world"] → tokenize each → merge

# SentencePiece approach
text = "Hello world" → treat as byte sequence → tokenize directly
```

### The Unigram Model

SentencePiece often uses a **unigram language model** approach:

1. Start with a large initial vocabulary
2. Assign probability to each token based on corpus
3. Iteratively remove tokens that hurt overall likelihood least
4. Stop when reaching desired vocabulary size

This is the reverse of BPE (which grows vocabulary), but achieves similar results.

### Handling Spaces

SentencePiece uses "▁" (a special underscore character) to represent spaces:

```
"Hello world" → ["▁Hello", "▁world"]
"New York" → ["▁New", "▁York"]
```

This allows reconstructing the original text exactly, including spacing.

### Language Independence

Because SentencePiece works on raw bytes:
- No language-specific pre-processing needed
- Works on Chinese, Japanese, code, or any Unicode
- Handles mixed-language text naturally

```
"Hello 世界" → ["▁Hello", "▁", "世", "界"]
```

This makes it popular for multilingual models.

## Byte-Level BPE

Modern LLMs like GPT-4 use **byte-level BPE**, combining insights from all approaches:

### How It Works

1. Convert text to UTF-8 bytes (256 possible base values)
2. Apply BPE on bytes rather than characters
3. Never encounter "unknown" tokens (any byte sequence works)

```python
# Byte-level example
text = "Hello"
utf8_bytes = text.encode('utf-8')  # b'Hello' → [72, 101, 108, 108, 111]

# BPE operates on these bytes
# Common patterns like [72, 101] might merge into single token
```

### Benefits

**Complete coverage**: Any valid byte sequence tokenizes. No "unknown token" situations.

**Unicode handling**: Emoji, unusual scripts, and special characters all work:
```
"🎉" → [240, 159, 142, 137] → might be one token or several
```

**Graceful degradation**: Unknown characters decompose to bytes rather than failing.

## Comparing Real Tokenizers

Let's see how different tokenizers handle the same text:

```python
from transformers import AutoTokenizer

text = "Tokenization is fascinating! 你好世界 🎉"

# GPT-2 (BPE)
gpt2 = AutoTokenizer.from_pretrained("gpt2")
print(gpt2.tokenize(text))
# ['Token', 'ization', 'Ġis', 'Ġfascinating', '!', 'Ġä½', 'ł', 'å¥', '½', 'ä¸', 'ĸ', 'çŀ', 'Į', 'ĠðŁ', 'İ', 'ī']

# BERT (WordPiece)
bert = AutoTokenizer.from_pretrained("bert-base-uncased")
print(bert.tokenize(text.lower()))
# ['token', '##ization', 'is', 'fascinating', '!', '[UNK]', '[UNK]']

# T5 (SentencePiece)
t5 = AutoTokenizer.from_pretrained("t5-base")
print(t5.tokenize(text))
# ['▁Tokenization', '▁is', '▁fascinating', '!', '▁', '你', '好', '世', '界', '▁', '🎉']
```

Notice the differences in handling English words, Chinese characters, and emoji.

## Tokenizer Training

Training a tokenizer involves:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Choose a model type
tokenizer = Tokenizer(models.BPE())

# 2. Define pre-tokenization (how to split initially)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3. Set up trainer with vocabulary size
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[PAD]", "[UNK]"])

# 4. Train on corpus
files = ["path/to/training/data.txt"]
tokenizer.train(files, trainer)

# 5. Save for later use
tokenizer.save("my-tokenizer.json")
```

The training corpus profoundly affects tokenization:
- Train on English → efficient for English
- Train on code → better handling of programming patterns
- Train on multilingual data → more balanced across languages

## Special Tokens

All tokenizers include special tokens with reserved meanings:

```python
# Common special tokens
"[CLS]"     # Classification token (BERT)
"[SEP]"     # Separator between sequences
"[PAD]"     # Padding for batch processing
"[UNK]"     # Unknown token (when all else fails)
"[MASK]"    # Masked token for pre-training
"<|endoftext|>"  # End of text (GPT)
"<|im_start|>"   # Chat format markers
"<|im_end|>"
```

These tokens have special token IDs and meanings in model processing.

## Key Takeaways

1. **BPE builds vocabulary by merging frequent pairs**, starting from characters and growing until reaching target size.

2. **WordPiece uses likelihood-based merging** and marks continuations with ##, powering BERT-family models.

3. **SentencePiece operates on raw bytes**, enabling language-independent tokenization without pre-tokenization.

4. **Byte-level BPE ensures complete coverage**—any byte sequence can be tokenized without unknown tokens.

5. **Training corpus shapes tokenizer behavior**: models trained on English may inefficiently tokenize other languages.

6. **Special tokens serve specific functions** in model architectures (classification, separation, masking).

## Further Reading

- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016) - BPE for NLP
- "SentencePiece: A simple and language independent subword tokenizer" (Kudo & Richardson, 2018)
- "Japanese and Korean Voice Search" (Schuster & Nakajima, 2012) - Original WordPiece paper
- Hugging Face Tokenizers library documentation
- "A Primer on Neural Network Models for Natural Language Processing" (Goldberg, 2016)
