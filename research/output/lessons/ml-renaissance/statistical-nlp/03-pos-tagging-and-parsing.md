# Statistical POS Tagging and Parsing

## Introduction

The rationalist approach to NLP had one undeniable strength: it produced interpretable structures. A rule-based parser didn't just say a sentence was probable—it showed you the grammatical relationships between words, the noun phrases and verb phrases, the subjects and objects. This linguistic structure seemed essential for real language understanding.

Could statistical methods produce structure, not just probabilities? The answer, developed through the 1990s, was a resounding yes. Statistical approaches to **part-of-speech tagging** and **syntactic parsing** showed that machines could learn grammatical structure from annotated corpora, often outperforming hand-crafted grammars while being faster and more robust.

This lesson explores how statistical methods conquered these fundamental NLP tasks, establishing techniques that remained dominant for two decades.

## Part-of-Speech Tagging: The Sequence Labeling Task

Every word in a sentence belongs to a grammatical category: noun, verb, adjective, preposition, and so on. **Part-of-speech (POS) tagging** is the task of assigning these categories to words in context.

Why "in context"? Because many words are ambiguous. "Book" can be a noun ("the book") or a verb ("book a flight"). "That" can be a determiner ("that cat"), pronoun ("I know that"), or complementizer ("I think that..."). The correct tag depends on the surrounding words.

The **Penn Treebank tagset**, developed in the early 1990s, became the standard for English. It defines 45 tags:
- NN (singular noun): "cat", "dog"
- NNS (plural noun): "cats", "dogs"
- VB (verb base form): "eat", "run"
- VBD (verb past tense): "ate", "ran"
- JJ (adjective): "big", "happy"
- RB (adverb): "quickly", "very"
- And 39 more...

```python
# Example POS-tagged sentence
sentence = "The/DT cat/NN sat/VBD on/IN the/DT mat/NN"

# Tags reveal grammatical structure
# DT = determiner, NN = noun, VBD = past-tense verb, IN = preposition
```

## Hidden Markov Models for Tagging

The breakthrough for statistical tagging came from applying **Hidden Markov Models (HMMs)**—the same technology that had revolutionized speech recognition.

In an HMM tagger:
- The **hidden states** are the POS tags
- The **observations** are the words
- **Transition probabilities** capture grammar: P(VBD | NN) models that verbs often follow nouns
- **Emission probabilities** capture lexical patterns: P("cat" | NN) models that "cat" is often a noun

```python
class HMMTagger:
    def __init__(self):
        self.transition = {}  # P(tag_j | tag_i)
        self.emission = {}    # P(word | tag)
        self.tags = set()

    def train(self, tagged_sentences):
        """Train from tagged corpus like Penn Treebank."""
        trans_counts = defaultdict(lambda: defaultdict(int))
        emit_counts = defaultdict(lambda: defaultdict(int))
        tag_counts = defaultdict(int)

        for sentence in tagged_sentences:
            prev_tag = '<START>'
            for word, tag in sentence:
                trans_counts[prev_tag][tag] += 1
                emit_counts[tag][word] += 1
                tag_counts[tag] += 1
                prev_tag = tag
                self.tags.add(tag)

        # Convert to probabilities with smoothing
        # ...

    def tag(self, words):
        """Find most likely tag sequence using Viterbi algorithm."""
        # Dynamic programming to find optimal path
        pass
```

The **Viterbi algorithm** efficiently finds the most likely tag sequence. Rather than considering all possible tag sequences (exponential in sentence length), it uses dynamic programming to compute the best path to each state at each position, running in O(n × T²) time for n words and T possible tags.

By the mid-1990s, HMM taggers trained on the Penn Treebank achieved accuracies around 96-97%—impressive given that human annotators agreed only about 98% of the time.

## Maximum Entropy and Feature-Rich Models

HMM taggers were limited in the features they could use. What if you wanted to consider not just the previous tag, but also:
- Word prefixes and suffixes ("running" ends in "-ing")
- Capitalization patterns
- The presence of digits
- Words before and after
- Whether the word is in a dictionary

**Maximum Entropy (MaxEnt) models**, also called logistic regression, allowed exactly this. They model the probability of a tag given arbitrary features of the context:

**P(tag | context) = (1/Z) × exp(Σᵢ λᵢ × fᵢ(tag, context))**

```python
# Maximum Entropy features for POS tagging
def extract_features(words, position, prev_tag):
    word = words[position]
    features = {
        f'word={word.lower()}': 1,
        f'suffix3={word[-3:]}': 1,
        f'prefix3={word[:3]}': 1,
        f'prev_tag={prev_tag}': 1,
        f'capitalized={word[0].isupper()}': 1,
        f'has_digit={any(c.isdigit() for c in word)}': 1,
    }
    if position > 0:
        features[f'prev_word={words[position-1].lower()}'] = 1
    if position < len(words) - 1:
        features[f'next_word={words[position+1].lower()}'] = 1
    return features
```

Adwait Ratnaparkhi's MaxEnt tagger (1996) demonstrated the power of this approach, achieving state-of-the-art results. The framework was flexible enough to incorporate linguistic insights (like suffix patterns) while learning feature weights from data.

## Statistical Parsing: Learning Syntax from Trees

POS tagging assigns categories to words, but sentences have deeper structure. "The cat chased the mouse" isn't just a sequence of tags—it has a subject ("the cat"), a verb ("chased"), and an object ("the mouse"). This hierarchical structure is represented as a **parse tree**.

```
              S
          /       \
        NP         VP
       /  \       /   \
      DT   NN    VBD   NP
      |    |      |   /   \
     The  cat  chased DT   NN
                      |    |
                    the  mouse
```

Rule-based parsers used hand-written grammars with hundreds or thousands of rules. But the Penn Treebank provided 40,000 parsed sentences—enough to learn a grammar from data.

## Probabilistic Context-Free Grammars (PCFGs)

The first statistical parsers used **Probabilistic Context-Free Grammars (PCFGs)**. A PCFG is a CFG where each rule has an associated probability:

```
S → NP VP      [1.0]       # S always rewrites as NP VP
NP → DT NN     [0.6]       # 60% of NPs are determiner + noun
NP → DT JJ NN  [0.3]       # 30% have an adjective too
NP → NNP       [0.1]       # 10% are proper nouns
VP → VBD NP    [0.7]       # 70% of VPs are transitive
VP → VBD       [0.3]       # 30% are intransitive
```

These probabilities are estimated from the Treebank by counting how often each rule is used:

**P(A → β) = Count(A → β) / Count(A)**

The probability of a parse tree is the product of all rule probabilities used to build it. The **CKY algorithm** (Cocke-Kasami-Younger) efficiently finds the most probable parse using dynamic programming.

```python
def pcfg_parse(sentence, grammar):
    """CKY parsing with PCFG."""
    n = len(sentence)
    # Table[i][j] stores best derivations for words i to j
    table = [[{} for _ in range(n+1)] for _ in range(n+1)]

    # Initialize with word → POS rules
    for i, word in enumerate(sentence):
        for tag, prob in grammar.lexical_rules(word):
            table[i][i+1][tag] = (prob, word)

    # Fill table bottom-up
    for span in range(2, n+1):
        for i in range(n - span + 1):
            j = i + span
            for k in range(i+1, j):
                for A, (B, C, rule_prob) in grammar.binary_rules():
                    if B in table[i][k] and C in table[k][j]:
                        prob = rule_prob * table[i][k][B][0] * table[k][j][C][0]
                        if A not in table[i][j] or prob > table[i][j][A][0]:
                            table[i][j][A] = (prob, B, C, k)

    return reconstruct_tree(table, 0, n, 'S')
```

## Lexicalized Parsing: Words Matter

Plain PCFGs had a serious flaw: they treated all NPs the same, all VPs the same. But syntactic structure depends heavily on specific words. "The man saw the dog with the telescope" is ambiguous—did the man use a telescope, or did the dog have a telescope? The answer depends on what "saw" and "telescope" typically combine with.

**Lexicalized parsers** addressed this by associating each phrase with a **head word**. The head of "the big red dog" is "dog"; the head of "saw the cat" is "saw". Rules now included head information:

```
VP(saw) → VBD(saw) NP(dog)
PP(with) → IN(with) NP(telescope)
```

Michael Collins's lexicalized parser (1997-1999) achieved dramatic improvements by modeling word-word dependencies. His models captured that:
- "Ate" prefers "food" as an object
- "With telescope" more naturally attaches to "saw" than to "dog"
- "President" is often modified by "of the United States"

The cost was data sparsity—word-specific rules are seen rarely. Collins developed sophisticated smoothing techniques, backing off from specific words to word classes when necessary.

## The Parsing Evaluation: PARSEVAL

How do you evaluate a parser? The **PARSEVAL metrics**, developed in the early 1990s, became standard:

- **Labeled Precision**: What fraction of brackets in the parse are correct?
- **Labeled Recall**: What fraction of gold-standard brackets are found?
- **F1 Score**: Harmonic mean of precision and recall

```python
def parseval(gold_tree, predicted_tree):
    """Calculate PARSEVAL metrics."""
    gold_brackets = extract_brackets(gold_tree)
    pred_brackets = extract_brackets(predicted_tree)

    correct = len(gold_brackets & pred_brackets)
    precision = correct / len(pred_brackets)
    recall = correct / len(gold_brackets)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

# A bracket is (start, end, label), e.g., (0, 2, 'NP')
```

By the late 1990s, the best statistical parsers achieved F1 scores around 88-90% on the Penn Treebank—far exceeding what rule-based systems had achieved.

## Impact and Legacy

Statistical tagging and parsing had profound impacts:

1. **Information Extraction**: POS tags help identify named entities, relations, events
2. **Machine Translation**: Parse structure guides translation reordering
3. **Sentiment Analysis**: Parsing reveals what modifies what ("not good" vs "not only good")
4. **Question Answering**: Syntax helps match questions to answer patterns

These techniques dominated NLP through the 2000s. Neural approaches would eventually surpass them, but the evaluation frameworks, datasets, and insights remained foundational.

## Key Takeaways

- POS tagging assigns grammatical categories to words in context, handling lexical ambiguity
- HMM taggers model tag sequences with transition and emission probabilities, decoded efficiently with the Viterbi algorithm
- Maximum Entropy models allow rich features beyond just the previous tag
- PCFGs assign probabilities to parse trees, enabling disambiguation of syntactic structure
- Lexicalized parsing incorporates word-specific preferences, dramatically improving accuracy
- PARSEVAL metrics became the standard for evaluating parsers

## Further Reading

- Ratnaparkhi, Adwait. "A Maximum Entropy Model for Part-of-Speech Tagging" (1996) - Feature-rich tagging
- Collins, Michael. "Head-Driven Statistical Models for Natural Language Parsing" (1999) - PhD thesis on lexicalized parsing
- Charniak, Eugene. *Statistical Language Learning* (1996) - Accessible introduction
- Marcus et al. "Building a Large Annotated Corpus of English: The Penn Treebank" (1993) - The dataset that enabled it all

---
*Estimated reading time: 11 minutes*
