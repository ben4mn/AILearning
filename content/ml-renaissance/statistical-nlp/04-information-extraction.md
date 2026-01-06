# Information Extraction and Named Entity Recognition

## Introduction

By the mid-1990s, the statistical revolution had transformed how we processed language at the sentence level. But real-world applications demanded more than tagging and parsing—they needed to extract meaningful information from text. Who did what to whom? What companies merged? What drugs treat which diseases?

**Information Extraction (IE)** emerged as the bridge between language processing and knowledge. Rather than parsing every grammatical detail, IE systems focused on finding specific types of information: names, events, relations, facts. And statistical methods, particularly **sequence labeling** approaches, proved remarkably effective at these tasks.

This lesson explores how statistical NLP tackled the extraction of structured information from unstructured text, establishing techniques that power search engines, question-answering systems, and knowledge bases to this day.

## The MUC Competitions

Information extraction crystallized as a field through the **Message Understanding Conferences (MUC)**, run by DARPA from 1987 to 1998. These competitions challenged teams to extract structured information from newswire text, with each year introducing new tasks and domains.

MUC-3 (1991) focused on Latin American terrorism reports. Systems had to extract:
- Incident type (bombing, kidnapping, attack)
- Perpetrators and targets
- Date and location
- Physical damage and casualties

Early MUC systems were rule-based nightmares—thousands of handcrafted patterns for every possible way something might be expressed. By MUC-6 (1995) and MUC-7 (1998), statistical methods began to dominate, particularly for the newly introduced **Named Entity Recognition** task.

## Named Entity Recognition: Finding Names in Text

**Named Entity Recognition (NER)** identifies and classifies proper names in text. The MUC-7 categories became standard:

- **PERSON**: "Albert Einstein", "Mary"
- **ORGANIZATION**: "IBM", "United Nations"
- **LOCATION**: "Paris", "Mount Everest"
- **DATE**: "January 5, 2026", "last Tuesday"
- **TIME**: "3:00 PM", "noon"
- **MONEY**: "$1.5 million", "fifty cents"
- **PERCENT**: "25%", "a third"

```python
# NER annotated sentence
text = """
<PERSON>Bill Gates</PERSON> founded <ORG>Microsoft</ORG> in
<LOCATION>Albuquerque</LOCATION> in <DATE>1975</DATE>.
"""

# Extracted entities:
entities = [
    ('Bill Gates', 'PERSON'),
    ('Microsoft', 'ORG'),
    ('Albuquerque', 'LOCATION'),
    ('1975', 'DATE')
]
```

Why is NER hard? Consider these challenges:

1. **Ambiguity**: "Washington" could be a person, state, or city
2. **Unknown names**: New people, companies, places appear constantly
3. **Context dependence**: "Apple" the company vs. "apple" the fruit
4. **Boundary detection**: Is it "New York" or "New York Times"?

## Sequence Labeling for NER

NER was formulated as a **sequence labeling** task. Each word gets a tag indicating whether it's part of an entity and what type. The **BIO notation** became standard:

- **B-TYPE**: Beginning of an entity of TYPE
- **I-TYPE**: Inside (continuation of) an entity
- **O**: Outside any entity

```python
# BIO-tagged sentence
words = ["Bill", "Gates", "founded", "Microsoft", "in", "1975", "."]
tags =  ["B-PER", "I-PER", "O",      "B-ORG",    "O",  "B-DATE", "O"]

# "Bill Gates" = PERSON (B-PER + I-PER)
# "Microsoft" = ORG (B-ORG alone)
# "1975" = DATE (B-DATE alone)
```

This formulation allowed NER to use the same machine learning techniques as POS tagging: HMMs, Maximum Entropy, and eventually Conditional Random Fields.

## Feature Engineering for NER

What makes a good named entity recognizer? The key was **feature engineering**—designing features that capture patterns indicating entity presence and type.

```python
def extract_ner_features(tokens, position, prev_label):
    """Extract features for NER at a given position."""
    word = tokens[position]
    features = {}

    # Word features
    features[f'word.lower={word.lower()}'] = 1
    features[f'word.isupper={word.isupper()}'] = 1
    features[f'word.istitle={word.istitle()}'] = 1
    features[f'word.isdigit={word.isdigit()}'] = 1

    # Prefix/suffix features
    for i in range(1, 5):
        if len(word) >= i:
            features[f'prefix{i}={word[:i].lower()}'] = 1
            features[f'suffix{i}={word[-i:].lower()}'] = 1

    # Shape features
    features[f'word.shape={get_shape(word)}'] = 1

    # Context features
    if position > 0:
        features[f'prev_word={tokens[position-1].lower()}'] = 1
        features[f'prev_label={prev_label}'] = 1
    if position < len(tokens) - 1:
        features[f'next_word={tokens[position+1].lower()}'] = 1

    # Gazetteer features (lists of known entities)
    if word in PERSON_NAMES:
        features['in_person_gazetteer'] = 1
    if word in COMPANY_NAMES:
        features['in_company_gazetteer'] = 1
    if word in LOCATION_NAMES:
        features['in_location_gazetteer'] = 1

    return features

def get_shape(word):
    """Convert word to shape: 'Apple' -> 'Xxxxx', 'U.S.' -> 'X.X.'"""
    shape = ''
    for c in word:
        if c.isupper():
            shape += 'X'
        elif c.islower():
            shape += 'x'
        elif c.isdigit():
            shape += 'd'
        else:
            shape += c
    return shape
```

**Gazetteers**—lists of known entity names—proved particularly valuable. Lists of cities, countries, company names, and personal names helped systems generalize beyond training data.

## Conditional Random Fields

By the early 2000s, **Conditional Random Fields (CRFs)** became the dominant model for sequence labeling. CRFs combined the advantages of HMMs (modeling sequential dependencies) with MaxEnt (rich feature engineering).

Unlike HMMs, which model the joint probability P(words, tags), CRFs directly model the conditional probability P(tags | words):

**P(y₁,...,yₙ | x₁,...,xₙ) = (1/Z) × exp(Σᵢ Σⱼ λⱼfⱼ(yᵢ₋₁, yᵢ, x, i))**

This allowed CRFs to use features of the entire observation sequence without independence assumptions.

```python
# CRFs capture richer dependencies than HMMs
# Example: A word is more likely to be B-ORG if:
# - It's capitalized
# - Previous word is "at" or "for"
# - Next word is "Inc." or "Corp."
# - It appears in a company name list
# ALL of these features can fire simultaneously

# CRF training finds weights that maximize conditional likelihood
# of correct labels given the observations
```

CRF-based NER systems achieved F1 scores above 90% on MUC datasets, setting performance standards that held for over a decade.

## Relation Extraction

NER finds entities, but real knowledge involves **relations** between entities. "Einstein worked at Princeton" expresses an EMPLOYMENT relation between a PERSON and an ORGANIZATION.

**Relation extraction** identifies these connections. MUC and later competitions (ACE, TAC-KBP) defined standard relation types:

- **EMPLOYMENT**: person works for organization
- **LOCATED-IN**: entity is in location
- **FOUNDER-OF**: person founded organization
- **PARENT-COMPANY**: organization owns organization
- **SPOUSE**: person married to person

Early statistical approaches used **supervised classification**: given a pair of entities in a sentence, extract features and classify the relation type (or NO-RELATION).

```python
def extract_relation_features(sentence, entity1, entity2):
    """Features for relation classification."""
    features = {}

    # Entity type features
    features[f'type1={entity1.type}'] = 1
    features[f'type2={entity2.type}'] = 1
    features[f'type_pair={entity1.type}_{entity2.type}'] = 1

    # Words between entities
    between_words = get_words_between(sentence, entity1, entity2)
    for word in between_words:
        features[f'between_word={word.lower()}'] = 1

    # Distance features
    distance = count_words_between(sentence, entity1, entity2)
    features[f'distance={min(distance, 10)}'] = 1

    # Syntactic path features (if parsed)
    if hasattr(sentence, 'parse'):
        path = get_dependency_path(sentence.parse, entity1, entity2)
        features[f'dep_path={path}'] = 1

    return features

# Training example:
# "Bill Gates founded Microsoft in 1975"
# Entity1: Bill Gates (PERSON)
# Entity2: Microsoft (ORG)
# Label: FOUNDER-OF
```

## Semi-Supervised and Distant Supervision

Labeled data for relation extraction was expensive and limited. Researchers developed approaches to leverage unlabeled text:

**Bootstrapping** (Riloff 1996, Brin 1998) started with a few seed examples, used them to find patterns, then used patterns to find more examples:

1. Seeds: (Einstein, Princeton), (Feynman, Caltech)
2. Find patterns: "X worked at Y", "X was a professor at Y"
3. Use patterns to find new pairs: (Chomsky, MIT)
4. Iterate

**Distant supervision** (Mintz et al. 2009) automatically labeled training data using knowledge bases:

```python
# Distant supervision for relation extraction
# Assumption: If KB says (Gates, Microsoft, FOUNDER), then any
# sentence mentioning both likely expresses that relation

kb_facts = {('Bill Gates', 'Microsoft'): 'FOUNDER'}

for sentence in corpus:
    for e1, e2 in entity_pairs(sentence):
        if (e1.text, e2.text) in kb_facts:
            # Automatically label as positive example
            relation = kb_facts[(e1.text, e2.text)]
            training_examples.append((sentence, e1, e2, relation))
```

This noisy but plentiful training data enabled large-scale relation extraction from the web.

## Event Extraction

The most ambitious IE task was **event extraction**: identifying complex events with multiple participants playing different roles.

From "Three people were killed when a bomb exploded in Baghdad on Tuesday":
- **Event type**: Bombing/Attack
- **Attacker**: Unknown (implicit)
- **Victim**: Three people
- **Instrument**: Bomb
- **Location**: Baghdad
- **Time**: Tuesday

Event extraction combined multiple components:
1. **Trigger identification**: Find words indicating events ("exploded", "killed")
2. **Argument extraction**: Find entities filling roles
3. **Role classification**: Assign roles (attacker, victim, etc.)
4. **Event coreference**: Link related events across sentences

This remains challenging even today, pushing the limits of what statistical NLP could achieve.

## Building Knowledge Bases

The ultimate goal of information extraction was populating **knowledge bases**—structured databases of facts. Projects like:

- **YAGO** (2007): Extracted facts from Wikipedia
- **Freebase** (2007): Combined extraction with crowdsourcing
- **DBpedia** (2007): Structured data from Wikipedia infoboxes
- **Knowledge Vault** (2014): Google's web-scale extraction

These knowledge bases powered question answering, search enhancement, and recommendation systems.

## Key Takeaways

- Information extraction bridges unstructured text and structured knowledge
- Named Entity Recognition identifies and classifies names using sequence labeling
- The BIO notation represents entity boundaries for sequence models
- Feature engineering—capturing capitalization, context, gazetteers—was crucial for NER
- Conditional Random Fields combined sequential modeling with rich features
- Relation extraction identifies connections between entities
- Distant supervision enabled learning from knowledge bases without manual annotation

## Further Reading

- Chinchor, Nancy. "MUC-7 Named Entity Task Definition" (1998) - The standard task definition
- Lafferty et al. "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data" (2001) - CRF introduction
- Mintz et al. "Distant supervision for relation extraction without labeled data" (2009) - Distant supervision breakthrough
- Sarawagi, Sunita. "Information Extraction" (2008) - Comprehensive survey

---
*Estimated reading time: 11 minutes*
