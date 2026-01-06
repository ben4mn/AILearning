# Practical Implications of Tokenization and Embeddings

## Introduction

Understanding tokenization and embeddings isn't just academic—it has real implications for anyone building with or using large language models. From API costs to multilingual fairness, from context window management to debugging strange model behaviors, the concepts we've explored translate directly into practical considerations.

In this lesson, we'll connect theory to practice, exploring how tokenization and embeddings affect everyday work with LLMs. We'll cover cost optimization, handling long documents, multilingual challenges, and techniques for getting the most out of these fundamental concepts.

## Managing Context Windows

### The Fundamental Constraint

Every interaction with an LLM must fit within its context window. This includes:

```
Total tokens = System prompt + Conversation history + Current input + Output

If Total tokens > Context window → Something must be cut
```

Understanding this is crucial for any application:

```python
# Example: Chat application with GPT-4 (8K context)
system_prompt = 200      # tokens for instructions
conversation_history = 0  # starts empty
user_message = 50        # current message
reserved_for_response = 1000  # leave room for output

available_for_history = 8000 - 200 - 50 - 1000  # = 6,750 tokens

# As conversation grows, old messages must be dropped or summarized
```

### Strategies for Long Contexts

**Truncation**: Simply cut oldest messages
```python
def truncate_history(messages, max_tokens):
    total = sum(count_tokens(m) for m in messages)
    while total > max_tokens:
        messages.pop(0)  # Remove oldest
        total = sum(count_tokens(m) for m in messages)
    return messages
```

**Summarization**: Compress old content
```python
def summarize_history(old_messages, new_messages, max_tokens):
    # Summarize old messages into condensed form
    old_summary = llm.summarize(old_messages, max_tokens=500)

    # Keep recent messages in full
    return [old_summary] + new_messages[-10:]
```

**Sliding window**: Keep most recent N messages
```python
def sliding_window(messages, window_size=10):
    return messages[-window_size:]
```

**Hierarchical summarization**: Multi-level compression
```python
# Very old → highly compressed summary
# Older → moderately compressed
# Recent → full detail
```

### Chunking for Long Documents

When processing documents longer than the context window:

```python
def chunk_document(text, chunk_size=2000, overlap=200):
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap prevents losing context at boundaries
    return chunks

# Process each chunk separately
results = [process_chunk(chunk) for chunk in chunks]
# Combine results
final = combine_results(results)
```

The overlap ensures that concepts split across chunk boundaries still get captured.

## Cost Optimization

### Understanding Token-Based Pricing

Most APIs charge per token:

```python
# Typical pricing (hypothetical, check current rates)
GPT_4_INPUT = 0.03 / 1000   # $0.03 per 1K input tokens
GPT_4_OUTPUT = 0.06 / 1000  # $0.06 per 1K output tokens

def estimate_cost(prompt_tokens, response_tokens):
    input_cost = prompt_tokens * GPT_4_INPUT
    output_cost = response_tokens * GPT_4_OUTPUT
    return input_cost + output_cost

# Example: 1000-word document analysis
# ~1300 input tokens, ~200 output tokens
cost = estimate_cost(1300, 200)  # ~$0.051
```

### Optimization Strategies

**Concise prompts**: Every word counts
```python
# Verbose (45 tokens)
prompt = """
I would like you to please analyze the following text and provide
me with a comprehensive summary of the main points and key takeaways.
Here is the text to analyze:
"""

# Concise (15 tokens)
prompt = "Summarize this text's main points:\n\n"
# Saves 30 tokens per request
```

**Cached system prompts**: Some APIs offer discounts for repeated prefixes
```python
# If your system prompt is always the same,
# it may be cached and charged at lower rates
```

**Batch processing**: Combine related requests
```python
# Instead of 10 separate API calls
results = [analyze(doc) for doc in documents]

# One call with all documents (if they fit)
prompt = "\n\n---\n\n".join(documents)
result = analyze_all(prompt)
```

**Choose appropriate models**: Bigger isn't always necessary
```python
# Use GPT-4 for complex reasoning
# Use GPT-3.5 for simple transformations
# Use fine-tuned smaller models for specific tasks

def select_model(task_complexity):
    if task_complexity == "high":
        return "gpt-4"
    elif task_complexity == "medium":
        return "gpt-3.5-turbo"
    else:
        return "fine-tuned-small-model"
```

### Token Counting in Practice

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    """Count tokens for accurate cost estimation."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Use before API calls
prompt = build_prompt(user_input)
token_count = count_tokens(prompt)

if token_count > MAX_TOKENS:
    prompt = truncate_prompt(prompt)
elif token_count * PRICE_PER_TOKEN > BUDGET:
    raise BudgetExceededError()
```

## Multilingual Challenges

### The Tokenization Tax

Languages other than English often require more tokens for equivalent content:

```python
# Same content, different languages
messages = {
    "en": "Hello, how are you today?",  # ~7 tokens
    "zh": "你好，你今天怎么样？",          # ~10 tokens
    "ar": "مرحبا، كيف حالك اليوم؟",      # ~12 tokens
    "hi": "नमस्ते, आज आप कैसे हैं?",        # ~15 tokens
}

# This means:
# - Higher costs for non-English users
# - Less content fits in context window
# - Potentially faster token limits hit
```

### Understanding the Disparity

The disparity arises because:

1. **Training data distribution**: Tokenizers trained mostly on English
2. **Script complexity**: Some scripts require more bytes
3. **Morphological differences**: Agglutinative languages have longer words

```python
# Japanese example
text = "東京は日本の首都です"  # "Tokyo is the capital of Japan"
# Might tokenize as: ["東", "京", "は", "日", "本", "の", "首", "都", "です"]
# Each character potentially becomes one token

# English equivalent: "Tokyo is the capital of Japan"
# Tokenizes as: ["Tokyo", " is", " the", " capital", " of", " Japan"]
# Much more efficient
```

### Mitigation Strategies

**Multilingual models**: Some models are trained for better multilingual efficiency
```python
# Models like mT5, BLOOM, or multilingual Claude
# have more balanced tokenization across languages
```

**Language-specific fine-tuning**: For specific language pairs
```python
# Train custom tokenizer on target language corpus
# Fine-tune model for specific language needs
```

**Preprocessing optimization**:
```python
# For some languages, transliteration before processing
# might be more token-efficient (with accuracy trade-offs)
```

## Embedding Applications

### Semantic Search Implementation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def index_documents(self, documents):
        """Create embeddings for all documents."""
        self.documents = documents
        self.embeddings = self.model.encode(documents)

    def search(self, query, top_k=5):
        """Find most similar documents."""
        query_embedding = self.model.encode([query])[0]

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        similarities /= np.linalg.norm(self.embeddings, axis=1)
        similarities /= np.linalg.norm(query_embedding)

        # Top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]

# Usage
searcher = SemanticSearch()
searcher.index_documents(["Doc about cats", "Doc about dogs", "Doc about taxes"])
results = searcher.search("pets and animals")
# Returns cat and dog docs, not tax doc
```

### Document Clustering

```python
from sklearn.cluster import KMeans

def cluster_documents(documents, n_clusters=5):
    """Group similar documents together."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)

    # Cluster based on embedding similarity
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)

    # Group documents by cluster
    grouped = {}
    for doc, cluster in zip(documents, clusters):
        if cluster not in grouped:
            grouped[cluster] = []
        grouped[cluster].append(doc)

    return grouped
```

### Duplicate Detection

```python
def find_near_duplicates(documents, threshold=0.95):
    """Find documents that are nearly identical."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)

    duplicates = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                duplicates.append((i, j, sim))

    return duplicates
```

## Debugging Token-Related Issues

### Strange Outputs

When LLM outputs seem wrong, consider token boundaries:

```python
# Issue: Model can't count letters
input = "How many letters in 'strawberry'?"
# Model sees: ["How", " many", " letters", " in", " '", "str", "aw", "berry", "'"]
# Letters aren't visible units—only tokens are!

# Solution: Have the model use code
input = """
Write Python code to count letters in 'strawberry':
print(len('strawberry'))
"""
```

### Arithmetic Errors

```python
# Issue: Math mistakes
input = "What is 7823 * 4521?"
# Numbers tokenize unpredictably
# "7823" might be ["782", "3"] or ["78", "23"]

# Solution: Use code execution
input = "Calculate using Python: 7823 * 4521"
```

### Context Window Exceeded

```python
# Error: "This model's maximum context length is 8192 tokens"

# Debug: Count tokens before sending
import tiktoken

def safe_send(prompt, max_tokens=8192):
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(prompt))

    if token_count > max_tokens:
        print(f"Warning: {token_count} tokens exceeds limit")
        # Truncate or summarize
        return truncate_to_fit(prompt, max_tokens)

    return send_to_api(prompt)
```

## Performance Considerations

### Embedding Computation

For large-scale applications, embedding computation matters:

```python
# Batch embedding is much faster than one-by-one
# Bad
embeddings = [model.encode(doc) for doc in documents]  # N API calls

# Good
embeddings = model.encode(documents, batch_size=32)  # Batched

# Pre-compute and cache embeddings for static content
import pickle

def get_or_compute_embeddings(documents, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    embeddings = model.encode(documents)
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings
```

### Vector Database Integration

For production semantic search:

```python
# Instead of numpy similarity search, use vector databases
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

# Add documents
collection.add(
    documents=["Doc 1 content", "Doc 2 content"],
    ids=["doc1", "doc2"]
)

# Search efficiently (even with millions of documents)
results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

Vector databases like ChromaDB, Pinecone, Weaviate, and Qdrant are optimized for embedding similarity search at scale.

## Key Takeaways

1. **Context window management is essential**—use truncation, summarization, and chunking strategies based on your application's needs.

2. **Token-based pricing means every word costs**—optimize prompts, batch when possible, and choose appropriate models.

3. **Multilingual applications face the tokenization tax**—non-English content often requires more tokens for equivalent meaning.

4. **Embeddings enable powerful applications**—semantic search, clustering, and duplicate detection become straightforward.

5. **Many strange LLM behaviors trace to tokenization**—arithmetic errors, counting mistakes, and boundary issues.

6. **Production systems need vector databases**—numpy similarity search doesn't scale to millions of documents.

## Further Reading

- tiktoken documentation (OpenAI's tokenizer library)
- Sentence-Transformers library documentation
- ChromaDB, Pinecone, and Weaviate documentation
- "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)
- "Scaling Data-Constrained Language Models" (Muennighoff et al., 2023)
