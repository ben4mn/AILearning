# Vector Databases and Embeddings for Search

## Introduction

Traditional search engines match keywords: if your query contains "salmon recipe," they find documents containing those words. But what if the best recipe says "grilled sockeye" without using "salmon" or "recipe"? Keyword search fails, but a human would recognize these as related.

**Semantic search** solves this by comparing *meanings* rather than words. At its heart is a simple idea: represent text as vectors (embeddings) where similar meanings are nearby in vector space. Then search becomes finding the nearest vectors to your query.

In this lesson, we'll explore how embeddings enable semantic search, how vector databases store and query these embeddings efficiently, and practical considerations for building retrieval systems.

## From Keywords to Semantics

### The Keyword Problem

Traditional search has fundamental limitations:

```python
# Query
query = "How do I fix a leaky faucet?"

# Document that answers perfectly
doc = "Repairing a dripping tap: First, turn off the water supply.
Then remove the handle and replace the worn washer..."

# Keyword search fails!
# "leaky" ≠ "dripping"
# "faucet" ≠ "tap"
# "fix" ≠ "repairing"
```

### The Semantic Solution

Embed both query and documents as vectors:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed query and documents
query_embedding = model.encode("How do I fix a leaky faucet?")
doc_embedding = model.encode("Repairing a dripping tap: First...")

# Compute similarity
similarity = cosine_similarity(query_embedding, doc_embedding)
# High similarity despite different words!
```

The embedding model learned that "leaky faucet" and "dripping tap" mean the same thing.

## How Embeddings Work for Search

### The Embedding Process

1. **Text → Vector**: Convert text to high-dimensional vector (e.g., 384 or 768 dimensions)
2. **Similar meanings → Nearby vectors**: Training ensures semantic similarity
3. **Search = Nearest neighbors**: Find vectors closest to query vector

```python
# Conceptual representation
embed("king")  = [0.2, 0.8, -0.1, 0.5, ...]  # 384 numbers
embed("queen") = [0.21, 0.79, -0.12, 0.48, ...] # Very similar
embed("fish")  = [-0.5, 0.1, 0.7, -0.3, ...]   # Very different
```

### Similarity Metrics

**Cosine similarity**: Measures angle between vectors (most common)
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# 1.0 = identical direction, 0 = orthogonal, -1 = opposite
```

**Euclidean distance**: Straight-line distance
```python
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
# 0 = identical, larger = more different
```

**Dot product**: Simple but affected by vector magnitude
```python
def dot_product(a, b):
    return np.dot(a, b)
```

Cosine similarity is usually preferred because it normalizes for vector length.

## Embedding Models

### Popular Choices

```python
# Open source models via sentence-transformers
from sentence_transformers import SentenceTransformer

# General purpose (good starting point)
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dim, fast
model = SentenceTransformer('all-mpnet-base-v2')  # 768 dim, better

# Multilingual
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Via APIs
import openai
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input="Your text here"
)
embedding = response.data[0].embedding
```

### Model Selection Trade-offs

| Factor | Smaller Models | Larger Models |
|--------|---------------|---------------|
| Speed | Faster | Slower |
| Memory | Less | More |
| Quality | Good enough for many tasks | Better for nuanced similarity |
| Dimensions | 384 | 768-1536 |
| Cost | Cheaper | More expensive |

For most applications, start with smaller models and upgrade only if needed.

## What Are Vector Databases?

### The Scale Problem

With millions of documents, comparing to every vector is too slow:

```python
# Naive search: O(n) comparisons
def naive_search(query_embedding, all_embeddings, k=10):
    similarities = []
    for doc_emb in all_embeddings:  # Millions of iterations!
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append(sim)
    return top_k(similarities, k)
```

**Vector databases** solve this with specialized data structures and algorithms.

### Approximate Nearest Neighbors (ANN)

Vector databases trade exact results for speed:

```python
# Exact: Check all 1 million documents
# Time: ~1 second

# Approximate: Use smart indexing
# Time: ~1 millisecond
# Accuracy: 95-99% of true nearest neighbors
```

Key algorithms:
- **HNSW** (Hierarchical Navigable Small World): Graph-based, very popular
- **IVF** (Inverted File Index): Clustering-based partitioning
- **LSH** (Locality Sensitive Hashing): Hash-based approximation
- **PQ** (Product Quantization): Compression for reduced memory

### Popular Vector Databases

**Pinecone**: Fully managed cloud service
```python
import pinecone

pinecone.init(api_key="YOUR_KEY")
index = pinecone.Index("my-index")

# Upsert vectors
index.upsert([
    ("id1", embedding1, {"text": "doc1 content"}),
    ("id2", embedding2, {"text": "doc2 content"}),
])

# Query
results = index.query(query_embedding, top_k=5)
```

**ChromaDB**: Easy to use, runs locally
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# Add documents (automatically embeds)
collection.add(
    documents=["Doc 1 text", "Doc 2 text"],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

**Weaviate**: Open source, feature-rich
```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Create schema
client.schema.create_class({
    "class": "Document",
    "properties": [{"name": "content", "dataType": ["text"]}]
})

# Add data
client.data_object.create({"content": "document text"}, "Document")

# Query
results = client.query.get("Document", ["content"]) \
    .with_near_text({"concepts": ["search query"]}) \
    .with_limit(5) \
    .do()
```

**Others**: Qdrant, Milvus, pgvector (PostgreSQL extension), FAISS (library)

## Building a Retrieval System

### The Basic Pipeline

```python
from sentence_transformers import SentenceTransformer
import chromadb

class SemanticRetriever:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")

    def add_documents(self, documents, ids):
        """Index documents for retrieval."""
        self.collection.add(
            documents=documents,
            ids=ids
        )

    def retrieve(self, query, top_k=5):
        """Find most relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results['documents'][0]

# Usage
retriever = SemanticRetriever()
retriever.add_documents(
    documents=["Doc 1 content", "Doc 2 content", ...],
    ids=["doc1", "doc2", ...]
)

relevant_docs = retriever.retrieve("What is machine learning?")
```

### Adding Metadata

Metadata enables filtering and context:

```python
collection.add(
    documents=["Python list comprehensions explained..."],
    ids=["doc1"],
    metadatas=[{
        "source": "python-docs",
        "category": "tutorial",
        "date": "2024-01-15",
        "author": "Guido"
    }]
)

# Query with filtering
results = collection.query(
    query_texts=["list manipulation"],
    n_results=5,
    where={"category": "tutorial"}  # Only tutorials
)
```

### Persistence and Scaling

```python
# Persistent storage (ChromaDB)
client = chromadb.PersistentClient(path="./chroma_data")

# For production scale, use managed services:
# - Pinecone: Auto-scaling, fully managed
# - Weaviate Cloud: Managed Weaviate
# - Qdrant Cloud: Managed Qdrant
```

## Practical Considerations

### Embedding Consistency

Always use the same model for indexing and querying:

```python
# BAD: Different models
index_model = SentenceTransformer('all-MiniLM-L6-v2')
query_model = SentenceTransformer('all-mpnet-base-v2')  # Different!
# Results will be poor

# GOOD: Same model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Use for both indexing and querying
```

### Batch Processing

Embed in batches for efficiency:

```python
# Slow: One at a time
embeddings = [model.encode(doc) for doc in documents]

# Fast: Batch processing
embeddings = model.encode(documents, batch_size=32, show_progress_bar=True)
```

### Updating Documents

Plan for document changes:

```python
# Option 1: Delete and re-add
collection.delete(ids=["doc1"])
collection.add(documents=[new_content], ids=["doc1"])

# Option 2: Upsert (add or update)
collection.upsert(
    documents=[new_content],
    ids=["doc1"]
)

# Option 3: Version in metadata
collection.add(
    documents=[new_content],
    ids=["doc1_v2"],
    metadatas=[{"version": 2, "supersedes": "doc1_v1"}]
)
```

## Key Takeaways

1. **Semantic search matches meanings, not keywords**, enabling better retrieval for natural language queries.

2. **Embeddings convert text to vectors** where similar meanings are nearby in vector space.

3. **Vector databases use ANN algorithms** to search millions of vectors in milliseconds.

4. **Multiple options exist**: Pinecone for managed, ChromaDB for simplicity, Weaviate/Qdrant for features.

5. **Consistency matters**: Use the same embedding model for indexing and querying.

6. **Metadata enables filtering** and provides context for retrieved documents.

## Further Reading

- "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018) - HNSW paper
- ChromaDB, Pinecone, and Weaviate documentation
- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019)
- "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (Wang et al., 2022)
