# Advanced RAG Techniques

## Introduction

The basic RAG pipeline—chunk, embed, retrieve, generate—works surprisingly well for many applications. But as you scale or tackle more challenging problems, you'll encounter limitations. Chunks may be too large or too small. The most relevant passage might not rank first. Simple semantic search misses keyword-specific queries.

In this lesson, we'll explore advanced techniques that address these limitations: sophisticated chunking strategies, reranking for improved precision, hybrid search combining semantic and keyword approaches, and query transformations that improve retrieval quality.

## Advanced Chunking Strategies

### The Chunking Trade-off

Chunk size presents a fundamental trade-off:

```
Small chunks (100-200 tokens):
✓ Precise retrieval
✓ Fit more in context
✗ May lose surrounding context
✗ More chunks to search

Large chunks (1000-2000 tokens):
✓ More context preserved
✓ Fewer chunks to manage
✗ Less precise retrieval
✗ May include irrelevant content
```

### Semantic Chunking

Instead of fixed-size chunks, split at semantic boundaries:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def semantic_chunk(text, max_size=500):
    """Chunk at sentence and paragraph boundaries."""
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_size = len(sent_text)

        # Check if adding this sentence exceeds limit
        if current_size + sent_size > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(sent_text)
        current_size += sent_size

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

### Hierarchical Chunking

Create chunks at multiple granularities:

```python
def hierarchical_chunk(document):
    """Create multi-level chunks."""
    chunks = []

    # Level 1: Full sections
    sections = split_by_headers(document)
    for section in sections:
        chunks.append({
            "text": section["text"],
            "level": "section",
            "title": section["title"]
        })

        # Level 2: Paragraphs within sections
        paragraphs = section["text"].split("\n\n")
        for para in paragraphs:
            if para.strip():
                chunks.append({
                    "text": para,
                    "level": "paragraph",
                    "parent_section": section["title"]
                })

    return chunks
```

### Parent-Child Retrieval

Retrieve small chunks but return larger context:

```python
class ParentChildRetriever:
    def __init__(self):
        # Small chunks for precise retrieval
        self.child_collection = create_collection("children")
        # Parent documents for context
        self.parent_docs = {}

    def index(self, documents):
        for doc in documents:
            # Store full document
            doc_id = generate_id(doc)
            self.parent_docs[doc_id] = doc["text"]

            # Create small chunks
            chunks = chunk_document(doc, chunk_size=200)
            for chunk in chunks:
                self.child_collection.add(
                    documents=[chunk["text"]],
                    ids=[f"{doc_id}_{chunk['index']}"],
                    metadatas=[{"parent_id": doc_id}]
                )

    def retrieve(self, query, top_k=5):
        # Find matching chunks
        results = self.child_collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Return parent documents for context
        parent_ids = set(
            r["parent_id"] for r in results["metadatas"][0]
        )
        return [self.parent_docs[pid] for pid in parent_ids]
```

## Reranking

### Why Rerank?

Initial retrieval is fast but imprecise. Reranking uses a more powerful model to re-order results:

```
Query: "What causes heart attacks?"

Initial retrieval (embedding similarity):
1. "Heart attacks occur when blood flow..." (relevant)
2. "The attack was sudden and unexpected..." (wrong "attack")
3. "Cardiac events are often preceded..." (relevant but different terms)
4. "Heart-healthy diets include..." (tangentially related)

After reranking:
1. "Heart attacks occur when blood flow..."
2. "Cardiac events are often preceded..."
3. "Heart-healthy diets include..."
4. (irrelevant result filtered out)
```

### Implementing Reranking

```python
from sentence_transformers import CrossEncoder

class RerankedRetriever:
    def __init__(self):
        self.collection = get_collection()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query, top_k=5, initial_k=20):
        # Step 1: Fast initial retrieval (get more than needed)
        initial_results = self.collection.query(
            query_texts=[query],
            n_results=initial_k
        )

        # Step 2: Rerank with cross-encoder
        documents = initial_results['documents'][0]
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)

        # Step 3: Sort by reranker scores
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k after reranking
        return [doc for doc, score in ranked[:top_k]]
```

### Cross-Encoder vs. Bi-Encoder

```
Bi-Encoder (embeddings):
- Encode query and documents separately
- Compare with vector similarity
- Fast: precompute document embeddings
- Less accurate: no query-document interaction

Cross-Encoder (reranking):
- Encode query and document together
- Model sees both simultaneously
- Slow: can't precompute
- More accurate: full attention between query and document
```

Use bi-encoders for initial retrieval, cross-encoders for reranking.

## Hybrid Search

### Combining Semantic and Keyword Search

Neither approach is perfect:

```
Semantic search fails:
Query: "Error code E-4521"
Semantic: Finds general "error handling" content
Needed: Exact match for "E-4521"

Keyword search fails:
Query: "How to fix a dripping tap"
Keyword: Finds "dripping" and "tap" but misses "leaky faucet"
Needed: Semantic understanding
```

**Hybrid search** combines both:

```python
def hybrid_search(query, collection, keyword_index, top_k=5, alpha=0.5):
    """Combine semantic and keyword search."""

    # Semantic search
    semantic_results = collection.query(
        query_texts=[query],
        n_results=top_k * 2
    )

    # Keyword search (BM25 or similar)
    keyword_results = keyword_index.search(query, limit=top_k * 2)

    # Combine scores (reciprocal rank fusion)
    combined_scores = {}

    for rank, doc_id in enumerate(semantic_results['ids'][0]):
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + alpha / (rank + 1)

    for rank, result in enumerate(keyword_results):
        doc_id = result['id']
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 - alpha) / (rank + 1)

    # Sort by combined score
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_id for doc_id, score in ranked[:top_k]]
```

### Reciprocal Rank Fusion (RRF)

A simple but effective method to combine rankings:

```python
def reciprocal_rank_fusion(rankings, k=60):
    """
    Combine multiple rankings using RRF.
    k is a constant (typically 60) that prevents top items
    from dominating too much.
    """
    scores = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

## Query Transformation

### The Problem

User queries are often not optimal for retrieval:

```
User query: "Why isn't my code working?"
Problems:
- Too vague
- No context about what "code" or "working" means
- Better query would be specific error or symptoms
```

### Query Expansion

Generate multiple query variations:

```python
def expand_query(query, llm):
    """Generate query variations for better coverage."""
    prompt = f"""
Generate 3 alternative ways to phrase this search query.
Keep the same intent but use different words.

Original query: {query}

Variations:
1.
2.
3.
"""
    response = llm.generate(prompt)
    variations = parse_variations(response)

    # Search with all variations
    all_results = []
    for q in [query] + variations:
        results = retrieve(q)
        all_results.extend(results)

    # Deduplicate and rank
    return deduplicate_and_rank(all_results)
```

### HyDE: Hypothetical Document Embeddings

Generate a hypothetical answer, then search for similar real documents:

```python
def hyde_retrieval(query, llm, collection):
    """
    Hypothetical Document Embeddings:
    1. Generate a hypothetical answer
    2. Embed the hypothetical answer
    3. Search for similar real documents
    """
    # Generate hypothetical answer
    hypothetical = llm.generate(f"""
Write a paragraph that would answer this question:
{query}

Even if you're not sure, write what a good answer might look like.
""")

    # Search using the hypothetical as query
    results = collection.query(
        query_texts=[hypothetical],
        n_results=5
    )

    return results
```

Why it works: The hypothetical answer contains terms and patterns similar to real answers, improving semantic matching.

### Query Decomposition

Break complex queries into simpler sub-queries:

```python
def decompose_query(query, llm):
    """Break complex query into retrievable sub-queries."""
    prompt = f"""
This query requires multiple pieces of information.
Break it into simpler questions that can each be answered
from a single document.

Query: {query}

Sub-questions:
"""
    sub_queries = llm.generate(prompt)

    # Retrieve for each sub-query
    all_contexts = []
    for sub_q in parse_sub_queries(sub_queries):
        results = retrieve(sub_q)
        all_contexts.extend(results)

    return all_contexts


# Example:
# Query: "Compare the climate policies of the US and EU"
# Sub-queries:
# 1. "What are the current US climate policies?"
# 2. "What are the current EU climate policies?"
```

## Self-Querying

### Extracting Metadata Filters

Let the LLM determine search filters from natural language:

```python
def self_querying_retrieval(query, llm, collection):
    """Extract structured filters from natural language query."""

    # Ask LLM to extract filters
    filter_prompt = f"""
Extract search parameters from this query.

Query: {query}

Extract:
- search_text: The core semantic search query
- date_filter: Any date constraints (null if none)
- category: Document category if specified (null if none)
- source: Specific source if mentioned (null if none)

Return as JSON.
"""

    filters = json.loads(llm.generate(filter_prompt))

    # Build query with filters
    where_clause = {}
    if filters.get("date_filter"):
        where_clause["date"] = {"$gte": filters["date_filter"]}
    if filters.get("category"):
        where_clause["category"] = filters["category"]
    if filters.get("source"):
        where_clause["source"] = filters["source"]

    # Search with extracted parameters
    results = collection.query(
        query_texts=[filters["search_text"]],
        where=where_clause if where_clause else None,
        n_results=5
    )

    return results


# Example:
# Query: "Find articles about climate change from 2023 in the science category"
# Extracted:
# - search_text: "climate change"
# - date_filter: "2023-01-01"
# - category: "science"
```

## Key Takeaways

1. **Chunking strategies significantly impact quality**: Consider semantic boundaries, hierarchical approaches, and parent-child retrieval.

2. **Reranking improves precision**: Use cross-encoders to re-order initial results from faster bi-encoder retrieval.

3. **Hybrid search combines the best of both worlds**: Semantic understanding plus exact keyword matching.

4. **Query transformation improves retrieval**: Expansion, HyDE, and decomposition help match user intent to documents.

5. **Self-querying extracts structure from natural language**: Let the LLM determine filters and parameters.

6. **There's no one-size-fits-all**: Different applications need different combinations of these techniques.

## Further Reading

- "Retrieve and Re-Rank: A Simple and Effective Method for Knowledge-Intensive NLP" (Ram et al., 2022)
- "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE paper, Gao et al., 2022)
- "Query Expansion by Prompting Large Language Models" (Wang et al., 2023)
- "Self-Query Retrieval" in LangChain documentation
- "Mastering RAG: Practical Techniques for Building Production RAG Systems" (Pinecone blog)
