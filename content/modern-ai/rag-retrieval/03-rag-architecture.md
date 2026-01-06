# RAG Architecture: The Retrieve-Then-Generate Pipeline

## Introduction

We've explored why retrieval matters and how vector databases enable semantic search. Now it's time to put the pieces together into a complete **Retrieval-Augmented Generation (RAG)** system. RAG combines the knowledge stored in a document collection with the reasoning capabilities of large language models to produce grounded, accurate responses.

In this lesson, we'll walk through the complete RAG pipeline, from document ingestion to response generation, understanding each component and how they work together.

## The Complete RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ OFFLINE: Document Ingestion                                     │
│ ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌────────────┐ │
│ │ Docs    │ →  │ Chunk    │ →  │ Embed     │ →  │ Store in   │ │
│ │ (PDF,   │    │ into     │    │ each      │    │ Vector DB  │ │
│ │  TXT,   │    │ pieces   │    │ chunk     │    │            │ │
│ │  etc)   │    │          │    │           │    │            │ │
│ └─────────┘    └──────────┘    └───────────┘    └────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ONLINE: Query Processing                                        │
│                                                                  │
│ ┌─────────────┐                                                 │
│ │ User Query  │                                                 │
│ └──────┬──────┘                                                 │
│        │                                                        │
│        ▼                                                        │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 1. RETRIEVE: Find relevant chunks                          │ │
│ │    - Embed query                                            │ │
│ │    - Search vector DB                                       │ │
│ │    - Get top-k chunks                                       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│        │                                                        │
│        ▼                                                        │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 2. AUGMENT: Build prompt with context                       │ │
│ │    - Format retrieved chunks                                │ │
│ │    - Add to prompt template                                 │ │
│ │    - Include user question                                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│        │                                                        │
│        ▼                                                        │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 3. GENERATE: LLM produces response                          │ │
│ │    - Process augmented prompt                               │ │
│ │    - Generate grounded answer                               │ │
│ │    - (Optionally) cite sources                              │ │
│ └─────────────────────────────────────────────────────────────┘ │
│        │                                                        │
│        ▼                                                        │
│ ┌─────────────┐                                                 │
│ │ Response    │                                                 │
│ └─────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Document Ingestion

### Loading Documents

First, gather and parse your documents:

```python
from pathlib import Path

def load_documents(directory):
    """Load documents from various formats."""
    documents = []

    for file_path in Path(directory).rglob("*"):
        if file_path.suffix == ".txt":
            text = file_path.read_text()
        elif file_path.suffix == ".pdf":
            text = extract_pdf_text(file_path)
        elif file_path.suffix == ".md":
            text = file_path.read_text()
        elif file_path.suffix == ".docx":
            text = extract_docx_text(file_path)
        else:
            continue

        documents.append({
            "text": text,
            "source": str(file_path),
            "metadata": extract_metadata(file_path)
        })

    return documents
```

### Chunking

Documents must be split into manageable pieces:

```python
def chunk_document(document, chunk_size=1000, overlap=200):
    """Split document into overlapping chunks."""
    text = document["text"]
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": document["source"],
                "chunk_index": len(chunks),
                "metadata": document["metadata"]
            })

        start = end - overlap

    return chunks
```

**Why chunking matters**:
- LLMs have context limits—can't process entire books
- Smaller chunks enable more precise retrieval
- Overlap prevents losing information at boundaries

### Embedding and Storage

```python
from sentence_transformers import SentenceTransformer
import chromadb

def build_index(documents):
    """Create searchable index from documents."""
    # Initialize
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./rag_index")
    collection = client.get_or_create_collection("documents")

    # Process all documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    # Add to vector database
    collection.add(
        documents=[c["text"] for c in all_chunks],
        ids=[f"{c['source']}_{c['chunk_index']}" for c in all_chunks],
        metadatas=[{
            "source": c["source"],
            "chunk_index": c["chunk_index"]
        } for c in all_chunks]
    )

    print(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")
    return collection
```

## Phase 2: Retrieval

When a user asks a question, find relevant chunks:

```python
def retrieve(query, collection, top_k=5):
    """Find chunks most relevant to the query."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    # Format results
    retrieved = []
    for i in range(len(results['documents'][0])):
        retrieved.append({
            "text": results['documents'][0][i],
            "source": results['metadatas'][0][i]['source'],
            "distance": results['distances'][0][i] if 'distances' in results else None
        })

    return retrieved
```

### Retrieval Quality Matters

Poor retrieval = poor answers, regardless of LLM quality:

```python
# If retrieval misses the relevant document...
query = "What is the company vacation policy?"
retrieved = ["Employee handbook intro", "Benefits overview", "Office locations"]
# ...the LLM has no relevant information to work with!

# Good retrieval enables good answers
retrieved = ["Annual leave: Employees receive 20 days PTO...",
             "Holiday schedule: The following dates are...",
             "Time-off request process: Submit requests..."]
# Now the LLM can give an accurate answer
```

## Phase 3: Augmentation

Build a prompt that combines the query with retrieved context:

```python
def build_prompt(query, retrieved_chunks):
    """Create the augmented prompt."""

    # Format context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    # Build the full prompt
    prompt = f"""Use the following context to answer the question. If the
answer is not contained in the context, say "I don't have enough
information to answer this question."

Context:
{context}

Question: {query}

Answer:"""

    return prompt
```

### Prompt Template Variations

Different use cases need different prompts:

```python
# Factual Q&A
qa_template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

# Summarization
summary_template = """
Summarize the key points from these documents:
{context}

Provide a concise summary:
"""

# Comparison
comparison_template = """
Based on the following information:
{context}

Compare and contrast the approaches described. Identify similarities,
differences, and trade-offs.
"""

# With citations
citation_template = """
Answer the question using the provided sources. Cite sources using
[Source N] notation.

Sources:
{context}

Question: {question}

Provide your answer with citations:
"""
```

## Phase 4: Generation

Send the augmented prompt to the LLM:

```python
import openai

def generate_response(prompt):
    """Generate response from LLM."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that
             answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content
```

## Complete RAG System

Putting it all together:

```python
class RAGSystem:
    def __init__(self, index_path="./rag_index"):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=index_path)
        self.collection = self.client.get_or_create_collection("documents")

    def ingest(self, documents):
        """Add documents to the index."""
        all_chunks = []
        for doc in documents:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)

        self.collection.add(
            documents=[c["text"] for c in all_chunks],
            ids=[f"{c['source']}_{c['chunk_index']}" for c in all_chunks],
            metadatas=[{"source": c["source"]} for c in all_chunks]
        )

    def query(self, question, top_k=5):
        """Answer a question using RAG."""
        # Retrieve
        retrieved = self.retrieve(question, top_k)

        # Augment
        prompt = self.build_prompt(question, retrieved)

        # Generate
        response = self.generate(prompt)

        return {
            "answer": response,
            "sources": [r["source"] for r in retrieved]
        }

    def retrieve(self, query, top_k):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return [{
            "text": results['documents'][0][i],
            "source": results['metadatas'][0][i]['source']
        } for i in range(len(results['documents'][0]))]

    def build_prompt(self, query, retrieved):
        context = "\n\n".join([
            f"[{r['source']}]: {r['text']}"
            for r in retrieved
        ])
        return f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"""

    def generate(self, prompt):
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


# Usage
rag = RAGSystem()

# Index your documents
documents = load_documents("./company_docs")
rag.ingest(documents)

# Ask questions
result = rag.query("What is the refund policy?")
print(result["answer"])
print("Sources:", result["sources"])
```

## Common Challenges

### Context Window Limits

Retrieved content must fit in the context window:

```python
def trim_context(retrieved_chunks, max_tokens=3000):
    """Ensure context fits in token limit."""
    total_tokens = 0
    kept_chunks = []

    for chunk in retrieved_chunks:
        chunk_tokens = count_tokens(chunk["text"])
        if total_tokens + chunk_tokens > max_tokens:
            break
        kept_chunks.append(chunk)
        total_tokens += chunk_tokens

    return kept_chunks
```

### Handling "I Don't Know"

When retrieved content doesn't contain the answer:

```python
prompt = """Answer based only on the provided context. If the answer
is not in the context, respond with "I cannot find this information
in the available documents."

Context:
{context}

Question: {question}
"""

# Better than hallucinating!
```

### Source Attribution

Enable users to verify information:

```python
def format_response_with_citations(answer, sources):
    """Add source information to response."""
    source_list = "\n".join([
        f"- [{i+1}] {source}"
        for i, source in enumerate(set(sources))
    ])

    return f"""{answer}

---
Sources:
{source_list}
"""
```

## Key Takeaways

1. **RAG has four phases**: document ingestion, retrieval, augmentation, and generation.

2. **Chunking is critical**: Documents must be split into pieces small enough for retrieval and context windows.

3. **Retrieval quality determines answer quality**: The LLM can only work with what you give it.

4. **Prompt templates shape behavior**: Different templates suit different use cases (Q&A, summarization, comparison).

5. **Handle edge cases**: Plan for when retrieved content doesn't contain the answer.

6. **Enable verification**: Include source attribution so users can check claims.

## Further Reading

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- LlamaIndex and LangChain documentation for RAG frameworks
- "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)
- "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2023)
