```python
# ============================================================
# COMMON SETUP (shared across all techniques)
# ============================================================

# pip install qdrant-client openai sentence-transformers rank-bm25 \
#            fastembed langchain-text-splitters cohere

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

client = OpenAI(api_key="OPENAI_API_KEY")

qdrant = QdrantClient(":memory:")  # local demo

COLLECTION = "advanced_rag"

qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding
```

---

# 1. Semantic Chunking

Instead of fixed-size chunks, split where topic meaning changes.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("all-MiniLM-L6-v2")

text = open("docs.txt").read()

# initial rough splits
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

rough_chunks = splitter.split_text(text)

# semantic similarity based merge/split
embeddings = encoder.encode(rough_chunks)

semantic_chunks = []
current = rough_chunks[0]

for i in range(1, len(rough_chunks)):
    sim = np.dot(embeddings[i-1], embeddings[i])

    if sim > 0.7:
        current += " " + rough_chunks[i]
    else:
        semantic_chunks.append(current)
        current = rough_chunks[i]

semantic_chunks.append(current)
```

---

# 2. Hierarchical Indexing

Store child chunks but return parent chunk.

```python
parent_docs = [
    {
        "parent_id": "doc_1",
        "text": full_document_text
    }
]

child_chunks = [
    {
        "parent_id": "doc_1",
        "chunk": chunk
    }
    for chunk in semantic_chunks
]

points = []

for idx, item in enumerate(child_chunks):
    points.append(
        PointStruct(
            id=idx,
            vector=embed(item["chunk"]),
            payload={
                "parent_id": item["parent_id"],
                "chunk": item["chunk"]
            }
        )
    )

qdrant.upsert(COLLECTION, points)

# retrieval
hits = qdrant.search(
    collection_name=COLLECTION,
    query_vector=embed("What is transformer attention?"),
    limit=3
)

# return parent docs instead
parent_ids = list(set(h.payload["parent_id"] for h in hits))
```

---

# 3. Metadata Filtering

Filter before semantic retrieval.

```python
payload = {
    "text": chunk,
    "author": "Sagar",
    "doc_type": "research",
    "year": 2025
}

# search only research docs
results = qdrant.search(
    collection_name=COLLECTION,
    query_vector=embed(query),
    query_filter=Filter(
        must=[
            FieldCondition(
                key="doc_type",
                match=MatchValue(value="research")
            )
        ]
    ),
    limit=5
)
```

---

# 4. Hybrid Search (BM25 + Vector)

Combine semantic + keyword search.

```python
from rank_bm25 import BM25Okapi
import numpy as np

docs = [c["chunk"] for c in child_chunks]

tokenized = [d.split() for d in docs]
bm25 = BM25Okapi(tokenized)

query = "multi head attention"

# BM25
bm25_scores = bm25.get_scores(query.split())

# vector search
vector_hits = qdrant.search(
    COLLECTION,
    query_vector=embed(query),
    limit=10
)

# combine scores
hybrid_results = []

for hit in vector_hits:
    text = hit.payload["chunk"]

    hybrid_score = (
        0.7 * hit.score +
        0.3 * bm25_scores[docs.index(text)]
    )

    hybrid_results.append((text, hybrid_score))

hybrid_results = sorted(
    hybrid_results,
    key=lambda x: x[1],
    reverse=True
)
```

---

# 5. Query Rewriting / Transformation

Use LLM to improve retrieval query.

```python
query = "how does it reduce hallucination?"

rewrite_prompt = f"""
Rewrite this query for better vector retrieval.

Query: {query}
"""

rewritten_query = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": rewrite_prompt}]
).choices[0].message.content

results = qdrant.search(
    COLLECTION,
    query_vector=embed(rewritten_query),
    limit=5
)
```

---

# 6. HyDE (Hypothetical Document Embeddings)

Generate hypothetical answer first.

```python
query = "How does RAG reduce hallucination?"

hyde_prompt = f"""
Write a detailed hypothetical answer.

Question: {query}
"""

hypothetical_doc = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": hyde_prompt}]
).choices[0].message.content

results = qdrant.search(
    COLLECTION,
    query_vector=embed(hypothetical_doc),
    limit=5
)
```

---

# 7. Query Decomposition

Break complex question into sub-queries.

```python
query = """
Compare RAG, Fine-tuning, and Prompt Engineering
for healthcare applications.
"""

decompose_prompt = f"""
Break into smaller searchable questions.

Query: {query}
"""

subqueries = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": decompose_prompt}]
).choices[0].message.content.split("\n")

all_results = []

for q in subqueries:
    hits = qdrant.search(
        COLLECTION,
        query_vector=embed(q),
        limit=3
    )

    all_results.extend(hits)
```

---

# 8. Reranking (Cross Encoder)

Bi-encoders retrieve fast.
Cross-encoders rerank accurately.

```python
# pip install cohere

import cohere

co = cohere.Client("COHERE_API_KEY")

retrieved_docs = [h.payload["chunk"] for h in hits]

reranked = co.rerank(
    query=query,
    documents=retrieved_docs,
    top_n=3,
    model="rerank-v3.5"
)

final_docs = [
    retrieved_docs[r.index]
    for r in reranked.results
]
```

---

# 9. Context Compression

Compress retrieved chunks before generation.

```python
large_context = "\n\n".join(final_docs)

compression_prompt = f"""
Extract only information relevant to the question.

Question:
{query}

Context:
{large_context}
"""

compressed_context = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": compression_prompt
        }
    ]
).choices[0].message.content
```

---

# 10. Self-RAG / Corrective RAG

Evaluate retrieval quality before answering.

```python
evaluation_prompt = f"""
Is this context sufficient to answer the question?

Question:
{query}

Context:
{compressed_context}

Answer only YES or NO.
"""

decision = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": evaluation_prompt}]
).choices[0].message.content

if "NO" in decision:
    # retry with rewritten query
    better_query = rewritten_query

    hits = qdrant.search(
        COLLECTION,
        query_vector=embed(better_query),
        limit=10
    )
```

---

# 11. Agentic RAG

Agents decide:

* retrieve
* search again
* summarize
* cite
* synthesize

```python
TOOLS = {
    "vector_search": lambda q: qdrant.search(
        COLLECTION,
        query_vector=embed(q),
        limit=5
    ),
}

SYSTEM_PROMPT = """
You are an agentic RAG system.

You can:
1. Search vector DB
2. Analyze retrieved context
3. Decide if more retrieval needed
4. Produce cited answer
"""

query = "Explain advantages of hybrid retrieval in enterprise RAG"

tool_results = TOOLS["vector_search"](query)

context = "\n\n".join([
    r.payload["chunk"]
    for r in tool_results
])

final_answer = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
            Question: {query}

            Context:
            {context}

            Answer with citations.
            """
        }
    ]
)

print(final_answer.choices[0].message.content)
```

---

# IMPORTANT REAL-WORLD NOTE

Production systems usually combine multiple techniques together:

```text
User Query
   ↓
Query Rewriting
   ↓
HyDE
   ↓
Hybrid Retrieval
   ↓
Metadata Filtering
   ↓
Reranking
   ↓
Context Compression
   ↓
Self Evaluation
   ↓
Final LLM Generation
```

---

# Enterprise Stack Recommendation

| Layer           | Best Choice            |
| --------------- | ---------------------- |
| Vector DB       | Qdrant                 |
| Embedding       | text-embedding-3-large |
| Hybrid Search   | BM25 + Dense           |
| Reranker        | Cohere Rerank          |
| Agent Framework | LangGraph              |
| Evaluation      | RAGAS                  |
| Compression     | LLM Compressor         |
| Retrieval       | Parent-Child Retriever |
