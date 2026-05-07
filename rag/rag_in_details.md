# RAG (Retrieval-Augmented Generation) — Complete Study Notes

---

## What is RAG and Why Does It Exist?

A Large Language Model (LLM) is trained on a fixed dataset up to a cutoff date. It cannot know about your company's internal documents, last week's news, or a private database. RAG solves this by giving the LLM a **retrieval mechanism** — it fetches relevant information at query time and injects it into the prompt before generation.

**Mental model:** Think of RAG as an open-book exam. The LLM is the student who is smart but can't memorize everything. The vector database is the textbook. RAG lets the student look up answers before writing.

---
<details>
<summary>Part 1 — Core Components (The Pipeline)</summary>
## Part 1 — Core Components (The Pipeline)

### 1.1 Indexing (The Setup Phase)

Indexing is the offline, one-time (or periodic) process of preparing your knowledge base. The three sub-steps are loading, chunking, and embedding.

---

### 1.2 Document Loaders

**What it is:** Tools that import raw data from various sources into the pipeline.

**Examples:**
- `LangChain` has loaders for PDFs (`PyPDFLoader`), websites (`WebBaseLoader`), SQL databases, Notion, Google Drive, CSV files, YouTube transcripts, and more.
- `LlamaIndex` provides similar abstractions called `Readers`.

**Code example (LangChain):**
```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()
# Returns a list of Document objects with .page_content and .metadata
```

**Key idea:** The loader's job is to produce a list of `Document` objects with raw text and metadata (filename, page number, source URL, etc.).

---

### 1.3 Chunking

**What it is:** Breaking large documents into smaller pieces so they fit within an LLM's context window and can be retrieved at a granular level.

**Why it matters:** A 100-page PDF can't be injected wholesale into a prompt. You need to retrieve only the relevant 3–5 paragraphs.

**Common strategies:**

| Strategy | Description | When to use |
|---|---|---|
| Fixed-size | Split every N tokens/characters | Simple, fast baseline |
| Recursive character | Try `\n\n`, then `\n`, then ` ` | General purpose |
| Semantic | Split at semantic shifts (embedding similarity drops) | High-quality, expensive |

**Code example:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
# chunk_overlap=50 ensures context isn't lost at boundaries
```

**Important:** `chunk_overlap` is critical. Without it, a sentence that spans two chunks might lose context.

---

### 1.4 Embeddings

**What it is:** Converting a text chunk into a high-dimensional numerical vector (e.g., 1536 dimensions for OpenAI's ada-002) that captures its *semantic meaning*.

**Key insight:** Chunks with similar meaning will have vectors that are geometrically close in this high-dimensional space. "How do I reset my password?" and "Steps to change account credentials" will have very similar vectors, even though they share almost no words.

**Popular embedding models:**
- `OpenAI text-embedding-ada-002` (1536 dims)
- `sentence-transformers/all-MiniLM-L6-v2` (384 dims, open-source, fast)
- `Cohere Embed v3`

**Code example:**
```python
from langchain.embeddings import OpenAIEmbeddings

embedder = OpenAIEmbeddings()
vector = embedder.embed_query("How do I reset my password?")
# Returns a list of 1536 floats
```

---

### 1.5 Vector Database

**What it is:** A specialized database designed to store embedding vectors and perform fast *approximate nearest neighbor* (ANN) search.

**How it works:** When you search, your query is also embedded into a vector, and the database finds the stored vectors geometrically closest to it (cosine similarity or dot product).

**Popular options:**

| Database | Notes |
|---|---|
| `Faiss` (Meta) | In-memory, blazing fast, no server required |
| `Weaviate` | Open-source, schema-based, hybrid search built in |
| `Pinecone` | Managed cloud service, simple API |
| `Chroma` | Great for local development, Python-native |
| `Qdrant` | High performance, rich filtering |

**Code example (storing + searching):**
```python
from langchain.vectorstores import Chroma

# Store chunks
vectordb = Chroma.from_documents(chunks, embedding=embedder)

# Retrieve
results = vectordb.similarity_search("password reset steps", k=5)
# Returns top-5 most similar chunks
```

---

### 1.6 Retrieval

**What it is:** At query time, the user's question is embedded and used to search the vector database. The top-K most semantically similar chunks are returned.

**The retrieval step bridges the indexed knowledge base and the LLM generation step.**

---

### 1.7 Generation

**What it is:** The retrieved chunks are formatted into a prompt alongside the user's original question, and the LLM generates a grounded, factual response.

**Basic prompt template:**
```
You are a helpful assistant. Use only the context below to answer the question.

Context:
{retrieved_chunk_1}
{retrieved_chunk_2}
...

Question: {user_query}
Answer:
```

**The LLM is now grounded** — it won't hallucinate because it has real source material to reference.

</details>
---
<details>
<summary>Part 2 — Advanced RAG Techniques</summary>

## Part 2 — Advanced RAG Techniques

### 2.1 Pre-Retrieval (Ingestion Optimization)

These techniques improve the *quality of what's stored* in your vector database.

---

#### 2.1.1 Semantic Chunking

**Problem with fixed-size chunking:** A 500-token window might cut a paragraph mid-thought, or lump together two completely unrelated topics.

**Solution:** Embed each sentence, then measure the *cosine similarity drop* between adjacent sentences. Where similarity drops sharply, a topic boundary exists — split there.

**Example:**
```
[Sentence 1: "Python is a high-level language."]
[Sentence 2: "It was created by Guido van Rossum."]  ← similar to S1, same chunk
[Sentence 3: "The Eiffel Tower was built in 1889."]  ← BIG similarity drop → NEW CHUNK
```

---

#### 2.1.2 Hierarchical Indexing (Parent-Child Chunks)

**Problem:** Small chunks retrieve precisely but lose surrounding context. Large chunks have context but match less precisely.

**Solution:** Index small chunks for search precision, but when a small chunk matches, return its larger *parent* chunk for context-rich generation.

```
Parent Chunk (512 tokens): "Chapter 3: Network Security... [full chapter]"
  ├── Child Chunk (128 tokens): "Firewalls filter traffic based on rules..."
  ├── Child Chunk (128 tokens): "VPNs encrypt data in transit..."
  └── Child Chunk (128 tokens): "Zero Trust assumes no implicit trust..."
```

Search matches the child → return the parent to the LLM. Best of both worlds.

---

#### 2.1.3 Metadata Filtering

**What it is:** Attaching structured metadata tags to each chunk, then using those tags to pre-filter the search space before semantic search runs.

**Why it matters:** If a user asks "What was our Q4 revenue policy in 2023?", you don't need to search documents from 2020–2022 at all. Filter those out first, then run similarity search on the rest.

**Example metadata schema:**
```json
{
  "source": "annual_report_2023.pdf",
  "date": "2023-12-31",
  "department": "finance",
  "document_type": "policy",
  "page": 14
}
```

**Weaviate query example:**
```python
results = vectordb.similarity_search(
    query="Q4 revenue policy",
    filter={"department": "finance", "date": {"$gte": "2023-01-01"}}
)
```

---

#### 2.1.4 Hybrid Search

**Problem:** Semantic search can miss exact matches. If a user searches for a specific product SKU like "X-7734-B", the embedding model may not understand it semantically, but a keyword search would find it instantly.

**Solution:** Combine semantic (vector) search with keyword-based search (BM25). Merge and deduplicate results using a score fusion technique like Reciprocal Rank Fusion (RRF).

```
BM25 results:  [doc_A (rank 1), doc_C (rank 2), doc_F (rank 3)]
Vector results: [doc_C (rank 1), doc_A (rank 2), doc_B (rank 3)]
Fused RRF:     [doc_A (combined best), doc_C, doc_B, doc_F]
```

This is now the standard baseline for production RAG systems.

---

### 2.2 Retrieval Optimization

These techniques improve *how you query* the vector database.

---

#### 2.2.1 Query Rewriting / Transformation

**Problem:** Users write queries in natural, colloquial language that may not match the formal language in your documents.

**User asks:** "what's the deal with the new leave policy"
**Document says:** "Section 4.2: Amended PTO and Leave Entitlements — Effective Q1 2024"

**Solution:** Use an LLM to rewrite the query before embedding it.

```python
rewrite_prompt = """Rewrite the following user question into a formal search query 
that would best match a corporate policy document.

User question: {query}
Formal search query:"""

# User: "what's the deal with the new leave policy"
# Rewritten: "Amended paid time off and leave entitlement policy 2024"
```

---

#### 2.2.2 HyDE — Hypothetical Document Embeddings

**Insight:** Instead of embedding the *question* (which is short and sparse), ask the LLM to *imagine what a perfect answer document looks like*, then embed that hypothetical document. It will land much closer to real answer documents in the vector space.

**Flow:**
```
User Query → LLM → Hypothetical Answer → Embed → Vector Search → Real Documents
```

**Example:**
- Query: "What causes inflation?"
- Hypothetical doc: "Inflation is primarily caused by excess money supply relative to goods, supply chain disruptions, and demand-pull pressures when consumers spend more than production capacity..."
- This hypothetical answer vector is far richer and more precise than the query vector alone.

---

#### 2.2.3 Query Decomposition

**Problem:** Complex multi-part questions can't be answered by a single retrieval step.

**User asks:** "Compare the tax implications of an LLC vs S-Corp for a startup with 3 founders and over $500K revenue."

**Solution:** Use an LLM to break this into sub-questions, retrieve independently, then synthesize.

```
Sub-question 1: "What are the tax implications of an LLC structure?"
Sub-question 2: "What are the tax implications of an S-Corp structure?"
Sub-question 3: "How does revenue above $500K affect entity tax treatment?"
Sub-question 4: "How does multi-founder ownership affect LLC vs S-Corp taxes?"
```

Each sub-question retrieves separately, and the answers are combined for a comprehensive final response.

---

### 2.3 Post-Retrieval (Context Processing)

These techniques improve the *quality of context* you pass to the LLM after retrieval.

---

#### 2.3.1 Reranking

**Problem:** The initial vector search ranks by approximate similarity (cosine distance), which is fast but imprecise. It doesn't understand the nuanced relationship between the query and each retrieved document.

**Solution:** Take the top-K retrieved chunks (e.g., 20) and pass them through a *cross-encoder* model that looks at the query and each document *together* (not independently) and scores true relevance. Keep only the top 3–5 for the final prompt.

**Cross-encoder vs Bi-encoder:**
- Bi-encoder: embeds query and doc separately → fast but less accurate
- Cross-encoder: processes query+doc together → slower but much more accurate

**Popular tool:** Cohere Rerank API, or `cross-encoder/ms-marco-MiniLM-L-6-v2` from HuggingFace.

```python
from cohere import Client

co = Client("your-api-key")
results = co.rerank(
    query="What causes inflation?",
    documents=[chunk.page_content for chunk in retrieved_chunks],
    top_n=3,
    model="rerank-english-v3.0"
)
```

---

#### 2.3.2 Context Compression

**Problem:** Retrieved chunks often contain a lot of irrelevant sentences surrounding the answer. Passing the entire chunk wastes tokens and can distract the LLM.

**Solution:** Use a small LLM to extract only the sentences from each chunk that are actually relevant to the query, discarding the rest.

**Example:**
- Retrieved chunk (300 tokens): Long paragraph about company history, founding, headquarters, CEO bio... and 2 sentences about remote work policy.
- Compressed output (40 tokens): Just the 2 sentences about remote work policy.

**LangChain implementation:**
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)
```

---

### 2.4 Generation & Evaluation

---

#### 2.4.1 Self-RAG / Corrective RAG

**Problem:** Standard RAG blindly trusts whatever was retrieved, even if the retrieved documents are irrelevant or contradictory.

**Self-RAG** introduces a *reflection* step: the system evaluates whether the retrieved context is actually useful before generating. If not, it retrieves again with a different query.

**Decision loop:**
```
Query → Retrieve → [Is context relevant?] → YES → Generate → [Is answer supported?] → YES → Return
                                          ↓ NO                                       ↓ NO
                                   Reformulate query                           Re-retrieve or flag
```

**Corrective RAG (CRAG)** goes further: if retrieved docs score low in relevance, it triggers a web search to supplement with fresher information.

---

#### 2.4.2 Agentic RAG

**What it is:** Rather than a fixed pipeline, an LLM *agent* dynamically decides how to retrieve information, which tools to use, and how many retrieval steps are needed.

**Example workflow:**
1. User asks: "Summarize our Q3 financials and compare with industry benchmarks."
2. Agent calls `search_internal_docs("Q3 financial report")`
3. Agent calls `web_search("SaaS industry revenue benchmarks Q3 2024")`
4. Agent synthesizes both sources with citations
5. Agent reflects: "Do I have enough data to answer?" → If not, retrieves more.

**Frameworks:** LangGraph, LlamaIndex Agents, AutoGen, CrewAI.

**Key difference from standard RAG:** Standard RAG is a linear pipeline. Agentic RAG is a *loop* with tools, memory, and decision-making.

---

## Summary: When to Use What

| Technique | Use when... |
|---|---|
| Semantic Chunking | Documents have varied topic density |
| Hierarchical Indexing | You need both search precision and contextual richness |
| Metadata Filtering | Documents have clear categorical structure (date, dept, type) |
| Hybrid Search | Users mix exact-match and conceptual queries |
| Query Rewriting | Users write informal/colloquial queries |
| HyDE | Query vectors don't match document vectors well |
| Query Decomposition | Users ask multi-part complex questions |
| Reranking | Retrieval precision is critical, latency budget allows extra step |
| Context Compression | Token budget is tight or LLM gets distracted by noise |
| Self-RAG / CRAG | High-stakes domains needing factual accuracy (legal, medical) |
| Agentic RAG | Multi-source synthesis, dynamic workflows, complex research tasks |

---
</details>
