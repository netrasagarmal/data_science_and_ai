# RAG (Retrieval-Augmented Generation) — Complete Study Notes

![RAG Pipeline](/static/rag_pipeline.png)
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

> Traditional RAG assumes the answer exists in one or two retrieved chunks.

In real enterprise systems, answers are often:

* spread across multiple documents
* require reasoning across sources
* require iterative retrieval
* require retrieval quality validation

Let's go deep into each.

---

# 1. Multi-Hop Retrieval

## What Problem Does It Solve?

Traditional RAG:

```text
Question
    ↓
Single Retrieval
    ↓
Answer
```

works for:

```text
"What is the leave policy?"
```

because answer exists in one chunk.

But consider:

```text
Which cloud provider hosts the application used by the Finance team?
```

The answer may require:

Document A:

```text
Finance team uses SAP Analytics.
```

Document B:

```text
SAP Analytics is deployed on Azure.
```

Neither document alone contains the answer.

Need:

```text
Hop 1:
Finance Team
    ↓
SAP Analytics

Hop 2:
SAP Analytics
    ↓
Azure
```

Final answer:

```text
Finance team's application is hosted on Azure.
```

---

# How Multi-Hop Retrieval Works

```text
User Question
      ↓
Retrieve Chunk A
      ↓
Extract Intermediate Entity
      ↓
New Query
      ↓
Retrieve Chunk B
      ↓
Combine Evidence
      ↓
Answer
```

---

## Example

Knowledge Base

### Document 1

```text
CEO of OpenAI is Sam Altman.
```

### Document 2

```text
Sam Altman invested in Helion Energy.
```

Question:

```text
Which energy company has received investment from OpenAI's CEO?
```

---

### Retrieval Step 1

Search:

```text
OpenAI CEO
```

Retrieve:

```text
CEO = Sam Altman
```

---

### Retrieval Step 2

New query:

```text
Sam Altman investments
```

Retrieve:

```text
Helion Energy
```

---

### Final Reasoning

```text
OpenAI CEO
      ↓
Sam Altman
      ↓
Helion Energy
```

Answer:

```text
Helion Energy
```

---

# Enterprise Example

Question:

```text
Which vendor provides the database used by the billing system?
```

---

Hop 1:

```text
Billing System
    ↓
PostgreSQL
```

---

Hop 2:

```text
PostgreSQL
    ↓
AWS RDS
```

---

Hop 3:

```text
AWS RDS
    ↓
Amazon
```

Final answer:

```text
Amazon provides the database service.
```

---

# Typical Implementation

Using agent workflow:

```python
while answer_not_found:
    retrieve()
    extract_entity()
    generate_new_query()
```

Very common in:

* LangGraph
* CrewAI
* AutoGen
* Agentic RAG

---

# 2. Cross-Document Retrieval

People often confuse this with Multi-Hop.

They are different.

---

## Multi-Hop

Requires sequential reasoning.

```text
Doc A → Doc B → Answer
```

---

## Cross-Document Retrieval

Requires combining information from many documents simultaneously.

```text
Doc A
Doc B
Doc C
 ↓
Combined Answer
```

---

## Example

Question:

```text
Compare Azure, AWS and GCP pricing strategies.
```

Documents:

### Doc A

```text
AWS pricing...
```

### Doc B

```text
Azure pricing...
```

### Doc C

```text
GCP pricing...
```

Need information from ALL documents.

---

Retrieval:

```text
AWS chunk
Azure chunk
GCP chunk
```

---

LLM combines:

```text
AWS → Pay as you go

Azure → Enterprise discounts

GCP → Sustained usage discounts
```

---

Final answer:

Comparison table.

---

# Enterprise Example

Question:

```text
Summarize all security incidents reported in 2025.
```

Incidents stored in:

```text
Incident_01.pdf
Incident_02.pdf
Incident_03.pdf
...
Incident_20.pdf
```

Need retrieval from:

```text
20 documents
```

then synthesis.

This is cross-document retrieval.

---

# Multi-Hop vs Cross-Document

| Multi-Hop                       | Cross-Document       |
| ------------------------------- | -------------------- |
| Sequential retrieval            | Parallel retrieval   |
| A → B → C                       | A + B + C            |
| Requires intermediate reasoning | Requires aggregation |
| Agentic workflows common        | RAG synthesis common |

---

# 3. Corrective RAG (CRAG)

CRAG = Corrective Retrieval-Augmented Generation

Paper idea:

> Before trusting retrieved documents, evaluate whether retrieval quality is good enough.

Traditional RAG:

```text
Retrieve
    ↓
Generate
```

Problem:

Bad retrieval → Hallucination

---

Example

Question:

```text
What is Qdrant?
```

Retrieved chunk:

```text
Pinecone is a managed vector database.
```

Poor retrieval.

Traditional RAG:

```text
Uses wrong chunk.
```

Produces bad answer.

---

CRAG introduces:

```text
Retrieve
    ↓
Evaluate Retrieval
    ↓
Good? → Continue
Bad?  → Fix Retrieval
```

---

# CRAG Workflow

```text
User Query
      ↓
Retrieve
      ↓
Evaluator
      ↓
Score
      ↓
Good?
 ┌────┴─────┐
 │          │
Yes         No
 │          │
Generate   Re-Retrieve
```

---

# Example

Question:

```text
How does Hybrid Search work?
```

Retrieved docs:

```text
Document about OCR
Document about Images
```

Evaluator says:

```text
Relevance = 0.2
```

Poor retrieval.

---

CRAG may:

### Option 1

Rewrite query

```text
Hybrid Search
    ↓
Dense + Sparse Retrieval
```

Retrieve again.

---

### Option 2

Search web.

---

### Option 3

Search another knowledge source.

---

### Option 4

Increase retrieval depth.

```text
Top 5
   ↓
Top 20
```

---

Then answer.

---

# Interview Explanation

A strong answer:

> CRAG adds a retrieval quality assessment layer. Instead of assuming retrieved documents are relevant, the system scores retrieval quality and triggers corrective actions such as query rewriting, re-retrieval, web search, or alternate knowledge source lookup before generation.

---

# 4. Self-RAG (Self-Reflection RAG)

CRAG evaluates retrieval.

Self-RAG evaluates itself.

---

Paper idea:

LLM learns to ask:

```text
Do I need retrieval?
Is retrieval sufficient?
Should I retrieve more?
Is my answer supported?
```

---

Traditional RAG

```text
Retrieve
Answer
Done
```

---

Self-RAG

```text
Retrieve
↓
Reflect
↓
Generate
↓
Reflect Again
↓
Revise
```

---

# Example

Question:

```text
What are the security controls used by the payment system?
```

Retrieved:

```text
Encryption at rest
Role-based access control
```

---

LLM Reflection

```text
Do I have enough information?

No.
```

---

Retrieves again.

Finds:

```text
MFA
Audit Logging
```

---

Now answer.

---

# Self-RAG Reflection Tokens

Paper introduces special decisions:

```text
Retrieve?
```

```text
Relevant?
```

```text
Supported?
```

```text
Complete?
```

---

Example Flow

```text
Question
    ↓
Need Retrieval?
    ↓
Retrieve
    ↓
Relevant?
    ↓
Generate Draft
    ↓
Supported?
    ↓
Final Answer
```

---

# Example

Question:

```text
How does our company deploy AI models?
```

Retrieved:

```text
Chunk about model deployment
```

Draft answer generated.

Reflection step:

```text
Is every claim supported?
```

Finds:

```text
Claim:
Uses Kubernetes

Evidence:
Missing
```

Answer revised:

```text
According to available documentation...
```

instead of hallucinating.

---

# CRAG vs Self-RAG

This is a favorite interview question.

| CRAG                       | Self-RAG                                |
| -------------------------- | --------------------------------------- |
| Focus on retrieval quality | Focus on retrieval + generation quality |
| External evaluator         | LLM self-reflection                     |
| Correct bad retrieval      | Correct reasoning and retrieval         |
| Simpler                    | More advanced                           |
| Retrieval-centric          | End-to-end self-checking                |

---

# How Modern Agentic RAG Combines All Four

A production-grade enterprise agent may look like:

```text
User Query
     ↓
Query Rewriting
     ↓
Multi-Hop Retrieval
     ↓
Cross-Document Retrieval
     ↓
CRAG Evaluation
     ↓
If Poor Retrieval
      └── Re-search
     ↓
Context Assembly
     ↓
LLM Generation
     ↓
Self-RAG Reflection
     ↓
Evidence Verification
     ↓
Final Answer
```

This is close to how advanced enterprise knowledge assistants, AI copilots, and research agents are built today: multi-hop retrieval discovers missing evidence, cross-document retrieval aggregates evidence, CRAG validates retrieval quality, and Self-RAG validates whether the final answer is actually supported by that evidence.



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

---

<details>
<summary>Multi Model Rag Systems</summary>
This is one of the most important design decisions in Multimodal RAG.

Many engineers make the mistake of treating everything as text and storing OCR output only. In production-grade multimodal RAG systems, the storage, embedding, and retrieval strategy depends heavily on the content type.

---

# First Principle

For every modality ask:

1. What should be stored?
2. What should be embedded?
3. Where should the original content live?
4. How should retrieval happen?

A common architecture is:

```text
                 Raw Files
                      │
                      ▼
                Object Storage
         (S3, Azure Blob, GCS, MinIO)
                      │
                      ▼
              Extraction Pipeline
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
   Text          Tables          Images
      │               │               │
      ▼               ▼               ▼
  Embeddings     Embeddings     Embeddings
      │               │               │
      └──────► Vector Database ◄──────┘
                    (Qdrant)

Metadata:
{
 file_id,
 page_no,
 chunk_type,
 object_storage_path
}
```

---

# 1. Tables from PDF / Documents

This is the most misunderstood part of RAG.

---

## Wrong Approach

```text
Revenue | 2023 | 100M
Revenue | 2024 | 120M
```

Convert to plain text and embed.

Problem:

```text
What was revenue growth?
```

Vector similarity often fails because table relationships are lost.

---

# Strategy 1: Table → Natural Language

Convert table to descriptive text.

Example table:

| Year | Revenue |
| ---- | ------- |
| 2023 | 100M    |
| 2024 | 120M    |

Convert to:

```text
Company revenue was 100 million in 2023
and increased to 120 million in 2024,
representing 20 percent growth.
```

Embed this text.

### Pros

Simple

### Cons

May lose detail

---

# Strategy 2: Store Structured Table + Summary

Production systems often store:

```python
{
  "table_json": {
      ...
  },

  "table_summary":
  "Revenue increased from 100M to 120M..."
}
```

Embed only summary.

Store actual table separately.

---

# Strategy 3: Table-Aware Embeddings (Recommended)

Modern approach.

Extract table as:

```python
DataFrame
```

Generate:

```python
table_summary
```

Store:

```text
Vector DB:
    summary embedding

Object Store:
    original table

Metadata:
    table_id
```

---

# Extraction Models

Popular options:

### OCR PDFs

* PaddleOCR
* Azure Document Intelligence
* Google Document AI
* Amazon Textract

### Native PDFs

* Camelot
* Tabula
* pdfplumber
* Unstructured.io

### Advanced

* Microsoft Table Transformer (TATR)
* Docling
* MinerU

---

# Embedding Models for Tables

Most teams don't directly embed tables.

Instead:

```text
Table
   ↓
LLM Summary
   ↓
Text Embedding
```

Models:

* OpenAI text-embedding-3-large
* BGE-M3
* Voyage-3
* Jina Embeddings v3

---

# Retrieval Flow

```text
Question
   ↓
Vector Search
   ↓
Table Summary Match
   ↓
Get table_id
   ↓
Load original table
   ↓
Pass to LLM
```

---

# 2. Images inside PDF

Suppose PDF contains:

```text
Page 1: Text
Page 2: Architecture Diagram
Page 3: Screenshot
```

Text embeddings alone miss the image meaning.

---

# Standard Practice

Extract image separately.

Store:

```python
{
    image_id,
    pdf_id,
    page_no,
    image_path
}
```

---

# Generate Image Caption

Use VLM:

Examples:

* GPT-4o
* GPT-4.1 Vision
* Gemini
* Claude Vision
* Qwen-VL
* InternVL

Generate:

```text
The diagram shows a RAG architecture
consisting of ingestion, vector storage,
retrieval and generation stages.
```

Embed caption.

---

# Store

Vector DB:

```python
{
   image_id,
   caption_embedding
}
```

Object Storage:

```text
s3://bucket/image123.png
```

---

# Better Approach: Multi-Vector Retrieval

Store:

```text
Caption Embedding

+
Image Embedding
```

---

# Image Embedding Models

### OpenAI

* text-embedding-3-large (caption route)

### CLIP Family

* CLIP
* OpenCLIP

### Modern Multimodal

* SigLIP
* ColPali
* ColQwen
* Jina CLIP v2

---

# Retrieval

```text
Question
    ↓
Embedding
    ↓
Search Image Captions
    ↓
Retrieve image
    ↓
Send image to VLM
```

or

```text
Question
     ↓
Cross-modal embedding
     ↓
Direct image retrieval
```

---

# 3. Image-Only Documents

Example:

```text
Medical image
Satellite image
Invoice image
Photo
Engineering drawing
```

No text exists.

---

# Strategy 1: Caption-Based Retrieval

Generate:

```text
A chest X-ray showing...
```

Embed caption.

Simple.

---

# Strategy 2: Visual Embedding Retrieval

Recommended.

Generate image embedding directly.

Models:

### CLIP

```text
Image -> vector
Text -> vector
```

Same embedding space.

---

Example:

```text
Query:
"dog playing in snow"

Text Embedding
       ↓

Search

Image Embeddings
```

Works without OCR.

---

# Best Models Today

### General

* SigLIP
* OpenCLIP
* Jina CLIP
* Nomic Vision

### Document Images

* ColPali
* ColQwen

These are becoming very popular in document RAG.

Reason:

No OCR required.

They understand:

* tables
* charts
* screenshots
* scanned docs

directly.

---

# ColPali Architecture

Instead of:

```text
Image
 ↓
OCR
 ↓
Embedding
```

Use:

```text
Image
 ↓
ColPali
 ↓
Embedding
```

Much better retrieval.

---

# 4. Videos

Video is actually:

```text
Video
 =

Frames
 +
Audio
 +
Temporal Context
```

Treating video as one embedding is usually a mistake.

---

# Standard Production Pipeline

```text
Video
   │
   ├── Extract Audio
   │
   ├── Extract Key Frames
   │
   └── Scene Detection
```

---

# Audio Processing

Convert speech:

```text
Audio
  ↓
Whisper
  ↓
Transcript
```

Embed transcript chunks.

---

# Visual Processing

Extract frames:

```text
Every 5 sec

or

Scene Change
```

Example:

```text
Frame 1
Frame 2
Frame 3
```

Generate captions.

```text
A person demonstrates a transformer architecture.
```

Embed captions.

---

# Storage

Object Storage:

```text
video.mp4
```

Metadata:

```python
{
   timestamp: 120,
   frame_id: 45,
   video_id: xyz
}
```

Vector DB:

```python
{
   transcript_embedding
}

{
   frame_caption_embedding
}
```

---

# Advanced Video RAG

Store multiple vectors:

```text
Transcript Vector

Frame Vector

Scene Summary Vector
```

---

# Video Embedding Models

### Audio

* Whisper
* Whisper Large V3

### Frame/Image

* CLIP
* SigLIP
* ColQwen

### End-to-End Video

* VideoCLIP
* Video-LLaVA
* InternVideo
* VideoPrism

---

# What Most Enterprise Multimodal RAG Systems Do Today

For PDFs containing text + tables + images:

```text
PDF
│
├── Text Chunks
│      ↓
│   Text Embedding
│
├── Tables
│      ↓
│   LLM Summary
│      ↓
│   Embedding
│
├── Images
│      ↓
│   VLM Caption
│      ↓
│   Embedding
│
└── Original Assets
       ↓
   S3/Blob Storage
```

Store all embeddings in a single collection:

```python
payload = {
    "type": "text" | "table" | "image",
    "page": 10,
    "source_file": "annual_report.pdf",
    "object_path": "s3://..."
}
```

At query time:

```text
User Query
     ↓
Hybrid Retrieval
     ↓
Text Matches
     +
Table Matches
     +
Image Matches
     ↓
Reranking
     ↓
Context Assembly
     ↓
LLM/VLM
```
</details>

---

<details>
<summary>Scenario Based Questions</summary>
These are exactly the types of architecture and system design questions that are now being asked in Senior AI Engineer / GenAI Engineer / AI Architect interviews.

The interviewer is usually not testing whether you know a specific library. They are testing whether you understand:

* Data extraction strategy
* Multimodal representation
* Storage architecture
* Retrieval architecture
* Tradeoffs
* Scalability

A 5-6 year experienced engineer should answer at architecture level first and implementation level second.

---

# Scenario 1

## Interview Question

> Suppose a PDF page contains text, tables and images together. How would you extract, store, embed and retrieve that information in a RAG system?

---

# How I would answer

First, I would not treat the PDF as a single text document because text, tables and images have different semantic characteristics and retrieval requirements.

My approach would be to separate the extraction pipeline by modality and then unify them during retrieval.

---

# Step 1: Document Parsing

Suppose page contains:

```text
------------------------------------------------
Revenue Report 2024

Text:
Revenue increased significantly this year.

Table:
Year     Revenue
2023     100M
2024     120M

Image:
Revenue growth chart
------------------------------------------------
```

I would use a document intelligence system such as:

* Azure Document Intelligence
* Google Document AI
* Docling
* MinerU
* Unstructured

to extract page layout information.

The parser should identify:

```python
{
    page_no:1,

    text_blocks:[...],

    tables:[...],

    images:[...]
}
```

At this stage I preserve document structure.

---

# Step 2: Text Processing

For text blocks:

```text
Revenue increased significantly this year.
```

I perform:

```text
Cleaning
↓
Semantic Chunking
↓
Embedding
```

Example:

```python
{
    type:"text",
    page:1,
    content:"Revenue increased significantly..."
}
```

Embedding:

```python
text-embedding-3-large
BGE-M3
Voyage
```

Store in Qdrant.

---

# Step 3: Table Processing

This is where many systems fail.

I do NOT directly flatten tables into text and embed.

Instead:

Extract:

```python
{
    "Year": [2023,2024],
    "Revenue":[100,120]
}
```

Generate table summary using LLM:

```text
Revenue increased from 100M in 2023
to 120M in 2024 representing
20% growth.
```

Store:

```python
{
    table_id,
    raw_table_json,
    table_summary
}
```

Embed:

```text
table_summary
```

Store original table separately.

Reason:

* retrieval becomes easier
* numerical relationships preserved

---

# Step 4: Image Processing

For image/chart:

```text
Revenue Growth Chart
```

I extract image.

Store image in object storage:

```text
S3
Azure Blob
GCS
```

Generate image description using VLM:

```text
The chart shows revenue growth
from 100M in 2023 to 120M in 2024.
```

Generate embedding from:

```text
caption
```

or

```text
caption + image embedding
```

Store in vector DB.

---

# Final Storage Design

## Vector Database

```python
[
 {
   id:"txt_1",
   type:"text",
   embedding:[...]
 },

 {
   id:"tbl_1",
   type:"table",
   embedding:[...]
 },

 {
   id:"img_1",
   type:"image",
   embedding:[...]
 }
]
```

---

## Object Storage

```text
pdfs/
images/
tables/
```

Store original assets.

---

# Metadata Design

```python
{
   document_id,
   page_number,
   chunk_type,
   object_path,
   source_pdf
}
```

This becomes extremely important later.

---

# Retrieval Flow

User asks:

```text
How much revenue growth occurred in 2024?
```

Query embedding generated.

Search Qdrant.

Potential matches:

```text
Text Chunk
Table Summary
Image Caption
```

Top results:

```text
1. Table Summary
2. Chart Caption
3. Text Paragraph
```

Then:

```text
table_id found
↓
load raw table
↓
image_id found
↓
load image
```

Final context:

```text
Text
+
Table
+
Image
```

sent to GPT-4o or Gemini.

---

# Architecture Diagram

```text
PDF
 │
 ├── Text
 │      ↓
 │   Chunking
 │      ↓
 │   Embedding
 │
 ├── Tables
 │      ↓
 │   Summary
 │      ↓
 │   Embedding
 │
 ├── Images
 │      ↓
 │   Caption
 │      ↓
 │   Embedding
 │
 └─────────────► Qdrant

Raw Assets
     ↓
S3 / Blob Storage
```

---

# Why This Is Good

Because:

* Each modality optimized separately
* Original fidelity preserved
* Retrieval quality improved
* Scales to millions of documents

---

# Scenario 2

## Interview Question

> Design a multimodal RAG system where users can upload documents, tables, images and reference videos, and during chat the system should retrieve the most relevant content regardless of modality.

---

# How I Would Answer

I would design it as a multimodal knowledge platform rather than a traditional text-only RAG system.

The key principle is:

```text
Everything becomes retrievable
through a common semantic layer.
```

---

# High-Level Architecture

```text
                User Upload
                      │
 ┌──────────────┬─────┴─────┬─────────────┐
 │              │           │             │
 PDF          Images      Tables       Videos
 │              │           │             │
 └──────────────┴───────────┴─────────────┘
                      │
               Processing Layer
                      │
             Embedding Generation
                      │
                 Vector DB
                      │
                  Chat Layer
```

---

# Ingestion Pipeline

---

## Document Text

Extract:

```text
Paragraphs
Sections
Headers
```

Perform:

```text
Semantic Chunking
Embedding
Store
```

---

## Tables

Extract:

```text
Structured Table
```

Generate:

```text
Table Summary
```

Store:

```text
Raw Table
+
Summary Embedding
```

---

## Images

Generate:

```text
Caption
```

Store:

```text
Image File
Caption Embedding
```

Optional:

```text
CLIP embedding
```

---

## Videos

Videos require special handling.

---

### Step 1

Extract audio.

```text
Video
 ↓
Audio
```

---

### Step 2

Transcribe.

```text
Whisper
 ↓
Transcript
```

---

### Step 3

Extract keyframes.

```text
Frame every N seconds
```

or

```text
Scene Change Detection
```

---

### Step 4

Caption keyframes.

```text
Frame Caption
```

---

### Store

```python
{
   transcript_embedding
}

{
   frame_caption_embedding
}

{
   scene_summary_embedding
}
```

---

# Storage Design

---

## Vector DB (Qdrant)

```python
{
  id,
  modality,
  embedding,
  metadata
}
```

Example:

```python
{
  id:"img_22",
  modality:"image"
}
```

```python
{
  id:"tbl_31",
  modality:"table"
}
```

```python
{
  id:"vid_11",
  modality:"video"
}
```

---

## Object Storage

Store originals.

```text
s3://docs
s3://images
s3://videos
s3://tables
```

Never store large binaries in vector DB.

---

# Retrieval Phase

Suppose user asks:

```text
Show me how invoice approval workflow works.
```

The answer may exist in:

* document text
* architecture diagram image
* process video

all simultaneously.

---

# Retrieval Strategy

I would use hybrid multimodal retrieval.

```text
Query
 ↓
Embedding
 ↓
Vector Search
```

Results:

```text
Text Chunk
Table Summary
Image Caption
Video Transcript
```

Retrieved together.

---

# Reranking

Apply reranker.

Example:

```text
Cohere Rerank
BGE Reranker
Jina Reranker
```

Across all modalities.

---

# Context Assembly

Build final context:

```text
Relevant Text

Relevant Table

Relevant Image

Relevant Video Segment
```

For video:

```text
Timestamp:
12:30 - 13:20
```

included.

---

# Final Generation

Use multimodal LLM:

* GPT-4o
* Gemini 2.5
* Claude Opus Vision

Provide:

```text
Text
+
Images
+
Tables
+
Video References
```

to generate final answer.

---

# What Would Impress Me As An Interviewer

A senior engineer should conclude with:

> I would not store raw PDFs, tables, images, or videos directly in the vector database. I would store embeddings and metadata in Qdrant, keep original assets in object storage such as S3, and use metadata references to reconstruct the relevant multimodal context at retrieval time. This separation of retrieval layer and storage layer is the standard enterprise architecture because it improves scalability, cost efficiency, and retrieval quality.

That final statement signals that you understand real-world multimodal RAG architecture rather than just vector search.


---


These four concepts are extremely important because they address one of the biggest limitations of traditional RAG:


</details>

---

<details>
<summary></summary>

</details>