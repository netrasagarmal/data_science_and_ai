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
<summary>Large enterprise Rag Claude</summary>
# Enterprise RAG System Design — Interview Prep Notes
*For an 8-YOE AI Engineer / Solutions Architect interview*

> **How to use this doc:** For each topic, don't just list strategies — pick 2-3, explain the tradeoff, and say which you'd choose *and why* for a given scale/domain. Interviewers at this level are testing judgment, not recall.

---

## 1. Multimodal Data Ingestion and Storage

### The core design decision
There are three fundamentally different architectural approaches. State this framing first in an interview — it shows you understand there's no single "right" answer.

| Approach | How it works | When to use |
|---|---|---|
| **Grounding to text** | Convert everything to text (image captioning/OCR, video → transcript + scene descriptions, audio → ASR transcript), then run standard text RAG | Most common in enterprise. Simplest, cheapest, most mature tooling. Good when images/video are supplementary, not primary content. |
| **Modality-specific embeddings, separate collections** | Text embedded with a text model, images with CLIP/vision embeddings, stored in **separate vector collections**, joined via shared metadata (`doc_id`, `page_number`) | When each modality needs native similarity search (e.g., "find images similar to this one") |
| **Unified multimodal embedding space** | Single model (CLIP, ImageBind, Google multimodal embeddings) embeds all modalities into one shared vector space | When true cross-modal retrieval is needed (text query → retrieve relevant image directly) — less mature, harder to tune |

### Ingestion pipeline per modality
- **Text:** parse → semantic/recursive chunk → embed
- **Images:** generate caption + run OCR (for text-in-image) → treat image as one atomic chunk; store bounding box + source page if extracted from a document
- **Video:** shot/scene detection → chunk by scene, aligned with transcript segments (sliding window keyed to timestamps)
- **Audio:** speaker-diarization + silence detection → chunk by speaker turn, timestamp-aligned transcript

### Maintaining cross-modal relationships — this is the part interviewers actually probe
1. **Metadata joins (most common):** every chunk, regardless of modality, carries `parent_doc_id`, `page_number`/`timestamp`, `modality`. Retrieval-time logic re-assembles the full context by pulling all chunks sharing a parent ID.
2. **Multi-vector records:** vector DBs like Qdrant/Weaviate support *named vectors* per point — one record, multiple vector fields (text_vector, image_vector). Search either, hydrate the same payload.
3. **Knowledge graph overlay:** for complex enterprise docs (10-K filings, technical manuals), a graph DB (Neo4j) models explicit relationships ("Figure 3 is referenced on Page 12, which is Section 4.2"). Overkill for simple use cases, valuable when documents have deep structural cross-references.

### Storage architecture
- **Vector DB:** embeddings + metadata pointers only — never raw media
- **Object storage (S3/GCS):** raw images/video/audio, referenced by URI in metadata
- **Relational/NoSQL:** document hierarchy, lineage, versioning

**Interview signal:** mention that raw media never lives in the vector DB — separating storage of "what to search" from "what to render" is a scale/cost decision interviewers expect you to know.

---

## 2. Handling File Updates in RAG Systems

### Change detection strategies
| Strategy | Description | Tradeoff |
|---|---|---|
| **Content hashing** | SHA256 of content; compare against stored hash to detect real changes | Cheap, avoids re-embedding unchanged docs |
| **CDC (Change Data Capture)** | Source system emits events (S3 event notifications, Kafka connectors, SharePoint webhooks) | Real-time, but requires source system support |
| **Batch polling** | Periodic scan/diff against last-known state | Simple, higher latency, works with any source |

### Update handling patterns
1. **Delete + re-insert** — simplest and safest; use when chunk boundaries shift (any content edit can change where chunks split)
2. **Deterministic upsert** — chunk ID = `hash(doc_id + chunk_index)`; allows idempotent re-ingestion without duplicates. This is the pattern I'd default to.
3. **Soft delete / tombstoning** — flag old vectors inactive via metadata instead of hard delete; purge in a batch job later. Avoids expensive delete operations and read/write race conditions at scale.

### Synchronization architecture
```
Source change → Message queue (Kafka/SQS) → Ingestion worker
→ Re-chunk affected doc → Re-embed → Upsert (deterministic ID)
→ Update metadata store → Invalidate cache
```
- **Idempotency is non-negotiable** — ingestion jobs will retry/replay; deterministic IDs prevent duplicate chunks
- **Cascading deletes** — when a source doc is deleted, all its chunks must go too; keep a deletion log for GDPR "right to be forgotten" compliance

### Special case: embedding model upgrades
Reindexing millions of chunks with a new embedding model needs **blue-green reindexing**: build the new collection in parallel, validate quality, then cut over via an alias swap. Never reindex in place — zero downtime requirement.

### Special case: point-in-time correctness
For finance/legal, don't just delete old versions — keep `effective_date`/`expiry_date` metadata and filter at query time. Enables answering "what did the policy say as of March 2023?"

---

## 3. Scaling Vector Databases to Millions/Billions of Chunks

### Indexing algorithms
| Algorithm | Used by | Notes |
|---|---|---|
| **HNSW** | Weaviate, Qdrant, Milvus, Pinecone | Industry default; great recall/latency tradeoff, memory-hungry |
| **IVF-PQ** | Faiss, Milvus | Product quantization compresses vectors; more memory-efficient, some recall loss |
| **DiskANN** | Microsoft, Milvus | On-disk index for billion-scale where full in-memory HNSW isn't feasible |
| **ScaNN** | Google | Optimized for very high throughput |

### Core scaling levers
1. **Sharding**
   - By tenant (multi-tenant SaaS isolation — most common in enterprise)
   - By data domain (reduces search space if category known upfront)
   - By time range (hot/cold tiering — recent data in fast shard, archive in cheaper storage)
2. **Quantization** (memory + speed)
   - Scalar quantization (float32→int8): ~4x memory reduction, minimal accuracy loss
   - Product quantization: higher compression, more accuracy tradeoff
   - Binary quantization: extreme compression for first-pass candidate filtering
3. **Two-stage retrieval** — this is the key scale pattern: fast approximate search over a compressed index to get top ~500-1000 candidates, then exact rescoring (full precision or cross-encoder rerank) on the much smaller candidate set. Avoids paying full-precision cost across the entire corpus.
4. **Filtered vector search** — metadata pre-filtering *must* happen natively inside the ANN index (not as a post-filter), or you lose recall at scale. Qdrant/Weaviate support this well; it's a real differentiator when evaluating vector DBs for enterprise scale.
5. **Caching** — semantic query cache (cache by embedding similarity, not exact string match), embedding cache for frequently-encountered content, read replicas for horizontal query scaling.

### Maintaining accuracy at scale
- Tune HNSW params (`ef_construction`, `M`, `ef_search`) — direct recall vs. latency tradeoff, needs periodic benchmarking against a labeled ground-truth set
- As corpus grows, near-duplicate content increases → pure vector recall degrades → **hybrid search (BM25 + vector) becomes more important, not less**, at scale
- Background compaction/reindexing jobs to prevent fragmentation from frequent upserts

**Interview signal:** "at scale, filtering has to happen inside the ANN search, not after it" is a strong, specific answer that separates senior candidates from mid-level ones.

---

## 4. RAG Evaluation and Benchmarking

### Evaluation vs. Benchmarking — the distinction interviewers want stated explicitly
- **Evaluation** = continuous, often ad-hoc assessment of *your specific system* on *your data/production traffic*, used to guide iteration
- **Benchmarking** = standardized comparison against *public datasets/baselines*, used for reproducible, objective comparison (e.g., choosing between embedding models)

### Retrieval metrics (need relevance judgments)
- **Precision@k / Recall@k**
- **MRR** (Mean Reciprocal Rank) — position of first relevant result
- **NDCG** — accounts for graded relevance and rank position
- **Hit Rate@k** — simpler binary "was anything relevant in top-k"

### Generation metrics
- **Faithfulness/Groundedness** — is the answer actually supported by retrieved context (hallucination check)
- **Answer relevance** — does it address the question
- **Context precision** — are retrieved chunks relevant
- **Context recall** — did retrieval surface everything needed

### Frameworks
- **RAGAS** — LLM-as-judge based, computes faithfulness/answer relevancy/context precision & recall
- **TruLens, DeepEval, ARES** — similar LLM-judge frameworks, differ in extensibility/integration

### LLM-as-judge — mention the caveats
Scalable, but watch for **verbosity bias** (judges favor longer answers) and **position bias**. Always calibrate against a human-labeled sample before trusting it in CI.

### Standard public benchmarks
| Benchmark | Measures |
|---|---|
| **MTEB** | Embedding model quality across tasks — use for embedding model selection |
| **BEIR** | Heterogeneous retrieval across domains |
| **Natural Questions, HotpotQA, TriviaQA** | QA and multi-hop reasoning |
| **MS MARCO** | Passage ranking |
| **RGB (Retrieval-Augmented Generation Benchmark)** | RAG-specific robustness: noise handling, negative rejection, counterfactual resistance |

### What a mature enterprise setup actually does
- Curate a **golden Q&A set** from real production queries, SME-verified — this is your regression test suite, run before every pipeline deployment
- **Online evaluation**: shadow deployment / A/B testing comparing pipeline versions on real traffic, tracked via engagement + explicit feedback

---

## 5. Detecting and Handling RAG Quality Drift

### Types of drift
- **Data drift** — source content changes, new terminology emerges
- **Embedding drift** — inconsistent embedding model versions across index
- **Query drift** — user query patterns/topics shift over time
- **Concept drift** — ground truth itself changes (e.g., policy updates)

### Monitoring signals
- Retrieval confidence score distribution over time (declining average similarity = signal)
- Feedback trend (rising thumbs-down rate)
- Rising "I don't know" / fallback response rate
- Latency degradation as index grows
- Statistical drift detection on query embedding distributions (KL divergence, Population Stability Index vs. historical baseline)

### Detection mechanisms
1. **Shadow evaluation pipeline** — run the golden dataset nightly/weekly, track metric trends, alert on threshold breach
2. **Anomaly detection** on retrieval score distributions (control charts / z-score)
3. **Query clustering** — periodically cluster incoming queries to spot emerging topics not covered by the knowledge base (a content-gap signal, not just a quality signal)

### Root cause isolation — this is the senior-level move
Evaluate **retrieval quality and generation quality independently**. If context precision/recall is stable but faithfulness drops → LLM/prompt issue. If context recall drops → stale index or chunking problem. Don't just say "quality dropped," show you can isolate *where* in the pipeline.

### Remediation
- Scheduled + drift-triggered reindexing
- Continuous learning loop from feedback (→ see #6)
- Formal process for embedding model upgrade evaluation
- Revisit chunking strategy if content-gap clusters keep recurring in the same topic

---

## 6. Designing a Production-Grade Feedback System

### Feedback types to collect
| Type | Examples | Signal strength |
|---|---|---|
| **Explicit** | Thumbs up/down, star rating, "was this helpful" | Clear but low volume (most users don't click) |
| **Implicit** | Dwell time, query reformulation rate, citation click-through, session abandonment | High volume, noisier — but **reformulation rate is one of the strongest negative signals** available |
| **Corrections** | User edits the answer, requests regeneration | Very high value, low volume |
| **Free-text comments** | Detailed complaints | Rich but needs NLP to extract signal |

### Architecture
Every feedback event must attach to the **full trace**: query, retrieved chunks, generated answer, model version, retrieval parameters. Without this, feedback is undebuggable. Use tracing tools (LangSmith, Arize Phoenix, or custom telemetry) — this is a logging/observability problem as much as an ML problem.

### Analysis pipeline
1. Aggregate quantitative trends by topic/category (thumbs-down rate segmented by document source, query type)
2. **LLM-based auto-categorization** of negative feedback into failure modes: retrieval miss vs. hallucination vs. incomplete answer vs. tone/style issue
3. Join feedback with the retrieval trace to find correlations — e.g., failures clustering around a specific document source or chunk type

### Closing the loop
- Negative feedback → curated into the **hard examples eval set** used for regression testing
- Human-in-the-loop review queue for low-confidence or negatively-rated interactions
- Corrected answers → few-shot examples or fine-tuning data
- Missed-retrieval cases → feed into a content-gap analysis process (maybe the answer genuinely doesn't exist in the KB yet)
- Click/feedback data → **learning-to-rank** signal for fine-tuning the reranker

### Guardrail to mention
Don't overfit to a vocal minority — sample-based human audits of the feedback pipeline itself, and filter for bot/gamed feedback before it influences the system.

---

## 7. Designing a RAG System for Millions of Users

This is the "put it all together" question. Structure your answer across these dimensions — naming all of them, even briefly, is what signals breadth at this level.

| Dimension | Key strategies |
|---|---|
| **Architecture** | Microservices split: ingestion (async, queue-based), retrieval, generation/orchestration, feedback — each independently scalable. API gateway in front. |
| **Concurrency/Scalability** | Stateless query service, horizontal autoscaling (K8s HPA), connection pooling to vector DB, async I/O so LLM calls don't block threads |
| **Latency** | Semantic query cache, embedding cache, LLM response cache for repeats, token streaming, **model routing** (cheap/fast model for simple queries, escalate to large model only when needed) |
| **Availability** | Multi-region deployment, vector DB replication (active-active/passive), circuit breakers on LLM API calls, graceful degradation (fall back to retrieval-only mode if generation is down) |
| **Cost** | Tiered model routing, embedding cache reuse, prompt caching (Anthropic/OpenAI native prompt caching), context compression to cut token spend, batch ingestion, spot instances for non-critical jobs |
| **Security** | PII detection/redaction at ingestion, **per-tenant data isolation** (critical — cross-tenant retrieval leakage is a top real-world incident category), **permission-aware retrieval** (filter results by the querying user's document ACLs, not just relevance), prompt injection defenses, encryption at rest/in transit |
| **Observability** | Distributed tracing across query→retrieval→generation, real-time dashboards (latency %iles, error rate, retrieval quality, cost/query), SLO-based alerting, per-tenant usage metering |
| **Data freshness** | Explicit real-time vs. batch ingestion decision per data source, event-driven updates for critical sources, staleness SLAs |
| **Retrieval quality** | Hybrid search + reranking + feedback loop (this is where sections 1-6 all plug in) |
| **Model selection** | Task-based routing, fallback chain, explicit cost/latency/quality tradeoff curve documented for stakeholders |
| **Failure handling** | Retries with exponential backoff, dead-letter queues for failed ingestion jobs, timeouts with fallback responses, regular chaos testing |

**Interview signal:** the single highest-leverage thing to say explicitly is **permission-aware retrieval** — most candidates forget that relevance ranking and access control are two separate filters, and conflating them is a real production security bug pattern.

---

## 8. Domain-Specific RAG Systems

The general framework interviewers want: beyond scale/latency/cost, every domain adds constraints in **data sensitivity, regulatory compliance, accuracy tolerance, freshness requirements, and explainability**. Name the domain-specific driver first, then the architectural consequence.

| Domain | Key driver | Architectural consequence |
|---|---|---|
| **Finance** | Regulatory (SEC/FINRA/SOX), zero hallucination tolerance on figures | Full audit trail per answer; numeric queries often need **structured/SQL-RAG hybrid** instead of pure vector search; point-in-time versioned data; near-real-time freshness for market data |
| **Healthcare** | HIPAA, clinical safety | PHI de-identification pipeline; mandatory human-in-the-loop (RAG as *assist*, never autonomous decision-maker); chunking aligned to medical ontologies (UMLS, SNOMED CT, ICD); retrieval restricted to approved/peer-reviewed sources only |
| **Media & Entertainment** | Copyright/licensing, personalization | Retrieval must respect content licensing per user/region; heavy multimodal (video/audio transcripts + scene metadata); breaking-content freshness; content moderation guardrails |
| **Telecom** | Massive interaction volume, legacy system integration | Structured (OSS/BSS, network logs) + unstructured (manuals) fusion; multilingual support; low-latency customer-facing bots at huge scale |
| **Supply Chain** | Real-time operational state | Near-real-time sync from ERP/inventory systems; often needs **hybrid RAG + live API/SQL query** rather than static vector retrieval; multi-party data sensitivity (supplier data) |
| **Energy** | Safety-critical accuracy, regulatory variation by region | Accuracy paramount for equipment/safety manuals; real-time IoT/sensor data integration for operational queries; explainability required for safety audits; infrequent but must-be-exact document updates |

**Interview signal:** the strongest answer pattern here is: *"the technical RAG stack barely changes across domains — what changes is where you put the guardrails: accuracy tolerance, human-in-the-loop points, and what data is even allowed to be retrieved."* That's the architect-level framing, not a domain-by-domain feature list.

---

## Quick Recap Table — One-Liner Per Topic

| # | Topic | One-line answer to lead with |
|---|---|---|
| 1 | Multimodal ingestion | Ground to text by default; use metadata joins to link modalities; unified embeddings only when true cross-modal search is required |
| 2 | File updates | Deterministic chunk IDs for idempotent upserts; soft-delete/tombstone at scale; blue-green reindex for model upgrades |
| 3 | Scaling vector DBs | Two-stage retrieval (compressed ANN → precise rescore) + filtering *inside* the ANN index, not after |
| 4 | Evaluation vs benchmarking | Evaluation = your system on your data; benchmarking = public datasets for objective comparison; use both |
| 5 | Quality drift | Isolate retrieval drift from generation drift before diagnosing root cause |
| 6 | Feedback systems | Log full trace with every feedback event; reformulation rate is your strongest implicit negative signal |
| 7 | Millions of users | Permission-aware retrieval + model routing are the two most forgotten pieces |
| 8 | Domain-specific | Stack stays the same; guardrails (accuracy bar, human-in-loop, access control) change per domain |
</details>

<details>
<summary>Large enterprise Rag GPT</summary>
The interviewer is usually testing whether you can reason about the **entire lifecycle of an enterprise RAG platform**:

> **Ingest → Understand → Normalize → Chunk → Index → Retrieve → Rerank → Generate → Evaluate → Observe → Learn → Re-index**

At enterprise scale, the important shift is:

> **RAG is not a vector database + LLM. It is a distributed information system with an AI retrieval layer.**

For example, a production enterprise RAG platform may have object storage, document processing, metadata/catalog services, sparse and dense indexes, ACL/security filters, query orchestration, reranking, LLM inference, caching, observability, evaluation, feedback and continuous re-indexing.

Modern search platforms explicitly combine lexical retrieval such as BM25 with vector retrieval and RRF/semantic reranking, rather than treating vector search as a replacement for classical search. ([Microsoft Learn][1])

---

# 0. The Mental Model You Should Use in an 8-Year Interview

Before going into your eight questions, establish this architecture mentally:

```text
                         ┌──────────────────────────────┐
                         │        DATA SOURCES           │
                         │                              │
                         │ PDFs │ HTML │ DB │ Images    │
                         │ Video│ Audio│ Email│ APIs    │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │     INGESTION PLATFORM       │
                         │                              │
                         │ Connectors / CDC / Events    │
                         │ Batch + Streaming            │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ CONTENT PROCESSING            │
                         │                              │
                         │ OCR / ASR / Parsing          │
                         │ Image understanding          │
                         │ Metadata extraction           │
                         │ Deduplication                 │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ CHUNKING + ENRICHMENT         │
                         │                              │
                         │ semantic / hierarchical       │
                         │ parent-child / contextual     │
                         │ metadata / ACLs               │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                ┌─────────────────┐           ┌─────────────────┐
                │ Sparse Index    │           │ Vector Index    │
                │ BM25            │           │ Embeddings      │
                │ Inverted Index  │           │ ANN             │
                └────────┬────────┘           └────────┬────────┘
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                              ┌──────────────────┐
                              │ QUERY PROCESSING │
                              │                  │
                              │ Rewrite          │
                              │ Expansion        │
                              │ Intent           │
                              │ ACL filtering    │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ HYBRID RETRIEVAL │
                              │ BM25 + Dense     │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ RERANKING        │
                              │ Cross Encoder /  │
                              │ Semantic Ranker  │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ CONTEXT BUILDING │
                              │ Compression      │
                              │ Deduplication    │
                              │ Parent expansion │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ LLM GENERATION   │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ RESPONSE         │
                              │ + CITATIONS      │
                              └────────┬─────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             ┌──────────────┐                     ┌───────────────┐
             │ OBSERVABILITY│                     │ USER FEEDBACK │
             └──────┬───────┘                     └───────┬───────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                              ┌──────────────────┐
                              │ EVALUATION       │
                              │ + DRIFT          │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ CONTINUOUS       │
                              │ IMPROVEMENT      │
                              └──────────────────┘
```

The rest of the answer explains how you should reason about each layer.

---

# 1. Multimodal Data Ingestion and Storage

This is actually **two different problems**:

1. How do I understand different modalities?
2. How do I preserve their relationships?

The second one is often missed in interviews.

---

## 1.1 Don't put everything directly into the vector DB

A common junior architecture is:

```text
PDF → chunks → embeddings → Vector DB
```

For enterprise systems, think:

```text
                    Source
                      │
                      ▼
              Object Storage
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    Raw/Immutable             Metadata DB
       Files
          │
          ▼
   Processing Pipeline
          │
     ┌────┼────┬────┐
     ▼    ▼    ▼    ▼
   Text Image Video Audio
     │    │    │    │
     └────┴────┴────┘
              │
              ▼
       Search / Vector Index
```

### Recommended storage separation

| Data                     | Recommended storage           |
| ------------------------ | ----------------------------- |
| Original PDF/video/audio | Object storage                |
| Extracted text           | Document store / search index |
| Metadata                 | Relational/document DB        |
| Embeddings               | Vector index                  |
| Search index             | Search engine                 |
| Relationships            | Metadata DB / graph           |
| Processing status        | Workflow DB                   |
| Versions                 | Object storage + metadata     |

**Never make the vector database your source of truth.**

Vector indexes should be treated as **derived indexes**.

---

# 1.2 Text ingestion

Typical pipeline:

```text
PDF
 ↓
Parser
 ↓
Layout detection
 ↓
OCR if required
 ↓
Text extraction
 ↓
Structure detection
 ↓
Chunking
 ↓
Metadata enrichment
 ↓
Embedding
 ↓
Indexing
```

Metadata could include:

```json
{
  "document_id": "DOC123",
  "version": 7,
  "chunk_id": "DOC123_v7_p12_c4",
  "page": 12,
  "section": "Financial Results",
  "source": "annual_report.pdf",
  "created_at": "...",
  "security_group": ["finance"],
  "language": "en",
  "tenant_id": "tenantA"
}
```

This metadata becomes extremely important later for:

* filtering
* security trimming
* versioning
* deletion
* traceability
* citations
* debugging
* evaluation

---

# 1.3 Image ingestion

Images should generally have **multiple representations**.

For example:

```text
Image
 ├── Original image
 ├── OCR text
 ├── Caption
 ├── Object/layout information
 └── Image embedding
```

Suppose a PDF contains:

> Figure 7 — Network architecture

You may extract:

```text
Text:
"Figure 7 shows the 5G core architecture..."

Image:
[architecture diagram]

Image caption:
"5G architecture consisting of AMF, SMF, UPF..."

Embedding:
[0.21, -0.03, ...]
```

This lets you answer questions that depend on the **diagram**, not only the surrounding text.

Modern multimodal search pipelines explicitly extract page text and images, generate descriptions, embed modalities, and retain references to the original images. ([Microsoft Learn][2])

---

# 1.4 Video ingestion

Don't embed an entire video.

Instead:

```text
Video
 │
 ├── Audio → ASR → Transcript
 │
 ├── Frames → Frame sampling
 │              ↓
 │         Image understanding
 │
 └── Metadata
       ├── timestamp
       ├── speaker
       └── scene
```

Then create temporal chunks:

```text
video_123
 ├── 00:00–02:00
 ├── 02:00–04:30
 ├── 04:30–06:10
```

Each chunk can have:

```text
Transcript
+
Visual description
+
Timestamp
+
Speaker
+
Embedding
```

### Example

Query:

> "What did the CEO say about AI investment?"

Retrieval should return:

```text
Video: earnings_call.mp4
Timestamp: 32:14–34:08
Speaker: CEO

Transcript:
"We expect AI infrastructure investment..."
```

The response can cite:

> Earnings call, 32:14.

---

# 1.5 Audio ingestion

Typical:

```text
Audio
 ↓
Speaker diarization
 ↓
ASR
 ↓
Timestamped transcript
 ↓
Semantic chunking
 ↓
Embedding
```

Metadata:

```text
speaker
timestamp
language
confidence
conversation_id
```

This is extremely useful for:

* call-center RAG
* customer support
* meeting assistants
* legal recordings
* earnings calls

---

# 1.6 The most important multimodal concept: Content lineage

Suppose:

```text
PDF
 ├── Page 10
 │    ├── Paragraph 1
 │    ├── Table 1
 │    └── Figure 3
```

You should preserve:

```text
Document
   ↓
Page
   ↓
Element
   ↓
Chunk
   ↓
Embedding
```

Think of it as a **content graph**.

```text
Document D1
     │
     ├── Page 10
     │     ├── Text chunk C1
     │     ├── Table T1
     │     └── Image I1
     │
     └── Page 11
           └── Text chunk C2
```

Now if C1 is retrieved, you can retrieve:

```text
C1
+
parent page
+
associated image
+
associated table
```

This is far better than treating every chunk as an isolated vector.

---

# 1.7 Multimodal RAG strategies

You should know at least these:

### Strategy A — Unimodal indexes

Separate:

```text
Text → text vector index
Image → image vector index
Audio → transcript vector index
Video → transcript + visual index
```

Query each separately.

**Pros:** simple, scalable.

**Cons:** difficult cross-modal retrieval.

---

### Strategy B — Shared multimodal embedding space

```text
Text ─┐
Image ├──→ same embedding space
Audio ┘
```

Query:

```text
text → embedding
```

and retrieve:

```text
text + images + other modalities
```

Useful for:

* product search
* visual search
* multimodal RAG

---

### Strategy C — Multi-index retrieval + fusion

```text
                Query
                  │
       ┌──────────┼───────────┐
       ▼          ▼           ▼
     BM25      Text Vector   Image Vector
       │          │           │
       └──────────┼───────────┘
                  ▼
              RRF / Fusion
                  ↓
               Reranker
```

This is often the safest enterprise approach.

---

# 2. Handling File Updates

This is where enterprise RAG differs significantly from a demo.

The question isn't:

> "How do I update the vector?"

The question is:

> **How do I maintain consistency between the source-of-truth, metadata, chunks, embeddings, search indexes and cached results?**

---

# 2.1 Treat every document as a versioned entity

Instead of:

```text
document_id = 123
```

think:

```text
document_id = 123
version = 1

document_id = 123
version = 2

document_id = 123
version = 3
```

Each chunk:

```text
doc_id
version
chunk_id
embedding_version
parser_version
chunking_version
```

Example:

```text
DOC123
V7
CHUNK42
embedding=v3
chunker=v5
parser=v4
```

This gives you reproducibility.

---

# 2.2 Strategy 1 — Delete and re-index

Simple:

```text
Old document
 ↓
Delete all chunks
 ↓
Process new document
 ↓
Create chunks
 ↓
Embed
 ↓
Insert
```

Good when:

* documents are small
* updates are infrequent
* simplicity matters

Bad for:

* huge documents
* high-frequency updates

---

# 2.3 Strategy 2 — Incremental re-indexing

Detect changed portions.

```text
Old document
       +
New document
       ↓
Diff
       ↓
Changed sections
       ↓
Rechunk only affected sections
```

Suppose:

```text
1000-page document

Only page 734 changed
```

Don't re-embed 1000 pages.

Only process:

```text
page 734
+ neighboring chunks if required
```

This saves:

* embedding cost
* processing time
* indexing load

---

# 2.4 Strategy 3 — Immutable versioned indexes

Create:

```text
index_v6
index_v7
```

Build v7 completely.

Then:

```text
                 Application
                     │
                     ▼
                  alias
                     │
             ┌───────┴───────┐
             ▼               ▼
          index_v6        index_v7
```

Once v7 is validated:

```text
alias → index_v7
```

This is effectively **blue/green indexing**.

Very useful for enterprise deployments because you avoid serving a partially updated index.

---

# 2.5 Strategy 4 — Tombstones for deletion

Instead of physically deleting immediately:

```text
DOC123
status = deleted
```

Then asynchronously clean:

```text
Search index
Vector index
Cache
Derived stores
```

This is useful in distributed systems because deletion isn't instantaneous across every system.

---

# 2.6 Strategy 5 — Event-driven updates

Source:

```text
SharePoint / S3 / DB
        │
        ▼
   Change event
        │
        ▼
      Kafka
        │
        ▼
 Processing workers
```

Events:

```text
DOCUMENT_CREATED
DOCUMENT_UPDATED
DOCUMENT_DELETED
DOCUMENT_REPLACED
```

Then downstream services react.

This gives you:

* scalability
* retries
* replayability
* decoupling

---

# 2.7 Exactly-once vs idempotency

An interviewer may ask:

> "What happens if the update event is processed twice?"

Don't say "Kafka guarantees exactly once."

Instead design for **idempotency**.

Example:

```text
document_id = D1
version = 7
```

Processing the same event twice should produce the same final state.

Use:

```text
document_id + version
```

as an idempotency key.

---

# 2.8 Critical enterprise concept: Embedding versioning

Suppose you change:

```text
embedding model A
```

to:

```text
embedding model B
```

You cannot casually mix them.

Track:

```text
embedding_model = text-embedding-v3
embedding_dimension = 1536
embedding_version = 3
```

Then build a new index.

---

# 3. Millions/Billions of Chunks

At this point the problem becomes a distributed systems problem.

---

# 3.1 First principle

Don't search everything.

You want:

```text
1 billion chunks
       ↓
metadata filtering
       ↓
100 million
       ↓
partition routing
       ↓
10 million
       ↓
ANN
       ↓
1000
       ↓
reranking
       ↓
20
```

The goal is **progressive narrowing**.

---

# 3.2 Partitioning

Partition by:

### Tenant

```text
tenant_A
tenant_B
tenant_C
```

### Geography

```text
EU
US
India
```

### Domain

```text
finance
HR
legal
engineering
```

### Time

```text
2024
2025
2026
```

### Data type

```text
text
image
video
audio
```

Choose partitions based on **query locality**, not arbitrary database convenience.

---

# 3.3 Sharding

When one node cannot handle the data:

```text
                Vector DB
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    Shard 1      Shard 2      Shard 3
    100M         100M         100M
```

Query:

```text
query → shards → local ANN → merge top-k
```

This is distributed top-k retrieval.

---

# 3.4 ANN — Approximate Nearest Neighbor

Exact search:

```text
query vs every vector
```

Complexity becomes too expensive.

ANN trades a little recall for massive speed improvement.

Important algorithms:

### HNSW

Graph-based.

Good:

* high recall
* low latency

Tradeoff:

* memory intensive

---

### IVF

Cluster vectors:

```text
1B vectors

       ↓

100k clusters
```

Query only selected clusters.

---

### PQ

Product Quantization compresses vectors.

Useful when:

```text
memory cost > latency requirement
```

---

### Hybrid ANN

Large systems often combine:

```text
IVF + PQ
```

or optimized HNSW variants.

Current large-scale benchmarks demonstrate that billion- and even 10-billion-vector systems can trade among recall, latency, throughput and index size using different ANN/index configurations. ([AlibabaCloud][3])

---

# 3.5 Vector compression

Suppose:

```text
1B vectors
×
1536 dimensions
×
4 bytes
```

Raw vector storage alone becomes enormous.

Use:

* FP16
* INT8
* scalar quantization
* binary quantization
* product quantization

Modern search systems expose vector quantization specifically to reduce index memory/storage requirements. ([Microsoft Learn][4])

---

# 3.6 Retrieval pipeline at scale

A good architecture:

```text
Query
 │
 ▼
Intent / Query rewriting
 │
 ▼
Security filter
 │
 ▼
Metadata filter
 │
 ├──── BM25 ───────┐
 │                 │
 └──── Dense ──────┤
                   ▼
                  RRF
                   │
                   ▼
               Top 100
                   │
                   ▼
                Reranker
                   │
                   ▼
                Top 10
```

Hybrid retrieval is increasingly standard because lexical search handles exact terms such as product codes, names and domain terminology while vector search handles conceptual similarity. RRF is commonly used to merge the rankings. ([Microsoft Learn][1])

---

# 3.7 Accuracy at billion scale

You should distinguish:

### Recall

Did I retrieve the relevant chunk?

### Precision

Are retrieved chunks actually relevant?

### Latency

How quickly?

### Throughput

How many queries/sec?

### Cost

How expensive?

These are competing objectives.

For example:

```text
HNSW efSearch ↑
      ↓
Recall ↑
      ↓
Latency ↑
```

Similarly:

```text
reranker candidates ↑
      ↓
quality ↑
      ↓
latency/cost ↑
```

This is the kind of tradeoff an 8-year candidate should explicitly mention.

---

# 4. RAG Evaluation vs Benchmarking

This distinction is **very important**.

## Evaluation

> "Is my system good?"

## Benchmarking

> "How does my system compare against another system or baseline under a standardized test?"

---

# 4.1 Evaluate at multiple levels

Don't evaluate only the final answer.

Use:

```text
                    RAG Evaluation
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Retrieval      Generation      System
```

---

# 4.2 Retrieval metrics

### Recall@K

```text
relevant documents retrieved
--------------------------------
total relevant documents
```

Example:

There are 5 relevant chunks.

Retriever finds 4 in top 10.

```text
Recall@10 = 4/5 = 0.8
```

---

### Precision@K

Of the top K, how many are relevant?

---

### MRR

Mean Reciprocal Rank.

If first relevant result appears at:

```text
rank 1 → 1
rank 2 → 0.5
rank 5 → 0.2
```

Useful when the first correct result matters.

---

### NDCG

Important when relevance has levels:

```text
0 = irrelevant
1 = somewhat relevant
2 = relevant
3 = highly relevant
```

It also rewards putting highly relevant results near the top.

---

# 4.3 Generation metrics

### Faithfulness

Does the answer actually follow from the retrieved context?

---

### Answer relevance

Does the answer answer the user's question?

---

### Correctness

Is it factually correct relative to a ground-truth answer?

---

### Citation accuracy

If the answer says:

> "Revenue increased 12%."

Does the citation actually support the claim?

This becomes extremely important in enterprise RAG.

---

# 4.4 RAGAS

RAGAS provides metrics such as:

* context precision
* context recall
* faithfulness
* answer relevancy
* factual correctness
* multimodal metrics

Its original goal was reference-free evaluation across retrieval and generation dimensions. ([arXiv][5])

---

# 4.5 ARES

ARES evaluates:

```text
context relevance
answer faithfulness
answer relevance
```

It uses automated LM judges combined with human-annotated samples. ([ACL Anthology][6])

---

# 4.6 Benchmark datasets

Important ones to know:

### BEIR

Primarily useful for evaluating **information retrieval** across diverse retrieval tasks.

Typical metrics:

```text
NDCG@10
Recall@K
MAP
MRR
```

---

### MTEB

Useful for evaluating embedding models across multiple tasks such as:

* retrieval
* semantic similarity
* classification
* clustering

---

### KILT

Useful for knowledge-intensive NLP tasks involving retrieval and downstream task performance.

ARES, for example, evaluates on knowledge-intensive tasks involving KILT, SuperGLUE and other datasets. ([ACL Anthology][6])

---

### Domain-specific benchmark

This is actually more important for enterprise.

Suppose you're building:

```text
Banking RAG
```

A public benchmark may not tell you whether:

> "What is the penalty for early withdrawal of this specific deposit?"

works correctly.

Build your own:

```text
500–5000 representative queries
+
gold documents
+
gold answers
+
expected citations
+
difficulty labels
```

---

# 4.7 Online evaluation

Offline:

```text
dataset → RAG → metrics
```

Online:

```text
real users
 ↓
queries
 ↓
answers
 ↓
feedback
 ↓
quality metrics
```

You need both.

---

# 5. Detecting RAG Quality Drift

This is a **production MLOps / LLMOps problem**.

Drift doesn't necessarily mean the model changed.

Your data may have changed.

---

# 5.1 Types of drift

### Data drift

Documents changed.

Example:

```text
old policies → new policies
```

---

### Query drift

User behavior changes.

Example:

```text
"What is X?"
```

becomes:

```text
"Compare X vs Y"
```

---

### Retrieval drift

Retriever starts returning poorer documents.

---

### Embedding drift

Embedding model changes.

---

### Generation drift

LLM provider/model changes.

---

### Distribution drift

The traffic mix changes.

---

# 5.2 Monitor the entire pipeline

```text
User
 │
 ▼
Query
 │
 ├── query length
 ├── language
 ├── intent
 └── complexity
 │
 ▼
Retriever
 │
 ├── recall
 ├── precision
 ├── top-k
 └── score distribution
 │
 ▼
Reranker
 │
 └── score distribution
 │
 ▼
LLM
 │
 ├── latency
 ├── token usage
 ├── refusal
 └── answer quality
 │
 ▼
User
 │
 └── feedback
```

---

# 5.3 Retrieval drift indicators

Watch:

```text
Recall@K ↓
Precision@K ↓
NDCG ↓
```

But in production you often don't have labels.

So use proxies:

### Retrieval score distribution

Suppose average similarity:

```text
0.82 → 0.61
```

Potential problem.

---

### Empty retrieval rate

```text
0.5% → 7%
```

Strong signal.

---

### Low-confidence retrieval

Percentage of queries where:

```text
top_score < threshold
```

---

### No-answer rate

Sudden increase may indicate:

* missing documents
* index failure
* embedding mismatch
* permissions issue

---

# 5.4 Generation drift indicators

Monitor:

* thumbs-down rate
* user corrections
* hallucination rate
* citation failures
* answer length
* refusal rate
* escalation rate

---

# 5.5 Root-cause analysis

Suppose:

```text
Answer quality ↓
```

Don't immediately change the LLM.

Trace:

```text
Answer ↓
    │
    ├── Retrieval good?
    │      │
    │      ├── NO → retriever/index issue
    │      │
    │      └── YES
    │
    └── Generation good?
           │
           ├── NO → prompt/model/context issue
           └── YES → perhaps evaluation/user behavior issue
```

This is one of the most important architectural concepts.

---

# 6. Production Feedback System

Don't only collect:

```text
👍 / 👎
```

That's too little information.

---

# 6.1 Feedback hierarchy

### Level 1

```text
👍
👎
```

Cheap.

---

### Level 2

Reason:

```text
Wrong answer
Poor retrieval
Outdated information
Missing information
Hallucination
Bad citation
Too verbose
```

---

### Level 3

Free-text:

> "The answer is wrong because the 2026 policy replaced the 2025 policy."

This is extremely valuable.

---

### Level 4

Correction:

```text
Expected answer:
...
```

Best training/evaluation signal.

---

# 6.2 Store complete interaction traces

Don't store only the answer.

Store:

```text
request_id
user_id / anonymized ID
tenant
query
query rewrite
retrieved chunks
retrieval scores
reranker scores
prompt version
model version
answer
citations
latency
token usage
feedback
timestamp
```

Then you can replay failures.

---

# 6.3 Automatically classify feedback

You can run an asynchronous LLM classifier:

```text
Feedback
   ↓
LLM classifier
   ↓
┌─────────────────────────────┐
│ failure_type                │
│ retrieval_failure           │
│ generation_failure          │
│ stale_information           │
│ missing_information         │
│ citation_failure            │
└─────────────────────────────┘
```

Then aggregate:

```text
40% retrieval
25% stale data
20% generation
10% citation
5% other
```

Now you know where to invest engineering effort.

---

# 6.4 Feedback → improvement loop

```text
User Feedback
      ↓
Classification
      ↓
Failure Dataset
      ↓
Root Cause
      ↓
Experiment
      ↓
Offline Evaluation
      ↓
A/B Test
      ↓
Production
```

Don't automatically fine-tune an LLM every time users give negative feedback.

Often the real fix is:

```text
better chunking
better metadata
better retrieval
better reranking
better indexing
```

---

# 7. RAG for Millions of Users

Now we're moving into **distributed architecture**.

---

# 7.1 Separate control plane and data plane

This is a very strong enterprise design answer.

### Control plane

Handles:

* configuration
* tenant management
* models
* prompts
* index versions
* policies
* evaluation
* deployment

### Data plane

Handles:

* user requests
* retrieval
* inference
* response

---

# 7.2 Stateless application layer

```text
                   Load Balancer
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       API Pod       API Pod       API Pod
```

No user session stored locally.

Use external:

* Redis
* DB
* object store

This allows horizontal scaling.

---

# 7.3 Asynchronous ingestion

Never let:

```text
user request
```

wait for:

```text
document parsing
embedding
indexing
```

Use:

```text
Upload
 ↓
Event
 ↓
Queue
 ↓
Workers
 ↓
Index
```

---

# 7.4 Caching

Multiple levels:

### Query cache

```text
query → response
```

---

### Embedding cache

```text
query → embedding
```

---

### Retrieval cache

```text
query → retrieved chunk IDs
```

---

### Reranking cache

```text
query + candidate IDs → ranking
```

---

### Semantic cache

Similar queries can share answers.

But be careful with:

* permissions
* user-specific context
* freshness

---

# 7.5 Latency budget

Suppose target:

```text
P95 < 2 sec
```

Create a budget:

```text
Gateway             50 ms
Query processing   100 ms
Embedding           80 ms
Retrieval          150 ms
Reranking          300 ms
LLM               1000 ms
Network             100 ms
--------------------------
Total              1780 ms
```

Now you know what to optimize.

This is much better than simply saying:

> "We need low latency."

---

# 7.6 Availability

Enterprise target might be:

```text
99.9%
99.99%
```

depending on business criticality.

Use:

* multi-AZ
* replicas
* health checks
* circuit breakers
* retries
* timeouts
* graceful degradation

---

# 7.7 Graceful degradation

Suppose vector DB fails.

Don't return:

```text
500 Internal Server Error
```

Potential fallback:

```text
Dense retrieval
     ↓ failure
BM25
     ↓ failure
Cached results
     ↓
"No reliable information available"
```

Never hallucinate simply because retrieval failed.

---

# 7.8 Security

This deserves an entire design layer.

Enterprise RAG needs:

```text
Authentication
       ↓
Authorization
       ↓
Security trimming
       ↓
Retrieval
```

Suppose:

```text
Employee A → Finance documents
Employee B → HR documents
```

A vector similarity search might find an HR document for Employee A.

The system must prevent it from entering the context.

So:

```text
Query
 ↓
Identity
 ↓
ACL filter
 ↓
Retriever
```

or carefully validated equivalent enforcement.

Security filters must be consistently applied across hybrid/vector retrieval paths; modern search documentation explicitly highlights this issue for security trimming. ([Microsoft Learn][7])

---

# 7.9 Tenant isolation

For SaaS:

```text
Tenant A
 ├── data
 ├── index
 └── policies

Tenant B
 ├── data
 ├── index
 └── policies
```

Strategies:

### Shared index + tenant filter

Cheaper.

### Partition per tenant

Better isolation.

### Separate index per tenant

Strong isolation but expensive.

### Separate infrastructure

For highly sensitive enterprise tenants.

Choose based on:

```text
security
scale
cost
noisy-neighbor risk
compliance
```

---

# 8. Domain-Specific RAG

This is where you demonstrate **architect-level maturity**.

Don't simply say:

> "Healthcare requires more security."

Explain how domain characteristics affect architecture.

---

# 8.1 Finance RAG

### Requirements

* accuracy
* auditability
* regulatory compliance
* data lineage
* temporal correctness
* explainability

Important concept:

> **Effective date**

A financial policy may have:

```text
valid_from
valid_to
```

Query:

> "What was the policy in March 2025?"

The system must retrieve the version valid **at that point in time**, not merely the latest document.

### Architecture additions

```text
Document
 ↓
Version
 ↓
Effective date
 ↓
Jurisdiction
 ↓
Product
 ↓
Regulation
```

Need:

* immutable audit logs
* citations
* provenance
* ACL
* PII protection
* human review for high-risk workflows

NIST's GenAI profile emphasizes provenance, privacy, security, evaluation and documented performance criteria as part of managing generative-AI risks. ([NIST Publications][8])

---

# 8.2 Healthcare RAG

Characteristics:

```text
high accuracy
+
privacy
+
clinical terminology
+
patient context
+
time sensitivity
```

Need:

* strong access control
* PHI protection
* audit logs
* medical ontology
* terminology normalization
* source provenance

Example:

```text
"MI"
```

could mean:

```text
Myocardial Infarction
Michigan
```

Domain-aware query understanding matters.

Also distinguish:

```text
clinical decision support
```

from:

```text
general medical information
```

The risk profile is different.

---

# 8.3 Telecom RAG

This is particularly interesting for enterprise architecture.

Data could include:

```text
network topology
alarms
tickets
customer data
billing
configuration
5G documentation
runbooks
logs
```

Now you have:

```text
structured + unstructured + real-time data
```

A pure vector RAG isn't sufficient.

Architecture:

```text
                    Query
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
    Vector         SQL/API       Knowledge Graph
    Search
       │              │              │
       └──────────────┼──────────────┘
                      ▼
                   Fusion
                      ↓
                    LLM
```

For:

> "Why is customer X experiencing poor 5G performance?"

You may need:

```text
customer profile
+
network KPIs
+
cell information
+
recent alarms
+
troubleshooting documents
```

This is **RAG + tools + structured retrieval**, not traditional document RAG.

---

# 8.4 Energy

Think:

```text
IoT telemetry
SCADA
maintenance records
engineering manuals
weather
historical incidents
```

You need:

* time-series retrieval
* real-time data
* equipment hierarchy
* spatial/geographical relationships
* engineering units
* safety

Example:

> "Why did turbine T-43 shut down?"

Requires:

```text
telemetry
+
alarm history
+
maintenance records
+
manual
```

Again:

> **RAG alone is insufficient.**

You need:

```text
RAG + APIs + time-series DB + knowledge graph
```

---

# 8.5 Supply Chain

Important dimensions:

* SKU
* supplier
* warehouse
* shipment
* geography
* dates
* inventory
* purchase orders

Queries are often:

```text
"What caused shipment X to be delayed?"
```

You need:

```text
ERP
+
WMS
+
TMS
+
documents
+
emails
```

This becomes **agentic retrieval**.

---

# 8.6 Media & Entertainment

Data:

```text
scripts
videos
subtitles
interviews
images
metadata
contracts
social media
```

Multimodal retrieval becomes extremely important.

Query:

> "Find all scenes where the protagonist is wearing a red jacket."

This is not conventional text RAG.

You need:

```text
video
 ↓
frame extraction
 ↓
visual embeddings
 ↓
scene segmentation
 ↓
multimodal retrieval
```

---

# 9. The Architecture Strategies You Should Know

For an 8-year interview, I would organize RAG strategies into these layers.

| Layer            | Strategies                                   |
| ---------------- | -------------------------------------------- |
| Ingestion        | batch, streaming, CDC, event-driven          |
| Storage          | object store, document DB, relational DB     |
| Parsing          | OCR, layout parsing, ASR, vision             |
| Chunking         | fixed, semantic, hierarchical, parent-child  |
| Metadata         | ACL, tenant, timestamp, source, version      |
| Embedding        | dense, multimodal, domain-specific           |
| Sparse retrieval | BM25, inverted index                         |
| Dense retrieval  | HNSW, IVF, ANN                               |
| Hybrid           | BM25 + vector                                |
| Fusion           | RRF, weighted fusion                         |
| Query            | rewrite, expansion, HyDE, decomposition      |
| Reranking        | cross-encoder, LLM reranker                  |
| Context          | compression, deduplication, parent expansion |
| Generation       | single-pass, multi-step, agentic             |
| Caching          | embedding, retrieval, semantic, response     |
| Scaling          | partitioning, sharding, replicas             |
| Security         | ACL, RBAC, ABAC, tenant isolation            |
| Freshness        | CDC, events, incremental indexing            |
| Evaluation       | offline + online                             |
| Monitoring       | quality + system metrics                     |
| Feedback         | implicit + explicit                          |
| Improvement      | experimentation + re-indexing                |
| Governance       | provenance, audit, compliance                |

---

# 10. The Most Important Enterprise RAG Trade-offs

For interviews, don't present technologies as universally "best."

Instead say:

> "It depends on the workload."

For example:

### BM25 vs Dense

```text
Exact terms/product codes → BM25
Semantic questions → Dense
Enterprise production → Hybrid
```

Modern hybrid search systems explicitly combine BM25/full-text and vector retrieval because their strengths are complementary. ([Microsoft Learn][1])

---

### HNSW vs IVF/PQ

```text
High recall + memory available
        → HNSW

Huge scale + memory pressure
        → IVF/PQ / quantization
```

---

### Small chunks vs large chunks

```text
Small chunks
→ retrieval precision ↑
→ context completeness ↓

Large chunks
→ context completeness ↑
→ retrieval precision ↓
```

Hence:

```text
small retrieval chunks
+
parent context expansion
```

can often be better.

---

### More retrieved documents

```text
K ↑
→ recall ↑
→ noise ↑
→ reranking cost ↑
→ context cost ↑
```

Therefore:

```text
retrieve 50
→ rerank 50
→ keep 8
→ context compression
→ LLM
```

is often better than directly sending 50 chunks.

---

# 11. A Strong Enterprise RAG Reference Architecture

If the interviewer says:

> **"Design a RAG system for millions of users."**

You can structure your answer like this:

```text
                         USERS
                           │
                           ▼
                  API Gateway / WAF
                           │
                           ▼
                 Authentication / IAM
                           │
                           ▼
                 RAG Orchestrator
                           │
            ┌──────────────┼───────────────┐
            │              │               │
            ▼              ▼               ▼
       Query Cache     Query Rewrite    Intent
            │              │               │
            └──────────────┼───────────────┘
                           │
                           ▼
                    Security Filter
                           │
                           ▼
                  ┌─────────────────┐
                  │ Hybrid Retrieval│
                  │                 │
                  │ BM25            │
                  │ Dense ANN       │
                  └────────┬────────┘
                           │
                           ▼
                      RRF / Fusion
                           │
                           ▼
                       Reranker
                           │
                           ▼
                  Context Optimizer
                  ┌────────┼────────┐
                  │        │        │
               Dedup    Compress  Parent
                                    │
                                    ▼
                              LLM Gateway
                           ┌──────┼──────┐
                           ▼      ▼      ▼
                         LLM-1  LLM-2  LLM-3
                           │
                           ▼
                    Guardrails / Citation
                           │
                           ▼
                       RESPONSE
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Feedback     Observability   Cache
              │            │
              └──────┬─────┘
                     ▼
                 Evaluation
                     │
                     ▼
             Continuous Improvement


DATA PLANE
──────────────────────────────────────────────

Sources
  │
  ▼
Connectors / CDC / Events
  │
  ▼
Kafka / Queue
  │
  ▼
Processing Workers
  │
  ├── Parser
  ├── OCR
  ├── ASR
  ├── Vision
  ├── Chunker
  ├── Metadata
  └── Embedding
  │
  ▼
Object Store ─── Metadata DB
  │
  ├────────────── Search Index
  │
  └────────────── Vector Index
```

---

# 12. What Makes This "8-Year Experience" Thinking?

A junior candidate says:

> "I'll use a vector database and GPT."

A 3–4 year candidate says:

> "I'll use hybrid search and reranking."

An experienced architect says:

> "First I'll establish the workload characteristics, tenant model, data sensitivity, freshness SLA, retrieval quality targets, latency SLO, availability target and cost envelope. Then I'll choose the retrieval architecture."

That's the mindset you want.

---

## The questions you should ask before choosing the architecture

### Data

* What sources?
* What modalities?
* How much data?
* How fast does it change?
* How many documents/chunks?

### Retrieval

* Keyword-heavy or semantic?
* Exact identifiers?
* Multi-hop questions?
* Cross-document reasoning?
* Temporal queries?

### Scale

* QPS?
* Peak concurrency?
* P95/P99 latency?
* Number of tenants?
* Number of vectors?

### Quality

* What does "correct" mean?
* What is acceptable recall?
* What is acceptable hallucination rate?
* Is citation mandatory?

### Security

* Who can access what?
* Tenant isolation?
* PII/PHI?
* Regulatory requirements?

### Freshness

```text
real time?
seconds?
minutes?
hours?
daily?
```

### Availability

```text
99%
99.9%
99.99%
```

### Cost

```text
cost/request
cost/user/month
embedding cost
LLM cost
storage cost
```

This is how you turn **"build a RAG"** into an actual **enterprise system-design problem**.

---

# 13. Your Overall Study Dependency

Given your goal of preparing for an **8-year AI Engineer interview**, I would study these topics in this exact order:

```text
                    INFORMATION RETRIEVAL
                           │
                           ▼
                 Classical Retrieval
                 ├── Boolean
                 ├── TF-IDF
                 ├── VSM
                 └── BM25
                           │
                           ▼
                    Dense Retrieval
                 ├── Embeddings
                 ├── Similarity
                 └── ANN
                           │
                           ▼
                   Vector Databases
                 ├── HNSW
                 ├── IVF
                 ├── PQ
                 ├── Sharding
                 └── Filtering
                           │
                           ▼
                    Hybrid Retrieval
                 ├── BM25
                 ├── Dense
                 └── RRF
                           │
                           ▼
                     Reranking
                 ├── Cross Encoder
                 └── LLM Reranker
                           │
                           ▼
                   Advanced Retrieval
                 ├── Query Rewrite
                 ├── Query Expansion
                 ├── HyDE
                 ├── Query Decomposition
                 └── Multi-hop Retrieval
                           │
                           ▼
                     RAG Architecture
                 ├── Chunking
                 ├── Metadata
                 ├── Context
                 └── Generation
                           │
                           ▼
                  Enterprise RAG
                 ├── Multimodal
                 ├── Versioning
                 ├── Security
                 ├── Multi-tenancy
                 ├── Freshness
                 └── Scaling
                           │
                           ▼
                 Production RAG
                 ├── Caching
                 ├── HA
                 ├── Observability
                 ├── Cost
                 └── Failure handling
                           │
                           ▼
                  RAG Evaluation
                 ├── Retrieval metrics
                 ├── Generation metrics
                 ├── RAGAS
                 ├── ARES
                 └── Domain evaluation
                           │
                           ▼
                  Feedback + Drift
                 ├── User feedback
                 ├── Quality monitoring
                 ├── Drift detection
                 └── Continuous improvement
                           │
                           ▼
                    Agentic RAG
                 ├── Tools
                 ├── APIs
                 ├── SQL
                 ├── Knowledge graphs
                 └── Multi-step reasoning
```

The crucial progression is:

> **IR → Retrieval → Vector Search → Hybrid Search → Reranking → RAG → Enterprise RAG → Production RAG → Evaluation → Agentic RAG**

And one final architectural principle I'd strongly recommend using in interviews:

> **Don't make RAG the universal data-access mechanism.**

For enterprise systems, use the right retrieval mechanism for the right information:

```text
Unstructured documents → RAG
Structured data          → SQL / APIs
Real-time metrics        → Time-series DB / APIs
Relationships            → Knowledge Graph
Images/video             → Multimodal retrieval
Transactional actions   → Tools/APIs
```

Then use an **orchestrator/agent** to combine them.

That distinction is often what separates a **"RAG developer" answer** from an **"enterprise AI architect" answer**.

NIST's current AI RMF guidance also frames AI risk management as a continuous lifecycle—govern, map, measure and manage—rather than a one-time model check, which is a useful mental model for production RAG governance as well. ([NIST AI Resource Center][9])

[1]: https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview?utm_source=chatgpt.com "Hybrid Search Overview - Azure AI Search | Microsoft Learn"
[2]: https://learn.microsoft.com/en-us/azure/search/multimodal-search-overview?utm_source=chatgpt.com "Multimodal Search Concepts and Guidance - Azure AI Search | Microsoft Learn"
[3]: https://www.alibabacloud.com/help/en/polardb/polardb-for-mysql/user-guide/performance-testing-based-on-a-dino-dataset-with-tens-of-billions-vectors?utm_source=chatgpt.com "PolarSearch IVF-PQ vs HNSW-DiskMode 10B vector benchmark - PolarDB - Alibaba Cloud Documentation Center"
[4]: https://learn.microsoft.com/en-us/azure/search/search-features-list?view=azuresql-db&utm_source=chatgpt.com "Features and Capabilities - Azure AI Search | Microsoft Learn"
[5]: https://arxiv.org/abs/2309.15217?utm_source=chatgpt.com "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
[6]: https://aclanthology.org/2024.naacl-long.20/?utm_source=chatgpt.com "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems - ACL Anthology"
[7]: https://learn.microsoft.com/en-us/azure/search/hybrid-search-how-to-query?utm_source=chatgpt.com "Create a Hybrid Query - Azure AI Search | Microsoft Learn"
[8]: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf?utm_source=chatgpt.com "Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile"
[9]: https://airc.nist.gov/airmf-resources/airmf/5-sec-core/?utm_source=chatgpt.com "AI RMF Core - AIRC"

</details>

<details>
<summary></summary>

</details>