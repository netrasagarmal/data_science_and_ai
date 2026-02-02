## Retrieval-Augmented Generation

Lets first understand what is Information Retrieval (IR)

<details>
<summary>Information Retrieval (IR)</summary>

Perfect topic choice 👍
Information Retrieval (IR) is **the backbone of search engines, recommendation systems, and modern RAG pipelines**. I’ll give you **clean, exam-ready + industry-ready study notes**, moving **from fundamentals → classical IR → semantic & neural IR → RAG-specific retrieval**, with **clear dependencies**, **examples**, and **real-world applications**.

I’ll start **from scratch** so nothing feels hand-wavy.

---

# 📘 INFORMATION RETRIEVAL (IR) — DETAILED STUDY NOTES

---

## 1️⃣ What is Information Retrieval (IR)?

### Definition

**Information Retrieval (IR)** is the science of **finding relevant information** from a large collection of **unstructured or semi-structured data** (text, documents, web pages, PDFs, emails, logs, etc.) in response to a **user query**.

> IR ≠ Database querying
> Databases return **exact matches**, IR returns **relevant matches**

---

### Core Problem IR Solves

Given:

* A **query** (user intent)
* A **document collection**
* A **relevance notion**

➡️ **Rank documents by relevance**, not just retrieve them.

---

### Example

**Query:**

> “How to fine-tune LLM for document QA?”

**IR system returns ranked documents like:**

1. LLM fine-tuning for QA (high relevance)
2. RAG vs Fine-tuning comparison
3. Transformers basics
4. NLP introduction (low relevance)

---

### Real-World Systems Using IR

| System                 | Role of IR                         |
| ---------------------- | ---------------------------------- |
| Google Search          | Rank web pages                     |
| ChatGPT + RAG          | Retrieve context before generation |
| Amazon Search          | Product relevance                  |
| LinkedIn Search        | People, jobs, posts                |
| Enterprise Search      | PDFs, contracts, invoices          |
| Legal / Medical Search | Case law, research papers          |

---

## 2️⃣ Key Components of an IR System

### High-Level Architecture

```
Documents → Preprocessing → Indexing → Retrieval → Ranking → Evaluation
```

---

### 1. Document Collection

* Web pages
* PDFs
* Logs
* Emails
* Knowledge base articles

---

### 2. Text Preprocessing (VERY IMPORTANT)

Dependency: **All IR models rely on clean text**

Includes:

* Tokenization
* Lowercasing
* Stopword removal
* Stemming / Lemmatization
* Normalization

**Example**

```
"Retrieving Documents Efficiently!"
→ ["retrieve", "document", "efficient"]
```

---

### 3. Indexing

Creates a structure to retrieve documents **fast**.

#### Inverted Index (Core IR Structure)

```
term → list of (doc_id, frequency)
```

Example:

```
"rag" → [(doc1, 3), (doc7, 1)]
```

Without indexing → search = O(N)
With indexing → search = O(1) or log scale

---

### 4. Query Processing

* Apply same preprocessing as documents
* Convert query into IR model representation

---

### 5. Ranking Function

Scores each document:

```
score(query, document)
```

Examples:

* TF-IDF
* BM25
* Cosine similarity
* Neural similarity

---

### 6. Evaluation

Check **how good** retrieval is using metrics like:

* Precision
* Recall
* NDCG
* MAP

---

## 3️⃣ Information Retrieval Models (Big Picture)

### IR Models evolve in **3 generations**

```
1️⃣ Classical IR (Lexical)
2️⃣ Semantic IR (Latent)
3️⃣ Neural IR (Dense / Hybrid)
```

| Generation | Key Idea              |
| ---------- | --------------------- |
| Classical  | Keyword matching      |
| Semantic   | Capture meaning       |
| Neural     | Learn representations |

---

## 4️⃣ Boolean Retrieval Model (FOUNDATION)

### Concept

Documents are retrieved using **binary logic**:

* AND
* OR
* NOT

---

### Example

Query:

```
LLM AND RAG NOT fine-tuning
```

Returns documents containing:
✔ LLM
✔ RAG
❌ fine-tuning

---

### Pros

* Simple
* Fast
* Exact filtering

---

### Cons

* No ranking
* No notion of relevance
* Too rigid

---

### Real-World Usage

* Filters in search UI
* Legal / compliance search
* Metadata filtering in RAG pipelines

---

## 5️⃣ Vector Space Model (VSM)

### Core Idea

Represent:

* Documents
* Queries
  as **vectors in a high-dimensional space**

Each dimension = a term

---

### Example

Vocabulary:

```
["rag", "llm", "retrieval", "training"]
```

Document:

```
D1 = [1, 1, 1, 0]
Query:
Q = [1, 1, 0, 0]
```

Similarity = **Cosine Similarity**

---

### Cosine Similarity Formula

```
cos(Q, D) = (Q · D) / (||Q|| ||D||)
```

---

### Why Cosine?

* Focuses on **direction**
* Ignores document length

---

### Real-World Application

* Early document ranking
* Search engines pre-neural era
* Still used in sparse retrieval

---

## 6️⃣ TF-IDF (Term Frequency – Inverse Document Frequency)

### Dependency

👉 Built on top of VSM

---

### Term Frequency (TF)

How often a term appears in a document

```
TF(t, d) = count(t in d)
```

---

### Inverse Document Frequency (IDF)

Penalizes common words

```
IDF(t) = log(N / df(t))
```

Where:

* N = total documents
* df(t) = documents containing term t

---

### TF-IDF Formula

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

---

### Example

If:

* “rag” appears many times in doc
* Appears in few documents overall

➡️ High TF-IDF → important term

---

### Pros

* Simple
* Interpretable
* Fast

---

### Cons

* No semantics
* “car” ≠ “automobile”
* Keyword-based only

---

### Real-World Usage

* Resume screening
* Early search engines
* Baseline RAG retrievers

---

## 7️⃣ Probabilistic Retrieval Models

### Core Idea

Rank documents by:

```
P(relevant | document, query)
```

---

### Binary Independence Model (BIM)

Assumptions:

* Terms are independent
* Document relevance is binary

---

### Why Important?

👉 **Theoretical foundation of BM25**

---

## 8️⃣ Okapi BM25 (Industry Standard Sparse Retriever)

### Dependency

TF-IDF + Probabilistic Model

---

### Key Improvements over TF-IDF

1. **Term frequency saturation**
2. **Document length normalization**

---

### BM25 Scoring Formula (Intuition)

* TF grows → score grows but saturates
* Long documents penalized
* Rare terms rewarded

---

### Why BM25 is Powerful

* Robust
* No training required
* Works extremely well for text search

---

### Example

Query:

```
"RAG architecture"
```

BM25 scores:

* Short, focused RAG doc → high score
* Long generic NLP doc → lower score

---

### Real-World Usage

* Elasticsearch
* OpenSearch
* Lucene
* RAG sparse retrievers

---

## 9️⃣ Latent Semantic Analysis (LSA / LSI)

### Problem It Solves

TF-IDF fails with:

* Synonyms
* Polysemy

---

### Core Idea

Reduce high-dimensional term space → **latent semantic space**

Uses:

```
SVD (Singular Value Decomposition)
```

---

### Example

“car”, “vehicle”, “automobile”
➡️ mapped closer in latent space

---

### Pros

* Captures hidden semantics
* Handles synonymy

---

### Cons

* Expensive
* Static
* Doesn’t scale well

---

### Real-World Use

* Academic search
* Early semantic IR systems

---

## 🔟 Graph-Based Retrieval: PageRank

### Core Idea

Importance of a page = links pointing to it

```
More high-quality incoming links → higher rank
```

---

### Real-World Use

* Google Search
* Web authority ranking

---

### RAG Context

* Used in **hybrid retrieval**
* Knowledge graph-based RAG

---

## 1️⃣1️⃣ Neural Retrieval Models (Modern IR)

### Motivation

Keywords ≠ meaning

---

### Dense Vector Representations

Documents → embeddings
Queries → embeddings

Similarity:

```
cosine(query_embedding, doc_embedding)
```

---

### Models Used

* Word2Vec
* GloVe
* FastText
* BERT
* Sentence-BERT
* E5, BGE, OpenAI embeddings

---

### Example

Query:

> “How to chunk documents for RAG?”

Retrieved even if document says:

> “Splitting long text for retrieval augmentation”

---

### Pros

* Semantic understanding
* Paraphrase handling

---

### Cons

* Requires vector DB
* Approximate search
* Harder to interpret

---

## 1️⃣2️⃣ Retrieval in RAG (CRITICAL SECTION)

### Why IR is Core to RAG

```
LLM ≠ knowledge base
IR = memory of RAG
```

---

### RAG Retrieval Pipeline

```
Query → Embed → Retrieve → Rerank → Generate
```

---

### Types of Retrieval in RAG

#### 1️⃣ Sparse Retrieval

* BM25
* TF-IDF

Pros: precise keywords
Cons: poor semantics

---

#### 2️⃣ Dense Retrieval

* Embedding similarity

Pros: semantic
Cons: misses exact keywords

---

#### 3️⃣ Hybrid Retrieval (BEST PRACTICE)

```
BM25 + Dense + Reranker
```

---

### Reranking Models

* Cross-encoders
* BERT re-rankers

---

### Real-World RAG Stack

| Layer     | Tool           |
| --------- | -------------- |
| Sparse    | BM25 (Elastic) |
| Dense     | FAISS / Milvus |
| Reranker  | Cross-encoder  |
| Generator | GPT / Gemini   |

---

## 1️⃣3️⃣ Evaluation Metrics (IR Quality)

### Precision

```
Relevant retrieved / total retrieved
```

---

### Recall

```
Relevant retrieved / total relevant
```

---

### NDCG

* Rewards correct ordering
* Used in search engines

---

### MAP

* Average precision across queries

---

## 🔗 Topic Dependency Graph (IMPORTANT)

```
IR Basics
   ↓
Text Preprocessing
   ↓
Indexing
   ↓
Boolean Model
   ↓
VSM
   ↓
TF-IDF
   ↓
Probabilistic Models
   ↓
BM25
   ↓
LSA
   ↓
Neural Retrieval
   ↓
Hybrid Retrieval
   ↓
RAG Retrieval Systems
```

---

## 🎯 Final Takeaway (Exam + Industry)

* **Classical IR** → fast, interpretable
* **Neural IR** → semantic understanding
* **Hybrid IR** → production RAG systems
* **BM25 + Dense + Reranker** = gold standard

---

</details>