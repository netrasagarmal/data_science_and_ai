# 📘 NLP Word Representations: One-Hot, Vector Space & Semantic Representations


## 🌐 What is a Word Embedding?

A **word embedding** is a **dense vector representation** of a word that captures its **meaning**, **semantic relationships**, and **contextual usage** in a fixed-dimensional space (e.g., 300 or 768 dimensions).


## 🧠 Why Represent Words as Vectors?

Computers don’t understand text—they process numbers.
To work with **natural language**, we must **numerically encode words** so machine learning models can understand and process them.

---

## 1. 🧩 **One-Hot Encoding**

### ➤ Definition:

Each word is represented by a **binary vector** of size equal to the vocabulary size. Only **one element is 1**, the rest are **0**.

### ➤ Example:

Vocabulary:
`["apple", "banana", "grape", "mango"]`

| Word   | One-Hot Vector |
| ------ | -------------- |
| apple  | \[1, 0, 0, 0]  |
| banana | \[0, 1, 0, 0]  |
| grape  | \[0, 0, 1, 0]  |
| mango  | \[0, 0, 0, 1]  |

### ✅ Pros:

* Simple
* Easy to implement

### ❌ Cons:

* **No semantic information** (e.g., “apple” and “mango” are unrelated)
* **High-dimensional** for large vocabularies
* **Sparse vectors** → inefficient computation

---

## 2. 📏 **Vector Space Model (VSM)**

### ➤ Definition:

Each word or document is represented as a **vector in an n-dimensional space**. The idea is that **semantically similar words/documents** will be **closer in the space**.

---

### 🔹 A. **Term Frequency-Inverse Document Frequency (TF-IDF)**

#### ➤ Purpose:

Weighs word importance in a document relative to its frequency across a corpus.

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)
$$

* **TF (term frequency)**: Number of times term appears in document
* **IDF (inverse document frequency)**: Penalizes common terms

#### ➤ Representation:

For a document, each word gets a weight → forms a dense vector.

---

### 🔹 B. **Count Vectorization (Bag of Words)**

* Count how many times each word appears in a document.
* Represent document as a vector of counts.

| Document                 | apple | banana | mango |
| ------------------------ | ----- | ------ | ----- |
| D1: "apple mango"        | 1     | 0      | 1     |
| D2: "banana mango mango" | 0     | 1      | 2     |

### ✅ Pros:

* Captures basic frequency
* Good for text classification with small datasets

### ❌ Cons:

* Still **sparse** and **high-dimensional**
* No context awareness or semantic meaning

---

## 3. 🌐 **Semantic Word Embeddings**

### ➤ Definition:

Represent words as **dense vectors** that **capture their meaning and relationships** based on their usage in language.

---

### 🔹 A. **Static Embeddings** (Word2Vec, GloVe, FastText)

Each word has **one vector** regardless of context.

#### 📌 Key Idea:

Words used in **similar contexts** have **similar vectors**.

* "king" - "man" + "woman" ≈ "queen"

#### Example (Word2Vec):

| Word   | Vector (simplified) |
| ------ | ------------------- |
| king   | \[0.51, 0.65, 0.13] |
| queen  | \[0.48, 0.60, 0.10] |
| banana | \[0.12, 0.05, 0.98] |

---

### 🔹 B. **Contextual Embeddings** (BERT, GPT, ELMo)

Each **word occurrence** gets a **different vector** depending on **surrounding context**.

#### ➤ Example:

* "The **bank** of the river was steep." → Embedding 1
* "He deposited money in the **bank**." → Embedding 2

Both are different even for the same word!

---

## 📊 Comparing Representations

| Feature            | One-Hot | TF-IDF / BoW | Static Embedding | Contextual Embedding |
| ------------------ | ------- | ------------ | ---------------- | -------------------- |
| Dense/Sparse       | Sparse  | Sparse       | Dense            | Dense                |
| Dimensionality     | High    | High         | Low (e.g., 300)  | Medium (768–2048)    |
| Captures semantics | ❌       | ❌            | ✅                | ✅✅                   |
| Context aware      | ❌       | ❌            | ❌                | ✅✅                   |
| Language evolution | ❌       | ❌            | ✅                | ✅✅                   |

---

## 🧠 Key Terms to Remember

* **Vector Space**: Geometric space where each word or doc is a point (vector).
* **Dimensionality**: Number of features in the vector (e.g., 300 for Word2Vec).
* **Semantic Similarity**: Measured by distance (cosine similarity) between vectors.
* **Context**: The words surrounding a target word that define its meaning.

---

## 🧪 Cosine Similarity (Used to compare word/document vectors)

$$
\text{cosine\_sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

* Ranges from -1 to 1
* Closer to 1 = more similar

---

## ✅ Summary Notes

| Type                  | Description                         | Captures Meaning? | Contextual? |
| --------------------- | ----------------------------------- | ----------------- | ----------- |
| One-hot               | Unique vector with one 1 and rest 0 | ❌                 | ❌           |
| BoW / TF-IDF          | Word count/statistics vector        | ❌                 | ❌           |
| Static Embeddings     | Dense vector learned via usage      | ✅                 | ❌           |
| Contextual Embeddings | Vector changes with sentence        | ✅✅                | ✅✅          |

---



---

Great! Let’s dive **deep into the differences between Static Embeddings and Contextual Embeddings**—two fundamental approaches to representing words in vector space.

---

## 🔍 Overview

| Aspect            | **Static Embeddings**                      | **Contextual Embeddings**                  |
| ----------------- | ------------------------------------------ | ------------------------------------------ |
| Representation    | One fixed vector per word                  | Dynamic vector that changes with context   |
| Examples          | Word2Vec, GloVe, FastText                  | BERT, RoBERTa, GPT, T5                     |
| Context Awareness | ❌ No                                       | ✅ Yes                                      |
| Architecture      | Shallow models                             | Transformer-based deep models              |
| Use Case          | Basic NLP tasks                            | Complex tasks like QA, summarization       |
| Output            | Same vector for every occurrence of a word | Different vector depending on the sentence |
| Vocabulary        | Fixed size, word-level                     | Subword-level (WordPiece, BPE)             |

---

## 🧱 1. STATIC EMBEDDINGS (Word2Vec / GloVe / FastText)

### ➤ Goal:

Convert every word into a **fixed-dimensional dense vector** capturing semantic similarity.

### 📌 Core Properties:

* Each word in the vocabulary has **one vector** irrespective of where it's used.
* Nearby words in meaning (e.g., “king” and “queen”) have vectors that are close in cosine similarity.

### 📘 Word2Vec - Skip-gram:

* Objective: Given a word, predict surrounding words.
* Training: Uses shallow neural network.

$$
\text{Maximize } \sum_{w \in V} \sum_{c \in \text{context}(w)} \log P(c | w)
$$

* After training, the weight matrix is the **embedding matrix**.
* Captures analogies:

  $$
  \text{vec("king")} - \text{vec("man")} + \text{vec("woman")} \approx \text{vec("queen")}
  $$

### ❌ Limitations:

* Same vector for "bank" in:

  * "He went to the **bank** to deposit money."
  * "He sat by the **bank** of the river."
* Cannot handle OOV (Out-of-Vocabulary) words unless trained explicitly.

---

## 🧠 2. CONTEXTUAL EMBEDDINGS (BERT, RoBERTa, GPT)

### ➤ Goal:

Generate **dynamic embeddings** that capture the **meaning of a word based on its context**.

### 📌 How it works:

* Use **Transformer architectures** (self-attention, positional embeddings).
* Each word’s embedding depends on the **entire sentence**.

For example:

* Embedding for **"bank"** in:

  * Sentence A: *“He went to the bank to withdraw cash.”*
  * Sentence B: *“He walked along the river bank.”*
  * 👉 **Different vectors** for "bank" because context is different.

### 🏗️ Under the Hood:

* Input → Subwords + Position + Segment embeddings
* Goes through multiple transformer layers
* Output:

  * A **different embedding** for each token
  * Each embedding is **context-sensitive**

### 📘 BERT Objective:

* Masked Language Modeling: Predict a masked word based on context.
* Example:

  ```
  Sentence: "The man went to the [MASK]"
  Model predicts: "store", "bank", etc. based on context.

  ```

### ✅ Advantages:

* Captures polysemy (same word, different meanings)
* Rich, deep representations useful for:

  * Sentiment analysis
  * Question answering
  * Named entity recognition
  * Semantic search

---

## 🧪 Example Comparison

### Sentence:

> "She went to the **bank** to withdraw cash."

### Static Embedding:

* `vec("bank")` → Same vector always

### Contextual Embedding:

* Transformer reads full sentence.
* Calculates attention over all words.
* Embedding for “bank” changes depending on surrounding words.

---

## 📊 Visualization Example (TSNE)

Imagine you visualize embeddings using TSNE:

* **Static**: Each word has one dot on the map
* **Contextual**: Each word may have **multiple dots** based on different contexts

E.g.:

* "bank" → multiple points (money-related, river-related)

---

## 🧠 Analogy

| Scenario              | Static                          | Contextual                           |
| --------------------- | ------------------------------- | ------------------------------------ |
| Word as a person      | Has only one fixed outfit       | Dresses based on situation           |
| "Apple" in a sentence | 🍎 and Apple Inc. → same vector | 🍎 vs Apple Inc. → different vectors |
| Vision                | Black-and-white                 | Full color and depth                 |

---

## 📌 Summary Table

| Feature              | Static Embedding            | Contextual Embedding                     |
| -------------------- | --------------------------- | ---------------------------------------- |
| Architecture         | Shallow (NNs, SVD)          | Transformers                             |
| Context Sensitivity  | No                          | Yes                                      |
| Polysemy Handling    | Poor                        | Excellent                                |
| Pretrained on        | Corpus (no context)         | Large corpora (context-aware)            |
| Real-world Use Cases | Word similarity, clustering | QA, chatbots, summarization, translation |

---

## 🔚 Final Thoughts

### Use Static Embeddings when:

* You need fast and simple representations.
* Training shallow models on small datasets.

### Use Contextual Embeddings when:

* You want high-quality, nuanced representations.
* Tasks involve semantics, syntax, or ambiguity.

---

Would you like a **real code demo** comparing both types using Python and HuggingFace? Or maybe an image showing how attention helps contextual models?

Let me know!




---

## 📦 Types of Embedding Models

| Category                  | Examples                  | Key Idea                                                        |
| ------------------------- | ------------------------- | --------------------------------------------------------------- |
| **Static Embeddings**     | Word2Vec, GloVe, FastText | One vector per word, fixed for all contexts                     |
| **Contextual Embeddings** | BERT, GPT, RoBERTa, T5    | Different vectors for each word occurrence depending on context |

---

## ⚙️ A. **Static Word Embedding: Word2Vec**

### 🔹 Word2Vec – Skip-gram Model

* Goal: Given a word, predict surrounding words (context).
* Architecture: Simple shallow neural network with one hidden layer.

### 🎯 Objective:

Maximize the probability of context words given a center word:

$$
\prod_{t=1}^{T} \prod_{-c \leq j \leq c, j \neq 0} P(w_{t+j} \mid w_t)
$$

where:

* $w_t$: target word
* $w_{t+j}$: context word
* $c$: window size

### 🧠 Model Details:

* **Input layer**: One-hot vector of size $V$ (vocab size).
* **Hidden layer**: Weight matrix $W$ of size $V \times D$. (This becomes the embedding matrix.)
* **Output layer**: Softmax over vocab.

### 📌 Training:

* Use **cross-entropy loss** or **negative sampling**.
* After training, rows of the weight matrix $W$ are the **word embeddings**.

---

## ⚙️ B. **Contextual Embeddings: BERT and Transformers**

These models generate different embeddings for the **same word** depending on the sentence.

---

### 🧱 Step-by-Step Architecture

#### 1. **Tokenization**

* Converts input sentence into tokens (subwords).
* e.g., `"Bank of river"` → `[CLS]`, `bank`, `of`, `river`, `[SEP]`

#### 2. **Input Embeddings**

Each token is represented as:

$$
\text{InputEmbedding} = \text{TokenEmbedding} + \text{PositionEmbedding} + \text{SegmentEmbedding}
$$

* **Token Embedding**: Lookup from vocab.
* **Position Embedding**: Adds info about position in the sentence.
* **Segment Embedding**: Used to differentiate sentence pairs in tasks like QA.

---

#### 3. **Transformer Layers (BERT has 12/24 layers)**

Each layer contains:

* **Multi-Head Self Attention**:

  * Each word "attends" to all others using a score:

    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
    $$
  * Where:

    * $Q = XW^Q$, $K = XW^K$, $V = XW^V$
    * $X$ is input from previous layer
* **Feed-Forward Layer**:

  * Two-layer MLP: $\text{ReLU}(XW_1 + b_1)W_2 + b_2$
* **Layer Norm**, **Residual Connections**

---

#### 4. **Final Output**

* Output of each token is a vector of size 768 (for BERT-base).
* Each word’s embedding encodes context.
* You can use:

  * `last_hidden_state[i]` → token i's contextual embedding
  * `CLS` token → sentence embedding

---

### 🧠 BERT Example (Code)

```python
from transformers import BertTokenizer, BertModel
import torch

sentence = "He went to the bank to deposit money."

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

tokens = tokenizer(sentence, return_tensors="pt")
output = model(**tokens)

embeddings = output.last_hidden_state  # shape: [1, seq_len, 768]
```

* `embeddings[0][3]` → contextual vector of token "bank"
* Same word "bank" in "river bank" will give a different vector!

---

## 📌 Summary Table

| Feature               | Word2Vec         | BERT                     |
| --------------------- | ---------------- | ------------------------ |
| Output                | Static vector    | Contextual vector        |
| Context Awareness     | ❌                | ✅                        |
| Architecture          | Shallow          | Deep transformer         |
| Token-Level           | Word             | Subword                  |
| Pretraining Objective | Skip-gram / CBOW | Masked Language Modeling |

---

## 🧠 Final Notes

* Embedding models **transform discrete tokens into continuous space**.
* Transformers use **attention** to encode rich semantics in embeddings.
* These embeddings can then be used for:

  * Classification
  * Similarity Search
  * Translation
  * Question Answering

---

Would you like:

* A visual flowchart of BERT embedding model?
* Code to extract sentence embeddings?
* Comparison with Sentence Transformers or OpenAI Embeddings?

Let me know how you'd like to continue!
