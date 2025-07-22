# Steps in Building Tokenizer

---

## 🔧 **1. Training a Tokenizer (e.g., BPE or WordPiece)**

* 🔹 **Input**: Raw text corpus (`.txt` file or string list)
* 🔹 **Goal**: Learn the most useful subword units from the corpus
* 🔹 **Steps**:

  * Split all words into **characters** with end-of-word marker `</w>`.
  * Count frequencies of **adjacent token pairs**.
  * Iteratively **merge most frequent pairs** to form subword tokens.
  * Repeat until reaching the target vocab size.

---

## 📁 **2. Creating the Vocabulary File (`vocab.txt` or `tokenizer.json`)**

* 🔹 Stores each learned subword token (like `"un"`, `"##happy"`, `"</w>"`).
* 🔹 Each token is assigned a unique **integer ID**:

  ```
  {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "the": 3,
    "##re": 4,
    ...
  }
  ```
* 🔹 Stored in:

  * `vocab.txt` (one token per line)
  * or `tokenizer.json` (structured for fast use by Hugging Face)

---

## 🗂️ **3. Vocabulary → Token ID Mapping**

* 🔹 Mapping is done using a dictionary:
  `token → id` and `id → token`
* 🔹 Special tokens (like `[PAD]`, `[CLS]`, `[SEP]`, `[UNK]`) are **reserved with fixed IDs**.

---

## ✨ **4. Tokenizing a New Sentence (Inference Time)**

* 🔹 Input: Raw sentence → `"Despite sse its title"`
* 🔹 Steps:

  1. Split sentence into words.
  2. For each word:

     * Apply **merge rules** learned during training (e.g., BPE).
     * Break into matching subwords from vocab.
  3. Add special tokens like `[CLS]` (start) and `[SEP]` (end).
  4. Convert subwords to their corresponding **input IDs**.
  5. Generate:

     * `input_ids`: numeric IDs of tokens
     * `attention_mask`: 1 for real tokens, 0 for padding
     * `token_type_ids`: segment indicators (0 or 1)

---

## 🧠 **5. Model Input**

The final output to be fed into a transformer model is a dictionary like:

```python
{
  'input_ids':       [101, 1234, 5678, 102],
  'attention_mask':  [1, 1, 1, 1],
  'token_type_ids':  [0, 0, 0, 0]
}
```

* `101` = \[CLS]
* `102` = \[SEP]

---

## 🧾 **Summary of Files Created During Tokenizer Training**

| File Name                 | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `vocab.txt`               | List of tokens with order matching their token ID      |
| `tokenizer.json`          | Fast tokenizer format (vocab + merge rules + metadata) |
| `tokenizer_config.json`   | Tokenizer type, casing info, special token settings    |
| `special_tokens_map.json` | Maps `[CLS]`, `[SEP]`, `[PAD]`, etc.                   |

---

## ✅ **Use Cases**

* 🔹 Training tokenizer → needed when building a new LLM or embedding model
* 🔹 Tokenizing sentences → needed during inference for all NLP models

---


