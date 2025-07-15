# Tokenization

Tokenization is the **first and fundamental step in Natural Language Processing (NLP)**. It involves **breaking down a text (sentence or document) into smaller pieces** called *tokens*, which can be **words, subwords, characters, or sentences**. These tokens are then used as input to NLP models.

---

### 🔹 Why Tokenization is Important

1. **Enables structured input for models** (e.g., transformer models expect numerical token IDs).
2. **Handles linguistic complexity** like punctuation, contractions, compound words.
3. **Improves model performance** by normalizing text (removing ambiguity).

---

### 🔹 Types of Tokenization in NLP

Let's break it down into types with examples.

---

## 1. **Word Tokenization**

👉 Splits text into individual words based on spaces or punctuation.

### 🔹 Example:

```python
sentence = "I'm learning NLP!"
tokens = ["I'm", "learning", "NLP", "!"]
```

🔸 Tools: `nltk.word_tokenize`, `split()`, spaCy

---

## 2. **Subword Tokenization**

👉 Breaks words into smaller meaningful units (subwords). Useful for handling **out-of-vocabulary words** (e.g., "unhappiness" → "un", "happi", "ness").

### 🔹 Example:

Sentence: `"unhappiness"`

* Subword Tokens (using Byte-Pair Encoding): `['un', 'happiness']`
* Or: `['un', 'happi', 'ness']`

🔸 Used in: BERT (WordPiece), GPT (BPE), RoBERTa (BPE), T5 (SentencePiece)

---

## 3. **Character Tokenization**

👉 Breaks down text into individual characters.

### 🔹 Example:

```python
sentence = "NLP"
tokens = ['N', 'L', 'P']
```

🔸 Useful for character-level models or spelling correction tasks.

---

## 4. **Sentence Tokenization**

👉 Splits text into individual sentences.

### 🔹 Example:

```python
text = "Hello world. I'm learning NLP. It's fun!"
tokens = ['Hello world.', "I'm learning NLP.", "It's fun!"]
```

🔸 Tools: `nltk.sent_tokenize`, spaCy

---

## 5. **Whitespace Tokenization**

👉 Splits text based purely on whitespace. Doesn't handle punctuation or contractions well.

### 🔹 Example:

```python
sentence = "Don't tokenize badly!"
tokens = ["Don't", "tokenize", "badly!"]
```

🔸 Not recommended for serious NLP tasks.

---

## 6. **Regex Tokenization**

👉 Uses regular expressions to extract specific patterns like hashtags, mentions, numbers.

### 🔹 Example:

Extracting hashtags:

```python
import re
text = "Let's #learn #NLP"
tokens = re.findall(r"#\w+", text)
# ['#learn', '#NLP']
```

---

### 🔹 Summary Table

| Type                    | Description                  | Example Input  | Example Tokens             |
| ----------------------- | ---------------------------- | -------------- | -------------------------- |
| Word Tokenization       | Split by words               | "NLP is fun!"  | \['NLP', 'is', 'fun', '!'] |
| Subword Tokenization    | Split into frequent subwords | "unhappiness"  | \['un', 'happi', 'ness']   |
| Character Tokenization  | Split into characters        | "NLP"          | \['N', 'L', 'P']           |
| Sentence Tokenization   | Split into sentences         | "Hi. Bye."     | \['Hi.', 'Bye.']           |
| Whitespace Tokenization | Split by space only          | "NLP is cool!" | \['NLP', 'is', 'cool!']    |
| Regex Tokenization      | Extract using regex patterns | "Hello @user!" | \['@user']                 |

---

### 🔹 Popular Tokenizer Libraries

* **NLTK**: Classic NLP tokenizer (sentence & word)
* **spaCy**: Fast, rule-based & ML-based tokenization
* **Hugging Face Tokenizers**:

  * WordPiece (BERT)
  * Byte Pair Encoding (GPT)
  * SentencePiece (T5, ALBERT)

---

### 🔹 Real-world Use Case: HuggingFace Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentence = "Tokenization is crucial in NLP."

tokens = tokenizer.tokenize(sentence)
input_ids = tokenizer.encode(sentence)

print("Tokens:", tokens)
print("Input IDs:", input_ids)
```

✅ Output:

```bash
Tokens: ['tokenization', 'is', 'crucial', 'in', 'nlp', '.']
Input IDs: [101, 19204, 2003, 8503, 1999, 17953, 1012, 102]
```

---
# Byte Pair Encoding (BPE) and WordPiece tokenization

**Byte Pair Encoding (BPE)** and **WordPiece** tokenization—both are subword tokenization techniques used in modern NLP models like GPT and BERT. They help handle:

* Unknown words (OOV issues)
* Rare words
* Vocabulary efficiency

---

## 🔷 1. Byte Pair Encoding (BPE) Tokenization

### ✅ Core Idea:

Byte Pair Encoding is a **data compression algorithm** adapted for NLP. It merges the **most frequent pairs of characters/subwords** in a corpus iteratively to create a vocabulary of subwords.

---

### 🛠️ How BPE Works (Step-by-Step):

#### 🔹 Step 1: Start with characters

Every word is initially split into characters (with an end-of-word symbol `</w>` for training).

#### 🔹 Step 2: Count frequency of symbol pairs

Find the most frequent adjacent character pairs.

#### 🔹 Step 3: Merge the most frequent pair

Combine those characters into a new subword token.

#### 🔹 Step 4: Repeat

Repeat until the desired vocabulary size is reached.

---

### 📌 Example:

**Corpus** (tiny set for illustration):

```text
low lowlow lower
```

Initial tokens (each word split into chars + `</w>`):

```
['l', 'o', 'w', '</w>']
['l', 'o', 'w', 'l', 'o', 'w', '</w>']
['l', 'o', 'w', 'e', 'r', '</w>']
```

#### 🔹 Iteration 1: Most frequent pair → `('l', 'o')`

New tokens:

```
['lo', 'w', '</w>']
['lo', 'w', 'lo', 'w', '</w>']
['lo', 'w', 'e', 'r', '</w>']
```

#### 🔹 Iteration 2: Merge `('lo', 'w')` → `low`

```
['low', '</w>']
['low', 'low', '</w>']
['low', 'e', 'r', '</w>']
```

#### 🔹 Iteration 3: Merge `('e', 'r')` → `er`

```
['low', '</w>']
['low', 'low', '</w>']
['low', 'er', '</w>']
```

#### 🔹 Iteration 4: Merge `('low', 'er')` → `lower`

```
['low', '</w>']
['low', 'low', '</w>']
['lower', '</w>']
```

📌 **Final vocabulary**:

```
['low', 'lower', 'lo', 'w', 'l', 'o', 'w', 'er']
```

---

### 🧠 Tokenizing New Word:

Word: `"lowlower"`

* Tokenized as: `['low', 'lower']`

If `"lowlowermost"` appears and is OOV:

* BPE breaks it as best match: `['low', 'lower', 'm', 'o', 's', 't']` (character fallback if needed)

---

## 🔶 2. WordPiece Tokenization

### ✅ Core Idea:

Similar to BPE but instead of merging based on **frequency alone**, it merges based on **likelihood** (probability of the merged word improving the language model). It's used in **BERT**.

---

### 🛠️ How WordPiece Works:

* Learns subword vocabulary from a corpus.
* Uses **Maximum Likelihood Estimation (MLE)** to choose merges.
* Prefixes subwords (not starting a word) with `##` to indicate continuation.

---

### 📌 Example:

Suppose we have:

**Vocabulary**:

```
['[UNK]', 'un', '##aff', '##able', 'affable', 'aff', '##able']
```

Tokenizing word: `"unaffable"`

WordPiece breaks it as:

```
['un', '##aff', '##able']
```

Each subword must exist in the vocab. If not, fallback is `[UNK]`.

---

### 🔹 Comparison with BPE:

| Feature                | BPE                     | WordPiece                  |
| ---------------------- | ----------------------- | -------------------------- |
| Merge rule             | Most frequent pair      | Most probable (likelihood) |
| Used in                | GPT, RoBERTa            | BERT, DistilBERT           |
| Subword marker         | No marker (`low`, `er`) | Uses `##` (`##er`)         |
| Unknown token fallback | Yes, character fallback | Yes, `[UNK]` token         |

---

## 🧪 HuggingFace Example (BERT = WordPiece, GPT = BPE)

```python
from transformers import AutoTokenizer

# WordPiece tokenizer (used in BERT)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(bert_tokenizer.tokenize("unaffable"))
# Output: ['un', '##aff', '##able']

# BPE tokenizer (used in GPT-2)
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(gpt_tokenizer.tokenize("unaffable"))
# Output might vary: ['un', 'aff', 'able'] or ['un', 'affable']
```

---

## ✅ Summary

| Tokenizer     | Type       | Used In          | Marker | Fallback         | Strength                             |
| ------------- | ---------- | ---------------- | ------ | ---------------- | ------------------------------------ |
| **BPE**       | Frequency  | GPT, RoBERTa     | No     | Char-level merge | Efficient and compressive            |
| **WordPiece** | Likelihood | BERT, DistilBERT | `##`   | `[UNK]`          | Better for modeling unseen sequences |

---

# Visualization of Tokenization & Training our own Tokenizer

---

## ✅ Part 1: **Visualization of Tokenization (Step-by-Step)**

Let’s take the word **`unaffable`** and see how **BPE** and **WordPiece** process it.

---

### 🔷 Word: `"unaffable"`

Let’s assume this is our vocabulary:

```
['[UNK]', 'un', 'aff', 'able', '##aff', '##able']
```

### 🟦 WordPiece Tokenization:

| Step | Input         | Matching Token       | Output So Far               |
| ---- | ------------- | -------------------- | --------------------------- |
| 1    | `"unaffable"` | `un` (start of word) | `['un']`                    |
| 2    | `"affable"`   | `##aff`              | `['un', '##aff']`           |
| 3    | `"able"`      | `##able`             | `['un', '##aff', '##able']` |

➡️ Final output: `['un', '##aff', '##able']`

> 📌 WordPiece uses `##` to mark continuation of a word.

---

### 🟩 BPE Tokenization (Example)

Assume BPE merges:

* `'a' + 'f' → 'af'`
* `'af' + 'f' → 'aff'`
* `'aff' + 'able' → 'affable'`
* `'un' + 'affable' → 'unaffable'`

| Step | Input                                 | Most Frequent Pair | Output                               |
| ---- | ------------------------------------- | ------------------ | ------------------------------------ |
| 1    | `'u','n','a','f','f','a','b','l','e'` | `'a','f'`          | `['u','n','af','f','a','b','l','e']` |
| 2    | `...`                                 | `'af','f'`         | `['u','n','aff','a','b','l','e']`    |
| 3    | `...`                                 | `'aff','able'`     | `['u','n','affable']`                |
| 4    | `...`                                 | `'un','affable'`   | `['unaffable']`                      |

➡️ Final output: `['unaffable']`

> 📌 BPE doesn’t use special markers, and builds vocabulary based on frequent subword merges.

---

## ✅ Part 2: **Train Your Own Tokenizer from Scratch (HuggingFace Tokenizers)**

You can build your own BPE or WordPiece tokenizer using Hugging Face's `tokenizers` library (not `transformers`, but lower-level).

---

### 🧰 Install Required Library

```bash
pip install tokenizers
```

---

### 🧪 Example: Train a BPE Tokenizer from Scratch

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.processors import TemplateProcessing

# 1. Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. Add pre-tokenizer (splits on whitespace)
tokenizer.pre_tokenizer = Whitespace()

# 3. Normalizer (optional but recommended)
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

# 4. Define trainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=100)

# 5. Train tokenizer on files
files = ["my_corpus.txt"]  # Your text corpus file
tokenizer.train(files, trainer)

# 6. Save the tokenizer
tokenizer.save("bpe_tokenizer.json")
```

---

### 🔍 Test the Tokenizer

```python
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

encoded = tokenizer.encode("unaffable person is rare")
print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)
```

---

### 📘 Sample Corpus: `my_corpus.txt`

```
unaffable
affable
unhappy
unreal
person
rare
```

---

### 🧱 Train WordPiece Tokenizer

```python
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=100)

tokenizer.train(files=["my_corpus.txt"], trainer=trainer)
tokenizer.save("wordpiece_tokenizer.json")
```

Then test with:

```python
tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")
encoded = tokenizer.encode("unaffable person is rare")
print("Tokens:", encoded.tokens)
```

---

## 📌 Key Takeaways

| Feature              | BPE                  | WordPiece                    |
| -------------------- | -------------------- | ---------------------------- |
| Merge strategy       | Frequent pair merges | Maximum likelihood           |
| Token marker         | None                 | `##subword` for continuation |
| Used in              | GPT, RoBERTa         | BERT, DistilBERT             |
| Library for training | `tokenizers` (HF)    | `tokenizers` (HF)            |

---

