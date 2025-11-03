<details>
  
<summary> Sequence Models </summary>

## **What are Sequence Models?**

**Definition:**
Sequence models are a class of models designed to process data where the order or *sequence* of elements matters.
They take in sequences as input and/or produce sequences as output.

**Examples of sequential data:**

* Text (sequence of words/characters)
* Audio (sequence of sound samples)
* Video (sequence of frames)
* Time-series (stock prices, weather data)

**Key property:**
The meaning of a data point depends on **previous** or **surrounding** elements.
Example:

> The word “bank” can mean a *river bank* or a *financial bank* depending on the **previous context**.

---

## **Why Sequence Models?**

Traditional neural networks (like Feedforward or CNNs) assume **independent inputs**, meaning they:

* Can’t remember what came before.
* Don’t capture **temporal** or **contextual** relationships.

**Sequence models solve this by:**

* Maintaining a *state* or *memory* across time steps.
* Learning dependencies between earlier and later elements.

**Applications:**

| Domain          | Example                               |
| --------------- | ------------------------------------- |
| NLP             | Language translation, text generation |
| Speech          | Speech recognition                    |
| Finance         | Stock prediction                      |
| Healthcare      | ECG signal analysis                   |
| Computer Vision | Video captioning                      |

---

## **Recurrent Neural Network (RNN)**

### **What is an RNN?**

An RNN is a type of neural network designed to handle **sequential** data by maintaining a *hidden state* that carries information from previous time steps.

Unlike feedforward networks that process inputs independently, RNNs process sequences one element at a time while preserving a memory of what’s been seen so far.

### **Architecture:**

At each time step `t`:

* Input: ( x_t )
* Hidden state (memory): ( h_t )
* Output: ( y_t )

**Equations:**

```
h_t = f(W_hh * h_(t-1) + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

Here:

* `f` is usually `tanh` or `ReLU`
* `h_(t-1)` carries information from the past → forms the "recurrence"

**Visualization (Unrolled form):**

```
x1 → [RNN Cell] → h1 → y1
x2 → [RNN Cell] → h2 → y2
x3 → [RNN Cell] → h3 → y3
         ↑ (shares weights)
```

### **Advantages:**

* Maintains temporal relationships.
* Can process sequences of variable length.

### **Limitations:**

* **Vanishing/Exploding Gradient**: Difficult to learn long-range dependencies.
* **Sequential computation**: Hard to parallelize during training.
* Struggles with long-term memory.

---

## **Different Types of RNN**

To address RNN limitations, several variants were introduced:

| Type                              | Description                                                                                            | Key Use / Advantage                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| **Vanilla RNN**                   | Basic RNN; processes sequence and maintains a single hidden state.                                     | Works for short-term dependencies.                            |
| **LSTM (Long Short-Term Memory)** | Introduces *cell state* and *gates* (input, forget, output) to preserve information over long periods. | Solves vanishing gradient problem; handles long dependencies. |
| **GRU (Gated Recurrent Unit)**    | Simplified LSTM with fewer gates (reset & update).                                                     | Faster and performs similarly to LSTM.                        |
| **Bidirectional RNN**             | Processes sequence in both forward and backward directions.                                            | Captures both past and future context.                        |
| **Deep RNN (Stacked RNN)**        | Multiple RNN layers stacked on top of each other.                                                      | Learns hierarchical sequence features.                        |

---

## **Backpropagation in RNN (Backpropagation Through Time - BPTT)**

### **What it is:**

Backpropagation in RNNs is known as **Backpropagation Through Time (BPTT)** because the network “unfolds” through each time step, and gradients are propagated **backward through all previous time steps**.

---

### **How it works:**

1. **Forward Pass:**
   The RNN processes inputs $x_1, x_2, …, x_T$ sequentially, generating hidden states $h_1, h_2, …, h_T$ and outputs $y_1, y_2, …, y_T$.

2. **Loss Calculation:**
   
   The total loss is usually the **sum of losses** at each time step:

   $L = \sum_{t=1}^{T} L_t$

4. **Backward Pass:**
   Gradients are computed **back in time**, starting from the last time step ( T ) and moving toward the first.
   Each hidden state $h_t$ depends on:

   * Current input $x_t$
   * Previous hidden state $h_{t-1}$

   So during backprop, the gradient must flow through both **time** and **layers**.

---

### **Main Problem:**

As gradients are repeatedly multiplied through many time steps:

* **If weights < 1:** Gradients shrink → **Vanishing Gradient Problem**
* **If weights > 1:** Gradients grow → **Exploding Gradient Problem**

This makes it hard for RNNs to learn **long-term dependencies** (e.g., words far apart in a sentence).

---

### **Key Takeaway:**

> BPTT helps RNNs learn temporal dependencies, but the gradient instability (vanishing/exploding) limits their ability to remember information across long sequences.

---

## **Advantages and Disadvantages of RNN**

| **Advantages**               | **Explanation**                                              |
| ---------------------------- | ------------------------------------------------------------ |
| Handles sequential data      | Processes data in order, maintaining temporal relationships. |
| Variable input/output length | Can handle sequences of varying sizes.                       |
| Shared parameters            | Same weights used across time steps → fewer parameters.      |
| Context awareness            | Maintains context via hidden states.                         |

| **Disadvantages**             | **Explanation**                              |
| ----------------------------- | -------------------------------------------- |
| Vanishing/exploding gradients | Difficult to capture long-term dependencies. |
| Sequential processing         | Hard to parallelize → slow training.         |
| Short-term memory             | Struggles with dependencies far in the past. |
| Expensive to train            | Requires full sequence unrolling in BPTT.    |

---
<details>
<summary>🧠 Mathematical Explanation — Forward and Backward Propagation in an RNN (Language Model)</summary>


## **1. Problem Setup**

We’ll build a **character-level or word-level language model** that predicts the **next token** given previous ones.

### Example:

Input sequence:

> “I love deep”

Output (next-word prediction):

> “learning”

We train the RNN so that:

$P(\text{learning} ,|, \text{I, love, deep})$

is maximized.

---

### Let’s define:

| Symbol                     | Meaning                                                     |
| -------------------------- | ----------------------------------------------------------- |
| $x_t$                    | Input vector at time step ( t ) (word embedding or one-hot) |
| $h_t$                    | Hidden state (memory) at time step ( t )                    |
| $y_t$                    | Output prediction (probability distribution over vocab)     |
| $W_{xh}, W_{hh}, W_{hy}$ | Learnable weight matrices                                   |
| $b_h, b_y$               | Bias vectors                                                |
| $T$                      | Length of input sequence                                    |

---

## **2. Forward Propagation (Step-by-Step)**

We unroll the RNN for each word in the sequence.

### **Equations:**

At each time step ( t ):

1. **Hidden state update:**
   
   $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$

   * Takes current input $x_t$
   * Combines it with previous hidden state $h_{t-1}$
   * Passes through tanh activation

2. **Output (unnormalized scores):**
   
   $o_t = W_{hy} h_t + b_y$

3. **Predicted probabilities (Softmax layer):**
   
   $\hat{y}_t = \text{softmax}(o_t)$
   
   where
   
   $\hat{y}_t^{(i)} = \frac{e^{o_t^{(i)}}}{\sum_j e^{o_t^{(j)}}}$
   

4. **Loss (Cross-Entropy):**
   If true next word at step ( t ) is represented as one-hot ( y_t ),
   
   $L_t = -\sum_i y_t^{(i)} \log(\hat{y}_t^{(i)})$
   

5. **Total loss across sequence:**

   $L = \sum_{t=1}^{T} L_t$

---

### **Example:**

Let’s say our input sentence (sequence length = 3):

> x₁ = “I” → x₂ = “love” → x₃ = “deep”

Our goal is to predict next tokens:

> y₁ = “love” → y₂ = “deep” → y₃ = “learning”

We’ll compute $h_1, h_2, h_3$ , then $\hat{y}_1, \hat{y}_2, \hat{y}_3 $, and finally the total loss.

---

## **3. Backward Propagation Through Time (BPTT)**

Now we compute gradients of loss ( L ) w.r.t all parameters
$( W_{xh}, W_{hh}, W_{hy}, b_h, b_y )$.

Because each $h_t$ depends on **previous** $h_{t-1}$,
we must backpropagate **through time**.

---

### **Gradients for the Output Layer**

At each time step ( t ):

$\frac{\partial L_t}{\partial o_t} = \hat{y}_t - y_t$

This is the gradient of the cross-entropy loss w.r.t the logits ( o_t ).

Then:

$\frac{\partial L_t}{\partial W_{hy}} = (\hat{y}_t - y_t) h_t^T$

$\frac{\partial L_t}{\partial b_y} = \hat{y}*t - y_t$

$\frac{\partial L_t}{\partial h_t} = W*{hy}^T (\hat{y}_t - y_t)$

### **Gradients for the Hidden State and Recurrent Weights**

Hidden state affects **future losses** too (through recurrence),
so the total gradient for $h_t$ is:

$\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + W_{hh}^T \frac{\partial L}{\partial h_{t+1}} \odot (1 - h_{t+1}^2)$

where:

* $(1 - h_{t+1}^2)$ is from derivative of tanh
* $\odot$ = element-wise multiplication

---

Then, gradients for parameters:

$\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} (1 - h_t^2) x_t^T$

$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} (1 - h_t^2) h_{t-1}^T$

$\frac{\partial L}{\partial b_h} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} (1 - h_t^2)$


---

### **Key Insight:**

The recursive dependence of $h_t$ on $h_{t-1}$ makes gradients flow *back through all previous time steps*.
When ( T ) is large:

$\prod_{t=1}^{T} W_{hh}^T$
can cause **gradients to vanish or explode** — the classic RNN training challenge.

## **4. Summary of the Complete Process**

| Step                  | Description                                    | Equation                                                                                                                          |
| --------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Forward Pass**      | Compute hidden states and outputs sequentially | $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$                                                                                  |
|                       | Compute prediction                             | $\hat{y}*t = \text{softmax}(W*{hy}h_t + b_y)$                                                                                   |
| **Loss**              | Compare prediction with ground truth           | $L_t = -y_t^T \log(\hat{y}_t)$                                                                                                  |
| **Backward Pass**     | Compute gradients of loss wrt outputs          | $\frac{\partial L_t}{\partial o_t} = \hat{y}_t - y_t$                                                                           |
|                       | Backpropagate to hidden states                 | $\frac{\partial L}{\partial h_t} = W_{hy}^T(\hat{y}*t - y_t) + W*{hh}^T\frac{\partial L}{\partial h_{t+1}} \odot (1-h_{t+1}^2)$ |
| **Parameter Updates** | Use gradient descent or Adam                   | $\theta = \theta - \eta \frac{\partial L}{\partial \theta}$                                                                     |

## **5. Intuitive Example — Tiny Network**

Let’s consider a toy RNN:

* Vocabulary = {“I”, “love”, “AI”}
* Input = one-hot vectors
* Output = next word prediction
* Sequence: “I → love → AI”

**Forward:**

1. ( h_1 = \tanh(W_{xh}x_1 + W_{hh}h_0) )
2. Predict next word ( \hat{y}*1 = \text{softmax}(W*{hy}h_1) )
3. Loss between ( \hat{y}_1 ) and target (“love”)
4. Repeat for next words.

**Backward:**

1. Compute output layer gradients: ( \hat{y}_t - y_t )
2. Propagate to hidden state through time (BPTT).
3. Update all weights via gradient descent.

## **6. Key Mathematical Takeaways**

| Concept                                 | Explanation                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| **Forward Propagation**                 | Computes activations through time using previous hidden states.               |
| **Backpropagation Through Time (BPTT)** | Unrolls the network over time to compute gradients recursively.               |
| **Gradient Flow**                       | Accumulates through multiplication by ( W_{hh} ) repeatedly.                  |
| **Training Objective**                  | Minimize total cross-entropy loss over sequence predictions.                  |
| **Vanishing Gradient**                  | When gradients shrink exponentially due to repeated tanh/sigmoid derivatives. |
| **Exploding Gradient**                  | When gradients grow uncontrollably (solved via clipping).                     |

---

## **7. Bridge to LSTM and Transformers**

* **LSTM** introduces *cell states and gates* to keep gradients flowing without vanishing.
* **GRU** simplifies LSTM using fewer gates.
* **Transformers** eliminate recurrence entirely — they compute dependencies via **Self-Attention**, enabling full parallelization and stable gradient flow.

</details>

---

## **What are LSTMs (Long Short-Term Memory Networks)?**

### **Definition:**

LSTM (proposed by Hochreiter & Schmidhuber, 1997) is an advanced RNN architecture designed to **remember information for long periods** and **combat vanishing gradients**.

It introduces a **cell state** and **gates** that control how information is added, forgotten, or output.

---

### **Core Idea:**

Instead of blindly passing hidden states through time, LSTM uses a **“memory cell”** that learns what to **keep**, **update**, or **forget**.

---

### **LSTM Cell Components:**

| Component             | Function                                                      | Equation (for reference)                                                     |
| --------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Forget Gate**       | Decides what information to discard from previous cell state. | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$                             |
| **Input Gate**        | Decides what new information to add.                          | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$                             |
| **Candidate Memory**  | Creates new candidate content for memory.                     | $\tilde{C}*t = \tanh(W_c \cdot [h*{t-1}, x_t] + b_c)$                      |
| **Cell State Update** | Updates memory using gates.                                   | $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$                                  |
| **Output Gate**       | Controls what to output.                                      | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) ), ( h_t = o_t * \tanh(C_t)$ |

---

### **Intuitive View:**

* **Forget gate:** “What should I erase from memory?”
* **Input gate:** “What new info should I store?”
* **Output gate:** “What part of my memory should I share?”

---

## **Why LSTM?**

RNNs forget long-term information due to vanishing gradients.
LSTMs **solve this** by:

* Keeping a **constant error flow** through the cell state.
* Using **gates** to regulate information flow.

This allows them to:

* Remember dependencies over long time spans (e.g., sentence context).
* Handle sequential problems more effectively (translation, speech, etc.).

---

### **Comparison with RNN:**

| Feature           | RNN                 | LSTM                 |
| ----------------- | ------------------- | -------------------- |
| Memory            | Short-term only     | Long-term memory     |
| Gradient issues   | Vanishing/exploding | Controlled via gates |
| Computation       | Simple              | More complex         |
| Long dependencies | Poor                | Excellent            |

---

## **Different Types of LSTM Models**

| **Type**                        | **Description**                                              | **Use Case**                            |
| ------------------------------- | ------------------------------------------------------------ | --------------------------------------- |
| **Vanilla LSTM**                | Standard LSTM layer; processes sequence in one direction.    | Basic sequence learning.                |
| **Stacked (Deep) LSTM**         | Multiple LSTM layers stacked to learn hierarchical features. | Complex temporal patterns.              |
| **Bidirectional LSTM (BiLSTM)** | Processes input in both forward and backward directions.     | Text, speech (context from both sides). |
| **Peephole LSTM**               | Allows gates to access the cell state directly.              | Improves timing-based tasks.            |
| **ConvLSTM**                    | Combines LSTM with CNNs; works on spatial-temporal data.     | Video prediction, weather forecasting.  |

---

## **Advantages and Disadvantages of LSTM**

| **Advantages**                    | **Explanation**                               |
| --------------------------------- | --------------------------------------------- |
| Solves vanishing gradient problem | Maintains long-term dependencies effectively. |
| Better context memory             | Learns patterns spanning many time steps.     |
| Flexible architecture             | Works with variable sequence lengths.         |
| Proven performance                | Effective in NLP, speech, time-series tasks.  |

| **Disadvantages**         | **Explanation**                                      |
| ------------------------- | ---------------------------------------------------- |
| Computationally expensive | Many gates → more parameters → slower training.      |
| Hard to parallelize       | Sequential nature limits GPU utilization.            |
| Risk of overfitting       | Especially on small datasets.                        |
| Complex tuning            | Many hyperparameters (layers, hidden size, dropout). |

---

### 🔑 **Key Takeaway Summary**

* **RNNs** struggle with long-term dependencies due to vanishing gradients.
* **LSTMs** overcome this using **cell state** and **gating mechanisms**.
* **LSTMs** improved sequence modeling but are **computationally heavy**, paving the way for **Transformers**, which replaced recurrence with **self-attention** — enabling **parallelism** and **global context learning**.

### **Quick Comparison (LSTM vs GRU vs RNN)**

| Feature                       | RNN  | LSTM                      | GRU               |
| ----------------------------- | ---- | ------------------------- | ----------------- |
| Gates                         | None | 3 (Input, Forget, Output) | 2 (Update, Reset) |
| Long-term memory              | ❌    | ✅                         | ✅                 |
| Computation speed             | Fast | Slow                      | Moderate          |
| Parameters                    | Few  | Many                      | Moderate          |
| Performance on long sequences | Poor | Excellent                 | Good              |

---

## **Language Model and Sequence Generation**

### **What is a Language Model (LM)?**

A **Language Model** predicts the likelihood of a sequence of words — that is, it assigns a **probability** to a sentence or sequence.

For a given sequence of words ( (w_1, w_2, ..., w_T) ),
the LM estimates:

$P(w_1, w_2, ..., w_T) = \prod_{t=1}^{T} P(w_t | w_1, w_2, ..., w_{t-1})$

So, it learns to predict **the next word given previous words** — this is the core idea behind **sequence generation**.

---

### **Intuition:**

If you train a model on a large text corpus, it learns probabilities such as:

> P("is" | "The sky") → high
> P("banana" | "The sky") → low

That’s how chatbots, text generators, and translators work — they predict **one token at a time**, conditioning on past context.

---

### **Types of Language Models:**

| Type                  | Description                                    | Examples                                 |
| --------------------- | ---------------------------------------------- | ---------------------------------------- |
| **Statistical LM**    | Based on n-gram probabilities.                 | Bigram, Trigram models                   |
| **Neural LM**         | Uses neural networks to model dependencies.    | RNN, LSTM, GRU, Transformer-based models |
| **Autoregressive LM** | Predicts next token sequentially.              | GPT, Llama                               |
| **Masked LM**         | Predicts masked (missing) words in a sentence. | BERT                                     |

---

### **Sequence Generation Process:**

1. Start with a **seed text** (“I love”)
2. Model predicts **next token** probabilities.
3. Choose top prediction (“music”).
4. Append it → “I love music”
5. Repeat the process until a stop token or max length.

**Decoding strategies:**

* Greedy Search
* Beam Search
* Sampling (Top-k, Top-p nucleus sampling)

---

### **Key Takeaway:**

> Language models learn to understand and generate sequences by modeling conditional probabilities of words.
> RNNs, LSTMs, and GRUs were early neural architectures for this — Transformers later revolutionized this with **parallel attention-based modeling**.

---

## **2. Vanishing Gradient Problem in RNN**

### **Where it happens:**

During **Backpropagation Through Time (BPTT)**, gradients are multiplied repeatedly through many time steps.

Each time step contributes a term like:

$\frac{\partial L}{\partial W} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$

If the activation function (like tanh or sigmoid) compresses values to between -1 and 1:

* Multiplying many small numbers → **gradients vanish (→ 0)**
* Multiplying many large numbers → **gradients explode (→ ∞)**

---

### **Result:**

* **Vanishing Gradient:** Model forgets early information → cannot learn long-term dependencies.
* **Exploding Gradient:** Model weights grow uncontrollably → unstable training.

---

### **Symptoms:**

* Training loss stops improving after some epochs.
* Model only learns **short-term patterns**.

---

### **Solutions:**

| Approach                            | How it helps                                                                           |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| **LSTM / GRU**                      | Use gating mechanisms to preserve long-term information.                               |
| **Gradient clipping**               | Caps gradients to avoid explosion.                                                     |
| **ReLU or LeakyReLU**               | Helps maintain gradient flow.                                                          |
| **Layer normalization / residuals** | Stabilizes training.                                                                   |
| **Transformers**                    | Completely remove recurrence, using self-attention to model all dependencies directly. |

---

### **Key Takeaway:**

> The vanishing gradient problem made RNNs unable to capture long-term dependencies — motivating the invention of **LSTMs** and **GRUs**, and eventually **Transformers**.

---

## **Gated Recurrent Unit (GRU)**

### **What is GRU?**

GRU (introduced by Cho et al., 2014) is a **simplified version of LSTM** that merges the **forget** and **input gates** into a single **update gate**, making it faster and computationally lighter while retaining long-term memory capabilities.

---

### **GRU Architecture:**

| Component                       | Purpose                                       | Equation                                                |
| ------------------------------- | --------------------------------------------- | ------------------------------------------------------- |
| **Update Gate (zₜ)**            | Controls how much past info to carry forward. | ( z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) )              |
| **Reset Gate (rₜ)**             | Controls how much past info to forget.        | ( r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) )              |
| **Candidate Hidden State (ĥₜ)** | New memory content.                           | ( \tilde{h}*t = \tanh(W_h \cdot [r_t * h*{t-1}, x_t]) ) |
| **Final Hidden State (hₜ)**     | Blends old and new information.               | ( h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t )       |

---

### **Intuition:**

* **Reset gate:** “How much of the old memory should I forget?”
* **Update gate:** “How much of the new information should I remember?”

---

### **Advantages over LSTM:**

* Fewer parameters (no separate cell state).
* Faster training and inference.
* Similar performance on many tasks.

---

### **Comparison Summary:**

| Feature            | LSTM                      | GRU                      |
| ------------------ | ------------------------- | ------------------------ |
| Gates              | 3 (input, forget, output) | 2 (update, reset)        |
| Memory cell        | Separate                  | Combined in hidden state |
| Speed              | Slower                    | Faster                   |
| Long-term learning | Excellent                 | Good                     |
| Parameters         | More                      | Fewer                    |

---

### **Key Takeaway:**

> GRUs simplify LSTMs by combining gates and removing the separate cell state — making them efficient and practical for many real-world sequence tasks.

---

## **Bidirectional RNN and Deep RNN**

---

### **Bidirectional RNN (BiRNN)**

#### **Concept:**

Standard RNNs read sequences in one direction — from **past → future**.
However, sometimes the **future context** also matters (especially in NLP).

**BiRNNs** process the sequence in **both directions**:

* One RNN moves forward (left → right)
* Another moves backward (right → left)

The outputs from both directions are **concatenated or combined**.

#### **Example:**

To predict a word in a sentence:

> “The cat sat on the ___”

The forward RNN knows: “The cat sat on the”
The backward RNN knows: “___ mat.”
→ Combined, they understand full context.

#### **Applications:**

* Speech recognition
* Text classification
* Named Entity Recognition (NER)
* Machine translation (encoder side)

#### **Limitation:**

* Can’t be used in **real-time generation tasks** (like speech or text generation) because future words aren’t available at inference.

### **Deep RNN (Stacked RNN)**

#### **Concept:**

A **Deep RNN** (or Stacked RNN) has multiple RNN/LSTM/GRU layers stacked on top of each other.
Each layer’s output becomes the next layer’s input, allowing the network to learn **hierarchical temporal patterns**.

```
Input → RNN Layer 1 → RNN Layer 2 → RNN Layer 3 → Output
```

* Lower layers capture **short-term** patterns.
* Higher layers capture **long-term / abstract** dependencies.

#### **Advantages:**

* Better feature extraction.
* Can model complex relationships across time.

#### **Disadvantages:**

* Higher computational cost.
* Greater risk of vanishing gradients in very deep stacks.

### **Key Takeaway:**

> * **Bidirectional RNNs** use both past and future context.
> * **Deep RNNs** build layered understanding of sequences.
>   These advances improved RNN expressiveness — but **Transformers** later achieved both long-term dependency and parallelism more efficiently.

### 🔑 **Overall Summary (Connecting to Transformers)**

| Challenge / Evolution       | Solution    | Key Feature                  |
| --------------------------- | ----------- | ---------------------------- |
| RNN: forgets long-term info | LSTM        | Gating + cell state          |
| LSTM: complex & slow        | GRU         | Simplified gating            |
| RNN: can’t see future       | BiRNN       | Bidirectional context        |
| RNN: limited hierarchy      | Deep RNN    | Layer stacking               |
| Still sequential & slow     | Transformer | Self-Attention + Parallelism |
</details>

