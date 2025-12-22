<details>
<summary> Multi-Model Transformers </summary> 

### **How Multi-Modal Transformers Learn Vision and Text Correlations**

Multi-modal transformers, such as **CLIP, Flamingo, BLIP, and Vision Transformers (ViTs) fused with text models like GPT or BERT**, process and align both visual and textual inputs. These models **do not rely on CNNs** in most cases but use **Vision Transformers (ViTs)** or hybrid architectures.

Here’s a step-by-step breakdown of how these models work:

---

## **1. Image Processing (Vision Encoding)**

- Multi-modal transformers **do not use CNNs in most cases**; instead, they use **Vision Transformers (ViT)**.
- ViT divides an image into **patches** (e.g., 16x16 or 32x32 pixels).
- These patches are **flattened into vectors** and **embedded** with positional encodings.
- The **transformer self-attention** mechanism then models relationships between patches.

### **When CNNs Are Used?**

- Some models (e.g., early versions of CLIP) used CNNs like ResNet as an image encoder before switching to ViTs.
- Hybrid models use **CNNs for feature extraction** and then pass embeddings into a transformer for relational learning.

---

## **2. Text Processing (Text Encoding)**

- The text input (e.g., a caption or query) is tokenized and passed into a **Transformer-based language model** like BERT, GPT, or T5.
- The transformer generates an embedding vector for each word based on **self-attention**.

---

## **3. Aligning Vision and Text (Cross-Modal Learning)**

### **(a) Latent Space Alignment**

- The image encoder (ViT or CNN) and the text encoder (BERT/GPT) create **separate embeddings**.
- These embeddings are projected into a **shared latent space** using a **contrastive loss function** (e.g., CLIP uses cosine similarity between image-text embeddings).

### **(b) Cross-Attention Mechanism**

- Some models, like **Flamingo or BLIP**, use **cross-attention layers** to explicitly model interactions between vision and text.
- Here, visual embeddings **attend** to text embeddings, enabling direct **image-text feature fusion**.

### **(c) Unified Transformer Architecture**

- Some models process both images and text within a **single transformer** using different token types.
- For example, **BLIP-2 and PaLI** allow images to be "tokenized" similarly to words, enabling seamless multi-modal reasoning.

---

## **4. Training Multi-Modal Transformers**

These models learn by:

- **Contrastive Learning (CLIP, ALIGN)**: Matching image-text pairs in a high-dimensional space.
- **Masked Language Modeling (BLIP, Flamingo)**: Predicting missing words based on vision input.
- **Vision-Language Pretraining (PaLI, OFA)**: Joint training on image-caption datasets.

---

## **Conclusion: Are CNNs Involved?**

- **CNNs are mostly NOT used** in modern multi-modal transformers.
- Instead, **ViTs replace CNNs** as image encoders.
- Some **hybrid models** may use CNNs for **early-stage feature extraction**, but the core processing happens with **transformers**.

Would you like an example of how a model like CLIP or Flamingo processes an image-text pair? 🚀

</details>
