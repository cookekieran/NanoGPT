# NanoGPT: Character-Level Language Model

## Overview

This project is a custom implementation of a **Generative Pre-trained Transformer (GPT)** built from scratch using PyTorch. The model is designed for **character-level text generation**, trained specifically on the text of *The Wizard of Oz*.

It demonstrates a deep understanding of the Transformer architecture, including:

- **Multi-head self-attention**
- **Residual connections**
- **Layer normalization**

---

## Technical Architecture

The model follows a modern **decoder-only Transformer** design with several key architectural features:

### Self-Attention Mechanism

Implements **scaled dot-product attention** with a causal mask to prevent the model from "looking into the future" during training.

### Multi-Head Attention

Utilizes multiple attention heads in parallel to allow the model to learn various relationships and dependencies within the text simultaneously.

### Residual Connections

Employs a "highway" architecture where the input is added back to the output of sub-layers to facilitate deeper network training and better gradient flow.

### Feed-Forward Networks

Each Transformer block includes a position-wise feed-forward network with:

- **ReLU activation**
- **Dimensionality expansion factor of 4**

### Normalization

Uses a **Pre-Norm architecture** with LayerNorm applied before the attention and feed-forward phases for improved training stability.

---

## Model Hyperparameters

The model was configured with the following parameters to balance performance and computational efficiency:

| Parameter | Value | Description |
|------------|--------|------------|
| **Batch Size** | 32 | Number of sequences processed in parallel |
| **Block Size** | 64 | Maximum context length for predictions |
| **Embedding Dim (n_embd)** | 128 | Dimensionality of character embeddings |
| **Attention Heads** | 4 | Number of parallel attention heads |
| **Layers** | 3 | Number of sequential Transformer blocks |
| **Dropout** | 0.2 | Probability of neuron deactivation for regularization |
| **Learning Rate** | 3e-4 | Step size for the AdamW optimizer |

---

## Training Performance

The model was trained for **500 iterations**, showing a consistent decrease in cross-entropy loss across both training and validation sets:

- **Initial Loss:** ~4.26  
- **Final Training Loss:** 2.175  
- **Final Validation Loss:** 2.191  
- **Total Parameters:** 620,873  

The convergence of training and validation loss indicates the model effectively learned the underlying patterns of the text without significant overfitting, aided by the 20% dropout rate.

---

## Data Pipeline

### Tokenization

A **character-level tokenizer** converts raw text into integers based on a unique vocabulary of 80+ characters found in the source text.

### Train/Validation Split

The dataset (~200,000 tokens) is split **80/20** into training and validation sets to monitor generalization.

### Positional Encoding

Since Transformers have no inherent sense of order, a **learnable positional embedding table** is used to provide the model with information about token locations.

---

## Usage

The model includes a generation function that uses a **multinomial distribution** to sample the next character based on predicted logits. This introduces controlled randomness, preventing the model from becoming stuck in repetitive loops.

### Sample Generation

```python
# Sample Generation
prompt = "The Great Wizard said "
context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
generated_indices = m.generate(context, max_new_tokens=50)
print(decode(generated_indices[0].tolist()))
```

---

## Future Scaling and Optimization

While the current model successfully learns basic character patterns and syntax from *The Wizard of Oz*, it is operating on a relatively small dataset of approximately 200,000 tokens. To transition from "structured gibberish" to coherent, human-like text generation, the following scaling strategies are the next logical steps for this architecture:

### 1. Data Expansion
The primary bottleneck for semantic understanding is dataset size. Moving from a single book to a massive, diverse corpus like **OpenWebText** or **Common Crawl** (billions of tokens) would allow the model to learn complex grammar, factual relationships, and various literary styles.

### 2. Increasing Model Capacity (Width & Depth)
The current "Nano" configuration uses 3 layers and 128-dimensional embeddings. Scaling to a "Base" or "Large" configuration (e.g., 12+ layers and 768+ embedding dimensions) would exponentially increase the model's ability to store parameters and recognize intricate long-range dependencies.


### 3. Transitioning to Sub-word Tokenization
The model currently predicts one character at a time. Implementing **Byte-Pair Encoding (BPE)**—the tokenization method used by industry-standard models like GPT-4—would allow the model to:
* Process common words or syllables as single units (tokens).
* Significantly reduce the computational cost per word.
* Focus its "attention" on high-level concepts rather than individual letters.

### 4. Expanding Context Window (Block Size)
The current model is limited to a context window of 64 characters. By increasing the **Block Size** to 1024 or higher, the model could "remember" the beginning of a long document while writing the conclusion, ensuring that the generated output remains contextually consistent over several paragraphs.