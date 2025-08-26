# Gemma 3 270M SLM from Scratch

## About The Project

This project is a complete, from-scratch implementation of a **270 Million parameter Small Language Model (SLM)** based on the **Gemma 3** architecture.



---

## Model Architecture


### Core Configuration

The model is configured with the following hyperparameters:

| Parameter        | Value    | Description                                  |
| ---------------- | -------- | -------------------------------------------- |
| `vocab_size`     | 50,257   | Size of the GPT-2 tokenizer vocabulary.      |
| `context_length` | 32,768   | Maximum token sequence length.               |
| `emb_dim`        | 640      | Dimension of token embeddings.               |
| `n_layers`       | 18       | Number of transformer blocks.                |
| `n_heads`        | 4        | Number of attention heads.                   |
| `n_kv_groups`    | 1        | Number of Key/Value groups for GQA.          |
| `head_dim`       | 256      | Dimension of each attention head.            |
| `hidden_dim`     | 2048     | Inner dimension of the Feed-Forward Network. |
| `sliding_window` | 512      | Attention window size for local layers.      |

### Key Architectural Features

* **RMSNorm (Root Mean Square Normalization):** A computationally efficient normalization technique used in place of standard LayerNorm for faster training.
* **Grouped Query Attention (GQA):** An efficient attention mechanism where all 4 query heads share a single Key/Value projection. This significantly reduces the memory footprint and speeds up inference compared to traditional Multi-Head Attention.
* **Rotary Positional Embeddings (RoPE):** Encodes token positions by rotating embeddings rather than adding them, providing strong performance on long sequences.
* **Hybrid Attention Mechanism:** The model's 18 layers use a mix of attention types for an optimal balance of efficiency and global context awareness:
    * **Sliding Window Attention:** Used in 15 layers for efficient local context processing.
    * **Full Attention:** Used in 3 layers (5, 11, 17) to capture long-range dependencies across the entire sequence.
* **GELU Feed-Forward Network:** The position-wise feed-forward network uses the GELU (Gaussian Error Linear Unit) activation function for non-linearity.

---

## Key Achievements & Metrics


### Model Scale

* **Total Parameters:** ~270,000,000
* **Vocabulary Size:** 50,257
* **Max Context Length:** 32,768 tokens
* **Transformer Layers:** 18

### Training & Optimization

* **Training Iterations (Target):** 150,000
* **Learning Rate:** `1e-4` (with 1,000-step linear warmup and cosine decay)
* **Effective Batch Size:** 1,024 (32 physical batch size x 32 gradient accumulation steps)
* **Optimizer:** AdamW (`β1=0.9`, `β2=0.95`, Weight Decay=`0.1`)
* **Precision:** `bfloat16` (mixed-precision)

### Performance

* **Initial Validation Loss:** ~9.71
* **Validation Loss after 2,000 iterations:** ~5.85
* **Training Environment:** Single T4 GPU
