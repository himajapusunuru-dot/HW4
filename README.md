# Homework 4 – CS5760 Natural Language Processing

**University of Central Missouri**
**Course:** CS5760 NLP | Spring 2026
**Name:** [Friend's Name]
**Student ID:** [Friend's ID]

---

## Overview

This repo contains the code for Homework 4. Three Jupyter Notebooks cover a character-level RNN language model, a mini Transformer Encoder built from scratch, and a from-scratch implementation of scaled dot-product attention. Short-answer questions are submitted separately on Bright Space.

---

## Files in This Repo

| File | Description |
|------|-------------|
| `Q1_RNN.ipynb` | Character-level GRU language model with temperature sampling |
| `Q2_transformer.ipynb` | Mini Transformer Encoder with multi-head attention and positional encoding |
| `Q3_attention.ipynb` | Scaled dot-product attention with softmax stability analysis |
| `README.md` | This file |

---

## Q1 – Character-Level RNN Language Model

### What it does
Trains a GRU-based model to predict the next character from a sequence of characters. Two training phases: toy corpus first, then a large public-domain novel.

### Data
- **Toy corpus:** Repeated short phrases (`hello`, `help`, common sentences) — about 6K characters total
- **Large corpus:** Downloads *Pride & Prejudice* from Project Gutenberg automatically (up to 150K characters)

### Model
```
Embedding layer (vocab → 64 dims)
↓
GRU (hidden=256, 2 layers, dropout=0.3)
↓
Linear (256 → vocab_size)
↓
Softmax (temperature-scaled at inference)
```

### Hyperparameters
- Sequence length: **75**, Batch size: **64**, Epochs: **15**
- Optimizer: Adam (lr=1e-3), LR scheduler: StepLR (γ=0.5 every 5 epochs)
- Gradient clipping at 5.0, 90/10 train/val split

### Teacher Forcing
During training, the actual ground-truth character is always fed as the next input — not the model's own prediction. This stabilizes training but creates a mismatch at inference time.

### Outputs
- Loss curves: `q1_loss_curves_toy.png` and `q1_loss_curves_large.png`
- Generated text at τ = 0.7, 1.0, 1.2 (seed: `"hello"` for toy, `"It is a truth"` for large corpus)

### Reflection
Sequence length of 75 gives the model enough context to start capturing word-level patterns, but gradient signals still weaken over long distances. Hidden size 256 is enough for the large corpus without overfitting given the dropout. Temperature is the most visible knob: at τ=0.7 output is repetitive but coherent, at τ=1.2 it's creative but sometimes breaks down — this reflects the softmax sharpening/flattening effect explained in the slides. The train/inference mismatch from teacher forcing gets amplified at higher temperatures.

---

## Q2 – Mini Transformer Encoder

### What it does
Implements a full Transformer Encoder stack from scratch — embedding, sinusoidal positional encoding, multi-head self-attention, feed-forward layers, and Add & Norm — and runs it on 10 custom sentences to produce contextual word embeddings.

### Dataset
10 manually written sentences (NLP-themed, e.g. `"attention is all you need"`). Whitespace tokenizer, vocab size 44, max length 6 with PAD tokens.

### Architecture breakdown

**Positional Encoding:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Non-learned; fixed for all positions.

**Each Encoder Layer has two sub-layers:**
1. Multi-Head Self-Attention (4 heads, d_k=16 per head) → Add & Norm
2. Feed-Forward Network (64 → 128 → ReLU → 64) → Add & Norm

**Full model:** Embedding(44, 64) → SinusoidalPE → 2× EncoderLayer → LayerNorm

### Outputs shown in notebook
- Token IDs for all 10 sentences
- Final embeddings shape: `(10, 6, 64)`
- Attention weight matrix printed for `"transformers use self attention"` (Layer 2, Head 1)
- Attention heatmap saved as `q2_attention_heatmap.png`

---

## Q3 – Scaled Dot-Product Attention

### What it does
Implements the core attention formula from scratch in PyTorch and runs 4 progressively complex tests.

### Formula
```
Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) @ V
```

### Function signature
```python
scaled_dot_product_attention(Q, K, V, mask=None)
# returns: output, weights, raw_scores, scaled_scores
```

### Test suite

**Test 1 – Deterministic (T=4, d_k=4)**
Hand-crafted Q, K, V matrices. Prints raw scores, scaled scores, the full 4×4 attention weight matrix, output vectors, and verifies all row sums equal 1.0.

**Test 2 – Random inputs (T=8, d_k=64)**
Q, K, V sampled from N(0,1). Prints full 8×8 attention matrix and outputs. Runs a softmax stability check comparing behavior before and after the √d_k scaling — demonstrates that unscaled scores saturate the softmax while scaled scores keep gradients healthy.

**Test 3 – Batched multi-head (B=2, H=4, T=6, d_k=16)**
Tests that the function handles arbitrary leading dimensions. Shape assertion and row-sum assertion both pass.

**Test 4 – Causal mask (T=5, d_k=8)**
Applies an upper-triangular boolean mask. Verifies that all blocked positions have weight exactly 0 after softmax.

---

## Running the Code

```bash
pip install torch numpy matplotlib
jupyter notebook
```

- Q1 needs internet for Phase 2 (auto-downloads from Project Gutenberg)
- Q2 and Q3 are fully self-contained, no downloads needed
- Seeds fixed at 42 everywhere — results are reproducible
