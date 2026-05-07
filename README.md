
## Overview

TThis repository contains three deep learning NLP implementations that trace the evolution of sequence modeling — from recurrent networks to the modern Transformer attention mechanism.

---

## Files in This Repo

| File | Description |
|------|-------------|
| `Q1_RNN.ipynb` | Character-level GRU language model with temperature sampling |
| `Q2_transformer.ipynb` | Mini Transformer Encoder with multi-head attention and positional encoding |
| `Q3_attention.ipynb` | Scaled dot-product attention with softmax stability analysis |
| `README.md` | This file |

1. char_rnn.ipynb — Character-Level GRU Language Model
A character-level language model that predicts the next character in a sequence, trained on both a toy corpus and Project Gutenberg's Pride & Prejudice.
Architecture:
Characters → Embedding → GRU (2 layers, h=256) → Dropout → Linear → Softmax
Key Features:

Teacher forcing during training
Gradient clipping (norm = 5.0) to prevent exploding gradients
Learning rate scheduling (StepLR, decay every 5 epochs)
Temperature-controlled text generation (τ = 0.7, 1.0, 1.2)
Two training phases: toy corpus and large corpus (150K chars)

Hyperparameters:
ParameterValueSequence length75Batch size64Embedding dim64Hidden size256GRU layers2Epochs15Learning rate1e-3
Temperature Effects:
τEffect0.7Sharp distribution → coherent but repetitive1.0True model distribution1.2Flat distribution → creative but noisy

2. transformer.ipynb — Mini Transformer Encoder
A from-scratch implementation of the Transformer encoder from "Attention Is All You Need" (Vaswani et al., 2017), trained on 10 short sentences.
Architecture:
Token IDs → Embedding → Sinusoidal PE → [EncoderLayer × 2] → LayerNorm → Output
Each Encoder Layer:
Input → Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm → Output
Key Components:
ComponentDetailsPositional EncodingSinusoidal (non-learned), generalizes to unseen lengthsMulti-Head Attention4 heads, d_k = 16 per headFeed-Forward Networkd_model=64 → d_ff=128 → d_model=64 with ReLUAdd & NormResidual connection + LayerNorm for stable gradientsPadding MaskPrevents attention to PAD tokens
Output:

Contextual embeddings of shape (10, 6, 64)
Attention heatmap saved as q2_attention_heatmap.png


3. attention.ipynb — Scaled Dot-Product Attention
A standalone, thoroughly tested implementation of the core attention formula:
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
The 5 Steps:
StepOperationPurpose1QKᵀCompute query-key similarity scores2÷ √d_kScale to prevent softmax saturation3Mask (optional)Block future or padding positions4SoftmaxConvert scores to probabilities5× VWeighted average of value vectors
Why √d_k scaling matters:
Before ScalingAfter ScalingScore std~7.4~0.93Avg max prob0.86 (saturated)0.33 (healthy)Entropy0.321.79
Four Test Cases:
TestSetupValidates1T=4, d_k=4 deterministicCorrectness (hand-checkable)2T=8, d_k=64 randomStability check before/after scaling3B=2, H=4, T=6, d_k=16Batched multi-head shape4T=5 causal maskAutoregressive masking (upper triangle)

🔗 How the Notebooks Connect
attention.ipynb          ←  core building block
       ↓
transformer.ipynb        ←  stacks attention into a full encoder
       ↓
char_rnn.ipynb           ←  RNN baseline (no attention, for comparison)
Together they cover the key architectural shift in NLP: from sequential RNNs to parallel Transformer attention that relates every token to every other token simultaneously.

🚀 How to Run
Requirements
bashpip install torch numpy matplotlib
Run in Jupyter
bashjupyter notebook
Run in Google Colab
All notebooks were developed and tested on Google Colab with T4 GPU. Simply upload and run — GPU is recommended for char_rnn.ipynb.

📊 Results Summary
char_rnn.ipynb
CorpusFinal Train LossFinal Val LossVal PerplexityToy0.05140.04541.0Pride & Prejudice0.50152.318710.2
transformer.ipynb

Vocabulary: 44 tokens
Output embeddings shape: (10, 6, 64)
Attention heatmap generated for: "transformers use self attention"

attention.ipynb

All 4 tests passed ✅
Causal masking verified (blocked position sum = 0.0)
Softmax saturation demonstrated and resolved


📚 Course Information
FieldDetailsUniversityUniversity of Central Missouri (UCM)CourseCS5760 — Natural Language ProcessingHomeworkHomework 2

📖 References

Vaswani et al. (2017) — Attention Is All You Need
Karpathy (2015) — The Unreasonable Effectiveness of Recurrent Neural Networks
Project Gutenberg — Pride and Prejudice (Jane Austen)
