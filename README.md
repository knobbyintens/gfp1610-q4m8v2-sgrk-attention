# SGRK-Attention

Reference implementation accompanying the paper
**SGRK-Attention: Gated Sparse Kernels for Efficient Linear Attention**
(preprint arXiv:2602.05678, Pollet, Anders and Ruiz, 2026).

SGRK-Attention reuses the sparse-gated recurrence machinery of earlier
recurrent-kernel work but replaces the local kernel bank with a linear-attention
surrogate. This repository implements the *SGRK-Attention* model only. It does
not reproduce the original SGRK paper, its identifier, or its experiments.

## Contents

* `sgrk_attn/model.py` — linear-attention kernel mixer
* `train.py` — long-horizon benchmark runner for the attention variant
