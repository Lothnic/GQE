# Reproduction Notes & Ambiguity Audit — GQE

This document tracks all specifications, assumptions, and unspecified details identified during the implementation of *Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention*.

## Ambiguity Audit

### 1. SPECIFIED (High Confidence)
These details are explicitly documented in the paper and followed directly:

| Choice | Value | Source |
|--------|-------|--------|
| **Routing Granularity** | Within each GQA group, select top-$k$ query-head experts per token | §1, §3.2 |
| **Softmax Position** | Softmax applied over group experts *before* top-$k$ selection | §3.2, Eq. 2 |
| **Output Construction** | Concat(O_i, \bar{o}_i, s(x_i)) W_O | §3.3, Eq. 6 |
| **Active Experts per Group ($k$)** | 1 active expert per group | §4.1 |
| **Attention configuration** | 16 query heads / 8 KV heads (G = 8 GQA groups) | §4.1 |
| **Shared Head** | 1 always-on shared attention head | §3.3, §4.2 |
| **Weighted-Sum Slot** | 1 renormalized weighted-sum slot using router probabilities | §3.3, Eq. 5 |
| **Auxiliary Loss** | Load-balancing loss computed using router probabilities and selection | §3.4 |
| **Pretraining Dataset** | FineWeb-Edu (subset of FineWeb2) | §4.1 |
| **Optimizer** | Fused AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=1\times 10^{-7}$) | §4.1, Table 1 |
| **LR Schedule** | WSD (Warmup-Stable-Decay) | §4.1, Table 1 |
| **Warm-up Tokens** | 3 Billion tokens (3BT) | §4.1, Table 1 |
| **Weight Decay** | 0.1 | §4.1, Table 1 |
| **Global Batch Size** | 1.05 Million tokens | §4.1, Table 1 |
| **Sequence Length** | 2048 | §4.1, Table 1 |
| **Precision** | Mixed precision BFloat16 | §4.1, Table 1 |
| **Loss Spike Mitigation** | ZClip | §4.1 |

---

### 2. PARTIALLY SPECIFIED (Judgment Call Made)
These details were mentioned in the paper but was not fully detailed:

| Choice | Our Decision | Paper Quote | Alternatives |
|--------|-------------|-------------|--------------|
| **ZClip implementation** | Running-statistic adaptive gradient norm clipping. | "...employ ZClip [15] to mitigate loss spikes" | Constant clipping (1.0) |
| **WSD Decay Portion** | Cosine decay over final 20% of training tokens. | "Learning-rate schedule: WSD" | Linear decay over last 10%, step decay |
| **Shared Head Keys/Values** | Default `shared_kv_policy: group_0` — attends to GQA group 0 KV (preserves dense GQA cache layout). Configurable: `dedicated` (extra shared K/V, 4-tuple cache) or `mean` (mean over group KVs). | "an always-on shared attention head that is computed for every token regardless of routing" | Dedicated KV; mean-pool all groups |

---

### 3. UNSPECIFIED (Our Defaults)
These design details were omitted from the paper. We adopted industry-standard pretraining defaults:

| Choice | Our Default | Rationale | Alternatives |
|--------|-------------|-----------|--------------|
| **Positional Embeddings** | Rotary Position Embeddings (RoPE) on **both** GQE and GQA | Necessary for long context (Figure 1); GQA baseline uses the same RoPE path for fair comparison. | Learned Absolute Positional Embeddings, Alibi |
| **Sparse vs dense Q projection** | Dense Q while training; sparse selected-expert Q at eval (`use_sparse_q: null`). Benchmarks force `use_sparse_q: true`. | Paper reduces active query-head compute; dense train path is numerically equivalent and simpler for autograd. | Always sparse; always dense |
| **FFN/MLP Structure** | standard GPT-style MLP with GELU | Standard for autoregressive Transformers at this scale. | SwiGLU, GLU |
| **Tokenizer** | GPT-2 BPE Tokenizer (vocab size 50,257) | Standard open-source tokenizer for English pretraining. | Custom SentencePiece, LLaMA BPE |
| **Auxiliary Loss Weight ($\alpha$)** | 0.01 | Standard MoE load-balancing weight (e.g., from Switch Transformer / GShard). | 0.05, 0.001 |
| **LayerNorm Position** | Pre-Normalization | Standard for stable pretraining of decoder-only models. | Post-Normalization |

---

## Table 2 ablation variants

Configured via `gqe.variant` in YAML (see `GQE_VARIANTS` in `src/gqe_attention.py`):

| Variant | Expert slots | Renorm weighted-sum slot | Shared head |
|---------|--------------|--------------------------|-------------|
| `full` | hard (unscaled) | yes | yes |
| `hard_only` | hard (unscaled) | no | no |
| `weighted_no_renorm` | scaled by router probs | no | no |

---

## Known Deviations

- **Embedding Parameter Tying**: We tie token embedding (`wte.weight`) and language model output projection (`lm_head.weight`) to match standard GPT pretraining practices.
- **Model Scale Option**: We introduced a 25M parameter model configuration (10% parameter scale) with a corresponding 3B token pretraining budget (10% token scale) to enable rapid local replication.
- **Sparse Q gather**: Selected-expert Q uses gathered weight rows + batched matmul (not a custom CUDA fused kernel). Correct and lower FLOPs than dense Q; wall-clock speedup depends on hardware/kernels.
