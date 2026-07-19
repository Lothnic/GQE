# Grouped Query Experts (GQE): Paper Understanding & Implementation Verification

**Paper:** *Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention*  
**Authors:** Vishesh Tripathi, Abhay Kumar (FrontiersMind)  
**arXiv:** [2606.20945v2](https://arxiv.org/abs/2606.20945)  
**Local source:** `2606.20945v2-2026-07-15_18-04-26.md`  
**Repo under review:** this codebase (`src/`, `configs/`, `tests/`)

This document has two parts:

1. **Paper understanding** — what GQE is, why it exists, how the math works, and what the experiments claim.  
2. **Implementation verification** — how faithfully this repo matches the paper, what was checked empirically, and where it diverges.

---

## Part I — Detailed Paper Understanding

### 1. Problem statement

Self-attention is both the strength and the bottleneck of Transformers:

- Pairwise token interactions scale as \(O(n^2)\) in sequence length \(n\).
- Standard multi-head / grouped-query attention still **activates every query head for every token**.

The paper’s motivation is token heterogeneity: content-bearing words and long-range dependencies may need specialized heads, while punctuation and stop words often do not. Spending full query-head compute on every token is wasteful, especially as context grows.

### 2. Core idea in one sentence

**GQE = MoE routing over query heads, constrained inside GQA groups, with dense unchanged KV caches.**

More precisely:

| Component | Dense GQA | GQE |
|-----------|-----------|-----|
| KV heads | Always dense | Always dense (same cache layout) |
| Query heads | All active | Top-\(k\) experts **per GQA group** |
| Output projection \(W_O\) | \(N\) head slots | \(kG + 2\) slots (experts + weighted sum + shared) |
| Goal | KV-cache efficiency | KV efficiency **plus** sparse query compute |

### 3. Background that GQE depends on

#### 3.1 Mixture-of-Experts (MoE)

Classic MoE (Shazeer et al., GShard, Switch) routes each token to a small subset of **MLP experts**. GQE moves that conditional-computation idea into the **attention** block, not the FFN.

#### 3.2 Grouped-Query Attention (GQA)

GQA sits between MHA and MQA:

- **MHA:** every query head has its own K/V head.
- **MQA:** all query heads share one K/V head.
- **GQA:** query heads are partitioned into \(G\) groups; each group shares one K/V head.

GQA reduces KV-cache memory/bandwidth for decoding, but it still runs **all** query heads. GQE keeps GQA’s KV structure and sparsifies only the query side.

#### 3.3 Related sparse-attention work (how GQE differs)

The paper contrasts GQE with head pruning, MoA, SwitchHead/MoH, MoMHA, LLaMA-MoE v2, etc. GQE’s design constraints:

1. **Routing granularity:** top-\(k\) *inside each fixed GQA group*, not over the full head set.  
2. **KV handling:** never sparsify or restructure the KV cache.  
3. **Training:** train router + experts jointly from scratch (not post-hoc prune/convert).

### 4. Method (Section 3) — formal construction

#### 4.1 Setup (§3.1)

Input sequence \(X \in \mathbb{R}^{n \times d}\). Standard MHA:

\[
\mathrm{MHA}(X) = \mathrm{Concat}(A_1(X),\ldots,A_H(X))\,W_O
\]

GQA partitions query heads into \(G\) groups that share KV heads, but still evaluates every query head.

#### 4.2 Within-group query experts (§3.2)

Notation used throughout the paper and this repo:

| Symbol | Meaning | Main experiment |
|--------|---------|-----------------|
| \(N\) | Total routed query-head experts | 16 |
| \(G\) | Number of GQA groups (= KV heads) | 8 |
| \(M = N/G\) | Experts per group | 2 |
| \(k\) | Active experts selected per group | 1 |

Each expert \(E_{g,m}\) owns its **own query projection** and produces an attention-head output against the **shared group KV head**. KV projections stay dense.

**Routing (Eqs. 2–3):**

\[
p_{i,g} = \mathrm{softmax}\big(r_g(x_i)\big) \in \mathbb{R}^{M}
\]

\[
\mathcal{K}_g(x_i) = \mathrm{TopK}_m(p_{i,g,m},\, k)
\]

Important design choice: **softmax is applied before top-\(k\)**, so selection is based on proper per-group probabilities.

Example from the paper: 8 experts in 4 groups \(\{a_1,a_2\}\ldots\{a_7,a_8\}\); router might pick \(a_1,a_3,a_5,a_7\) (one per group).

#### 4.3 Output construction with shared head (§3.3) — the critical part

Let \(o_{i,g,m} = E_{g,m}(x_i)\) for selected experts, and \(s(x_i)\) be an **always-on shared head**.

**Slot A — hard concatenation (unscaled):**

\[
O_i = \{ o_{i,g,m} : g \in \{1..G\},\, m \in \mathcal{K}_g(x_i) \}
\]

Selected head outputs are **not** probability-scaled here; they are ordinary head slots.

**Slot B — renormalized weighted sum (router learning signal):**

Selected probabilities across groups generally do **not** sum to 1 after top-\(k\). The paper adds one extra slot:

\[
\bar{o}_i = \sum_g \sum_{m \in \mathcal{K}_g(x_i)} w_{i,g,m}\, o_{i,g,m},
\quad
w_{i,g,m} = \frac{p_{i,g,m}}{\sum_{g'}\sum_{m'\in\mathcal{K}_{g'}(x_i)} p_{i,g',m'}}
\]

This is the **differentiable path** from the LM loss back to the router. Without it, hard top-\(k\) selection is discrete and gives the router a weak learning signal.

**Slot C — always-on shared head** \(s(x_i)\): anchors the layer while routed experts specialize.

**Final output (Eq. 6):**

\[
y_i = \mathrm{Concat}(O_i,\, \bar{o}_i,\, s(x_i))\, W_O
\]

For the main setting \(N=16\), \(G=8\), \(k=1\):

- 8 hard expert slots  
- 1 weighted-sum slot  
- 1 shared slot  
- **10 slots total** before \(W_O\)

So \(W_O\) goes from \(16 d_h \times d\) (dense GQA) to \(10 d_h \times d\) (GQE). Comparisons keep training budget, data, KV layout, and head dim fixed, but are **not** exactly parameter-matched on \(W_O\).

#### 4.4 Auxiliary load-balancing loss (§3.4)

In addition to LM loss, a standard MoE-style load-balancing auxiliary loss prevents collapse onto the same expert within each group. The paper describes it qualitatively (encourage balance of selected-head usage) but does **not** give a closed-form equation; implementers typically use Switch/GShard-style:

\[
\mathcal{L}_{\mathrm{aux},g} = M \sum_{m=1}^{M} f_{g,m}\, P_{g,m}
\]

where \(f_{g,m}\) is the fraction of tokens selecting expert \(m\) in group \(g\), and \(P_{g,m}\) is mean router probability for that expert.

#### 4.5 Compute profile (§3.5)

- Routed active experts: \(kG\) out of \(N\)  
- Plus 1 shared head attention  
- Weighted-sum slot **reuses** selected expert outputs (no extra attention)

Active query-attention fraction:

\[
\frac{kG + 1}{N}
\]

Main setting: \((8+1)/16 = 9/16 \approx 56\%\).  
The informal “50%” claim refers to **routed** experts only (\(kG/N = 1/2\)), not total query-attention ops.

Speedup is not free: router overhead, dispatch efficiency, hardware utilization, sequence length, and smaller \(W_O\) all matter. Benefit grows with sequence length because attention’s quadratic term dominates fixed routing cost.

### 5. Experiments and claims (Section 4)

#### 5.1 Setup

| Item | Paper value |
|------|-------------|
| Scale | ~250M parameters |
| Tokens | 30B (fixed budget) |
| Data | FineWeb-Edu (FineWeb2 subset) |
| Attention layout | 16 query / 8 KV heads |
| \(k\) | 1 per group |
| Seq length | 2048 |
| Optimizer | Fused AdamW \(\beta_1=0.9,\beta_2=0.95,\epsilon=10^{-7}\) |
| LR schedule | WSD; max LR \(10^{-5}\)–\(5\cdot10^{-4}\) |
| Warmup | 3B tokens |
| Global batch | 1.05M tokens |
| Precision | bf16 |
| Spike mitigation | ZClip |
| Downstream | HellaSwag, PIQA, ARC-Easy |
| Throughput | Prefill latency vs context length |

#### 5.2 Accuracy (Table 2) — causal ablation story

| Variant | Avg (HS / ARC-E / PIQA) | Interpretation |
|---------|-------------------------|----------------|
| GQA baseline (all 16 heads) | 55.86 | Dense query compute |
| Weighted concat, no renorm slot | 55.18 | Weak / wrong router signal |
| Hard concat only | 55.43 | Still missing differentiable router path |
| **GQE (renorm + shared head)** | **56.04** | Matches/exceeds baseline at ~½ routed experts |

**Takeaway:** sparsity alone does **not** preserve quality. You need:

1. Renormalized weighted-sum slot for router gradients, and  
2. Always-on shared head for stability.

#### 5.3 Throughput (Figure 1)

Prefill speedup (GQA time / GQE time):

- ~\(1.15\times\) at 2k (overhead-dominated)  
- ~\(1.67\)–\(1.80\times\) from 4k upward (long-context regime)

### 6. Limitations (Section 5)

- Only validated at 250M / 30B tokens; small accuracy margin needs multi-seed / larger-scale confirmation.  
- Small per-group expert pool (\(M=2\)) limits specialization headroom.  
- Future work: larger models, larger \(M\), comparison to non-Transformer long-context architectures (e.g. Mamba).

### 7. Intuition summary

```
Token x_i
   │
   ├─► Router r_g(x_i) ──softmax──► top-k experts per GQA group
   │
   ├─► Selected query experts ──attn vs dense group KV──► hard slots O_i  (unscaled)
   │                                              └──► weighted slot ō_i (renorm p)
   │
   └─► Shared head s(x_i) ──always on──► shared slot
                                          │
                                          ▼
                         Concat(O_i, ō_i, s) W_O  →  y_i
```

GQE keeps a **pool** of attention patterns but only **pays** for the subset each token needs, without touching the dense GQA KV cache.

---

## Part II — Implementation Verification

### 1. Repository map → paper sections

| Code | Paper role |
|------|------------|
| `src/router.py` | §3.2 routing (Eq. 2–3), §3.4 aux loss |
| `src/gqe_attention.py` | §3.1–3.3 GQE layer + GQA baseline |
| `src/loss.py` | LM + scaled aux objective |
| `src/model.py` | Decoder stack wrapping GQE/GQA |
| `src/utils.py` | RoPE (assumed), ZClip (§4.1) |
| `src/train.py` | §4.1 WSD, AdamW, bf16, FineWeb training |
| `src/data.py` | FineWeb-Edu streaming packing |
| `src/evaluate.py` | HellaSwag / PIQA / ARC-Easy |
| `src/benchmark.py` | §4.3 prefill latency |
| `configs/base.yaml` | 250M / 30B paper config skeleton |
| `configs/25m_config.yaml` | Scaled-down local replica |
| `REPRODUCTION_NOTES.md` | Ambiguity audit |

### 2. Equation-by-equation checklist

| Paper claim | Implementation | Status |
|-------------|----------------|--------|
| Softmax over group experts **before** top-\(k\) (§3.2) | `GroupRouter`: `softmax` then `topk` on last dim | **Match** |
| Within-group routing only | logits reshaped to `(…, G, M)` | **Match** |
| Dense KV, one head per group | `k_proj`/`v_proj` → `(…, G, d_head)` always | **Match** |
| Each expert owns query projection | packed `q_proj` for \(N\) experts (+ shared) | **Match** (dense pack; see notes) |
| Hard concat unscaled (§3.3) | `o_flat` from selected outputs, no ×p | **Match** |
| Renormalized weighted sum (Eq. 5) | gather probs → sum over all selected → scale | **Match** |
| Shared head always-on | optional extra Q head + attention | **Match** (KV source assumed) |
| \(y = \mathrm{Concat}(O,\bar o,s) W_O\) (Eq. 6) | `torch.cat` then `o_proj` | **Match** |
| \(W_O\) slots = \(kG+2\) | for \(k=1,G=8\): **10 slots**, `in_features=640` | **Match** |
| Active fraction \((kG+1)/N = 9/16\) | structural (8 expert attns + 1 shared) | **Match** |
| Aux load-balancing (§3.4) | Switch-style \(M\sum f\cdot P\) per group | **Match in spirit** (formula not in paper) |
| LM + aux joint training | `GQELoss` | **Match** |
| GQA baseline all heads | `GQAAttention` | **Match** (missing RoPE parity) |
| Ablation knobs (Table 2) | `use_shared_head`, `use_weighted_sum_slot` | **Partial** (no pure “weighted concat without renorm” path) |
| FineWeb-Edu, AdamW, WSD, ZClip, bf16 | `train.py` / configs | **Match** (details partially specified) |

### 3. Empirical checks run for this verification

Environment: `uv pip install -r requirements.txt` into `./venv`, then:

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/ -v
```

**Result: 6/6 tests passed** (router shapes/probs, aux collapse case, GQE/GQA/block/model shapes).

Additional manual checks:

| Check | Result |
|-------|--------|
| \(G=8, M=2, N=16\), slots \(=10\), \(W_O\) in \(=640\) | Pass |
| Active fraction \((8+1)/16 = 0.5625\) | Pass |
| Router weight receives gradient with full GQE (weighted slot on) | **Yes** |
| Hard-only (no weighted slot, no shared): router gets **no** grad from output loss | **Yes** (matches paper’s “weak learning signal” claim) |
| Causal mask: perturb last token → past positions unchanged | Pass (`0.00e+00` max delta) |
| KV cache decode step lengthens cache \(3 \to 4\) | Pass |
| Softmax probabilities sum to 1 per group | Pass |
| Base config param count | GQE ≈ **231.2M**, GQA ≈ **236.2M** (paper “250M scale”) |

### 4. What is implemented correctly (high confidence)

1. **Routing semantics** — per-group softmax → top-\(k\), indices used to gather Q (and thus attention outputs).  
2. **Output algebra** — hard slots + renorm weighted sum + shared head + smaller \(W_O\).  
3. **Router learning story** — empirically, the weighted-sum path is what backprops into the router; hard routing alone does not.  
4. **KV density** — cache shape is GQA-compatible `(batch, seq, G, d_head)`.  
5. **Training stack** — WSD, AdamW betas/eps, weight decay, FineWeb streaming pack, aux coefficient, ZClip hook, bf16 autocast.  
6. **Eval / bench harnesses** — zero-shot LL scoring on the three paper tasks; prefill latency sweep structure for Figure 1 style plots.

### 5. Deviations, gaps, and risks

#### 5.1 Issues (status after fixes)

| Issue | Severity | Status |
|-------|----------|--------|
| **GQA baseline lacks RoPE; GQE has RoPE** | High | **Fixed** — `GQAAttention` uses the same `RotaryEmbedding` / `apply_rotary_pos_emb` path as GQE. |
| **Query projection is dense** | Medium | **Fixed** — sparse selected-expert Q (`_project_queries_sparse`) at eval by default; dense while training; benchmarks force `use_sparse_q: true`. Dense vs sparse numerically matched in tests. |
| **Shared head KV source unspecified** | Medium | **Fixed (configurable)** — `shared_kv_policy`: `group_0` (default) \| `dedicated` \| `mean`. |
| **Table 2 ablations not first-class** | Low–Medium | **Fixed** — `gqe.variant`: `full` \| `hard_only` \| `weighted_no_renorm`. |
| **Aux loss coefficient = 0.01** | Low | Unchanged — unspecified in paper; standard Switch-like default. |
| **Architecture dims for “250M”** | Low | Unchanged — ~231M with assumed dims. |
| **ZClip implementation** | Low–Medium | Unchanged — simplified adaptive clip, not line-audited vs ZClip paper. |

#### 5.2 Unspecified paper details filled with standards

Documented in `REPRODUCTION_NOTES.md` and configs:

- **RoPE** for long-context (needed for Figure 1 lengths).  
- **Pre-norm** decoder blocks, GELU MLP.  
- **GPT-2 BPE** tokenizer (vocab 50257).  
- **Embedding/LM-head weight tying**.  
- **WSD decay** as cosine over the final portion of training.

#### 5.3 What this repo does *not* claim to reproduce yet

- Full 30B-token / 250M training run results (Table 2 numbers).  
- Measured \(1.7\)–\(1.8\times\) long-context speedups on the paper’s hardware.  
- Multi-seed statistical significance.  
- Exact fused-kernel sparse attention dispatch used for production-scale timing.

### 6. Verdict

| Layer | Verdict |
|-------|---------|
| **Algorithmic core (§3.2–3.3)** | **Faithful.** Equations for routing, hard concat, renorm weighted sum, shared head, and \(W_O\) slot count match the paper and pass gradient/shape tests. |
| **Auxiliary loss (§3.4)** | **Reasonable standard realization** of an underspecified objective. |
| **Systems / training (§4)** | **Structurally aligned** with hyperparameters and data; several engineering choices are documented assumptions. |
| **Fair baseline & speed claims** | **Addressed in code** — GQA has RoPE parity; GQE eval/bench uses sparse selected-expert Q. Wall-clock speedups still depend on kernels/hardware. |
| **End-to-end paper numbers** | **Not verified** — architecture is ready; full reproduction would require the 30B-token train + eval + long-context bench. |

**Bottom line:** algorithmic core is faithful; previously flagged fairness/sparsity/ablation gaps are fixed in-repo. Full Table 2 / Figure 1 numerical reproduction still requires large-scale training and hardware-matched benchmarks.

### 8. How to re-run verification

```bash
# deps (project README convention)
uv pip install -r requirements.txt --python ./venv/bin/python

# unit tests
PYTHONPATH=. ./venv/bin/python -m pytest tests/ -v

# dry-run training (3 steps)
PYTHONPATH=. ./venv/bin/python src/train.py --config configs/25m_config.yaml --dry_run

# prefill latency skeleton
PYTHONPATH=. ./venv/bin/python src/benchmark.py --config configs/25m_config.yaml --max_len 4096
```

---

## Appendix A — Main-setting numbers at a glance

| Quantity | Value |
|----------|-------|
| Query experts \(N\) | 16 |
| KV / groups \(G\) | 8 |
| Experts/group \(M\) | 2 |
| Top-\(k\) | 1 |
| Hard slots | 8 |
| Weighted-sum slots | 1 |
| Shared slots | 1 |
| \(W_O\) input slots | 10 |
| Routed active fraction | \(8/16 = 50\%\) |
| Total query-attn fraction | \(9/16 \approx 56\%\) |
| Paper quality claim | GQE avg 56.04 vs GQA 55.86 @ 30B tokens |
| Paper speed claim | ~\(1.7\)–\(1.8\times\) prefill at long context |

## Appendix B — Mental model for reading the code

When reading `GQEAttention.forward`:

1. **Project everything dense** (Q experts + optional shared Q; full K/V).  
2. **Route** token → top-\(k\) expert indices per group.  
3. **Gather** only selected Q; run causal attention vs group K/V.  
4. **Build slots:** hard concat → renorm mix of those same outputs → shared-head attention.  
5. **Project** reduced concatenation through \(W_O\).  
6. **Return** output, optional KV cache, and layer aux loss for `GQELoss`.

That pipeline *is* the paper’s Section 3, with the practical caveat that step 1 is denser than an idealized sparse deployment of step 3 alone.
