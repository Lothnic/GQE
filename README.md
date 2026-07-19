# Grouped Query Experts (GQE)

This repository contains a PyTorch implementation of the **Grouped Query Experts (GQE)** attention mechanism, as proposed in:
> **Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention** (FrontiersMind, 2026)

GQE integrates Mixture-of-Experts (MoE) routing directly into Grouped-Query Attention (GQA). Within each GQA group, a router selects $k$ query-head experts per token, while keeping all key-value (KV) heads dense and unchanged. This achieves compute reduction and prefill acceleration while preserving GQA's cache benefits.

## Core Features

- **GQE Attention Layer** (`src/gqe_attention.py`): Per-group routing, dense Q projection with index gather, fused single-SDPA shared head, renorm weighted-sum slot, and Table 2 variants (`full` / `hard_only` / `weighted_no_renorm`).
- **GQA Baseline**: Same RoPE path as GQE for fair quality and latency comparison.
- **Group Router** (`src/router.py`): Top-k query expert routing with softmax-before-selection and load-balancing auxiliary loss.
- **Long Context Support**: Rotary Position Embeddings (RoPE) on both GQE and GQA.
- **Pretraining Pipeline** (`src/train.py`): Features mixed-precision training, gradient accumulation, adaptive gradient clipping via **ZClip**, and a Warmup-Stable-Decay (WSD) scheduler.
- **Evaluation Harness** (`src/evaluate.py`): Measures zero-shot accuracy on PIQA, ARC-Easy, and HellaSwag.
- **Throughput Benchmark** (`src/benchmark.py`): Compares prefill latencies between GQE and a dense GQA baseline.
- **Interactive Inference** (`src/inference.py`): Autoregressively generates text from a prompt, demonstrating model capability and verifying correct KV cache reuse.

---

## Empirical Comparison: GQE vs GQA (25M Model Scale)

Both GQE and GQA were trained under identical pretraining budgets (300M tokens on FineWeb-Edu) and evaluated on quality and prefill latency.

### 1. Zero-Shot Downstream Accuracy (100 validation samples)
| Task | GQE Accuracy (%) | GQA Baseline Accuracy (%) | GQE Score Count | GQA Score Count |
| :--- | :---: | :---: | :---: | :---: |
| **PIQA** | 53.00% | 58.00% | 53 / 100 | 58 / 100 |
| **ARC-Easy** | 27.00% | 27.00% | 27 / 100 | 27 / 100 |
| **HellaSwag** | 34.00% | 31.00% | 34 / 100 | 31 / 100 |
| **Average** | **38.00%** | **38.67%** | — | — |

*GQE retains competitive downstream performance compared to the fully-dense GQA baseline (within 0.67% average accuracy) while saving active query-head compute.*

---

### 2. Prefill Latency (25M config, 16K sequence length)

Measured on RTX 4060 Mobile (8 GB). GQE uses dense Q projection with `torch.gather` index selection (not per-expert weight-row gather), and the shared head is fused into a single `F.scaled_dot_product_attention` call alongside the routed heads.

| Precision | GQA (ms) | GQE (ms) | Speedup |
| :---: | :---: | :---: | :---: |
| fp32 | — | — | **1.12x** |
| bf16 | — | — | **1.16x** |

*Earlier passes using per-expert weight-row gathering and a separate SDPA call for the shared head were substantially slower (~0.4x at 16K). Switching to a dense Q projection with index gather on the result, and fusing the shared head into the routed heads' SDPA call, recovered positive speedup.*

---

### 3. Gap vs Paper Claim

The paper reports **1.7–1.8× prefill speedup** at long context lengths (Figure 1). My best results on the 25M-parameter scale config reach ~1.16× at 16K with bf16, which is still short of that target which might be due to the smaller model size.

**Likely factors for the gap:**

| Factor | Impact |
|--------|--------|
| **Model scale** | Paper reports at 250M params (32 layers, d_model=1024). Our 25M config (6 layers, d_model=512) has far lower attention FLOPs, so routing overhead is a larger fraction of total time. At 250M, the crossover should occur at shorter sequence lengths. |
| **Kernel efficiency** | The paper likely uses custom fused CUDA kernels for sparse expert dispatch, whereas we use stock PyTorch `torch.gather` + `F.scaled_dot_product_attention`. A fused kernel that skips the full Q projection entirely (projecting only selected expert rows) would reduce both compute and memory traffic. |
| **Router overhead** | The router is a full `(B, S, d_model) × (d_model, G*M)` matmul per layer. This is O(d_model²) FLOPs per token — non-trivial at small scales. Fusing the router into the Q projection or using a cheaper router might help. |
| **Output projection** | GQE's W_O projects (kG+2) = 10 slots vs GQA's 16. This is already smaller, but the paper accounts for this — it's a genuine saving at any scale. |

---

## Code Structure

```
├── configs/
│   ├── base.yaml               # 250M parameter base configuration
│   ├── 25m_config.yaml         # 25M parameter 10% scale GQE model configuration
│   ├── 25m_gqa.yaml            # 25M parameter 10% scale GQA baseline configuration
│   └── bench_longctx_config.yaml # Tiny benchmark config for long-context sweeps (≤65536)
├── src/
│   ├── model.py                # Decoder-only Transformer model
│   ├── gqe_attention.py        # GQE & GQA attention layers
│   ├── router.py               # GroupRouter & auxiliary loss
│   ├── loss.py                 # Joint LM + routing loss
│   ├── data.py                 # FineWeb-Edu dataset stream & packing
│   ├── train.py                # Pretraining pipeline
│   ├── evaluate.py             # PIQA / ARC-E / HellaSwag zero-shot evaluation
│   ├── benchmark.py            # Latency benchmark
│   ├── inference.py            # Autoregressive text generation
│   └── utils.py                # RoPE and ZClip utilities
└── tests/                      # Shape and routing verification tests
```

---

## Installation

Ensure you have Python 3.10+ and `uv` installed, then run:

```bash
uv pip install -r requirements.txt
```

Verify the installation by running the test suite:

```bash
PYTHONPATH=. pytest tests/ -v
```

---

## How to Run

### 1. Pretraining (25M Model)
To train a 25M parameter GQE model on FineWeb-Edu:

```bash
PYTHONPATH=. python3 src/train.py --config configs/25m_config.yaml --ckpt_dir checkpoints/25m
```

To train the dense GQA baseline under the exact same pretraining setup:

```bash
PYTHONPATH=. python3 src/train.py --config configs/25m_gqa.yaml --ckpt_dir checkpoints/25m_gqa
```

For a quick test/dry run (runs 3 steps and exits):
```bash
PYTHONPATH=. python3 src/train.py --config configs/25m_config.yaml --dry_run
```

To run on Modal (e.g. detached pretraining):
```bash
# GQE model
modal run --detach modal_train.py --config configs/25m_config.yaml

# GQA baseline
modal run --detach modal_train.py --config configs/25m_gqa.yaml
```

### 2. Throughput Benchmarking
Compare prefill latencies between GQE and standard GQA baseline across different context lengths:

```bash
# Standard sweep (25M config, naive attention, up to 16384)
PYTHONPATH=. python3 src/benchmark.py --config configs/25m_config.yaml --max_len 16384

# Long-context sweep with Flash Attention 2 + bf16 (reaches 65536 on 8 GB VRAM)
PYTHONPATH=. python3 src/benchmark.py --config configs/bench_longctx_config.yaml --max_len 65536 --half
```

Additional flags:
- `--half` — wraps forward passes in `bf16` autocast; enables FA2 and halves attention-map VRAM
- `--warmup N` — number of untimed warm-up iterations (default 3)
- `--runs N` — number of timed iterations to average (default 10)

### 3. Evaluation
Evaluate your pretrained checkpoint on zero-shot validation benchmarks (PIQA, ARC-Easy, HellaSwag):

```bash
# GQE model evaluation
PYTHONPATH=. python3 src/evaluate.py --config configs/25m_config.yaml --checkpoint checkpoints/25m/model_step_573.pt --max_samples 100

# GQA baseline evaluation
PYTHONPATH=. python3 src/evaluate.py --config configs/25m_gqa.yaml --checkpoint checkpoints/25m_gqa/model_step_573.pt --max_samples 100
```

### 4. Text Generation (Inference)
Run text generation from a custom prompt (falls back to random weights if no checkpoint is supplied to verify structure):

```bash
PYTHONPATH=. python3 src/inference.py --config configs/25m_config.yaml --prompt "Deep learning is" --max_tokens 30
```

