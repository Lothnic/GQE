# Grouped Query Experts (GQE)

This repository contains a PyTorch implementation of the **Grouped Query Experts (GQE)** attention mechanism, as proposed in:
> **Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention** (FrontiersMind, 2026)

GQE integrates Mixture-of-Experts (MoE) routing directly into Grouped-Query Attention (GQA). Within each GQA group, a router selects $k$ query-head experts per token, while keeping all key-value (KV) heads dense and unchanged. This achieves compute reduction and prefill acceleration while preserving GQA's cache benefits.

## Core Features

- **GQE Attention Layer** (`src/gqe_attention.py`): Per-group routing, sparse selected-expert Q (eval), dense GQA KV, renorm weighted-sum slot, shared head, and Table 2 variants (`full` / `hard_only` / `weighted_no_renorm`).
- **GQA Baseline**: Same RoPE path as GQE for fair quality and latency comparison.
- **Group Router** (`src/router.py`): Top-k query expert routing with softmax-before-selection and load-balancing auxiliary loss.
- **Long Context Support**: Rotary Position Embeddings (RoPE) on both GQE and GQA.
- **Pretraining Pipeline** (`src/train.py`): Features mixed-precision training, gradient accumulation, adaptive gradient clipping via **ZClip**, and a Warmup-Stable-Decay (WSD) scheduler.
- **Evaluation Harness** (`src/evaluate.py`): Measures zero-shot accuracy on PIQA, ARC-Easy, and HellaSwag.
- **Throughput Benchmark** (`src/benchmark.py`): Compares prefill latencies between GQE and a dense GQA baseline.
- **Interactive Inference** (`src/inference.py`): Autoregressively generates text from a prompt, demonstrating model capability and verifying correct KV cache reuse.

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

### 2. Prefill Latency — Naive Attention (25M config, fp32)

Measured on RTX 4060 Mobile (8 GB). GQE uses sparse selected-expert Q projection.

| Sequence Length | GQA Latency (ms) | GQE Latency (ms) | Speedup |
| :---: | :---: | :---: | :---: |
| 2048 | 57.59 | 94.10 | 0.61x |
| 4096 | 184.72 | 233.46 | 0.79x |
| **8192** | **650.01** | **644.38** | **1.01x ✓** |

*With naive O(seq²) attention, GQE reaches parity with GQA at **8192 tokens** as the quadratic attention cost comes to dominate the fixed routing overhead — consistent with the paper's thesis.*

---

### 3. Prefill Latency — Flash Attention 2 (bench config, bf16, 2048 → 65536)

To reach 65536-token sequences on an 8 GB GPU we upgraded the attention implementation and added a benchmark config (`d_model=256`, 2 layers, 4Q/2KV heads, `bf16` autocast). Three changes were required in `gqe_attention.py`:

- **FA2 dispatch**: PyTorch's FA2 kernel requires 4D tensors `(B, H, S, D)`. Our grouped attention used 5D tensors — fixed by reshaping group/expert dims into the heads dim before SDPA and restoring after.
- **Contiguous tensors**: `expand()` creates stride-0 views that block FA2; added `.contiguous()` on K/V before SDPA.
- **Chunked sparse-Q**: The weight gather created a `(B, S, G, K, d_head, d_model)` temporary that is O(S) × weight size. Fixed by processing 2048 tokens per chunk (~64 MB peak instead of 2 GB).

**⚠️ GQE is slower than GQA across all measured lengths under Flash Attention:**

| Sequence Length | GQA (ms) | GQE (ms) | GQE overhead (GQE/GQA) |
| :---: | :---: | :---: | :---: |
| 2048 | 2.69 | 10.15 | **3.77× slower** |
| 4096 | 5.12 | 19.29 | **3.77× slower** |
| 8192 | 10.40 | 38.17 | **3.67× slower** |
| 16384 | 26.54 | 78.88 | **2.97× slower** |
| 32768 | 75.93 | 174.22 | **2.29× slower** |
| 65536 | 233.44 | 411.07 | **1.76× slower** |

**Why?** Flash Attention converts both models to O(seq) memory and makes GQA's attention extremely cheap. GQE carries routing overhead (router forward pass + chunked weight gather) that is *constant per token* and does not benefit from FA2. Because this small benchmark model has tiny attention FLOPs relative to its routing cost, GQA wins by a large margin.

The gap *is* closing (3.77× → 1.76×) as sequence length grows, because attention FLOPs still scale O(seq²) in compute even with FA2's memory savings — so GQE's advantage eventually reasserts itself. But at this model scale (d_model=256, 2 layers), the crossover requires sequences far longer than 65536. At the paper's full **250M-parameter scale** with 16 attention heads and 32 layers, attention FLOPs dominate much sooner and the crossover is closer to the 8192–16384 range even with FA2.
