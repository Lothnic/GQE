# src/benchmark.py
# §3.5 & §4.3 — Prefill Latency Benchmarking (GQE vs GQA)

import time
import torch
import yaml
import argparse
from contextlib import nullcontext
from typing import Dict, Any, List

from src.model import GQETransformerLM

def benchmark_prefill(
    model: torch.nn.Module,
    seq_len: int,
    device: torch.device,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    use_half: bool = False,
) -> float:
    """Measures the average prefill (forward pass) latency for a given sequence length.

    Args:
        model: The model to benchmark (already moved to device).
        seq_len: Sequence length for the prefill pass.
        device: Target device.
        warmup_runs: Number of untimed warm-up iterations.
        benchmark_runs: Number of timed iterations to average.
        use_half: If True, wrap forward passes in torch.autocast(bf16) to halve
                  attention-map VRAM and allow longer sequences on limited GPUs.
    """
    model.eval()

    # batch size = 1, as is typical for prefill/decoding benchmarks
    input_ids = torch.randint(0, model.vocab_size, (1, seq_len), dtype=torch.long, device=device)

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_half else nullcontext()

    # Warmup runs to compile/cache kernels
    with torch.no_grad(), amp_ctx:
        for _ in range(warmup_runs):
            _ = model(input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # If CUDA is available, use CUDA events for maximum precision
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad(), amp_ctx:
            for _ in range(benchmark_runs):
                _ = model(input_ids)
        end_event.record()

        torch.cuda.synchronize()
        avg_latency_ms = start_event.elapsed_time(end_event) / benchmark_runs
        return avg_latency_ms
    else:
        # CPU timing fallback
        start_time = time.perf_counter()
        with torch.no_grad(), amp_ctx:
            for _ in range(benchmark_runs):
                _ = model(input_ids)
        end_time = time.perf_counter()
        avg_latency_ms = ((end_time - start_time) / benchmark_runs) * 1000.0
        return avg_latency_ms


def run_benchmark(
    config_path: str,
    max_seq_len: int = 32768,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    use_half: bool = False,
):
    """Run a GQA vs GQE prefill latency sweep.

    Args:
        config_path: Path to the YAML model config.
        max_seq_len: Highest sequence length to include in the sweep.
        warmup_runs: Untimed warm-up iterations per measurement.
        benchmark_runs: Timed iterations to average per measurement.
        use_half: Wrap forward passes in bf16 autocast. Halves the attention-map
                  memory footprint (seq² × heads × 2 bytes instead of × 4), which
                  lets an 8 GB GPU reach seq=16384–32768 without OOM.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_tag = "bf16" if use_half else "fp32"
    print(f"Benchmarking on device: {device}  |  precision: {precision_tag}")

    # Sequence lengths to sweep (paper Figure 1 x-axis)
    seq_lengths = [2048, 4096, 8192, 16384, 32768, 65536]
    seq_lengths = [s for s in seq_lengths if s <= max_seq_len]

    # Setup baseline GQA config
    import copy
    gqa_config = copy.deepcopy(config)
    gqa_config["model"]["type"] = "gqa"
    gqa_config["model"]["vocab_size"] = 32000

    gqe_config = copy.deepcopy(config)
    gqe_config["model"]["type"] = "gqe"
    gqe_config["model"]["vocab_size"] = 32000

    print("\n--- Initializing Models ---")
    gqa_model = GQETransformerLM(gqa_config).to(device)
    gqe_model = GQETransformerLM(gqe_config).to(device)
    gqa_model.eval()
    gqe_model.eval()

    print(f"Hidden dim: {config['model']['d_model']}")
    print(f"Layers:     {config['model']['n_layers']}")
    print(f"Q Heads:    {config['model']['n_query_heads']} | KV Heads: {config['model']['n_kv_heads']}")
    print("GQA: dense RoPE attention.  GQE: dense Q projection + routed attention.")

    print("\n--- Running Latency Sweep ---")
    print(f"{'Sequence Len':<15}{'GQA Latency (ms)':<20}{'GQE Latency (ms)':<20}{'Speedup (GQA/GQE)':<20}")
    print("-" * 75)

    for seq_len in seq_lengths:
        # Pre-clear VRAM before each length so memory from prior iteration does not
        # affect whether the next length OOMs.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        try:
            gqa_ms = benchmark_prefill(
                gqa_model, seq_len, device,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
                use_half=use_half,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

            gqe_ms = benchmark_prefill(
                gqe_model, seq_len, device,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
                use_half=use_half,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

            speedup = gqa_ms / gqe_ms
            print(f"{seq_len:<15}{gqa_ms:<20.2f}{gqe_ms:<20.2f}{speedup:<20.2f}x")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{seq_len:<15}{'OOM':<20}{'OOM':<20}{'-':<20}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            else:
                raise e

    print("-" * 75)
    print("Benchmark complete. Note: Speedup ratios > 1.0x indicate GQE is faster.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GQE throughput benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to model config yaml")
    parser.add_argument("--max_len", type=int, default=32768, help="Max sequence length to benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warm-up iterations")
    parser.add_argument("--runs", type=int, default=10, help="Timed iterations to average")
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help=(
            "Run forward passes under bf16 autocast. Halves attention-map VRAM "
            "(seq²×heads×2 bytes vs ×4) so an 8 GB GPU can reach seq=16384–32768 "
            "without OOM. Ratios remain valid because both models are cast equally."
        ),
    )
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config,
        max_seq_len=args.max_len,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        use_half=args.half,
    )
