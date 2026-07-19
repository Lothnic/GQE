"""
modal_train.py — GQE pretraining on Modal A100 GPU.

Usage:
    # Dry-run (smoke test, 3 steps):
    modal run modal_train.py --config configs/25m_config.yaml --dry-run

    # Full 25M training run:
    modal run modal_train.py --config configs/25m_config.yaml

    # Full 250M base training run:
    modal run modal_train.py --config configs/base.yaml

    # Resume from a checkpoint stored in the Modal volume:
    modal run modal_train.py --config configs/25m_config.yaml --resume checkpoints/model_step_500.pt

Checkpoints are written to and read from a persistent Modal Volume mounted at /vol.
"""

import os
import modal

# ---------------------------------------------------------------------------
# Image — install all dependencies on top of an official CUDA image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML stack
        "torch>=2.3.0",
        "torchvision",
        # HuggingFace stack (needed for FineWeb streaming + tokenizer)
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "huggingface_hub>=0.23.0",
        # Misc utilities
        "pyyaml",
        "tqdm",
        "numpy",
        "fsspec>=2023.5.0",
    )
    # Make the local src/ package and configs/ available inside the container
    # NOTE: add_local_* must come last — no build steps allowed after them
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)

# ---------------------------------------------------------------------------
# Persistent volume — checkpoints survive across runs
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("gqe-checkpoints", create_if_missing=True)
VOLUME_MOUNT = "/vol"
CKPT_DIR_IN_VOL = f"{VOLUME_MOUNT}/checkpoints"

# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------
app = modal.App("gqe-training", image=image)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    gpu="L4",                            # 24 GB L4 — free-tier accessible, ~$0.80/hr
    timeout=60 * 60 * 12,               # 12 h max (Modal hard cap for a single run)
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("huggingface-token")],  # HF_TOKEN env var
    # Keep the container warm between retries so we don't re-download data
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=10.0),
)
def train_on_modal(
    config_path: str = "configs/25m_config.yaml",
    resume_path: str | None = None,
    dry_run: bool = False,
):
    """Run GQE pretraining on a Modal A100.

    Args:
        config_path:  Path to a YAML config file (relative to /root inside the container).
        resume_path:  Optional path to a checkpoint stored in the Modal Volume,
                      e.g. "checkpoints/model_step_500.pt".  If provided, the path
                      is resolved relative to VOLUME_MOUNT (/vol).
        dry_run:      If True, stop after 3 optimizer steps (for smoke-testing).
    """
    import sys
    # Make /root the package root so `from src.xxx import ...` works
    sys.path.insert(0, "/root")

    # Hugging Face auth (needed to stream FineWeb-Edu)
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("[modal] Logged in to Hugging Face Hub.")
    else:
        print("[modal] WARNING: HF_TOKEN not set. Streaming may fail for gated datasets.")

    # Resolve resume path against the volume mount
    resolved_resume: str | None = None
    if resume_path:
        resolved_resume = os.path.join(VOLUME_MOUNT, resume_path)
        if not os.path.exists(resolved_resume):
            print(f"[modal] WARNING: Resume checkpoint not found at {resolved_resume}, starting fresh.")
            resolved_resume = None
        else:
            print(f"[modal] Resuming from {resolved_resume}")

    # Checkpoint directory lives inside the persistent volume
    os.makedirs(CKPT_DIR_IN_VOL, exist_ok=True)

    # Import and run the local train() function
    # We monkey-patch the checkpoint_dir to point at the volume.
    from src.train import train

    train(
        config_path=config_path,
        checkpoint_dir=CKPT_DIR_IN_VOL,
        resume_path=resolved_resume,
        dry_run=dry_run,
    )

    # Commit volume writes so they're visible to future runs
    volume.commit()
    print("[modal] Training complete. Checkpoints committed to volume.")


# ---------------------------------------------------------------------------
# Local entrypoint — thin CLI wrapper
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    config: str = "configs/25m_config.yaml",
    resume: str = "",
    dry_run: bool = False,
):
    """
    Args:
        --config    Path to YAML config (default: configs/25m_config.yaml)
        --resume    Checkpoint path relative to the Modal volume root, e.g.
                    checkpoints/model_step_500.pt  (leave empty to start fresh)
        --dry-run   Run only 3 steps for a quick smoke test
    """
    print(f"[local] Dispatching GQE training to Modal A100")
    print(f"[local]   config   : {config}")
    print(f"[local]   resume   : {resume or '(none)'}")
    print(f"[local]   dry_run  : {dry_run}")

    f_call = train_on_modal.spawn(
        config_path=config,
        resume_path=resume if resume else None,
        dry_run=dry_run,
    )
    print(f"[local] Spawned remote function call. ID: {f_call.object_id}")
    print(f"[local] You can view logs using: modal app logs ap-xxxxx or on the dashboard.")

