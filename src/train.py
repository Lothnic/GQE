# src/train.py
# §4.1 — Full pretraining script with WSD schedule and ZClip

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import math
from typing import Dict, Any

from src.model import GQETransformerLM
from src.data import get_dataloader
from src.loss import GQELoss
from src.utils import ZClip

class WSDScheduler:
    """§4.1, Table 1 — Warmup-Stable-Decay (WSD) Learning-Rate Scheduler
    
    Linearly warms up LR, keeps it stable, then decays it (linear or cosine decay)
    based on the number of cumulative tokens processed during training.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        warmup_tokens: int,
        total_tokens: int,
        stable_ratio: float = 0.7,
        min_lr_ratio: float = 0.05
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_tokens = warmup_tokens
        self.total_tokens = total_tokens
        self.min_lr = max_lr * min_lr_ratio
        
        # Calculate tokens for each phase
        self.decay_tokens = total_tokens * (1.0 - stable_ratio - (warmup_tokens / total_tokens))
        self.stable_end_tokens = total_tokens - self.decay_tokens
        
        assert warmup_tokens < self.stable_end_tokens, "Warmup tokens exceed stable phase starting point"

    def get_lr(self, current_tokens: int) -> float:
        if current_tokens < self.warmup_tokens:
            # Linear warmup phase
            return self.max_lr * (current_tokens / max(1, self.warmup_tokens))
        elif current_tokens < self.stable_end_tokens:
            # Stable phase
            return self.max_lr
        elif current_tokens < self.total_tokens:
            # Decay phase (Cosine decay to min_lr)
            decay_progress = (current_tokens - self.stable_end_tokens) / max(1, self.decay_tokens)
            decay_progress = min(1.0, max(0.0, decay_progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        else:
            return self.min_lr

    def step(self, current_tokens: int):
        lr = self.get_lr(current_tokens)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train(config_path: str, checkpoint_dir: str, resume_path: str = None, dry_run: bool = False):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Extract configs
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer to get actual vocab size
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.get("tokenizer", "gpt2"))
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")
    
    # Override model vocab_size with tokenizer size if it is different
    model_cfg["vocab_size"] = vocab_size
    
    # Initialize model
    model = GQETransformerLM(config)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params / 1e6:.2f}M (Trainable: {trainable_params / 1e6:.2f}M)")
    
    # Optimizer configuration (§4.1, Table 1)
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.get("max_lr", 5.0e-4),
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.95)),
        eps=float(train_cfg.get("epsilon", 1e-7)),
        weight_decay=train_cfg.get("weight_decay", 0.1)
    )
    
    # Loss Function
    loss_fn = GQELoss(aux_loss_coeff=config["gqe"].get("aux_loss_coeff", 0.01))
    
    # Token calculation & gradient accumulation setup
    global_batch_tokens = train_cfg["global_batch_tokens"]
    max_seq_len = model_cfg["max_seq_len"]
    
    # Determine local batch size
    # Allow configuring device_batch_size in config, fallback to 4 on CPU / 2 on CUDA to prevent OOM
    device_batch_size = train_cfg.get("device_batch_size", 4 if device.type == "cpu" else 2)
    accum_steps = max(1, global_batch_tokens // (device_batch_size * max_seq_len))
    actual_global_batch_size = device_batch_size * accum_steps
    actual_global_batch_tokens = actual_global_batch_size * max_seq_len
    
    print(f"Local batch size: {device_batch_size}")
    print(f"Gradient accumulation steps: {accum_steps}")
    print(f"Global batch size: {actual_global_batch_size} sequences ({actual_global_batch_tokens} tokens)")
    
    # Initialize Dataloader
    dataloader = get_dataloader(config, batch_size=device_batch_size)
    
    # Scheduler
    total_tokens = train_cfg["total_tokens"]
    warmup_tokens = train_cfg["warmup_tokens"]
    scheduler = WSDScheduler(
        optimizer=optimizer,
        max_lr=train_cfg["max_lr"],
        warmup_tokens=warmup_tokens,
        total_tokens=total_tokens
    )
    
    # AMP Grad Scaler (using new torch.amp API to avoid deprecation warnings)
    scaler = torch.amp.GradScaler(device.type, enabled=(train_cfg["precision"] == "bf16" or train_cfg["precision"] == "fp16"))
    
    # ZClip initialization (§4.1)
    use_zclip = train_cfg.get("use_zclip", True)
    if use_zclip:
        zclip = ZClip(base_clip=1.0)
        print("Using ZClip adaptive gradient clipping.")
    
    # State tracking
    current_tokens = 0
    step = 0
    start_step = 0
    
    # Resume from checkpoint
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_tokens = checkpoint.get('current_tokens', 0)
        step = checkpoint.get('step', 0)
        start_step = step
        if use_zclip and 'zclip_state' in checkpoint:
            zclip.running_norm_mean = checkpoint['zclip_state']['mean']
            zclip.running_norm_std = checkpoint['zclip_state']['std']
            
    # Training Loop
    model.train()
    optimizer.zero_grad()
    
    # Warmup data iteration
    data_iter = iter(dataloader)
    
    pbar = tqdm(total=total_tokens, initial=current_tokens, desc="Tokens")
    
    while current_tokens < total_tokens:
        accum_loss = 0.0
        accum_lm_loss = 0.0
        accum_aux_loss = 0.0
        
        # Step through accumulation
        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with AMP autocast (using device-aware torch.amp.autocast to avoid warnings)
            with torch.amp.autocast(device_type=device.type, enabled=(train_cfg["precision"] == "bf16"), dtype=torch.bfloat16):
                logits, _, total_aux_loss = model(input_ids)
                loss, lm_loss, aux_loss = loss_fn(logits, labels, total_aux_loss)
                
                # Scale loss by accumulation steps
                loss = loss / accum_steps
                
            # Backward pass
            scaler.scale(loss).backward()
            
            accum_loss += loss.item() * accum_steps
            accum_lm_loss += lm_loss.item() / accum_steps
            accum_aux_loss += aux_loss.item() / accum_steps
            
        # Step optimizer
        scaler.unscale_(optimizer)
        
        # Gradient Clipping (ZClip or standard clip)
        if use_zclip:
            clip_val = zclip(model.parameters())
        else:
            clip_val = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Update tokens and step scheduler
        step_tokens = device_batch_size * accum_steps * max_seq_len
        current_tokens += step_tokens
        lr = scheduler.step(current_tokens)
        
        pbar.update(step_tokens)
        step += 1
        
        # Print progress and log metrics
        if step % 10 == 0 or current_tokens >= total_tokens:
            print(
                f"\nStep {step} | Tokens {current_tokens / 1e9:.3f}B | "
                f"Loss: {accum_loss:.4f} | LM Loss: {accum_lm_loss * accum_steps:.4f} | "
                f"Aux Loss: {accum_aux_loss * accum_steps:.4f} | LR: {lr:.2e} | Clip Limit: {clip_val:.2f}"
            )
            
        # Save periodic checkpoint
        if step % 500 == 0 or current_tokens >= total_tokens:
            ckpt_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save({
                'step': step,
                'current_tokens': current_tokens,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'zclip_state': {'mean': zclip.running_norm_mean, 'std': zclip.running_norm_std} if use_zclip else None,
                'config': config
            }, ckpt_path)
            
        # Dry run escape (used for quick testing)
        if dry_run and step >= 3:
            print("Dry run completed successfully after 3 steps.")
            break
            
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GQE Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry_run", action="store_true", help="Run 3 steps for verification")
    args = parser.parse_args()
    
    train(config_path=args.config, checkpoint_dir=args.ckpt_dir, resume_path=args.resume, dry_run=args.dry_run)
