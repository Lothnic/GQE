# src/utils.py
# Helper utilities: Rotary Position Embeddings (RoPE) and ZClip

import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    """§4.1 [ASSUMPTION] — Rotary Position Embeddings (RoPE) for long-context capability.
    
    Provides rotary position embeddings to queries and keys.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Calculate inverse frequency
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(max_position_embeddings, device=torch.device("cpu"))

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # Outer product to get frequencies per position
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate frequencies to match full head dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple:
        # Check if we need to expand the cache
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(max_seq_len := max(seq_len, self.max_seq_len_cached * 2), device=x.device)
            
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half of the hidden dimension for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """Applies RoPE to query and key tensors.
    
    Args:
        q: Query tensor, shape (batch, seq, ...)
        k: Key tensor, shape (batch, seq, ...)
        cos: Cosine tensor, shape (seq, d_head)
        sin: Sine tensor, shape (seq, d_head)
    """
    # Reshape cos/sin to align with query/key shapes: (1, seq, 1, ..., d_head)
    # Match the query/key shape dimensions by adding unsqueezed dimensions
    cos = cos.view(1, cos.shape[0], *([1] * (q.ndim - 3)), cos.shape[1])
    sin = sin.view(1, sin.shape[0], *([1] * (q.ndim - 3)), sin.shape[1])
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ZClip:
    """§4.1 — ZClip: Adaptive Spike Mitigation for LLM Pre-Training
    
    Ref: arXiv:2504.02507
    Mitigates gradient spikes dynamically during training.
    """
    def __init__(self, base_clip: float = 1.0, beta: float = 0.999):
        self.base_clip = base_clip
        self.beta = beta
        self.running_norm_mean = None
        self.running_norm_std = None

    def __call__(self, parameters) -> float:
        """Computes grad norm and clips gradients adaptively."""
        # Convert generator to list to avoid partial/empty generator consumption
        param_list = list(parameters)
        if not param_list:
            return 0.0
            
        # Calculate standard global grad norm
        total_norm = torch.nn.utils.clip_grad_norm_(param_list, max_norm=float('inf'))
        
        # Track statistics
        if self.running_norm_mean is None:
            self.running_norm_mean = total_norm.item()
            self.running_norm_std = 0.1 * total_norm.item()
        else:
            # Running updates
            delta = total_norm.item() - self.running_norm_mean
            self.running_norm_mean += (1 - self.beta) * delta
            self.running_norm_std = math.sqrt(
                self.beta * (self.running_norm_std ** 2) + (1 - self.beta) * (delta ** 2)
            )
            
        # Determine adaptive threshold (e.g. mean + 3 * std)
        # Limit the clip threshold to prevent massive sudden spike propagation
        threshold = min(self.base_clip, self.running_norm_mean + 3 * self.running_norm_std)
        
        # Apply the clip
        torch.nn.utils.clip_grad_norm_(param_list, max_norm=threshold)
        return threshold
