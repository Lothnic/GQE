# tests/test_router.py
# Verification of router output correctness and load-balancing auxiliary loss

import pytest
import torch
import torch.nn.functional as F
from src.router import GroupRouter, compute_auxiliary_loss

def test_router_outputs():
    d_model = 128
    n_groups = 4
    n_experts_per_group = 3
    top_k = 1
    batch_size = 2
    seq_len = 16
    
    router = GroupRouter(
        d_model=d_model,
        n_groups=n_groups,
        n_experts_per_group=n_experts_per_group,
        top_k=top_k
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    selected_experts, router_probs = router(x)
    
    # Selected experts should be long indices
    assert selected_experts.dtype == torch.long
    assert selected_experts.shape == (batch_size, seq_len, n_groups, top_k)
    # Check that indices are valid [0, M-1]
    assert torch.all(selected_experts >= 0)
    assert torch.all(selected_experts < n_experts_per_group)
    
    # Router probabilities shape
    assert router_probs.shape == (batch_size, seq_len, n_groups, n_experts_per_group)
    # Check that probabilities sum to 1.0 along the expert dimension (softmax verification)
    probs_sum = router_probs.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-6)

def test_auxiliary_loss_computation():
    batch_size = 2
    seq_len = 10
    n_groups = 2
    n_experts = 4
    top_k = 1
    
    # Toy probabilities (all tokens route to expert 0)
    probs = torch.zeros(batch_size, seq_len, n_groups, n_experts)
    probs[..., 0] = 1.0 # collapse case
    
    selected = torch.zeros(batch_size, seq_len, n_groups, top_k, dtype=torch.long)
    
    loss = compute_auxiliary_loss(probs, selected, top_k)
    # With perfect collapse (all tokens to expert 0):
    # f_g_0 = 1.0, P_g_0 = 1.0. Other experts are 0.
    # L_aux_g = M * (1.0 * 1.0) = M = 4.0
    assert torch.allclose(loss, torch.tensor(float(n_experts)), atol=1e-5)
    
    # Differentiability check
    # Require grad on logits (leaf tensor) to simulate backprop
    logits = torch.randn(batch_size, seq_len, n_groups, n_experts, requires_grad=True)
    probs_with_grad = torch.softmax(logits, dim=-1)
    selected_mock = torch.randint(0, n_experts, (batch_size, seq_len, n_groups, top_k))
    
    loss_val = compute_auxiliary_loss(probs_with_grad, selected_mock, top_k)
    loss_val.backward()
    
    # Gradients should have flowed back to the leaf tensor (logits)
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)
