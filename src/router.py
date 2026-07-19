# src/router.py
# §3.2 & §3.4 — Grouped Query Expert Routing and Auxiliary Loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupRouter(nn.Module):
    """§3.2 — Within-Group Query Experts Routing
    
    For each routing unit x_i (token representation) and each group g,
    the router produces scores over the M experts, applies softmax, and
    selects the top-k highest-scoring experts.
    """
    def __init__(self, d_model: int, n_groups: int, n_experts_per_group: int, top_k: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        self.n_experts_per_group = n_experts_per_group
        self.top_k = top_k
        
        # §3.2 — Router function r_g(x_i) implemented as a single linear projection for efficiency
        # Projection output dimension: G * M (number of groups * experts per group)
        self.router_weights = nn.Linear(d_model, n_groups * n_experts_per_group, bias=False)
        
        # Initialize router weights to small values to avoid early collapse
        nn.init.normal_(self.router_weights.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input token representations of shape (batch_size, seq_len, d_model)
        Returns:
            selected_experts: LongTensor of shape (batch_size, seq_len, n_groups, top_k)
                containing indices of selected experts in [0, n_experts_per_group - 1].
            router_probs: FloatTensor of shape (batch_size, seq_len, n_groups, n_experts_per_group)
                containing the softmax probabilities for all experts.
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute scores: (batch_size, seq_len, n_groups * n_experts_per_group)
        logits = self.router_weights(x)
        
        # Reshape to separate groups and experts: (batch_size, seq_len, n_groups, n_experts_per_group)
        logits = logits.view(batch_size, seq_len, self.n_groups, self.n_experts_per_group)
        
        # §3.2, Eq. 2 — Apply softmax over the group's experts before selection
        router_probs = F.softmax(logits, dim=-1)
        
        # §3.2, Eq. 3 — Select the top-k highest-scoring experts in each group
        # topk returns (values, indices) along the specified dimension
        _, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        
        return selected_experts, router_probs

def compute_auxiliary_loss(router_probs: torch.Tensor, selected_experts: torch.Tensor, top_k: int) -> torch.Tensor:
    """§3.4 — Routing Auxiliary Loss
    
    Computes a standard load-balancing auxiliary loss to prevent the router
    from collapsing onto the same expert in each group.
    
    Formula:
        For each group g:
            L_aux_g = M * sum_{m=1}^M (f_g_m * P_g_m)
        where:
            f_g_m = fraction of tokens where expert m was selected in group g
            P_g_m = mean router probability for expert m in group g
            M = number of experts per group
            
    Args:
        router_probs: FloatTensor of shape (batch_size, seq_len, n_groups, n_experts_per_group)
        selected_experts: LongTensor of shape (batch_size, seq_len, n_groups, top_k)
        top_k: number of active experts selected per group (usually 1)
        
    Returns:
        aux_loss: Scalar FloatTensor representing the load-balancing loss.
    """
    batch_size, seq_len, n_groups, n_experts = router_probs.shape
    total_tokens = batch_size * seq_len
    
    # Flatten batch and sequence dimensions for easier token-wise operations
    # Shape: (total_tokens, n_groups, n_experts)
    flat_probs = router_probs.view(total_tokens, n_groups, n_experts)
    # Shape: (total_tokens, n_groups, top_k)
    flat_selected = selected_experts.view(total_tokens, n_groups, top_k)
    
    # Calculate P_g_m: average routing probability for expert m in group g
    # Shape: (n_groups, n_experts)
    P_g = flat_probs.mean(dim=0)
    
    # Calculate f_g_m: fraction of tokens where expert m was selected in group g
    # Create one-hot indicators of selected experts
    # Shape: (total_tokens, n_groups, top_k, n_experts)
    one_hot = F.one_hot(flat_selected, num_classes=n_experts).float()
    
    # Sum over top_k dimension to get selection indicator: (total_tokens, n_groups, n_experts)
    selection_indicator = one_hot.sum(dim=2)
    
    # Calculate fraction of tokens: (n_groups, n_experts)
    f_g = selection_indicator.mean(dim=0)
    
    # Compute the auxiliary loss per group: M * sum(f_g * P_g)
    # Shape: (n_groups,)
    group_aux_loss = n_experts * torch.sum(f_g * P_g, dim=-1)
    
    # Return the average auxiliary loss across all groups
    return group_aux_loss.mean()
