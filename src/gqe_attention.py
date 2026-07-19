# src/gqe_attention.py
# §3.1, §3.2 & §3.3 — Grouped Query Experts (GQE) Attention Layer

from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.router import GroupRouter, compute_auxiliary_loss
from src.utils import RotaryEmbedding, apply_rotary_pos_emb

# Table 2 ablation presets (§4.2)
GQE_VARIANTS = {
    # Full GQE: hard expert slots + renorm weighted-sum slot + shared head
    "full": {
        "use_shared_head": True,
        "use_weighted_sum_slot": True,
        "scale_expert_slots": False,
    },
    # Hard concat only: selected expert outputs, no renorm slot, no shared head
    "hard_only": {
        "use_shared_head": False,
        "use_weighted_sum_slot": False,
        "scale_expert_slots": False,
    },
    # Weighted concat without renormalized slot: scale expert slots by router probs
    "weighted_no_renorm": {
        "use_shared_head": False,
        "use_weighted_sum_slot": False,
        "scale_expert_slots": True,
    },
}

SharedKVPolicy = Literal["group_0", "dedicated", "mean"]


class GQEAttention(nn.Module):
    """§3.1, §3.2, §3.3 — Grouped Query Experts Attention

    Within each GQA group, a router selects k query-head experts per token,
    while all key-value (KV) heads remain dense and unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        d_head: int,
        top_k: int = 1,
        use_shared_head: bool = True,
        use_weighted_sum_slot: bool = True,
        scale_expert_slots: bool = False,
        variant: Optional[str] = None,
        shared_kv_policy: SharedKVPolicy = "group_0",
        use_sparse_q: Optional[bool] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if variant is not None:
            if variant not in GQE_VARIANTS:
                raise ValueError(
                    f"Unknown GQE variant '{variant}'. "
                    f"Choose from: {list(GQE_VARIANTS)}"
                )
            preset = GQE_VARIANTS[variant]
            use_shared_head = preset["use_shared_head"]
            use_weighted_sum_slot = preset["use_weighted_sum_slot"]
            scale_expert_slots = preset["scale_expert_slots"]

        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.top_k = top_k
        self.use_shared_head = use_shared_head
        self.use_weighted_sum_slot = use_weighted_sum_slot
        self.scale_expert_slots = scale_expert_slots
        self.shared_kv_policy = shared_kv_policy
        self.variant = variant

        self.rotary_emb = RotaryEmbedding(d_head)

        self.n_groups = n_kv_heads
        assert n_query_heads % n_kv_heads == 0, "Query heads must be divisible by KV heads (groups)"
        self.n_experts_per_group = n_query_heads // n_kv_heads
        self.n_routed_experts = n_query_heads

        self.router = GroupRouter(
            d_model=d_model,
            n_groups=self.n_groups,
            n_experts_per_group=self.n_experts_per_group,
            top_k=top_k,
        )

        self.total_q_heads = self.n_routed_experts + (1 if use_shared_head else 0)
        self.q_proj = nn.Linear(d_model, self.total_q_heads * d_head, bias=False)

        self.k_proj = nn.Linear(d_model, self.n_groups * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_groups * d_head, bias=False)

        if use_shared_head and shared_kv_policy == "dedicated":
            self.k_shared_proj = nn.Linear(d_model, d_head, bias=False)
            self.v_shared_proj = nn.Linear(d_model, d_head, bias=False)
        else:
            self.k_shared_proj = None
            self.v_shared_proj = None

        self.n_output_slots = top_k * self.n_groups
        if use_weighted_sum_slot:
            self.n_output_slots += 1
        if use_shared_head:
            self.n_output_slots += 1

        self.o_proj = nn.Linear(self.n_output_slots * self.d_head, d_model, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        past_group_kv = None
        past_shared_kv = None
        if past_key_value is not None:
            if len(past_key_value) == 2:
                past_group_kv = (past_key_value[0], past_key_value[1])
            elif len(past_key_value) == 4:
                past_group_kv = (past_key_value[0], past_key_value[1])
                past_shared_kv = (past_key_value[2], past_key_value[3])
            else:
                raise ValueError(
                    f"Unexpected past_key_value length {len(past_key_value)}"
                )

        selected_experts, router_probs = self.router(x)
        aux_loss = compute_auxiliary_loss(router_probs, selected_experts, self.top_k)

        q_all = self.q_proj(x)
        q_routed = q_all[..., :self.n_routed_experts * self.d_head].view(
            batch_size, seq_len, self.n_groups, self.n_experts_per_group, self.d_head
        )
        gather_indices = selected_experts.unsqueeze(-1).expand(
            -1, -1, -1, -1, self.d_head
        )
        q_selected = torch.gather(q_routed, dim=3, index=gather_indices)
        q_shared = None
        if self.use_shared_head:
            q_shared = q_all[..., self.n_routed_experts * self.d_head:].view(
                batch_size, seq_len, self.d_head
            )

        k = self.k_proj(x).view(batch_size, seq_len, self.n_groups, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_groups, self.d_head)

        if past_group_kv is not None:
            cached_k, cached_v = past_group_kv
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        kv_seq_len = k.shape[1]

        cos, sin = self.rotary_emb(k, kv_seq_len)
        q_cos, q_sin = cos[kv_seq_len - seq_len:], sin[kv_seq_len - seq_len:]
        q_selected_rot, _ = apply_rotary_pos_emb(q_selected, q_selected, q_cos, q_sin)
        _, k_rot = apply_rotary_pos_emb(k, k, cos, sin)

        GK = self.n_groups * self.top_k
        q_t = q_selected_rot.permute(0, 2, 3, 1, 4).contiguous()
        k_t = k_rot.permute(0, 2, 1, 3).unsqueeze(2)
        v_t = v.permute(0, 2, 1, 3).unsqueeze(2)

        if self.top_k > 1:
            k_t = k_t.expand(-1, -1, self.top_k, -1, -1).contiguous()
            v_t = v_t.expand(-1, -1, self.top_k, -1, -1).contiguous()
        else:
            k_t = k_t.contiguous()
            v_t = v_t.contiguous()

        q4 = q_t.reshape(batch_size, GK, seq_len, self.d_head)
        k4 = k_t.reshape(batch_size, GK, kv_seq_len, self.d_head)
        v4 = v_t.reshape(batch_size, GK, kv_seq_len, self.d_head)

        new_shared_cache = None
        if self.use_shared_head:
            q_shared_rot, _ = apply_rotary_pos_emb(
                q_shared.unsqueeze(2), q_shared.unsqueeze(2), q_cos, q_sin
            )
            q_shared_4d = q_shared_rot.squeeze(2).unsqueeze(1)

            if self.shared_kv_policy == "group_0":
                k_shared_4d = k4[:, 0:1, :, :]
                v_shared_4d = v4[:, 0:1, :, :]
            elif self.shared_kv_policy == "mean":
                k_mean = k_rot.mean(dim=2, keepdim=True)
                v_mean = v.mean(dim=2, keepdim=True)
                k_shared_4d = k_mean.permute(0, 2, 1, 3).contiguous()
                v_shared_4d = v_mean.permute(0, 2, 1, 3).contiguous()
            elif self.shared_kv_policy == "dedicated":
                k_new = self.k_shared_proj(x).view(batch_size, seq_len, self.d_head)
                v_new = self.v_shared_proj(x).view(batch_size, seq_len, self.d_head)
                if past_shared_kv is not None:
                    pk, pv = past_shared_kv
                    k_new = torch.cat([pk, k_new], dim=1)
                    v_new = torch.cat([pv, v_new], dim=1)
                cos_s, sin_s = self.rotary_emb(k_new, k_new.shape[1])
                _, k_shared_rot = apply_rotary_pos_emb(k_new, k_new, cos_s, sin_s)
                k_shared_4d = k_shared_rot.unsqueeze(1)
                v_shared_4d = v_new.unsqueeze(1)
                new_shared_cache = (k_new, v_new) if use_cache else None

            q4 = torch.cat([q4, q_shared_4d], dim=1)
            k4 = torch.cat([k4, k_shared_4d], dim=1)
            v4 = torch.cat([v4, v_shared_4d], dim=1)

        is_causal = seq_len > 1 or past_group_kv is not None
        dropout_p = self.dropout_layer.p if self.training else 0.0
        o4 = F.scaled_dot_product_attention(
            q4, k4, v4,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        if self.use_shared_head:
            o_routed = o4[:, :GK, :, :]
            o_shared = o4[:, GK:, :, :].squeeze(1)
        else:
            o_routed = o4

        o_selected = o_routed.view(
            batch_size, self.n_groups, self.top_k, seq_len, self.d_head
        ).permute(0, 3, 1, 2, 4).contiguous()

        selected_probs = torch.gather(router_probs, dim=3, index=selected_experts)

        if self.scale_expert_slots:
            o_experts = o_selected * selected_probs.unsqueeze(-1)
        else:
            o_experts = o_selected
        o_flat = o_experts.view(
            batch_size, seq_len, self.n_groups * self.top_k * self.d_head
        )
        output_slots = [o_flat]

        if self.use_weighted_sum_slot:
            denominator = selected_probs.sum(dim=(2, 3), keepdim=True) + 1e-9
            renorm_weights = selected_probs / denominator
            o_weighted_sum = torch.sum(
                renorm_weights.unsqueeze(-1) * o_selected, dim=(2, 3)
            )
            output_slots.append(o_weighted_sum)

        if self.use_shared_head:
            output_slots.append(o_shared)

        o_final = torch.cat(output_slots, dim=-1)
        output = self.o_proj(o_final)

        new_past = None
        if use_cache:
            if (
                self.use_shared_head
                and self.shared_kv_policy == "dedicated"
                and new_shared_cache is not None
            ):
                new_past = (k, v, new_shared_cache[0], new_shared_cache[1])
            else:
                new_past = (k, v)

        return output, new_past, aux_loss


class GQAAttention(nn.Module):
    """§2.2 — Standard Grouped-Query Attention (GQA) Baseline

    Evaluates all query heads for every token without routing.
    Shares one KV head within each GQA query group.
    Uses the same RoPE path as GQE for fair comparison.
    """

    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        d_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head

        self.n_groups = n_kv_heads
        assert n_query_heads % n_kv_heads == 0, "Query heads must be divisible by KV heads"
        self.group_size = n_query_heads // n_kv_heads

        self.rotary_emb = RotaryEmbedding(d_head)

        self.q_proj = nn.Linear(d_model, n_query_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_query_heads * d_head, d_model, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(
            batch_size, seq_len, self.n_groups, self.group_size, self.d_head
        )
        k = self.k_proj(x).view(batch_size, seq_len, self.n_groups, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_groups, self.d_head)

        if past_key_value is not None:
            cached_k, cached_v = past_key_value
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        new_past_key_value = (k, v) if use_cache else None
        kv_seq_len = k.shape[1]

        # RoPE parity with GQE
        cos, sin = self.rotary_emb(k, kv_seq_len)
        q_cos, q_sin = cos[kv_seq_len - seq_len :], sin[kv_seq_len - seq_len :]
        q_rot, _ = apply_rotary_pos_emb(q, q, q_cos, q_sin)
        _, k_rot = apply_rotary_pos_emb(k, k, cos, sin)

        # Flash Attention via F.scaled_dot_product_attention — avoids materialising
        # the full O(seq²) attention map, so long-context runs fit in 8 GB VRAM.
        # FA2 requires exactly 4D tensors; reshape (B, G, group_size, S, d_h) →
        # (B, G*group_size, S, d_h), run SDPA, then restore.
        q_t = q_rot.permute(0, 2, 3, 1, 4).contiguous()  # (B, G, group_size, S, d_h)
        k_t = k_rot.permute(0, 2, 1, 3).unsqueeze(2).expand(
            -1, -1, self.group_size, -1, -1
        ).contiguous()  # (B, G, group_size, T, d_h)
        v_t = v.permute(0, 2, 1, 3).unsqueeze(2).expand(
            -1, -1, self.group_size, -1, -1
        ).contiguous()  # (B, G, group_size, T, d_h)

        H = self.n_groups * self.group_size  # = n_query_heads
        q4 = q_t.reshape(batch_size, H, seq_len, self.d_head)
        k4 = k_t.reshape(batch_size, H, kv_seq_len, self.d_head)
        v4 = v_t.reshape(batch_size, H, kv_seq_len, self.d_head)

        is_causal = seq_len > 1 or past_key_value is not None
        dropout_p = self.dropout_layer.p if self.training else 0.0
        out_t = F.scaled_dot_product_attention(
            q4, k4, v4,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )  # (B, H, S, d_h)
        out = out_t.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_len, self.n_query_heads * self.d_head
        )
        output = self.o_proj(out)

        aux_loss = torch.tensor(0.0, device=x.device)
        return output, new_past_key_value, aux_loss
