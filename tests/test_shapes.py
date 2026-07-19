# Shape and logical verification tests for GQE Attention and Transformer Blocks

import pytest
import torch
from src.gqe_attention import GQEAttention, GQAAttention, GQE_VARIANTS
from src.model import GQETransformerBlock, GQETransformerLM


def test_gqe_attention_shapes():
    # Setup test variables (16 query heads, 8 KV heads -> 8 GQA groups, 2 experts/group)
    d_model = 1024
    n_query_heads = 16
    n_kv_heads = 8
    d_head = 64
    batch_size = 2
    seq_len = 128

    attn = GQEAttention(
        d_model=d_model,
        n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        top_k=1,
        variant="full",
        use_sparse_q=False,
    )

    assert attn.n_groups == 8
    assert attn.n_experts_per_group == 2
    # Active slots: 8 (selected experts) + 1 (weighted sum) + 1 (shared) = 10 slots
    assert attn.n_output_slots == 10
    assert attn.o_proj.weight.shape == (d_model, 10 * d_head)

    x = torch.randn(batch_size, seq_len, d_model)
    output, new_cache, aux_loss = attn(x, use_cache=True)

    assert output.shape == (batch_size, seq_len, d_model)
    assert new_cache[0].shape == (batch_size, seq_len, n_kv_heads, d_head)
    assert new_cache[1].shape == (batch_size, seq_len, n_kv_heads, d_head)
    assert aux_loss.ndim == 0


def test_gqa_attention_shapes_and_rope():
    d_model = 512
    n_query_heads = 8
    n_kv_heads = 4
    d_head = 64
    batch_size = 2
    seq_len = 64

    attn = GQAAttention(
        d_model=d_model,
        n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
    )
    # RoPE parity with GQE
    assert hasattr(attn, "rotary_emb")

    x = torch.randn(batch_size, seq_len, d_model)
    output, _, aux_loss = attn(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert aux_loss.item() == 0.0


def test_transformer_block_shapes():
    d_model = 256
    n_query_heads = 8
    n_kv_heads = 4
    d_head = 32
    d_ff = 512
    batch_size = 2
    seq_len = 32

    block = GQETransformerBlock(
        d_model=d_model,
        n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        d_ff=d_ff,
        use_gqe=True,
        top_k=1,
        variant="full",
    )

    x = torch.randn(batch_size, seq_len, d_model)
    output, new_cache, aux_loss = block(x, use_cache=True)

    assert output.shape == (batch_size, seq_len, d_model)
    assert new_cache[0].shape == (batch_size, seq_len, n_kv_heads, d_head)


def test_full_model_logits():
    config = {
        "model": {
            "d_model": 128,
            "n_layers": 2,
            "n_query_heads": 4,
            "n_kv_heads": 2,
            "d_head": 32,
            "d_ff": 256,
            "vocab_size": 1000,
            "max_seq_len": 64,
            "type": "gqe",
        },
        "gqe": {
            "top_k": 1,
            "variant": "full",
            "shared_kv_policy": "group_0",
            "use_sparse_q": False,
            "aux_loss_coeff": 0.01,
        },
    }

    model = GQETransformerLM(config)

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    logits, new_caches, total_aux_loss = model(input_ids, use_cache=True)

    assert logits.shape == (batch_size, seq_len, 1000)
    assert len(new_caches) == 2
    assert new_caches[0][0].shape == (batch_size, seq_len, 2, 32)
    assert total_aux_loss.ndim == 0


def test_table2_variants_slot_counts():
    """§4.2 Table 2 — output slot counts for each ablation variant."""
    common = dict(d_model=128, n_query_heads=8, n_kv_heads=4, d_head=16, top_k=1)

    full = GQEAttention(**common, variant="full")
    hard = GQEAttention(**common, variant="hard_only")
    weighted = GQEAttention(**common, variant="weighted_no_renorm")

    # kG=4 experts; full adds renorm + shared → 6; others → 4
    assert full.n_output_slots == 6
    assert hard.n_output_slots == 4
    assert weighted.n_output_slots == 4
    assert full.use_weighted_sum_slot and full.use_shared_head
    assert not hard.use_weighted_sum_slot and not hard.use_shared_head
    assert weighted.scale_expert_slots and not weighted.use_weighted_sum_slot

    x = torch.randn(2, 8, 128)
    for attn in (full, hard, weighted):
        y, _, aux = attn(x)
        assert y.shape == (2, 8, 128)
        assert aux.ndim == 0


def test_shared_kv_policies():
    x = torch.randn(1, 6, 64)
    for policy in ("group_0", "mean", "dedicated"):
        attn = GQEAttention(
            d_model=64,
            n_query_heads=4,
            n_kv_heads=2,
            d_head=16,
            top_k=1,
            variant="full",
            shared_kv_policy=policy,
            use_sparse_q=False,
        )
        y, cache, _ = attn(x, use_cache=True)
        assert y.shape == (1, 6, 64)
        if policy == "dedicated":
            assert len(cache) == 4
            assert cache[2].shape == (1, 6, 16)
        else:
            assert len(cache) == 2

        # decode step
        y2, cache2, _ = attn(torch.randn(1, 1, 64), past_key_value=cache, use_cache=True)
        assert y2.shape == (1, 1, 64)
        assert cache2[0].shape[1] == 7


def test_gqa_causal_mask():
    torch.manual_seed(1)
    attn = GQAAttention(d_model=64, n_query_heads=4, n_kv_heads=2, d_head=16)
    attn.eval()
    x = torch.randn(1, 5, 64)
    with torch.no_grad():
        y1, _, _ = attn(x)
        x2 = x.clone()
        x2[0, -1] += 5.0
        y2, _, _ = attn(x2)
    assert (y1[0, :-1] - y2[0, :-1]).abs().max().item() == 0.0
