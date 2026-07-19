# src/model.py
# Full Transformer Decoder with GQE Attention

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any

from src.gqe_attention import GQEAttention, GQAAttention

class GQETransformerMLP(nn.Module):
    """[ASSUMPTION] — Standard GPT-style MLP with GELU activation and d_ff hidden dimension."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.c_fc(x))
        h = self.c_proj(h)
        return self.dropout(h)


class GQETransformerBlock(nn.Module):
    """Transformer decoder block using GQE Attention or standard GQA Attention."""
    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int,
        use_gqe: bool = True,
        top_k: int = 1,
        use_shared_head: bool = True,
        use_weighted_sum_slot: bool = True,
        scale_expert_slots: bool = False,
        variant: str | None = None,
        shared_kv_policy: str = "group_0",
        use_sparse_q: bool | None = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        if use_gqe:
            self.attn = GQEAttention(
                d_model=d_model,
                n_query_heads=n_query_heads,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
                top_k=top_k,
                use_shared_head=use_shared_head,
                use_weighted_sum_slot=use_weighted_sum_slot,
                scale_expert_slots=scale_expert_slots,
                variant=variant,
                shared_kv_policy=shared_kv_policy,
                use_sparse_q=use_sparse_q,
                dropout=dropout
            )
        else:
            self.attn = GQAAttention(
                d_model=d_model,
                n_query_heads=n_query_heads,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
                dropout=dropout
            )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GQETransformerMLP(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # Pre-normalization and self-attention
        norm_x = self.ln_1(x)
        attn_out, new_past_kv, aux_loss = self.attn(
            norm_x,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        x = x + self.dropout_1(attn_out)
        
        # Pre-normalization and MLP
        x = x + self.dropout_2(self.mlp(self.ln_2(x)))
        
        return x, new_past_kv, aux_loss


class GQETransformerLM(nn.Module):
    """GQE Decoder-Only Transformer Language Model.
    
    Includes token embedding, blocks, and language modeling head.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        model_cfg = config["model"]
        gqe_cfg = config.get("gqe", {})
        
        self.d_model = model_cfg["d_model"]
        self.vocab_size = model_cfg["vocab_size"]
        self.n_layers = model_cfg["n_layers"]
        
        # Determine architecture type
        self.use_gqe = model_cfg.get("type", "gqe") != "gqa"
        
        # Word token embedding
        self.wte = nn.Embedding(self.vocab_size, self.d_model)
        self.emb_dropout = nn.Dropout(model_cfg.get("dropout", 0.0))
        
        # Transformer Blocks
        # gqe.variant may be "full" | "hard_only" | "weighted_no_renorm" (Table 2)
        # or omit and set individual flags.
        self.blocks = nn.ModuleList([
            GQETransformerBlock(
                d_model=self.d_model,
                n_query_heads=model_cfg["n_query_heads"],
                n_kv_heads=model_cfg["n_kv_heads"],
                d_head=model_cfg["d_head"],
                d_ff=model_cfg["d_ff"],
                use_gqe=self.use_gqe,
                top_k=gqe_cfg.get("top_k", 1),
                use_shared_head=gqe_cfg.get("use_shared_head", True),
                use_weighted_sum_slot=gqe_cfg.get("use_weighted_sum_slot", True),
                scale_expert_slots=gqe_cfg.get("scale_expert_slots", False),
                variant=gqe_cfg.get("variant", None),
                shared_kv_policy=gqe_cfg.get("shared_kv_policy", "group_0"),
                use_sparse_q=None,
                dropout=model_cfg.get("dropout", 0.0)
            )
            for _ in range(self.n_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(self.d_model)
        
        # LM head (output projections)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Tie embeddings (standard weight sharing)
        self.lm_head.weight = self.wte.weight
        
        # Apply standard weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            past_key_values: Optional list of past key-value caches per layer
            use_cache: If True, returns updated past_key_values cache
            
        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
            new_past_key_values: List of updated KV caches per layer if use_cache is True
            total_aux_loss: Scalar FloatTensor representing sum of load-balancing losses
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.wte(input_ids)
        x = self.emb_dropout(x)
        
        # Forward through transformer layers
        new_past_kvs = [] if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        
        use_checkpoint = self.config.get("training", {}).get("gradient_checkpointing", False) and self.training
        
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if use_checkpoint:
                # Trade compute for memory using gradient checkpointing
                x, new_kv, layer_aux_loss = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    past_kv,
                    use_cache,
                    use_reentrant=False
                )
            else:
                x, new_kv, layer_aux_loss = block(
                    x,
                    past_key_value=past_kv,
                    use_cache=use_cache
                )
                
            if use_cache:
                new_past_kvs.append(new_kv)
                
            total_aux_loss = total_aux_loss + layer_aux_loss
            
        # Final norm & head projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, new_past_kvs, total_aux_loss
