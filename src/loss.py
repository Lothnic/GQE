# src/loss.py
# §3.4 — GQE Loss Calculation (LM Loss + Load-Balancing Aux Loss)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class GQELoss(nn.Module):
    """§3.4 — GQE Joint Loss Function
    
    Combines standard Autoregressive Cross-Entropy Language Modeling Loss
    with the Load-Balancing routing auxiliary loss.
    """
    def __init__(self, aux_loss_coeff: float = 0.01):
        super().__init__()
        self.aux_loss_coeff = aux_loss_coeff
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, total_aux_loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
            labels: LongTensor of shape (batch_size, seq_len)
            total_aux_loss: Scalar FloatTensor sum of layer-wise auxiliary losses
            
        Returns:
            total_loss: Combined loss (LM Loss + aux_loss_coeff * aux_loss)
            lm_loss: Autoregressive cross-entropy loss
            aux_loss: Scaled auxiliary loss
        """
        # Flat shapes for cross entropy
        # logits: (batch_size * seq_len, vocab_size)
        # labels: (batch_size * seq_len)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        
        lm_loss = self.cross_entropy(flat_logits, flat_labels)
        
        # Scale the auxiliary load-balancing loss
        scaled_aux_loss = self.aux_loss_coeff * total_aux_loss
        
        # §3.4 — Total loss determines the routing behavior together with the LM loss
        total_loss = lm_loss + scaled_aux_loss
        
        return total_loss, lm_loss, scaled_aux_loss
