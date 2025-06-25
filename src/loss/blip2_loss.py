import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np
import logging

class MultiModalITLoss(nn.Module):
    """Combined ITC / ITG / ITM loss class.

    Args:
        temperature (float): Softmax temperature for ITC.
        itc_weight (float): Weight of ITC term.
        itg_weight (float): Weight of ITG term.
        itm_weight (float): Weight of ITM term.
        vocab_size (int, optional): Needed for ITG cross‑entropy.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        itc_weight: float = 1.0,
        itg_weight: float = 1.0,
        itm_weight: float = 1.0,
        vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.itc_weight = itc_weight
        self.itg_weight = itg_weight
        self.itm_weight = itm_weight
        self.ce = CrossEntropyLoss(reduction="mean") if vocab_size is not None else None
        # simple linear head for ITM (matching vs. non‑matching)
        self.itm_head = nn.Linear(vocab_size if vocab_size else 768, 2)

    def forward(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        lm_logits: Optional[torch.Tensor] = None,
        lm_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss.
        img_feats, txt_feats: (B, D) normalized feature vectors.
        lm_logits, lm_labels: for ITG.
        """
        # ITC (InfoNCE)
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        logits_per_img = img_feats @ txt_feats.t() / self.temperature
        logits_per_txt = logits_per_img.t()
        targets = torch.arange(img_feats.size(0), device=img_feats.device)
        itc_loss = (self.ce(logits_per_img, targets) + self.ce(logits_per_txt, targets)) / 2

        # ITM  – positive pairs are aligned indices
        concat_feats = torch.cat([img_feats, txt_feats], dim=-1)  # (B, 2D)
        itm_logits = self.itm_head(concat_feats)  # (B, 2)
        itm_labels = torch.ones(img_feats.size(0), dtype=torch.long, device=img_feats.device)
        itm_loss = self.ce(itm_logits, itm_labels)

        # ITG – language modelling cross‑entropy
        if lm_logits is not None and lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            itg_loss = self.ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            itg_loss = torch.tensor(0.0, device=img_feats.device)

        total = self.itc_weight * itc_loss + self.itm_weight * itm_loss + self.itg_weight * itg_loss
        return total, {"itc": itc_loss, "itm": itm_loss, "itg": itg_loss}
