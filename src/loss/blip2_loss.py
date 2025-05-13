import torch.nn.functional as F
import torch

# === Helper functions for ITC and ITM loss calculation ===
def compute_itc_loss(image_feats: torch.Tensor, text_feats: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Compute InfoNCE-based contrastive loss for image-text similarity.
    image_feats: (B, D) normalized image embeddings
    text_feats:  (B, D) normalized text embeddings
    tau:         scalar temperature (learnable or fixed)
    """
    # similarity logits
    logits_i2t = image_feats @ text_feats.T / tau.clamp(min=1e-6)
    logits_t2i = logits_i2t.T
    # targets are diagonal
    B = image_feats.size(0)
    targets = torch.arange(B, device=image_feats.device)
    # cross-entropy in both directions
    loss_i2t = F.cross_entropy(logits_i2t, targets)
    loss_t2i = F.cross_entropy(logits_t2i, targets)
    return 0.5 * (loss_i2t + loss_t2i)

def compute_itm_loss(itm_logits: torch.Tensor, itm_labels: torch.Tensor) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for ITM head.
    itm_logits: (B, 2) raw logits from itm_head
    itm_labels: (B,)   binary labels (1=match, 0=non-match)
    """
    return F.cross_entropy(itm_logits, itm_labels)