"""
losses.py
---------
Loss functions for asymmetric DDI prediction.

  FocalLoss               — down-weights common classes (type 89 = 54.9%)
  asymmetry_regularization — penalizes symmetric predictions on directed pairs
  combined_loss           — focal + λ·asymmetry_reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for extreme class imbalance in multi-class classification.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    With γ=2 and inverse-frequency alpha:
    - A correctly-classified type-89 pair (p≈0.95) contributes ~400× less loss
      than a misclassified rare-type pair.
    - The model cannot profitably ignore rare classes.

    Parameters
    ----------
    alpha : FloatTensor (num_classes,) or None
        Per-class weights. If None, uniform (no reweighting).
        Recommended: compute_class_weights() from data_loader.py.
    gamma : float
        Focusing exponent. γ=0 recovers weighted CE. γ=2 is standard.
    num_classes : int
    """

    def __init__(self, alpha=None, gamma: float = 2.0, num_classes: int = 95):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

        if alpha is None:
            self.register_buffer('alpha', torch.ones(num_classes))
        else:
            alpha = alpha.float()
            self.register_buffer('alpha', alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C) raw unnormalized scores — NO Sigmoid/Softmax before this
        targets : (B,)   integer class indices
        returns : scalar loss
        """
        # Standard cross-entropy per sample (no reduction)
        ce = F.cross_entropy(logits, targets, reduction='none')   # (B,)

        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce)                                        # (B,)

        # Per-class alpha weight for each sample
        alpha_t = self.alpha[targets]                              # (B,)

        focal = alpha_t * (1.0 - pt) ** self.gamma * ce           # (B,)
        return focal.mean()


def asymmetry_regularization(
    decoder,
    h_perp: torch.Tensor,
    h_vuln: torch.Tensor,
    lambda_asym: float = 0.1,
    margin: float = 0.5,
) -> torch.Tensor:
    """Penalize when the model assigns similar probabilities to (A→B) and (B→A).

    For a 100% unidirectional dataset every pair carries directional signal.
    A symmetric model (predicting the same distribution in both directions)
    is wasteful — this regularizer pushes against it.

    Strategy: compute KL(P(A→B) || P(B→A)) and penalize if KL < margin.
    We want KL to be *large* (distinct predictions), so:
        reg = relu(margin - KL)

    Parameters
    ----------
    decoder : CliffordDDIDecoder (or any module with forward(h_a, h_b))
    h_perp  : (B, D) perpetrator embeddings
    h_vuln  : (B, D) victim embeddings
    lambda_asym : regularization coefficient
    margin  : minimum desired KL divergence (default 0.5 nats)

    Returns
    -------
    scalar regularization loss
    """
    logits_fwd = decoder(h_perp, h_vuln)            # (B, C)
    logits_rev = decoder(h_vuln, h_perp)            # (B, C)

    log_p_fwd = F.log_softmax(logits_fwd, dim=-1)   # (B, C)
    p_rev     = F.softmax(logits_rev,    dim=-1)    # (B, C)

    # KL(P_fwd || P_rev) = Σ P_fwd * log(P_fwd / P_rev)
    kl = F.kl_div(log_p_fwd, p_rev, reduction='batchmean')

    # Penalize when KL is too small (model is being symmetric)
    return lambda_asym * F.relu(margin - kl)


def combined_loss(
    focal_fn: FocalLoss,
    logits: torch.Tensor,
    targets: torch.Tensor,
    decoder=None,
    h_perp: torch.Tensor = None,
    h_vuln: torch.Tensor = None,
    lambda_asym: float = 0.1,
    use_asym_reg: bool = True,
) -> tuple:
    """Compute focal loss + optional asymmetry regularization.

    Returns (total_loss, focal_loss_value, asym_reg_value).
    """
    fl = focal_fn(logits, targets)

    if use_asym_reg and decoder is not None:
        ar = asymmetry_regularization(decoder, h_perp, h_vuln, lambda_asym)
    else:
        ar = torch.tensor(0.0, device=logits.device)

    return fl + ar, fl, ar
