"""
losses.py
---------
Loss functions for asymmetric DDI prediction.

v2: Switched from FocalLoss (over-corrected, killed type-89) to standard
    CrossEntropyLoss with label_smoothing=0.1.
v3: Fixed asymmetry regularization — replaced ReLU-margin (collapsed at epoch 3)
    with continuous -λ·KL that always provides gradient pressure.

  FocalLoss               — kept for ablation, default γ lowered to 0.5
  asymmetry_regularization — continuous KL maximization (always active)
  combined_loss           — CE + λ·asymmetry_reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for extreme class imbalance in multi-class classification.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    WARNING: With γ=2 + inverse-freq weights + WeightedRandomSampler, this
    completely suppressed type-89 (54.9% of data) in v1. Use γ≤0.5 or
    switch to standard CE with label smoothing.

    Parameters
    ----------
    alpha : FloatTensor (num_classes,) or None
    gamma : float (default 0.5, lowered from 2.0)
    num_classes : int
    """

    def __init__(self, alpha=None, gamma: float = 0.5, num_classes: int = 95):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

        if alpha is None:
            self.register_buffer('alpha', torch.ones(num_classes))
        else:
            alpha = alpha.float()
            self.register_buffer('alpha', alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        focal = alpha_t * (1.0 - pt) ** self.gamma * ce
        return focal.mean()


def asymmetry_regularization(
    decoder,
    h_perp: torch.Tensor,
    h_vuln: torch.Tensor,
    lambda_asym: float = 0.1,
) -> torch.Tensor:
    """Encourage the model to produce different predictions for (A→B) vs (B→A).

    For a 100% unidirectional dataset every pair carries directional signal.
    A symmetric model (predicting the same distribution in both directions)
    is wasteful — this regularizer pushes against it.

    v3 strategy: continuous bounded reward -λ·KL/(1+KL) that ALWAYS provides gradient.
    The old v2 approach used relu(margin - KL) which hit zero gradient by
    epoch 3 (margin=0.5 nats is trivially exceeded with 95 classes).

    Now: reg = -λ · KL/(1 + KL)
    This is always negative (reduces total loss when KL increases),
    providing continuous pressure to maximize asymmetry. The KL/(1+KL)
    form saturates at -λ, preventing the reward from exploding at large KL.

    Parameters
    ----------
    decoder : CliffordDDIDecoder (or any module with forward(h_a, h_b))
    h_perp  : (B, D) perpetrator embeddings
    h_vuln  : (B, D) victim embeddings
    lambda_asym : regularization coefficient

    Returns
    -------
    scalar regularization loss (negative = reward for asymmetry)
    """
    logits_fwd = decoder(h_perp, h_vuln)            # (B, C)
    logits_rev = decoder(h_vuln, h_perp)            # (B, C)

    log_p_fwd = F.log_softmax(logits_fwd, dim=-1)   # (B, C)
    p_rev     = F.softmax(logits_rev,    dim=-1)    # (B, C)

    # KL(P_fwd || P_rev) — measures how different the two directions are
    kl = F.kl_div(log_p_fwd, p_rev, reduction='batchmean')

    # Bounded continuous reward: -KL/(1+KL) — always provides gradient
    # Saturates at -lambda_asym, preventing explosion at large KL
    # Gradient: -1/(1+KL)^2 → diminishing returns, stable training
    return -lambda_asym * (kl / (1.0 + kl))


def combined_loss(
    loss_fn,
    logits: torch.Tensor,
    targets: torch.Tensor,
    decoder=None,
    h_perp: torch.Tensor = None,
    h_vuln: torch.Tensor = None,
    lambda_asym: float = 0.1,
    use_asym_reg: bool = True,
) -> tuple:
    """Compute classification loss + optional asymmetry regularization.

    loss_fn can be FocalLoss, nn.CrossEntropyLoss, or any (logits, targets) → scalar.

    Returns (total_loss, cls_loss_value, asym_reg_value).
    """
    cl = loss_fn(logits, targets)

    if use_asym_reg and decoder is not None:
        ar = asymmetry_regularization(decoder, h_perp, h_vuln, lambda_asym)
    else:
        ar = torch.tensor(0.0, device=logits.device)

    return cl + ar, cl, ar
