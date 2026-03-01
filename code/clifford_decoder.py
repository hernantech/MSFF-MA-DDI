"""
clifford_decoder.py
-------------------
Asymmetric DDI decoder using Cl(3,0) geometric products.

Core idea (v3 — grade-aware readout):
    score(A→B, r) = w · (φ_perp(h_A) ⊙ T_r ⊙ φ_vuln(h_B))

where φ_perp and φ_vuln are DIFFERENT learned projections (→ asymmetry),
T_r is a learned relation-specific multivector, and w is a learned 8-dim
grade-weighting vector that reads out ALL components of the multivector.

v3 changes (asymmetry fix):
  ✓ Use all 8 multivector components instead of scalar-only ⟨·⟩₀
    (the scalar part is SYMMETRIC: ⟨a⊙b⟩₀ = ⟨b⊙a⟩₀, discarding the
     antisymmetric signal from bivectors and pseudoscalar)
  ✓ Learned grade-weighted projection preserves non-commutative structure
  ✓ Increased relation transform init from 0.02 → 0.1 for stronger
    non-commutative signal from the start

Asymmetry is now structurally guaranteed by:
    1. φ_perp ≠ φ_vuln (different role projections)
    2. Non-commutative Clifford product (a⊙t ≠ t⊙a)
    3. Grade-aware readout preserving antisymmetric bivector/pseudoscalar

Parameter efficiency:
    Relation transforms: 95 × K × 8 = 6,080 params  (K=8 multivectors)
    vs asymmetric bilinear: 95 × 512 × 512 = 24.9M params
    → 4,100× reduction in the relational parameter budget
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from clifford_algebra import CliffordProduct


class CliffordDDIDecoder(nn.Module):
    """Asymmetric DDI decoder using Cl(3,0) geometric products.

    For each pair (perpetrator A → victim B):
        1. Project h_A → K multivectors via φ_perp  (perpetrator role)
        2. Project h_B → K multivectors via φ_vuln  (victim role)
        3. Compute triple product: (φ_perp(h_A)) ⊙ T_r ⊙ (φ_vuln(h_B))
           for each of R relation types
        4. Grade-weighted readout over all 8 Clifford components
        5. Mean over K multivectors → per-relation logit

    Parameters
    ----------
    input_dim    : drug embedding dimension from encoder (default 512)
    num_relations: number of DDI types (95 for DRGATAN)
    num_mv       : number of parallel multivectors per drug (K=8 default)
                   more → more capacity; fewer → less overfitting for rare types
    hidden_dim   : hidden layer in role projections (default 256)
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_relations: int = 95,
        num_mv: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_mv = num_mv
        self.num_relations = num_relations
        self.input_dim = input_dim

        # ── Role-specific projections ─────────────────────────────────────────
        # These MUST be different — this is where asymmetry is born.
        # h_drug → K multivectors ∈ Cl(3,0)  (each 8-dim)
        self.proj_perpetrator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_mv * 8),
        )
        self.proj_victim = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_mv * 8),
        )

        # ── Relation-specific transforms: T_r ∈ Cl(3,0)^K ───────────────────
        # num_relations × num_mv × 8  =  95 × 8 × 8  =  6,080 params
        # Init at 0.1 (not 0.02) for stronger non-commutative signal
        self.relation_transforms = nn.Parameter(
            torch.randn(num_relations, num_mv, 8) * 0.1
        )

        # ── Grade-weighted readout (v3) ───────────────────────────────────────
        # Learned 8-dim weights over all Clifford grades instead of scalar-only.
        # Bivector (indices 4,5,6) and pseudoscalar (index 7) carry the
        # antisymmetric directional signal that ⟨·⟩₀ discards.
        #
        # Init: [1, 0, ..., 0] — warm-start from v2 scalar-only behavior.
        # Optimizer learns to activate antisymmetric grades during training.
        gw_init = torch.zeros(8)
        gw_init[0] = 1.0
        self.grade_weights = nn.Parameter(gw_init)

        # ── Clifford geometric product (stateless, just carries the matrix) ──
        self.clifford = CliffordProduct()

    def forward(
        self,
        h_perp: torch.Tensor,
        h_vuln: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-relation logits for a batch of directed pairs.

        Parameters
        ----------
        h_perp : (B, input_dim)  perpetrator drug embeddings
        h_vuln : (B, input_dim)  victim drug embeddings

        Returns
        -------
        logits : (B, num_relations)  — raw logits, no activation applied
        """
        B = h_perp.shape[0]
        R = self.num_relations
        K = self.num_mv

        # Project each drug to K multivectors in Cl(3,0)
        m_perp = self.proj_perpetrator(h_perp).reshape(B, K, 8)  # (B, K, 8)
        m_vuln = self.proj_victim(h_vuln).reshape(B, K, 8)       # (B, K, 8)

        # Expand for all R relation types: (B, R, K, 8)
        # .expand() is memory-free (no copy)
        m_perp_e = m_perp.unsqueeze(1).expand(B, R, K, 8)
        m_vuln_e = m_vuln.unsqueeze(1).expand(B, R, K, 8)
        T_r_e    = self.relation_transforms.unsqueeze(0).expand(B, R, K, 8)

        # Flatten batch dims for efficient geometric product
        shape = (B * R * K, 8)
        a = m_perp_e.reshape(shape)
        t = T_r_e.reshape(shape)
        b = m_vuln_e.reshape(shape)

        # Triple product: (a ⊙ t) ⊙ b
        result = self.clifford.triple(a, t, b)  # (B*R*K, 8)

        # Grade-weighted readout (v3): use ALL 8 components, not just scalar
        # grade_weights learns which components matter for scoring:
        #   indices 0     : scalar   (symmetric)
        #   indices 1,2,3 : vectors  (mixed)
        #   indices 4,5,6 : bivectors (ANTISYMMETRIC — directional signal)
        #   index   7     : pseudoscalar (ANTISYMMETRIC)
        result = result.reshape(B, R, K, 8)       # (B, R, K, 8)
        weighted = (result * self.grade_weights).sum(dim=-1)  # (B, R, K)

        # Mean over K multivectors → per-relation logit
        logits = weighted.mean(dim=-1)             # (B, R)

        return logits

    def extra_repr(self) -> str:
        return (
            f'input_dim={self.input_dim}, '
            f'num_relations={self.num_relations}, '
            f'num_mv={self.num_mv}'
        )


class CliffordDDIDecoderWithHead(nn.Module):
    """Variant with an MLP classification head for additional mixing capacity.

    Useful when num_mv is small (e.g. K=4) and you need more capacity.
    With K=8 the base decoder is usually sufficient.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_relations: int = 95,
        num_mv: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.core = CliffordDDIDecoder(input_dim, num_relations, num_mv, hidden_dim)
        # Light residual head: LayerNorm + skip connection
        self.head = nn.Sequential(
            nn.LayerNorm(num_relations),
            nn.Linear(num_relations, num_relations),
        )

    def forward(self, h_perp: torch.Tensor, h_vuln: torch.Tensor) -> torch.Tensor:
        logits = self.core(h_perp, h_vuln)
        return self.head(logits) + logits   # residual


# ── Sanity checks ─────────────────────────────────────────────────────────────

def _verify_decoder():
    """Verify that score(A→B) ≠ score(B→A) for random inputs."""
    torch.manual_seed(0)
    dec = CliffordDDIDecoder(input_dim=64, num_relations=95, num_mv=4)
    dec.eval()

    h_A = torch.randn(8, 64)
    h_B = torch.randn(8, 64)

    with torch.no_grad():
        s_AB = dec(h_A, h_B)   # (8, 95)
        s_BA = dec(h_B, h_A)   # (8, 95)

    # Predictions must differ for asymmetry to hold
    preds_AB = s_AB.argmax(dim=-1)
    preds_BA = s_BA.argmax(dim=-1)
    direction_diff = (preds_AB != preds_BA).float().mean().item()

    print(f'CliffordDDIDecoder verification:')
    print(f'  score(A→B) ≠ score(B→A) on {direction_diff*100:.0f}% of random pairs')
    print(f'  Max |score(A→B) - score(B→A)|: {(s_AB - s_BA).abs().max().item():.4f}')

    params = sum(p.numel() for p in dec.parameters())
    print(f'  Total parameters: {params:,}')

    # Compare to asymmetric bilinear baseline
    bilinear_params = 95 * 64 * 64
    print(f'  Asymmetric bilinear would need: {bilinear_params:,} params for M_r alone')
    print(f'  Reduction: {bilinear_params / params:.1f}×')


if __name__ == '__main__':
    _verify_decoder()
