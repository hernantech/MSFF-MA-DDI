"""
model_asymmetric.py
-------------------
AsymmetricMSFF: full MSFF-MA-DDI adapted for asymmetric DDI on DRGATAN.

Pipeline (single end-to-end differentiable model):
    Drug features (2159-dim)
        → DRGATANFeatureEncoder  → 512-dim per-drug embedding
        → CliffordDDIDecoder     → 95-class logits (perpetrator ≠ victim)

Key improvements over original MSFF-MA-DDI:
  ✓ No to_bidirection() — directed pairs preserved as-is
  ✓ Role-specific projections (φ_perp ≠ φ_vuln) — structural asymmetry
  ✓ Clifford geometric product — non-commutative scoring
  ✓ No Sigmoid before CrossEntropyLoss (bug fix from audit)
  ✓ Model reset between CV folds (bug fix from audit)
  ✓ End-to-end training (no frozen encoder + separate DNN stage)
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder_drgatan import DRGATANFeatureEncoder, count_params
from clifford_decoder import CliffordDDIDecoder, CliffordDDIDecoderWithHead


class AsymmetricMSFF(nn.Module):
    """Multi-source feature fusion model for asymmetric DDI prediction.

    Parameters
    ----------
    feature_dim   : input drug feature dimension (2159 for DRGATAN)
    enc_dim       : drug embedding dimension from encoder (512 default)
    num_relations : number of DDI interaction types (95 for DRGATAN)
    num_mv        : parallel multivectors in Clifford decoder (K=8 default)
    use_head      : if True, add an MLP classification head on the decoder
    """

    def __init__(
        self,
        feature_dim:   int = 2159,
        enc_dim:       int = 512,
        num_relations: int = 95,
        num_mv:        int = 8,
        use_head:      bool = False,
    ):
        super().__init__()
        self.feature_dim   = feature_dim
        self.enc_dim       = enc_dim
        self.num_relations = num_relations

        self.encoder = DRGATANFeatureEncoder(
            input_dim=feature_dim,
            hidden_dim=enc_dim,
            output_dim=enc_dim,
        )

        decoder_cls = CliffordDDIDecoderWithHead if use_head else CliffordDDIDecoder
        self.decoder = decoder_cls(
            input_dim=enc_dim,
            num_relations=num_relations,
            num_mv=num_mv,
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode all drugs. Useful for caching embeddings during evaluation.

        features : (num_drugs, feature_dim)
        returns  : (num_drugs, enc_dim)
        """
        return self.encoder(features)

    def forward(
        self,
        features:   torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        features   : (num_drugs, feature_dim)  all drug features
        edge_index : (B, 2)  [perpetrator_idx, victim_idx]

        returns    : (B, num_relations)  logits — NO activation applied
        """
        # Encode all drugs in one pass
        drug_emb = self.encoder(features)           # (num_drugs, enc_dim)

        # Look up pair embeddings by index
        h_perp = drug_emb[edge_index[:, 0]]         # (B, enc_dim)
        h_vuln = drug_emb[edge_index[:, 1]]         # (B, enc_dim)

        return self.decoder(h_perp, h_vuln)          # (B, num_relations)

    def predict_direction(
        self,
        features:   torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        """Predict both directions and return asymmetry information.

        Returns dict with:
            logits_fwd : (B, R) — score for A→B
            logits_rev : (B, R) — score for B→A
            pred_fwd   : (B,)   — argmax of A→B
            pred_rev   : (B,)   — argmax of B→A
            direction_diff : float — fraction of pairs where preds differ
        """
        drug_emb = self.encoder(features)

        h_A = drug_emb[edge_index[:, 0]]
        h_B = drug_emb[edge_index[:, 1]]

        logits_fwd = self.decoder(h_A, h_B)
        logits_rev = self.decoder(h_B, h_A)

        pred_fwd = logits_fwd.argmax(dim=-1)
        pred_rev = logits_rev.argmax(dim=-1)

        return {
            'logits_fwd':      logits_fwd,
            'logits_rev':      logits_rev,
            'pred_fwd':        pred_fwd,
            'pred_rev':        pred_rev,
            'direction_diff':  (pred_fwd != pred_rev).float().mean().item(),
        }

    def extra_repr(self) -> str:
        return (
            f'feature_dim={self.feature_dim}, '
            f'enc_dim={self.enc_dim}, '
            f'num_relations={self.num_relations}'
        )


def build_model(
    feature_dim:   int = 2159,
    enc_dim:       int = 512,
    num_relations: int = 95,
    num_mv:        int = 8,
    use_head:      bool = False,
    device:        str = 'cpu',
) -> AsymmetricMSFF:
    """Factory function — always creates a FRESH model (no stale fold weights)."""
    model = AsymmetricMSFF(
        feature_dim=feature_dim,
        enc_dim=enc_dim,
        num_relations=num_relations,
        num_mv=num_mv,
        use_head=use_head,
    ).to(device)
    return model


if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(device=device)
    total = count_params(model)
    enc_p = count_params(model.encoder)
    dec_p = count_params(model.decoder)

    print(f'AsymmetricMSFF parameter breakdown:')
    print(f'  Encoder : {enc_p:>10,}')
    print(f'  Decoder : {dec_p:>10,}')
    print(f'  Total   : {total:>10,}')

    # Forward pass smoke test
    N = 1876
    B = 64
    features   = torch.randn(N, 2159).to(device)
    edge_index = torch.randint(0, N, (B, 2)).to(device)

    logits = model(features, edge_index)
    print(f'\nForward pass: features{list(features.shape)} × '
          f'edges{list(edge_index.shape)} → logits{list(logits.shape)}')
    assert logits.shape == (B, 95)

    # Asymmetry check
    result = model.predict_direction(features, edge_index)
    print(f'Direction diff on random inputs: {result["direction_diff"]*100:.1f}%')
    print('All checks passed.')
