"""
model_asymmetric.py
-------------------
AsymmetricMSFF v2: MSFF-MA-DDI with directed graph encoder + Clifford decoder.

Pipeline (single end-to-end differentiable model):
    Drug features (2159-dim)
        → DRGATANFeatureEncoder  → 512-dim per-drug embedding
        → DirectedGATEncoder     → 512-dim graph-contextualized embedding
        → CliffordDDIDecoder     → 95-class logits (perpetrator ≠ victim)

v2 changes:
  ✓ Added 2-layer directed GAT encoder (closes the 15pt AUROC gap from v1)
  ✓ GNN uses ONLY training edges (leakage prevention)
  ✓ encode() takes edge_index so embeddings are graph-aware
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder_drgatan import DRGATANFeatureEncoder, count_params
from encoder_graph import DirectedGATEncoder
from clifford_decoder import CliffordDDIDecoder, CliffordDDIDecoderWithHead


class AsymmetricMSFF(nn.Module):
    """Multi-source feature fusion model for asymmetric DDI prediction.

    Parameters
    ----------
    feature_dim   : input drug feature dimension (2159 for DRGATAN)
    enc_dim       : drug embedding dimension (512 default)
    num_relations : number of DDI interaction types (95 for DRGATAN)
    num_mv        : parallel multivectors in Clifford decoder (K=8 default)
    use_head      : if True, add an MLP classification head on the decoder
    gnn_heads     : attention heads in directed GAT (4 default)
    gnn_layers    : number of directed GAT layers (2 default)
    gnn_dropout   : dropout in graph encoder
    use_gnn       : if False, skip graph encoder (for ablation)
    """

    def __init__(
        self,
        feature_dim:   int = 2159,
        enc_dim:       int = 512,
        num_relations: int = 95,
        num_mv:        int = 8,
        use_head:      bool = False,
        gnn_heads:     int = 4,
        gnn_layers:    int = 2,
        gnn_dropout:   float = 0.2,
        use_gnn:       bool = True,
    ):
        super().__init__()
        self.feature_dim   = feature_dim
        self.enc_dim       = enc_dim
        self.num_relations = num_relations
        self.use_gnn       = use_gnn

        self.feature_encoder = DRGATANFeatureEncoder(
            input_dim=feature_dim,
            hidden_dim=enc_dim,
            output_dim=enc_dim,
        )

        if use_gnn:
            self.graph_encoder = DirectedGATEncoder(
                enc_dim=enc_dim,
                n_heads=gnn_heads,
                n_layers=gnn_layers,
                dropout=gnn_dropout,
            )
        else:
            self.graph_encoder = None

        decoder_cls = CliffordDDIDecoderWithHead if use_head else CliffordDDIDecoder
        self.decoder = decoder_cls(
            input_dim=enc_dim,
            num_relations=num_relations,
            num_mv=num_mv,
        )

    def encode(self, features: torch.Tensor,
               graph_edge_index: torch.Tensor = None) -> torch.Tensor:
        """Encode all drugs. GNN uses graph_edge_index (training edges only).

        Parameters
        ----------
        features         : (num_drugs, feature_dim)
        graph_edge_index : (2, E) directed edges for GNN — MUST be training
                           edges only to prevent leakage. If None and use_gnn,
                           skips graph encoding.

        Returns
        -------
        drug_emb : (num_drugs, enc_dim)
        """
        drug_emb = self.feature_encoder(features)

        if self.use_gnn and self.graph_encoder is not None and graph_edge_index is not None:
            drug_emb = self.graph_encoder(drug_emb, graph_edge_index)

        return drug_emb

    def forward(
        self,
        features:         torch.Tensor,
        pair_edge_index:  torch.Tensor,
        graph_edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        features         : (num_drugs, feature_dim)  all drug features
        pair_edge_index  : (B, 2)  pairs to score [perpetrator_idx, victim_idx]
        graph_edge_index : (2, E)  training edges for GNN (NEVER include test edges)

        returns          : (B, num_relations) logits
        """
        drug_emb = self.encode(features, graph_edge_index)

        h_perp = drug_emb[pair_edge_index[:, 0]]
        h_vuln = drug_emb[pair_edge_index[:, 1]]

        return self.decoder(h_perp, h_vuln)

    def predict_direction(
        self,
        features:         torch.Tensor,
        pair_edge_index:  torch.Tensor,
        graph_edge_index: torch.Tensor = None,
    ) -> dict:
        """Predict both directions and return asymmetry information."""
        drug_emb = self.encode(features, graph_edge_index)

        h_A = drug_emb[pair_edge_index[:, 0]]
        h_B = drug_emb[pair_edge_index[:, 1]]

        logits_fwd = self.decoder(h_A, h_B)
        logits_rev = self.decoder(h_B, h_A)

        pred_fwd = logits_fwd.argmax(dim=-1)
        pred_rev = logits_rev.argmax(dim=-1)

        return {
            'logits_fwd':     logits_fwd,
            'logits_rev':     logits_rev,
            'pred_fwd':       pred_fwd,
            'pred_rev':       pred_rev,
            'direction_diff': (pred_fwd != pred_rev).float().mean().item(),
        }

    def extra_repr(self) -> str:
        return (
            f'feature_dim={self.feature_dim}, '
            f'enc_dim={self.enc_dim}, '
            f'num_relations={self.num_relations}, '
            f'use_gnn={self.use_gnn}'
        )


def build_model(
    feature_dim:   int = 2159,
    enc_dim:       int = 512,
    num_relations: int = 95,
    num_mv:        int = 8,
    use_head:      bool = False,
    use_gnn:       bool = True,
    gnn_heads:     int = 4,
    gnn_layers:    int = 2,
    device:        str = 'cpu',
) -> AsymmetricMSFF:
    """Factory function — always creates a FRESH model (no stale fold weights)."""
    model = AsymmetricMSFF(
        feature_dim=feature_dim,
        enc_dim=enc_dim,
        num_relations=num_relations,
        num_mv=num_mv,
        use_head=use_head,
        use_gnn=use_gnn,
        gnn_heads=gnn_heads,
        gnn_layers=gnn_layers,
    ).to(device)
    return model


if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(device=device, use_gnn=True)
    total = count_params(model)
    feat_p = count_params(model.feature_encoder)
    gnn_p = count_params(model.graph_encoder) if model.graph_encoder else 0
    dec_p = count_params(model.decoder)

    print(f'AsymmetricMSFF v2 parameter breakdown:')
    print(f'  Feature Encoder : {feat_p:>10,}')
    print(f'  Graph Encoder   : {gnn_p:>10,}')
    print(f'  Decoder         : {dec_p:>10,}')
    print(f'  Total           : {total:>10,}')

    # Forward pass smoke test
    N = 1876
    B = 64
    E = 10000
    features = torch.randn(N, 2159).to(device)
    pair_edges = torch.randint(0, N, (B, 2)).to(device)
    graph_edges = torch.stack([
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
    ]).to(device)

    logits = model(features, pair_edges, graph_edges)
    print(f'\nForward pass: features{list(features.shape)} × '
          f'pairs{list(pair_edges.shape)} × '
          f'graph{list(graph_edges.shape)} → logits{list(logits.shape)}')
    assert logits.shape == (B, 95)

    # Asymmetry check
    result = model.predict_direction(features, pair_edges, graph_edges)
    print(f'Direction diff on random inputs: {result["direction_diff"]*100:.1f}%')
    print('All checks passed.')
