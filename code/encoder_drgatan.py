"""
encoder_drgatan.py
------------------
Drug feature encoder for the DRGATAN dataset.

Input: 2,159-dim continuous drug features (continuous values in [0,1])
Output: 512-dim drug embeddings

Architecture:
    Feature MLP branch:  2159 → 512  (global drug representation)
    Conv1D branch:       2159 → 128   (local pattern extraction, treats features
                                       as a 1D signal — "pseudo-sequence" branch)
    Cross-attention:     feature-branch attends to conv-branch output
    Output:              512-dim with residual connection

The two-branch design preserves the MSFF "multi-source fusion" identity
while adapting to the DRGATAN feature format (no SMILES available).
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing from the same code directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MultiHeadAttention import MultiHeadAttentionBlock


class DRGATANFeatureEncoder(nn.Module):
    """Encode 2159-dim DRGATAN drug features into 512-dim embeddings.

    Parameters
    ----------
    input_dim  : feature vector dimension (2159 for DRGATAN)
    hidden_dim : intermediate MLP dimension
    output_dim : final embedding dimension (512 default)
    n_attn_heads : number of attention heads in cross-branch fusion
    """

    def __init__(
        self,
        input_dim:    int = 2159,
        hidden_dim:   int = 512,
        output_dim:   int = 512,
        n_attn_heads: int = 4,
        dropout:      float = 0.2,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim

        # ── Branch 1: Global MLP ──────────────────────────────────────────────
        # Captures the overall drug feature profile
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # ── Branch 2: 1D CNN "pseudo-sequence" ───────────────────────────────
        # Treats the 2159-dim feature vector as a 1D signal.
        # Captures local co-occurrence patterns (e.g. correlated target-binding
        # features at adjacent positions in the feature vector).
        conv_out = 128
        self.conv_branch = nn.Sequential(
            # (B, 1, 2159) → (B, 32, 538)
            nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(32),
            # (B, 32, 538) → (B, 64, 178)
            nn.Conv1d(32, 64, kernel_size=6, stride=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(64),
            # (B, 64, 178) → (B, 128, 88)
            nn.Conv1d(64, conv_out, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(conv_out),
            # Pool to fixed size: (B, 128, 1)
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),          # (B, 128)
        )

        # ── Cross-branch attention ────────────────────────────────────────────
        # Feature MLP branch (query, 512-dim) attends to Conv branch (key/val, 128-dim)
        # Uses the existing MultiHeadAttentionBlock from the codebase.
        # Output dim = output_dim (atte_dim parameter).
        self.fusion_attn = MultiHeadAttentionBlock(
            sim_dim=output_dim,     # query dimension (feature branch)
            smi_dim=conv_out,       # key/value dimension (conv branch)
            atte_dim=output_dim,    # output dimension per head
            n_heads=n_attn_heads,
        )

        # ── Final output projection ───────────────────────────────────────────
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features : (N, 2159)  all drug features (N can be num_drugs or batch)
        returns  : (N, 512)   drug embeddings
        """
        # Branch 1: global MLP
        feat_emb = self.feature_mlp(features)       # (N, 512)

        # Branch 2: 1D CNN over feature-as-sequence
        conv_in  = features.unsqueeze(1)             # (N, 1, 2159)
        conv_emb = self.conv_branch(conv_in)          # (N, 128)

        # Cross-attention: feature branch (query) attends to conv branch (kv)
        # MultiHeadAttentionBlock expects 3-D inputs (N, seq_len, dim)
        fused, _ = self.fusion_attn(
            feat_emb.unsqueeze(1),   # (N, 1, 512) — query
            conv_emb.unsqueeze(1),   # (N, 1, 128) — key/value
        )
        fused = fused.squeeze(1)                     # (N, 512)

        # Residual + projection
        out = self.out_proj(fused + feat_emb)        # (N, 512)
        return self.out_norm(out)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    enc = DRGATANFeatureEncoder(input_dim=2159, output_dim=512)
    print(f'DRGATANFeatureEncoder parameters: {count_params(enc):,}')

    x = torch.randn(16, 2159)
    out = enc(x)
    print(f'Input:  {x.shape}')
    print(f'Output: {out.shape}')
    assert out.shape == (16, 512), f'Unexpected output shape: {out.shape}'
    print('Shape check passed.')
