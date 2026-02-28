"""
encoder_graph.py
----------------
Directed Graph Attention encoder for DDI interaction graphs.

2-layer directed GAT that contextualizes drug embeddings using the
interaction graph topology. Each layer aggregates separately from:
  - In-neighbors  (drugs that attack me   → "victim context")
  - Out-neighbors (drugs I attack         → "perpetrator context")

Design choices (from Gemini discussion):
  ✓ Shared linear W between in/out (parameter efficiency)
  ✓ Separate attention vectors a_in, a_out (directional awareness)
  ✓ Binary message passing — no edge types in GNN (decoder handles types)
  ✓ 2 layers (matches DRGATAN paper; avoids over-smoothing on 1876 nodes)
  ✓ Residual connections + LayerNorm for training stability

Input:  (num_drugs, enc_dim) initial embeddings + (2, num_edges) edge_index
Output: (num_drugs, enc_dim) contextualized embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectedGATLayer(nn.Module):
    """Single directed graph attention layer.

    Aggregates separately from in-neighbors and out-neighbors with
    independent attention coefficients but shared feature transform W.

    Parameters
    ----------
    in_dim   : input feature dimension
    out_dim  : output feature dimension per head
    n_heads  : number of attention heads
    dropout  : attention coefficient dropout
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0, f"out_dim {out_dim} must be divisible by n_heads {n_heads}"

        # Shared feature transform (parameter efficient)
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # Self-loop transform
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)

        # Separate attention vectors for in/out directions
        # Each: (n_heads, 2 * head_dim) for computing attn(src, dst)
        self.a_in = nn.Parameter(torch.randn(n_heads, 2 * self.head_dim) * 0.01)
        self.a_out = nn.Parameter(torch.randn(n_heads, 2 * self.head_dim) * 0.01)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def _attention_aggregate(self, h_transformed, edge_index, attn_vec,
                             num_nodes, direction='in'):
        """Compute attention-weighted aggregation for one direction.

        Parameters
        ----------
        h_transformed : (N, n_heads, head_dim) — transformed node features
        edge_index    : (2, E) — [src, dst] directed edges
        attn_vec      : (n_heads, 2*head_dim) — attention parameters
        num_nodes     : int
        direction     : 'in' (aggregate from sources to targets)
                        or 'out' (aggregate from targets to sources)
        """
        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)

        if direction == 'in':
            # For node i, aggregate from its in-neighbors (drugs that point TO i)
            # msg: src → dst, so we aggregate messages AT dst
            sender_idx = src_idx
            receiver_idx = dst_idx
        else:
            # For node i, aggregate from its out-neighbors (drugs i points TO)
            # We reverse: aggregate AT src from dst
            sender_idx = dst_idx
            receiver_idx = src_idx

        h_sender = h_transformed[sender_idx]      # (E, n_heads, head_dim)
        h_receiver = h_transformed[receiver_idx]   # (E, n_heads, head_dim)

        # Attention scores: concat(h_sender, h_receiver) @ a
        edge_feat = torch.cat([h_sender, h_receiver], dim=-1)  # (E, n_heads, 2*head_dim)
        e = (edge_feat * attn_vec.unsqueeze(0)).sum(dim=-1)    # (E, n_heads)
        e = self.leaky_relu(e)

        # Softmax over edges pointing to same receiver
        # Sparse softmax via scatter
        e_max = _scatter_max(e, receiver_idx, num_nodes)          # (N, n_heads)
        e_stable = e - e_max[receiver_idx]                        # (E, n_heads)
        alpha = torch.exp(e_stable)                               # (E, n_heads)
        alpha_sum = _scatter_sum(alpha, receiver_idx, num_nodes)  # (N, n_heads)
        alpha_norm = alpha / (alpha_sum[receiver_idx] + 1e-8)     # (E, n_heads)
        alpha_norm = self.dropout(alpha_norm)

        # Weighted message
        msg = h_sender * alpha_norm.unsqueeze(-1)  # (E, n_heads, head_dim)
        out = _scatter_sum(msg, receiver_idx, num_nodes)  # (N, n_heads, head_dim)
        return out

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        h          : (N, in_dim) node features
        edge_index : (2, E) directed edges [src, dst]
        returns    : (N, out_dim)
        """
        N = h.shape[0]

        # Shared feature transform
        h_proj = self.W(h).view(N, self.n_heads, self.head_dim)  # (N, H, D)

        # Self contribution
        h_self = self.W_self(h)  # (N, out_dim)

        # In-neighbor aggregation (who attacks me)
        h_in = self._attention_aggregate(
            h_proj, edge_index, self.a_in, N, direction='in'
        )  # (N, H, D)

        # Out-neighbor aggregation (who I attack)
        h_out = self._attention_aggregate(
            h_proj, edge_index, self.a_out, N, direction='out'
        )  # (N, H, D)

        # Combine: self + in + out
        h_in_flat = h_in.reshape(N, -1)    # (N, out_dim)
        h_out_flat = h_out.reshape(N, -1)  # (N, out_dim)
        combined = h_self + h_in_flat + h_out_flat + self.bias

        return self.norm(combined)


class DirectedGATEncoder(nn.Module):
    """2-layer directed GAT encoder for DDI graphs.

    Parameters
    ----------
    enc_dim   : embedding dimension (512)
    n_heads   : attention heads per layer (4)
    n_layers  : number of GAT layers (2)
    dropout   : dropout rate
    """

    def __init__(self, enc_dim: int = 512, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                DirectedGATLayer(enc_dim, enc_dim, n_heads, dropout)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        h          : (N, enc_dim) initial drug embeddings
        edge_index : (2, E) directed edges [src, dst] — TRAINING EDGES ONLY
        returns    : (N, enc_dim) contextualized embeddings
        """
        for layer in self.layers:
            h_new = layer(h, edge_index)
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual
        return h


# ── Scatter utilities (no torch_scatter dependency) ──────────────────────────

def _scatter_sum(src: torch.Tensor, index: torch.Tensor,
                 num_nodes: int) -> torch.Tensor:
    """Scatter-add src into output[index]. Works for arbitrary trailing dims."""
    out_shape = (num_nodes,) + src.shape[1:]
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    idx = index
    # Expand index to match src dimensions
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    return out.scatter_add(0, idx, src)


def _scatter_max(src: torch.Tensor, index: torch.Tensor,
                 num_nodes: int) -> torch.Tensor:
    """Scatter-max src into output[index]. For numerical stability in softmax."""
    out_shape = (num_nodes,) + src.shape[1:]
    out = torch.full(out_shape, float('-inf'), dtype=src.dtype, device=src.device)
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    return out.scatter_reduce(0, idx, src, reduce='amax')


# ── Verification ─────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = 1876  # DRGATAN drug count
    E = 10000  # subset of edges
    enc_dim = 512

    encoder = DirectedGATEncoder(enc_dim=enc_dim).to(device)
    print(f'DirectedGATEncoder parameters: {count_params(encoder):,}')

    h = torch.randn(N, enc_dim).to(device)
    edge_index = torch.stack([
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
    ]).to(device)

    out = encoder(h, edge_index)
    print(f'Input:  {h.shape}')
    print(f'Edges:  {edge_index.shape}')
    print(f'Output: {out.shape}')
    assert out.shape == (N, enc_dim)
    print('Shape check passed.')

    # Verify gradient flow
    loss = out.sum()
    loss.backward()
    print('Gradient flow check passed.')
