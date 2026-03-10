"""Multi-head scaled dot-product attention blocks for cross-attention between feature sources."""

import torch
from torch import nn


class Query(torch.nn.Module):
    """Linear projection layer for computing attention queries."""

    def __init__(self, input_dim, hidden_dim):
        super(Query, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)


class Key(torch.nn.Module):
    """Linear projection layer for computing attention keys."""

    def __init__(self, input_dim, hidden_dim):
        super(Key, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)


class Value(torch.nn.Module):
    """Linear projection layer for computing attention values."""

    def __init__(self, input_dim, hidden_dim):
        super(Value, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)


class AttentionBlock(torch.nn.Module):
    """Single-head scaled dot-product attention.

    Computes cross-attention where queries come from one feature source
    and keys/values come from another.

    Args:
        query_dim: Dimensionality of the query input features.
        key_value_dim: Dimensionality of the key/value input features.
        attention_dim: Dimensionality of the projected query/key/value space.
    """

    def __init__(self, query_dim, key_value_dim, attention_dim):
        super(AttentionBlock, self).__init__()
        self.query = Query(query_dim, attention_dim)
        self.key = Key(key_value_dim, attention_dim)
        self.value = Value(key_value_dim, attention_dim)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([attention_dim])).float())
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, query_input, key_value_input):
        """Compute scaled dot-product attention.

        Args:
            query_input: Tensor of query features.
            key_value_input: Tensor of key/value features.

        Returns:
            Tuple of (attended_output, attention_weights).
        """
        query = self.query(query_input)
        key = self.key(key_value_input)
        value = self.value(key_value_input)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)).float()
        attention_scores = attention_scores / self.scale.float()
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attended_output = torch.matmul(attention_weights, value)
        attended_output = self.dropout(attended_output)

        return attended_output, attention_weights


class MultiHeadAttentionBlock(torch.nn.Module):
    """Multi-head attention that runs multiple AttentionBlocks in parallel and concatenates results.

    Args:
        query_dim: Dimensionality of the query input features.
        key_value_dim: Dimensionality of the key/value input features.
        attention_dim: Dimensionality per attention head.
        n_heads: Number of parallel attention heads.
    """

    def __init__(self, query_dim, key_value_dim, attention_dim, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = nn.ModuleList([
            AttentionBlock(query_dim, key_value_dim, attention_dim)
            for _ in range(n_heads)
        ])
        self.fc = nn.Linear(n_heads * attention_dim, attention_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, query_input, key_value_input):
        """Run all attention heads, concatenate outputs, and project.

        Args:
            query_input: Tensor of query features.
            key_value_input: Tensor of key/value features.

        Returns:
            Tuple of (projected_output, averaged_attention_weights).
        """
        head_outputs = []
        attentions = []
        for head in self.heads:
            attended_output, attention_weights = head(query_input, key_value_input)
            head_outputs.append(attended_output)
            attentions.append(attention_weights)

        concatenated_heads = torch.cat(head_outputs, dim=-1)
        projected_output = self.fc(concatenated_heads)
        averaged_attention = torch.stack(attentions).mean(dim=0)

        return projected_output, averaged_attention
