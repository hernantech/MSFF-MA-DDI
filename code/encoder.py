"""Encoder architectures for drug SMILES sequences using CNN and self-attention."""

import numpy as np
import torch
import torch.nn as nn
from MultiHeadAttention import *


# Mapping of SMILES characters to integer indices (1-64), 0 reserved for padding.
SMILES_CHAR_TO_INDEX = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64,
}

VOCAB_SIZE = 65  # 64 characters + 1 padding index


def encode_smiles(smiles_string, char_to_index, max_length):
    """Convert a SMILES string to a fixed-length integer array.

    Each character is mapped to its index via char_to_index. The result is
    zero-padded (or truncated) to max_length.

    Args:
        smiles_string: Raw SMILES string.
        char_to_index: Dict mapping characters to integer indices.
        max_length: Fixed output length.

    Returns:
        Numpy array of shape (max_length,) with integer-encoded characters.
    """
    encoded = np.zeros(max_length, dtype=np.int64)
    for i, ch in enumerate(smiles_string[:max_length]):
        encoded[i] = char_to_index[ch]
    return encoded


def positional_encoding(embed_dim, max_length, embedding):
    """Add sinusoidal positional encoding to an embedding tensor.

    Uses sine for even dimensions and cosine for odd dimensions, with
    frequencies decreasing geometrically across dimensions.

    Args:
        embed_dim: Dimensionality of the embedding vectors.
        max_length: Maximum sequence length supported.
        embedding: Input tensor of shape (..., seq_len, embed_dim).

    Returns:
        Tensor with positional encoding added, same shape as input.
    """
    positional_matrix = torch.zeros((max_length, embed_dim))
    position_scales = torch.arange(max_length, dtype=torch.float32).reshape(
        -1, 1) / torch.pow(max_length, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim)
    positional_matrix[:, 0::2] = torch.sin(position_scales)
    positional_matrix[:, 1::2] = torch.cos(position_scales)
    embedding = embedding + positional_matrix[:embedding.shape[-2], :].to(embedding.device)
    return embedding


class CnnSelfAttnEncoder(torch.nn.Module):
    """Encodes SMILES sequences using multi-head self-attention followed by a 3-layer CNN.

    Pipeline: character embedding (64D) -> positional encoding -> 8-head self-attention
    -> 3-layer CNN (40 -> 80 -> 160 filters) -> global max pool -> 160D output.

    The three CNN kernel sizes (4, 6, 8) capture chemical substructure patterns
    at different scales.
    """

    def __init__(self):
        super(CnnSelfAttnEncoder, self).__init__()
        self.embed_dim = 64
        self.num_filters = 40
        self.max_seq_length = 100
        self.kernel_sizes = [4, 6, 8]
        self.attention_dim = 64
        self.attention_heads = 8

        self.char_embedding = nn.Embedding(VOCAB_SIZE, self.embed_dim, padding_idx=0)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.num_filters,
                      kernel_size=self.kernel_sizes[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.num_filters, out_channels=self.num_filters * 2,
                      kernel_size=self.kernel_sizes[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.num_filters * 2, out_channels=self.num_filters * 4,
                      kernel_size=self.kernel_sizes[2]),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool1d(
            self.max_seq_length - self.kernel_sizes[0] - self.kernel_sizes[1] - self.kernel_sizes[2] + 3)

        self.attention = MultiHeadAttentionBlock(
            self.attention_dim, self.attention_dim, self.attention_dim, self.attention_heads)

    def forward(self, smiles_list):
        """Encode a batch of SMILES strings into fixed-length feature vectors.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Tensor of shape (N, num_filters * 4) with pooled CNN features (160D).
        """
        device = next(self.parameters()).device

        # Encode all SMILES to integer indices, pad/truncate to max_seq_length: (N, 100)
        encoded = np.stack([encode_smiles(s, SMILES_CHAR_TO_INDEX, self.max_seq_length)
                            for s in smiles_list])
        token_indices = torch.from_numpy(encoded).to(device)

        # Character embedding: (N, 100, 64)
        embeddings = self.char_embedding(token_indices)

        # Positional encoding: (N, 100, 64)
        embeddings = positional_encoding(self.embed_dim, self.max_seq_length, embeddings)

        # Self-attention: (N, 100, 64)
        embeddings, attention_weights = self.attention(embeddings, embeddings)

        # Permute for CNN: (N, 64, 100) — channels first
        embeddings = embeddings.permute(0, 2, 1)

        # CNN: (N, 160, reduced_length)
        cnn_output = self.cnn(embeddings.float())

        # Global max pool over sequence dimension: (N, 160)
        pooled_features = cnn_output.max(dim=-1).values

        return pooled_features
    