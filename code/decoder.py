"""Decoder architectures for predicting drug-drug interaction events from concatenated drug embeddings."""

import torch
from torch import nn


class MlpDecoder(torch.nn.Module):
    """Three-layer MLP decoder for binary drug-drug pair prediction.

    Takes two drug feature vectors, concatenates them, and passes through
    three MLP blocks (with dropout and ReLU) to produce a single sigmoid score.

    Args:
        input_dim: Dimensionality of each individual drug feature vector.
    """

    def __init__(self, input_dim):
        super(MlpDecoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Linear(int(input_dim * 2), int(input_dim)),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Linear(int(input_dim), int(input_dim // 2)),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Linear(int(input_dim // 2), 1),
            nn.Sigmoid())

    def forward(self, drug_feature, disease_feature):
        """Predict interaction score for a drug-disease pair.

        Args:
            drug_feature: Tensor of shape (batch, input_dim).
            disease_feature: Tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, 1) with sigmoid scores.
        """
        pair_feature = torch.cat([drug_feature, disease_feature], dim=1)
        hidden = self.layer1(pair_feature)
        hidden = self.layer2(hidden)
        output = self.layer3(hidden)
        return output


class DenseDecoder(torch.nn.Module):
    """Dense decoder with batch normalization for binary classification.

    Architecture: Linear(input_dim -> 400) -> BN -> ReLU
                  -> Linear(400 -> 200) -> BN -> ReLU
                  -> Linear(200 -> 2) -> Sigmoid.

    Args:
        input_dim: Dimensionality of the input feature vector.
    """

    def __init__(self, input_dim):
        super(DenseDecoder, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 400),
            nn.BatchNorm1d(400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 2),
            torch.nn.Sigmoid(),
        )

    def forward(self, feature):
        """Forward pass through the dense network.

        Args:
            feature: Input tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, 2) with sigmoid scores.
        """
        return self.network(feature)


class FulconDecoder(torch.nn.Module):
    """Fully connected decoder with dropout and batch normalization for binary classification.

    Architecture: Linear(input_dim -> 128) -> Dropout(0.3) -> BN -> ReLU
                  -> Linear(128 -> 64) -> Dropout(0.3) -> BN -> ReLU
                  -> Linear(64 -> 2) -> Dropout(0.3) -> Sigmoid.

    Args:
        input_dim: Dimensionality of the input feature vector.
    """

    def __init__(self, input_dim):
        super(FulconDecoder, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.Dropout(p=0.3),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(p=0.3),
            nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Dropout(p=0.3),
            torch.nn.Sigmoid(),
        )

    def forward(self, feature):
        """Forward pass through the fully connected network.

        Args:
            feature: Input tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, 2) with sigmoid scores.
        """
        return self.network(feature)


class MultiClassFulconDecoder(torch.nn.Module):
    """Fully connected decoder for 65-class DDI event type classification.

    Architecture: Linear(input_dim -> 256) -> BN -> ReLU
                  -> Linear(256 -> 128) -> BN -> ReLU
                  -> Linear(128 -> 65) -> Sigmoid.

    Args:
        input_dim: Dimensionality of the input feature vector.
    """

    def __init__(self, input_dim):
        super(MultiClassFulconDecoder, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 65),
            torch.nn.Sigmoid(),
        )

    def forward(self, feature):
        """Forward pass through the multi-class network.

        Args:
            feature: Input tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, 65) with per-class sigmoid scores.
        """
        return self.network(feature)


class BinaryFulconDecoder(torch.nn.Module):
    """Fully connected decoder with batch normalization for single-score binary prediction.

    Architecture: Linear(input_dim -> 128) -> BN -> ReLU
                  -> Linear(128 -> 64) -> BN -> ReLU
                  -> Linear(64 -> 1) -> Sigmoid.

    Args:
        input_dim: Dimensionality of the input feature vector.
    """

    def __init__(self, input_dim):
        super(BinaryFulconDecoder, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, feature):
        """Forward pass through the binary prediction network.

        Args:
            feature: Input tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, 1) with a sigmoid score.
        """
        return self.network(feature)
