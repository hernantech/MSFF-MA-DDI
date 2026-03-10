"""High-level model architectures that combine encoders and decoders for DDI prediction.

Note: FusionNet, FusionFeatureExtractor, DrugFeatureExtractor, and NetworkFeatureExtractor
reference external encoder classes (DD_Encoder, Graph_Atte_Encoder1, Graph_Encoder) that
are not defined in this codebase. These models are not currently used in the active pipeline.
"""

from decoder import *
import torch
import torch.nn.functional as F


class FusionNet(torch.nn.Module):
    """End-to-end DDI prediction model combining feature fusion encoder with a decoder.

    Encodes drug features via FusionFeatureExtractor, concatenates feature vectors
    for each drug pair, and predicts interaction via FulconDecoder.

    Args:
        node_feature_dim: Dimensionality of input node features.
        hidden_dim1: First hidden layer size for the DD_Encoder.
        hidden_dim2: Second hidden layer size for the DD_Encoder.
        decoder_input_dim: Input dimensionality for the FulconDecoder.
    """

    def __init__(self, node_feature_dim, hidden_dim1, hidden_dim2, decoder_input_dim):
        super(FusionNet, self).__init__()
        self.encoder = FusionFeatureExtractor(node_feature_dim, hidden_dim1, hidden_dim2)
        self.decoder = FulconDecoder(decoder_input_dim)

    def forward(self, node_features, edge_index, similarity_embedding, smiles_data, row, col):
        """Encode drugs and predict interaction scores for given drug pairs.

        Args:
            node_features: Node feature matrix of shape (num_drugs, node_feature_dim).
            edge_index: Edge index tensor for the drug graph.
            similarity_embedding: Pre-computed drug similarity embeddings.
            smiles_data: SMILES-based drug features.
            row: Source drug indices for pairs to predict.
            col: Target drug indices for pairs to predict.

        Returns:
            Tensor of shape (num_pairs, 2) with predicted interaction scores.
        """
        drug_features = self.encoder(node_features, edge_index, smiles_data, similarity_embedding)
        pair_features = torch.cat([drug_features[row, :], drug_features[col, :]], dim=1)
        prediction = self.decoder(pair_features)
        return prediction


class FusionFeatureExtractor(torch.nn.Module):
    """Two-stage feature extractor: graph encoder followed by attention-based encoder.

    First applies DD_Encoder to produce graph-based drug features, normalizes them,
    then passes through Graph_Atte_Encoder1 for attention-refined features.

    Args:
        node_feature_dim: Dimensionality of input node features.
        hidden_dim1: First hidden layer size for DD_Encoder.
        hidden_dim2: Second hidden layer size for DD_Encoder.
    """

    def __init__(self, node_feature_dim, hidden_dim1, hidden_dim2):
        super(FusionFeatureExtractor, self).__init__()
        self.graph_encoder = DD_Encoder(node_feature_dim, hidden_dim1, hidden_dim2)
        self.attention_encoder = Graph_Atte_Encoder1()

    def forward(self, node_features, edge_index, smiles_data, similarity_embedding):
        """Extract fused drug features through graph and attention encoding.

        Args:
            node_features: Node feature matrix of shape (num_drugs, node_feature_dim).
            edge_index: Edge index tensor for the drug graph.
            smiles_data: SMILES-based drug features.
            similarity_embedding: Pre-computed drug similarity embeddings.

        Returns:
            Tensor of shape (num_drugs, feature_dim) with attention-refined drug features.
        """
        graph_features = self.graph_encoder(node_features, edge_index)
        graph_features = F.normalize(graph_features)
        refined_features = self.attention_encoder(smiles_data, graph_features)
        return refined_features


class DrugFeatureExtractor(torch.nn.Module):
    """Simple drug feature extractor using a graph encoder.

    Wraps Graph_Encoder to produce drug feature representations from input data.
    """

    def __init__(self):
        super(DrugFeatureExtractor, self).__init__()
        self.graph_encoder = Graph_Encoder()

    def forward(self, data):
        """Extract drug features from input data.

        Args:
            data: Input drug data for the graph encoder.

        Returns:
            Tensor of drug feature representations.
        """
        drug_features = self.graph_encoder(data)
        return drug_features


class NetworkFeatureExtractor(torch.nn.Module):
    """Drug feature extractor using only the DD_Encoder graph network.

    Produces L2-normalized drug features from node features and graph structure.

    Args:
        node_feature_dim: Dimensionality of input node features.
        hidden_dim1: First hidden layer size for DD_Encoder.
        hidden_dim2: Second hidden layer size for DD_Encoder.
    """

    def __init__(self, node_feature_dim, hidden_dim1, hidden_dim2):
        super(NetworkFeatureExtractor, self).__init__()
        self.graph_encoder = DD_Encoder(node_feature_dim, hidden_dim1, hidden_dim2)

    def forward(self, node_features, edge_index):
        """Extract normalized drug features from the graph network.

        Args:
            node_features: Node feature matrix of shape (num_drugs, node_feature_dim).
            edge_index: Edge index tensor for the drug graph.

        Returns:
            Tensor of shape (num_drugs, hidden_dim2) with L2-normalized drug features.
        """
        drug_features = self.graph_encoder(node_features, edge_index)
        drug_features = F.normalize(drug_features, dim=0)
        return drug_features
