"""Utility functions for graph edge processing, evaluation metrics, and data loading."""

from sklearn import metrics
import scipy.sparse as sp
import numpy as np
import torch
import csv


def remove_bidirection(edge_index, edge_type):
    """Remove duplicate edges from a bidirectional graph, keeping only edges where source > target.

    Args:
        edge_index: Tensor of shape (2, num_edges) with source and target node indices.
        edge_type: Tensor of edge type labels, or None.

    Returns:
        Filtered edge_index (and edge_type if provided) with one direction removed.
    """
    mask = edge_index[0] > edge_index[1]
    keep_set = mask.nonzero(as_tuple=False).view(-1)

    if edge_type is None:
        return edge_index[:, keep_set]
    else:
        return edge_index[:, keep_set], edge_type[keep_set]


def to_bidirection(edge_index, edge_type=None):
    """Convert directed edges to bidirectional by adding reverse edges.

    Args:
        edge_index: Tensor of shape (2, num_edges) with source and target node indices.
        edge_type: Tensor of edge type labels, or None.

    Returns:
        Expanded edge_index (and edge_type if provided) with both directions.
    """
    reversed_edges = edge_index.clone()
    reversed_edges[0, :], reversed_edges[1, :] = edge_index[1, :], edge_index[0, :]
    if edge_type is None:
        return torch.cat([edge_index, reversed_edges], dim=1)
    else:
        return torch.cat([edge_index, reversed_edges], dim=1), torch.cat([edge_type, edge_type])


def get_range_list(edge_list):
    """Compute start/end index ranges for concatenated edge lists.

    Args:
        edge_list: List of edge tensors, each of shape (2, num_edges_i).

    Returns:
        Tensor of shape (len(edge_list), 2) with (start, end) column indices.
    """
    ranges = []
    start = 0
    for edges in edge_list:
        ranges.append((start, start + edges.shape[1]))
        start += edges.shape[1]
    return torch.tensor(ranges)


def process_edges(raw_edge_list, train_ratio=0.9):
    """Split edges into train/test sets and convert to bidirectional.

    Each edge type is independently sampled: each edge is included in the
    training set with probability train_ratio, otherwise placed in the test set.

    Args:
        raw_edge_list: List of edge index tensors, one per edge type.
        train_ratio: Probability of assigning each edge to the training set.

    Returns:
        Tuple of (train_edge_index, train_edge_types, train_range,
                  test_edge_index, test_edge_types, test_range).
    """
    train_list = []
    test_list = []
    train_label_list = []
    test_label_list = []

    for edge_type_idx, edge_index in enumerate(raw_edge_list):
        train_mask = np.random.binomial(1, train_ratio, edge_index.shape[1])
        test_mask = 1 - train_mask
        train_set = train_mask.nonzero()[0]
        test_set = test_mask.nonzero()[0]

        train_list.append(edge_index[:, train_set])
        test_list.append(edge_index[:, test_set])

        train_label_list.append(torch.ones(2 * train_set.size, dtype=torch.long) * edge_type_idx)
        test_label_list.append(torch.ones(2 * test_set.size, dtype=torch.long) * edge_type_idx)

    train_list = [to_bidirection(idx) for idx in train_list]
    test_list = [to_bidirection(idx) for idx in test_list]

    train_range = get_range_list(train_list)
    test_range = get_range_list(test_list)

    train_edge_index = torch.cat(train_list, dim=1)
    test_edge_index = torch.cat(test_list, dim=1)

    train_edge_types = torch.cat(train_label_list)
    test_edge_types = torch.cat(test_label_list)

    return train_edge_index, train_edge_types, train_range, test_edge_index, test_edge_types, test_range


def sparse_id(n):
    """Create an n x n sparse identity matrix as a torch sparse tensor.

    Args:
        n: Size of the identity matrix.

    Returns:
        Sparse FloatTensor of shape (n, n) with ones on the diagonal.
    """
    indices = torch.LongTensor([list(range(n)), list(range(n))])
    values = torch.FloatTensor([1.0] * n)
    return torch.sparse.FloatTensor(indices, values, torch.Size((n, n)))


def dense_id(n):
    """Create an n x n dense identity matrix as a torch tensor.

    Args:
        n: Size of the identity matrix.

    Returns:
        Dense FloatTensor of shape (n, n) with ones on the diagonal.
    """
    diag_indices = list(range(n))
    ones = [1.0] * n
    sparse_matrix = sp.coo_matrix((ones, (diag_indices, diag_indices)), shape=(n, n), dtype=float)
    return torch.Tensor(sparse_matrix.todense())


def auprc_auroc_ap(target_tensor, score_tensor):
    """Compute AUPRC, AUROC, and average precision from prediction scores.

    Args:
        target_tensor: Ground truth binary labels (torch tensor).
        score_tensor: Predicted scores (torch tensor).

    Returns:
        Tuple of (auprc, auroc, average_precision).
    """
    labels = target_tensor.detach().cpu().numpy()
    predictions = score_tensor.detach().cpu().numpy()
    auroc = metrics.roc_auc_score(labels, predictions)
    average_precision = metrics.average_precision_score(labels, predictions)
    precision_values, recall_values, _ = metrics._ranking.precision_recall_curve(labels, predictions)
    auprc = metrics._ranking.auc(recall_values, precision_values)

    return auprc, auroc, average_precision


def uniform(size, tensor):
    """Initialize a tensor with uniform distribution scaled by 1/sqrt(size).

    Args:
        size: Fan-in size used to compute the bound.
        tensor: Tensor to initialize in-place, or None (no-op).
    """
    bound = 1.0 / np.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def dict_ep_to_nparray(metrics_dict, num_epochs):
    """Convert a dict of per-epoch metrics to a numpy array.

    Args:
        metrics_dict: Dict mapping epoch index to [auprc, auroc, ap] lists.
        num_epochs: Total number of epochs (determines array width).

    Returns:
        Numpy array of shape (3, num_epochs) with rows for AUPRC, AUROC, and AP.
    """
    metrics_array = np.zeros(shape=(3, num_epochs))
    for epoch_idx, [prc, roc, ap] in metrics_dict.items():
        metrics_array[0, epoch_idx] = prc
        metrics_array[1, epoch_idx] = roc
        metrics_array[2, epoch_idx] = ap
    return metrics_array


def load_csv(filename, dtype):
    """Load a CSV file into a numpy matrix, skipping the header row and first column.

    Args:
        filename: Path to the CSV file.
        dtype: Data type string - 'int' for integers, anything else for floats.

    Returns:
        Numpy matrix of the parsed data.
    """
    matrix_data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row_vector in csvreader:
            if dtype == 'int':
                matrix_data.append(list(map(int, row_vector[1:])))
            else:
                matrix_data.append(list(map(float, row_vector[1:])))
    return np.matrix(matrix_data)
