"""
evaluate.py
-----------
Evaluation metrics for asymmetric DDI prediction.

Standard metrics (reproducing DGAT-DDI / DRGATAN / MGKAN baselines):
    accuracy, micro-AUROC, micro-AUPR, macro-F1, macro-precision, macro-recall

Asymmetry-specific metrics (new):
    direction_accuracy — fraction of pairs where pred(A→B) ≠ pred(B→A)
    asymmetry_score    — mean |P(A→B) - P(B→A)|

Per-relation-type F1 grouped by frequency:
    dominant, top-4, mid-frequency, rare (<10 examples)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# ── AUPR helper ────────────────────────────────────────────────────────────────

def _binary_aupr(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def micro_aupr(y_true_onehot, y_score):
    """Micro-averaged AUPR (same as in original MSFF-MA-DDI)."""
    y_true_flat  = y_true_onehot.ravel()
    y_score_flat = y_score.ravel()
    return _binary_aupr(y_true_flat, y_score_flat)


# ── Standard evaluation ────────────────────────────────────────────────────────

def evaluate_standard(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_score: np.ndarray,
    num_classes: int = 95,
) -> dict:
    """Compute the 6 standard metrics used in the DDI literature.

    Parameters
    ----------
    y_true  : (N,)     integer ground-truth labels
    y_pred  : (N,)     integer predicted labels (argmax of scores)
    y_score : (N, C)   softmax probability scores
    num_classes : int

    Returns
    -------
    dict with keys: accuracy, aupr, auroc, f1, precision, recall
    """
    classes = np.arange(num_classes)
    y_onehot = label_binarize(y_true, classes=classes)

    # Handle edge case: some classes absent from y_true in small test folds
    present = np.unique(y_true)
    y_score_safe = y_score

    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'aupr':      micro_aupr(y_onehot, y_score_safe),
        'auroc':     roc_auc_score(y_onehot, y_score_safe,
                                   average='micro', multi_class='ovr'),
        'f1':        f1_score(y_true, y_pred, average='macro',
                              labels=present, zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro',
                                     labels=present, zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='macro',
                                  labels=present, zero_division=0),
    }


# ── Asymmetry metrics ──────────────────────────────────────────────────────────

@torch.no_grad()
def direction_accuracy(
    model,
    features:         torch.Tensor,
    edge_index:       torch.Tensor,
    graph_edge_index: torch.Tensor = None,
) -> float:
    """Fraction of pairs where pred(A→B) ≠ pred(B→A).

    A perfectly symmetric model scores 0.0.
    A model that has learned directional patterns scores closer to 1.0.
    """
    result = model.predict_direction(features, edge_index, graph_edge_index)
    return result['direction_diff']


@torch.no_grad()
def asymmetry_score(
    model,
    features:         torch.Tensor,
    edge_index:       torch.Tensor,
    graph_edge_index: torch.Tensor = None,
) -> float:
    """Mean absolute difference in predicted probabilities for (A→B) vs (B→A).

    Range [0, 1]. Higher = more directional predictions.
    """
    drug_emb = model.encode(features, graph_edge_index)
    h_A = drug_emb[edge_index[:, 0]]
    h_B = drug_emb[edge_index[:, 1]]

    p_fwd = F.softmax(model.decoder(h_A, h_B), dim=-1)
    p_rev = F.softmax(model.decoder(h_B, h_A), dim=-1)

    return (p_fwd - p_rev).abs().mean().item()


# ── Per-type F1 grouped by frequency ──────────────────────────────────────────

def per_type_f1(
    y_true:        np.ndarray,
    y_pred:        np.ndarray,
    train_labels:  np.ndarray,
    num_classes:   int = 95,
    thresholds:    dict = None,
) -> dict:
    """Per-relation-type F1 grouped by training-set frequency.

    Groups (DRGATAN defaults):
        dominant      : type 89 (120K examples)
        top4          : 4 most frequent types (~82% of data)
        mid           : 10–1000 training examples
        rare          : <10 training examples (32 types in DRGATAN)

    Parameters
    ----------
    y_true       : (N,) ground-truth labels
    y_pred       : (N,) predicted labels
    train_labels : (M,) training set labels (used to determine frequency groups)
    num_classes  : int
    thresholds   : dict with keys 'rare' (default 10) and 'mid' (default 1000)

    Returns
    -------
    dict: { 'dominant': f1, 'top4': f1, 'mid': f1, 'rare': f1, 'per_class': array }
    """
    if thresholds is None:
        thresholds = {'rare': 10, 'mid': 1000}

    counts = np.bincount(train_labels, minlength=num_classes)
    sorted_types = np.argsort(counts)[::-1]

    dominant_types = [sorted_types[0]]
    top4_types     = sorted_types[:4].tolist()
    rare_types     = np.where(counts < thresholds['rare'])[0].tolist()
    mid_types      = np.where(
        (counts >= thresholds['rare']) & (counts < thresholds['mid'])
    )[0].tolist()

    per_class_f1 = f1_score(
        y_true, y_pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0,
    )

    def group_f1(type_list):
        if not type_list:
            return float('nan')
        return float(np.mean([per_class_f1[t] for t in type_list]))

    return {
        'dominant': group_f1(dominant_types),
        'top4':     group_f1(top4_types),
        'mid':      group_f1(mid_types),
        'rare':     group_f1(rare_types),
        'per_class': per_class_f1,
    }


# ── Full evaluation pass ───────────────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(
    model,
    features:         torch.Tensor,
    edge_index:       torch.Tensor,
    labels:           torch.Tensor,
    train_labels:     np.ndarray,
    num_classes:      int = 95,
    batch_size:       int = 2048,
    device:           str = 'cpu',
    graph_edge_index: torch.Tensor = None,
) -> dict:
    """Run the full evaluation suite on a test split.

    graph_edge_index: (2, E) TRAINING edges for GNN (prevents test leakage).

    Returns dict containing all standard + asymmetry metrics.
    """
    model.eval()
    features   = features.to(device)
    edge_index = edge_index.to(device)
    if graph_edge_index is not None:
        graph_edge_index = graph_edge_index.to(device)

    # Encode all drugs once (GNN uses training edges only)
    drug_emb = model.encode(features, graph_edge_index)

    all_logits = []
    all_labels = []
    N = len(labels)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        ei_b = edge_index[start:end]
        h_p  = drug_emb[ei_b[:, 0]]
        h_v  = drug_emb[ei_b[:, 1]]
        logits_b = model.decoder(h_p, h_v)
        all_logits.append(logits_b.cpu())
        all_labels.append(labels[start:end].cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_labels, dim=0).numpy()

    y_score = F.softmax(logits, dim=-1).numpy()
    y_pred  = logits.argmax(dim=-1).numpy()

    std = evaluate_standard(y_true, y_pred, y_score, num_classes)

    # Asymmetry metrics on a subset
    sample_size = min(N, 4096)
    idx_sample  = np.random.choice(N, sample_size, replace=False)
    ei_sample   = edge_index[idx_sample].to(device)

    dir_acc  = direction_accuracy(model, features, ei_sample, graph_edge_index)
    asym_sc  = asymmetry_score(model, features, ei_sample, graph_edge_index)

    type_f1 = per_type_f1(y_true, y_pred, train_labels, num_classes)

    return {
        **std,
        'direction_accuracy': dir_acc,
        'asymmetry_score':    asym_sc,
        'f1_dominant':        type_f1['dominant'],
        'f1_top4':            type_f1['top4'],
        'f1_mid':             type_f1['mid'],
        'f1_rare':            type_f1['rare'],
        'per_class_f1':       type_f1['per_class'],
    }


def print_metrics(metrics: dict, prefix: str = ''):
    """Pretty-print the evaluation metrics dict."""
    indent = '  ' + prefix
    print(f'{indent}Accuracy         : {metrics["accuracy"]:.4f}')
    print(f'{indent}Micro-AUROC      : {metrics["auroc"]:.4f}')
    print(f'{indent}Micro-AUPR       : {metrics["aupr"]:.4f}')
    print(f'{indent}Macro-F1         : {metrics["f1"]:.4f}')
    print(f'{indent}Macro-Precision  : {metrics["precision"]:.4f}')
    print(f'{indent}Macro-Recall     : {metrics["recall"]:.4f}')
    if 'direction_accuracy' in metrics:
        print(f'{indent}Direction Acc    : {metrics["direction_accuracy"]:.4f}')
        print(f'{indent}Asymmetry Score  : {metrics["asymmetry_score"]:.4f}')
        print(f'{indent}F1 [dominant]    : {metrics["f1_dominant"]:.4f}')
        print(f'{indent}F1 [top-4]       : {metrics["f1_top4"]:.4f}')
        print(f'{indent}F1 [mid-freq]    : {metrics["f1_mid"]:.4f}')
        print(f'{indent}F1 [rare <10]    : {metrics["f1_rare"]:.4f}')
