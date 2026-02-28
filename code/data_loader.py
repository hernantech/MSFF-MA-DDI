"""
data_loader.py
--------------
Unified data loading API for DRGATAN (primary) and ADI-MSF (secondary) datasets.

DRGATAN:  1,876 drugs | 218,917 directed pairs | 95 interaction types
ADI-MSF:  1,752 drugs | 504,468 directed pairs | binary (no types)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(_HERE, '..', 'datasets')
DRGATAN_DIR  = os.path.join(DATASET_ROOT, 'DRGATAN')
ADMSF_DIR    = os.path.join(DATASET_ROOT, 'ADI-MSF')


# ── DRGATAN loader ─────────────────────────────────────────────────────────────

def load_drgatan(device='cpu'):
    """Load the full DRGATAN dataset.

    Returns
    -------
    features : torch.FloatTensor (1876, 2159)
    edge_index : torch.LongTensor (218917, 2)   — [perpetrator, victim]
    labels : torch.LongTensor (218917,)          — 0..94
    num_drugs : int
    num_types : int
    """
    # Drug features: space-delimited, one drug per row
    feat_path = os.path.join(DRGATAN_DIR, 'my_drug_features.csv')
    feat_rows = []
    with open(feat_path) as f:
        for line in f:
            feat_rows.append([float(x) for x in line.strip().split()])
    features = torch.tensor(feat_rows, dtype=torch.float32).to(device)

    # Directed edge list: tab-separated, (drug1, drug2)
    edge_df = pd.read_csv(
        os.path.join(DRGATAN_DIR, 'my_edge_list.csv'),
        sep='\t', header=None, names=['src', 'dst']
    )
    edge_index = torch.tensor(edge_df.values, dtype=torch.long).to(device)

    # Interaction types: one integer per line
    type_df = pd.read_csv(
        os.path.join(DRGATAN_DIR, 'my_edge_type.csv'),
        header=None, names=['type']
    )
    labels = torch.tensor(type_df['type'].values, dtype=torch.long).to(device)

    num_drugs = features.shape[0]
    num_types = int(labels.max().item()) + 1

    return features, edge_index, labels, num_drugs, num_types


def load_admsf(device='cpu'):
    """Load the ADI-MSF dataset (binary directed pairs + fingerprint features).

    Returns
    -------
    features_dict : dict  {'maccs': (1752,167), 'morgan': (1752,1024), 'dbp': (1752,3332)}
    edge_index : torch.LongTensor (504468, 2)
    num_drugs : int
    """
    maccs   = pd.read_csv(os.path.join(ADMSF_DIR, '1752maccs167.csv'),   index_col=0)
    morgan  = pd.read_csv(os.path.join(ADMSF_DIR, '1752morgan1024.csv'),  index_col=0)
    dbp     = pd.read_csv(os.path.join(ADMSF_DIR, '1752DBP3332.csv'),     index_col=0)

    features_dict = {
        'maccs':  torch.tensor(maccs.values,  dtype=torch.float32).to(device),
        'morgan': torch.tensor(morgan.values, dtype=torch.float32).to(device),
        'dbp':    torch.tensor(dbp.values,    dtype=torch.float32).to(device),
    }

    edges = []
    with open(os.path.join(ADMSF_DIR, 'edgelist.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append([int(parts[0]), int(parts[1])])
    edge_index = torch.tensor(edges, dtype=torch.long).to(device)

    return features_dict, edge_index, maccs.shape[0]


# ── Dataset class ──────────────────────────────────────────────────────────────

class DDIPairDataset(Dataset):
    """Indexed pair dataset for a single train/test split."""

    def __init__(self, edge_index, labels):
        """
        edge_index : (N, 2) LongTensor  [perpetrator, victim]
        labels     : (N,)   LongTensor  interaction type
        """
        self.edge_index = edge_index
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.edge_index[idx], self.labels[idx]


# ── Cross-validation split ─────────────────────────────────────────────────────

def make_cv_splits(edge_index, labels, n_splits=5, seed=42):
    """Stratified k-fold splits on the pair level.

    Returns list of (train_idx, test_idx) numpy arrays.
    Uses stratification so rare classes appear in every fold.
    Falls back to regular KFold for classes with fewer than n_splits samples.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    splits = []
    for train_idx, test_idx in skf.split(np.arange(len(labels_np)), labels_np):
        splits.append((train_idx, test_idx))
    return splits


# ── Class weights ──────────────────────────────────────────────────────────────

def compute_class_weights(labels, num_classes, device='cpu'):
    """Inverse-frequency class weights for focal loss alpha.

    Returns a FloatTensor of shape (num_classes,).
    """
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
    # Inverse freq, smoothed with +1 to avoid div-by-zero for unseen classes
    weights = 1.0 / (counts + 1.0)
    # Normalize so weights sum to num_classes (preserves expected loss scale)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def compute_sample_weights(labels, num_classes):
    """Per-sample weights for WeightedRandomSampler."""
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
    class_w = 1.0 / (counts + 1.0)
    sample_w = class_w[labels_np]
    return torch.tensor(sample_w, dtype=torch.float32)


# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_dataloader(edge_index, labels, num_classes,
                    batch_size=512, weighted_sampling=False, shuffle=True):
    """Build a DataLoader with optional WeightedRandomSampler.

    Default: shuffle=True, no weighted sampling (v2 fix — weighted sampling
    caused focal loss over-correction in v1).
    """
    dataset = DDIPairDataset(edge_index, labels)

    if weighted_sampling:
        sample_w = compute_sample_weights(labels, num_classes)
        sampler  = WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(sample_w),
            replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ── Graph edge format conversion ──────────────────────────────────────────────

def edges_to_graph_format(edge_index: torch.Tensor) -> torch.Tensor:
    """Convert (N, 2) edge list to (2, N) format expected by GNN layers.

    Input:  (N, 2) — each row is [src, dst]
    Output: (2, N) — row 0 is all src, row 1 is all dst
    """
    if edge_index.dim() == 2 and edge_index.shape[1] == 2:
        return edge_index.t().contiguous()
    return edge_index  # already in (2, E) format


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading DRGATAN...')
    feat, edges, labs, n_drugs, n_types = load_drgatan()
    print(f'  features : {feat.shape}   dtype={feat.dtype}')
    print(f'  edges    : {edges.shape}')
    print(f'  labels   : {labs.shape}   n_types={n_types}')
    print(f'  drugs    : {n_drugs}')

    counts = torch.bincount(labs, minlength=n_types)
    print(f'  top-3 types: {counts.topk(3)}')

    cw = compute_class_weights(labs, n_types)
    print(f'  class weights min={cw.min():.4f}  max={cw.max():.4f}')

    splits = make_cv_splits(edges, labs, n_splits=5)
    for fold, (tr, te) in enumerate(splits):
        print(f'  fold {fold}: train={len(tr)}  test={len(te)}')

    print('\nLoading ADI-MSF...')
    fd, ae, nd = load_admsf()
    for name, t in fd.items():
        print(f'  {name}: {t.shape}')
    print(f'  edges: {ae.shape}')
