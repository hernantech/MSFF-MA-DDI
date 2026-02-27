"""
train_drgatan.py
----------------
5-fold cross-validation training loop for AsymmetricMSFF on DRGATAN.

Key design choices (addressing all audit issues):
  ✓ Model re-instantiated fresh per fold (no data leakage across folds)
  ✓ No to_bidirection() — directed pairs preserved exactly as in dataset
  ✓ FocalLoss (γ=2) + inverse-frequency alpha (class imbalance)
  ✓ WeightedRandomSampler (over-samples rare types each epoch)
  ✓ Asymmetry regularization loss
  ✓ Early stopping with validation patience
  ✓ Cosine annealing LR schedule (actually fires)
  ✓ Gradient clipping for training stability
  ✓ No Sigmoid before CE anywhere in the pipeline

Usage:
    python train_drgatan.py [--device cuda] [--folds 5] [--epochs 200]
                            [--batch 512] [--enc-dim 512] [--num-mv 8]
                            [--lambda-asym 0.1] [--no-asym-reg]
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader     import (load_drgatan, make_cv_splits, make_dataloader,
                              compute_class_weights)
from model_asymmetric import build_model
from losses           import FocalLoss, combined_loss
from evaluate         import full_evaluation, print_metrics


# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--folds',       type=int,   default=5)
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--batch',       type=int,   default=512)
    p.add_argument('--patience',    type=int,   default=20,
                   help='Early stopping patience (epochs without improvement)')
    p.add_argument('--enc-dim',     type=int,   default=512)
    p.add_argument('--num-mv',      type=int,   default=8,
                   help='Number of parallel multivectors in Clifford decoder')
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--weight-decay',type=float, default=1e-4)
    p.add_argument('--gamma',       type=float, default=2.0,
                   help='Focal loss focusing parameter')
    p.add_argument('--lambda-asym', type=float, default=0.1,
                   help='Asymmetry regularization coefficient')
    p.add_argument('--no-asym-reg', action='store_true',
                   help='Disable asymmetry regularization (ablation)')
    p.add_argument('--use-head',    action='store_true',
                   help='Add MLP head to Clifford decoder')
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--save-dir',    default='../results/clifford_runs')
    return p.parse_args()


# ── Training step ──────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, drug_features, optimizer, focal_fn,
                device, lambda_asym, use_asym_reg, grad_clip=1.0):
    """One training epoch. Returns mean total/focal/asym losses."""
    model.train()
    drug_features = drug_features.to(device)

    total_losses, focal_losses, asym_losses = [], [], []

    for batch_edges, batch_labels in dataloader:
        batch_edges  = batch_edges.to(device)
        batch_labels = batch_labels.to(device)

        # Encode all drugs, look up pair embeddings
        drug_emb = model.encoder(drug_features)
        h_perp   = drug_emb[batch_edges[:, 0]]
        h_vuln   = drug_emb[batch_edges[:, 1]]

        logits = model.decoder(h_perp, h_vuln)

        loss, fl, ar = combined_loss(
            focal_fn, logits, batch_labels,
            decoder=model.decoder,
            h_perp=h_perp,
            h_vuln=h_vuln,
            lambda_asym=lambda_asym,
            use_asym_reg=use_asym_reg,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_losses.append(loss.item())
        focal_losses.append(fl.item())
        asym_losses.append(ar.item())

    return np.mean(total_losses), np.mean(focal_losses), np.mean(asym_losses)


# ── Validation step ────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, dataloader, drug_features, focal_fn, device):
    """Compute validation loss (focal only, no reg). Returns mean loss."""
    model.eval()
    drug_features = drug_features.to(device)

    losses = []
    for batch_edges, batch_labels in dataloader:
        batch_edges  = batch_edges.to(device)
        batch_labels = batch_labels.to(device)

        drug_emb = model.encoder(drug_features)
        h_perp   = drug_emb[batch_edges[:, 0]]
        h_vuln   = drug_emb[batch_edges[:, 1]]
        logits   = model.decoder(h_perp, h_vuln)

        loss = focal_fn(logits, batch_labels)
        losses.append(loss.item())

    return np.mean(losses)


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'Device: {device}')
    print(f'Loading DRGATAN...')
    t0 = time.time()
    drug_features, edge_index, labels, num_drugs, num_types = load_drgatan(device=device)
    print(f'  {num_drugs} drugs, {len(labels):,} pairs, {num_types} types  '
          f'({time.time()-t0:.1f}s)')

    # Class weights for focal loss (computed from full dataset)
    class_weights = compute_class_weights(labels, num_types, device=device)
    focal_fn = FocalLoss(alpha=class_weights, gamma=args.gamma,
                         num_classes=num_types).to(device)

    # 5-fold stratified CV splits
    splits = make_cv_splits(edge_index, labels, n_splits=args.folds, seed=args.seed)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f'\n{"="*60}')
        print(f'Fold {fold+1}/{args.folds}  '
              f'(train={len(train_idx):,}  test={len(test_idx):,})')
        print(f'{"="*60}')

        train_edges  = edge_index[train_idx]
        train_labels = labels[train_idx]
        test_edges   = edge_index[test_idx]
        test_labels  = labels[test_idx]

        # ── Fresh model per fold (fix: no data leakage) ───────────────────────
        model = build_model(
            feature_dim=drug_features.shape[1],
            enc_dim=args.enc_dim,
            num_relations=num_types,
            num_mv=args.num_mv,
            use_head=args.use_head,
            device=str(device),
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

        # DataLoaders with WeightedRandomSampler (oversamples rare types)
        train_loader = make_dataloader(
            train_edges, train_labels, num_types,
            batch_size=args.batch, weighted_sampling=True,
        )
        # Use an 80/20 subset of training data as validation proxy
        val_size  = max(len(train_idx) // 5, 1)
        val_idx   = np.random.choice(len(train_idx), val_size, replace=False)
        val_loader = make_dataloader(
            train_edges[val_idx], train_labels[val_idx], num_types,
            batch_size=args.batch * 2, weighted_sampling=False,
        )

        # ── Training with early stopping ──────────────────────────────────────
        best_val_loss  = float('inf')
        patience_count = 0
        best_state     = None

        for epoch in range(1, args.epochs + 1):
            t_ep = time.time()
            train_loss, fl, ar = train_epoch(
                model, train_loader, drug_features, optimizer, focal_fn,
                device, args.lambda_asym,
                use_asym_reg=not args.no_asym_reg,
            )
            val_loss = validate(model, val_loader, drug_features, focal_fn, device)
            scheduler.step()

            if epoch % 10 == 0 or epoch <= 5:
                lr_now = optimizer.param_groups[0]['lr']
                print(f'  Ep {epoch:3d}/{args.epochs}  '
                      f'train={train_loss:.4f}  (fl={fl:.4f} ar={ar:.4f})  '
                      f'val={val_loss:.4f}  lr={lr_now:.2e}  '
                      f't={time.time()-t_ep:.1f}s')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(f'  Early stop at epoch {epoch}  '
                          f'(best val loss {best_val_loss:.4f})')
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Final test evaluation ─────────────────────────────────────────────
        print(f'\n  Evaluating on test fold...')
        metrics = full_evaluation(
            model, drug_features, test_edges, test_labels,
            train_labels=train_labels.cpu().numpy(),
            num_classes=num_types,
            device=str(device),
        )
        print_metrics(metrics, prefix=f'[fold {fold+1}] ')
        fold_results.append(metrics)

        # Save fold model
        ckpt_path = os.path.join(args.save_dir, f'fold{fold+1}_model.pt')
        torch.save({
            'model_state': best_state,
            'args':        vars(args),
            'fold':        fold,
            'metrics':     {k: v for k, v in metrics.items()
                            if k != 'per_class_f1'},
        }, ckpt_path)
        print(f'  Saved checkpoint: {ckpt_path}')

    # ── Aggregate across folds ────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'5-FOLD CROSS-VALIDATION SUMMARY')
    print(f'{"="*60}')

    scalar_keys = ['accuracy', 'auroc', 'aupr', 'f1', 'precision', 'recall',
                   'direction_accuracy', 'asymmetry_score',
                   'f1_dominant', 'f1_top4', 'f1_mid', 'f1_rare']

    summary = {}
    for key in scalar_keys:
        vals = [r[key] for r in fold_results if key in r]
        if vals:
            summary[key] = {'mean': np.mean(vals), 'std': np.std(vals)}
            print(f'  {key:<22}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    # Save summary CSV
    import csv
    csv_path = os.path.join(args.save_dir, 'cv_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'std'])
        for key, stats in summary.items():
            writer.writerow([key, f'{stats["mean"]:.6f}', f'{stats["std"]:.6f}'])
    print(f'\nSummary saved to {csv_path}')

    return summary


if __name__ == '__main__':
    main()
