"""
train_drgatan.py
----------------
5-fold cross-validation training loop for AsymmetricMSFF v2 on DRGATAN.

v2 changes (from Gemini discussion):
  ✓ Graph-aware: GNN encodes ALL drugs using TRAINING edges only (no leakage)
  ✓ CrossEntropyLoss + label_smoothing=0.1 (replaces FocalLoss that killed type-89)
  ✓ No WeightedRandomSampler (caused over-correction with focal loss)
  ✓ Drug embeddings computed ONCE per batch via full-graph GNN pass
  ✓ --no-gnn flag for ablation (proves graph encoder matters)
  ✓ --focal flag to optionally use mild focal loss (γ=0.5)
  ✓ TensorBoard logging (loss curves, metrics per fold, LR schedule)

Usage:
    python train_drgatan.py [--epochs 200] [--folds 5] [--batch 512]
    tensorboard --logdir ../logs/tensorboard --bind_all
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (load_drgatan, make_cv_splits, make_dataloader,
                          compute_class_weights, edges_to_graph_format)
from model_asymmetric import build_model
from losses import FocalLoss, combined_loss
from evaluate import full_evaluation, print_metrics


# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--folds',        type=int,   default=5)
    p.add_argument('--epochs',       type=int,   default=200)
    p.add_argument('--batch',        type=int,   default=512)
    p.add_argument('--patience',     type=int,   default=20)
    p.add_argument('--enc-dim',      type=int,   default=512)
    p.add_argument('--num-mv',       type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smooth', type=float, default=0.1,
                   help='Label smoothing for CE loss (default 0.1)')
    p.add_argument('--focal',        action='store_true',
                   help='Use FocalLoss instead of CE (ablation)')
    p.add_argument('--gamma',        type=float, default=0.5,
                   help='Focal loss gamma (only if --focal)')
    p.add_argument('--lambda-asym',  type=float, default=0.1)
    p.add_argument('--no-asym-reg',  action='store_true')
    p.add_argument('--use-head',     action='store_true')
    p.add_argument('--no-gnn',       action='store_true',
                   help='Disable graph encoder (ablation)')
    p.add_argument('--gnn-heads',    type=int,   default=4)
    p.add_argument('--gnn-layers',   type=int,   default=2)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--save-dir',     default='../results/clifford_runs_v2')
    p.add_argument('--tb-dir',       default='../logs/tensorboard',
                   help='TensorBoard log directory')
    p.add_argument('--run-name',     default=None,
                   help='TensorBoard run name (auto-generated if not set)')
    return p.parse_args()


def make_run_name(args):
    """Generate a descriptive TensorBoard run name from args."""
    if args.run_name:
        return args.run_name
    parts = ['v2']
    if args.no_gnn:
        parts.append('no-gnn')
    else:
        parts.append(f'gnn{args.gnn_layers}L{args.gnn_heads}H')
    if args.focal:
        parts.append(f'focal-g{args.gamma}')
    else:
        parts.append(f'ce-ls{args.label_smooth}')
    parts.append(f'mv{args.num_mv}')
    if args.no_asym_reg:
        parts.append('no-asym')
    parts.append(time.strftime('%m%d-%H%M'))
    return '_'.join(parts)


# ── Training step ──────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, drug_features, graph_edge_index,
                optimizer, loss_fn, device, lambda_asym, use_asym_reg,
                grad_clip=1.0):
    """One training epoch with graph-aware encoding."""
    model.train()
    drug_features = drug_features.to(device)
    graph_edge_index = graph_edge_index.to(device)

    total_losses, cls_losses, asym_losses = [], [], []

    for batch_edges, batch_labels in dataloader:
        batch_edges  = batch_edges.to(device)
        batch_labels = batch_labels.to(device)

        drug_emb = model.encode(drug_features, graph_edge_index)

        h_perp = drug_emb[batch_edges[:, 0]]
        h_vuln = drug_emb[batch_edges[:, 1]]

        logits = model.decoder(h_perp, h_vuln)

        loss, cl, ar = combined_loss(
            loss_fn, logits, batch_labels,
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
        cls_losses.append(cl.item())
        asym_losses.append(ar.item())

    return np.mean(total_losses), np.mean(cls_losses), np.mean(asym_losses)


# ── Validation step ────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, dataloader, drug_features, graph_edge_index,
             loss_fn, device):
    """Validation loss. GNN uses TRAINING edges only (no test leakage)."""
    model.eval()
    drug_features = drug_features.to(device)
    graph_edge_index = graph_edge_index.to(device)

    losses = []
    for batch_edges, batch_labels in dataloader:
        batch_edges  = batch_edges.to(device)
        batch_labels = batch_labels.to(device)

        drug_emb = model.encode(drug_features, graph_edge_index)
        h_perp   = drug_emb[batch_edges[:, 0]]
        h_vuln   = drug_emb[batch_edges[:, 1]]
        logits   = model.decoder(h_perp, h_vuln)

        loss = loss_fn(logits, batch_labels)
        losses.append(loss.item())

    return np.mean(losses)


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # TensorBoard
    run_name = make_run_name(args)
    tb_path = os.path.join(args.tb_dir, run_name)
    os.makedirs(tb_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_path)
    print(f'TensorBoard: {tb_path}')

    # Log hyperparameters
    writer.add_text('config', str(vars(args)))

    print(f'Device: {device}')
    print(f'Loading DRGATAN...')
    t0 = time.time()
    drug_features, edge_index, labels, num_drugs, num_types = load_drgatan(device=device)
    print(f'  {num_drugs} drugs, {len(labels):,} pairs, {num_types} types  '
          f'({time.time()-t0:.1f}s)')

    # Loss function
    if args.focal:
        class_weights = compute_class_weights(labels, num_types, device=device)
        loss_fn = FocalLoss(alpha=class_weights, gamma=args.gamma,
                            num_classes=num_types).to(device)
        print(f'  Loss: FocalLoss(gamma={args.gamma})')
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth).to(device)
        print(f'  Loss: CrossEntropyLoss(label_smoothing={args.label_smooth})')

    print(f'  GNN: {"enabled" if not args.no_gnn else "DISABLED (ablation)"}')
    print(f'  Asym reg: {"enabled" if not args.no_asym_reg else "disabled"}')

    # 5-fold stratified CV splits
    splits = make_cv_splits(edge_index, labels, n_splits=args.folds, seed=args.seed)

    fold_results = []
    global_step = 0  # continuous step counter across folds

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f'\n{"="*60}')
        print(f'Fold {fold+1}/{args.folds}  '
              f'(train={len(train_idx):,}  test={len(test_idx):,})')
        print(f'{"="*60}')

        train_edges  = edge_index[train_idx]
        train_labels = labels[train_idx]
        test_edges   = edge_index[test_idx]
        test_labels  = labels[test_idx]

        # Convert training edges to (2, E) format for GNN
        graph_edge_index = edges_to_graph_format(train_edges).to(device)
        print(f'  Graph edges for GNN: {graph_edge_index.shape}')

        # ── Fresh model per fold ───────────────────────────────────────────
        model = build_model(
            feature_dim=drug_features.shape[1],
            enc_dim=args.enc_dim,
            num_relations=num_types,
            num_mv=args.num_mv,
            use_head=args.use_head,
            use_gnn=not args.no_gnn,
            gnn_heads=args.gnn_heads,
            gnn_layers=args.gnn_layers,
            device=str(device),
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Model parameters: {n_params:,}')

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

        # DataLoaders — NO weighted sampling (v2 fix)
        train_loader = make_dataloader(
            train_edges, train_labels, num_types,
            batch_size=args.batch, weighted_sampling=False, shuffle=True,
        )
        # Validation subset (20% of training data)
        val_size = max(len(train_idx) // 5, 1)
        val_idx  = np.random.choice(len(train_idx), val_size, replace=False)
        val_loader = make_dataloader(
            train_edges[val_idx], train_labels[val_idx], num_types,
            batch_size=args.batch * 2, weighted_sampling=False, shuffle=False,
        )

        # ── Training with early stopping ──────────────────────────────────
        best_val_loss  = float('inf')
        patience_count = 0
        best_state     = None

        for epoch in range(1, args.epochs + 1):
            t_ep = time.time()
            train_loss, cl, ar = train_epoch(
                model, train_loader, drug_features, graph_edge_index,
                optimizer, loss_fn, device, args.lambda_asym,
                use_asym_reg=not args.no_asym_reg,
            )
            val_loss = validate(
                model, val_loader, drug_features, graph_edge_index,
                loss_fn, device,
            )
            scheduler.step()
            global_step += 1

            lr_now = optimizer.param_groups[0]['lr']

            # ── TensorBoard logging ────────────────────────────────────────
            writer.add_scalars('loss/train_val', {
                f'train/fold{fold+1}': train_loss,
                f'val/fold{fold+1}': val_loss,
            }, global_step)
            writer.add_scalar(f'fold{fold+1}/loss/train', train_loss, epoch)
            writer.add_scalar(f'fold{fold+1}/loss/val', val_loss, epoch)
            writer.add_scalar(f'fold{fold+1}/loss/cls', cl, epoch)
            writer.add_scalar(f'fold{fold+1}/loss/asym_reg', ar, epoch)
            writer.add_scalar(f'fold{fold+1}/lr', lr_now, epoch)

            if epoch % 10 == 0 or epoch <= 5:
                print(f'  Ep {epoch:3d}/{args.epochs}  '
                      f'train={train_loss:.4f}  (cl={cl:.4f} ar={ar:.4f})  '
                      f'val={val_loss:.4f}  lr={lr_now:.2e}  '
                      f't={time.time()-t_ep:.1f}s')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(f'  Early stop at epoch {epoch}  '
                          f'(best val loss {best_val_loss:.4f})')
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Final test evaluation ─────────────────────────────────────────
        print(f'\n  Evaluating on test fold...')
        metrics = full_evaluation(
            model, drug_features, test_edges, test_labels,
            train_labels=train_labels.cpu().numpy(),
            num_classes=num_types,
            device=str(device),
            graph_edge_index=graph_edge_index,
        )
        print_metrics(metrics, prefix=f'[fold {fold+1}] ')
        fold_results.append(metrics)

        # ── Log fold metrics to TensorBoard ────────────────────────────────
        metric_keys = ['accuracy', 'auroc', 'aupr', 'f1', 'precision',
                       'recall', 'direction_accuracy', 'asymmetry_score',
                       'f1_dominant', 'f1_top4', 'f1_mid', 'f1_rare']
        for key in metric_keys:
            if key in metrics:
                writer.add_scalar(f'test/{key}', metrics[key], fold + 1)

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

    # ── Aggregate across folds ────────────────────────────────────────────
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

    # Log final summary as hparams
    hparam_dict = {
        'lr': args.lr, 'batch': args.batch, 'enc_dim': args.enc_dim,
        'num_mv': args.num_mv, 'gnn_layers': args.gnn_layers,
        'gnn_heads': args.gnn_heads, 'label_smooth': args.label_smooth,
        'use_gnn': not args.no_gnn, 'asym_reg': not args.no_asym_reg,
        'focal': args.focal, 'gamma': args.gamma,
    }
    metric_dict = {f'hparam/{k}': v['mean'] for k, v in summary.items()}
    writer.add_hparams(hparam_dict, metric_dict)

    # Save summary CSV
    import csv
    csv_path = os.path.join(args.save_dir, 'cv_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'mean', 'std'])
        for key, stats in summary.items():
            w.writerow([key, f'{stats["mean"]:.6f}', f'{stats["std"]:.6f}'])
    print(f'\nSummary saved to {csv_path}')

    writer.close()
    return summary


if __name__ == '__main__':
    main()
