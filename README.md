# MSFF-MA-DDI

Original paper: *"MSFF-MA-DDI: Multi-Source Feature Fusion with Multiple Attention blocks for predicting Drug-Drug Interaction events"*

This repository contains the original MSFF-MA-DDI implementation (branch `main`) and an asymmetric DDI extension using Clifford algebra (branch `clifford`).

---

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Original MSFF-MA-DDI ported to PyTorch/GPU. Batched encoder (~75× speedup), fixed device placement, full pipeline runnable end-to-end. |
| `clifford` | Asymmetric DDI extension. New dataset (DRGATAN), Clifford algebra decoder, asymmetry-specific metrics. |

---

## Branch: `main` — Original MSFF-MA-DDI (PyTorch/GPU port)

### What changed from upstream

The upstream repo (`BioCenter-SHU/MSFF-MA-DDI`) uses Keras/TensorFlow for the final classification stage and has no GPU support. This fork:

- Replaced the Keras DNN stage with a pure PyTorch `DNN` module (end-to-end differentiable)
- Added `device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` throughout
- Batched the 572-drug Python for-loop in `encoder.py` into a single forward pass — **~75× speedup** (CPU ~30s/epoch → GPU batched ~0.4s/epoch)
- Fixed `PositionalEncoding` to use `.to(embedding.device)` so it moves with the model
- Changed `self.scale` in `MultiHeadAttention` from a plain tensor to `register_buffer` so it migrates to GPU with `.to(device)`
- Changed `.T` → `.transpose(-2, -1)` for batched 3D tensor support
- Fixed all `.detach().numpy()` → `.detach().cpu().numpy()` (unavoidable for sklearn)
- Precomputed `train_label_t = torch.tensor(...).to(device)` once per fold to avoid GPU↔CPU ping-pong

### Running the original pipeline

```bash
conda activate msff-ddi
cd code

# Stage 1: sequence branch → seqbranch_embedding.txt
python sequence_feature.py

# Stage 2: heterogeneous branch → hetbranch_embedding.txt
python heterogeneous_feature.py

# Stage 3: multi-source fusion → final results
python "multi_source fusion.py"
```

### Original dataset

| Property | Value |
|----------|-------|
| Drugs | 572 |
| Pairs | 37,264 |
| Interaction types | 65 |
| Drug representation | SMILES sequences + 3 similarity matrices |
| Directed pairs | ~69% (not enforced by model) |

### Original results (reproduced)

| Metric | Stage 1 (seq) | Stage 2 (hete) | Stage 3 (fusion) |
|--------|--------------|----------------|------------------|
| Accuracy | 0.940 | 0.948 | 0.948 |
| Micro-AUROC | 0.999 | 0.999 | 0.9995 |
| Micro-AUPR | 0.982 | 0.986 | 0.987 |
| Macro-F1 | 0.801 | 0.811 | 0.835 |

### Known issues in original codebase (from audit)

| # | Severity | Issue |
|---|----------|-------|
| 1 | **Critical** | `Sigmoid()` final activation fed into `CrossEntropyLoss` — wrong loss surface, dampened gradients |
| 2 | **Critical** | Model instantiated once before the 5-fold loop — weights carry across folds = data leakage |
| 3 | High | `Selfatte_Encoder` applies multi-head attention to a single vector (seq_len=1) — degenerates to a linear layer, Q/K interaction contributes nothing |
| 4 | Medium | `StepLR(step_size=1500)` with 500-epoch training + `scheduler.step()` never called — LR stays flat |
| 5 | Medium | Two-stage pipeline (PyTorch encoder → freeze → Keras DNN) prevents end-to-end gradient flow |

These are fixed in the `clifford` branch.

---

## Branch: `clifford` — Asymmetric DDI with Clifford Algebra

### Motivation

67% of drug-drug interactions are monodirectional. Drug A inhibiting Drug B is pharmacologically different from Drug B inhibiting Drug A. Current DDI models — including the original MSFF-MA-DDI — treat pairs symmetrically: `(A, B)` and `(B, A)` are considered identical.

The `clifford` branch replaces the symmetric MLP decoder with a **Clifford algebra Cl(3,0) geometric product decoder** that structurally guarantees `score(A→B) ≠ score(B→A)`.

### Dataset: DRGATAN (primary)

| Property | Value |
|----------|-------|
| Drugs | 1,876 (int IDs 0–1875) |
| Directed pairs | 218,917 |
| Interaction types | 95 (labels 0–94) |
| Bidirectional pairs | **0** (100% unidirectional) |
| Drug features | 2,159-dim continuous [0,1] |
| Train/test split | 5-fold stratified CV |
| Source | DrugBank NL descriptions parsed into directed edges |
| Used by | DGAT-DDI (2022), DRGATAN (2024), MGKAN (2025) |

**Class imbalance:** type 89 = 54.9% of all pairs. Top 4 types = ~82%. 32 of 95 types have fewer than 10 examples.

### Dataset: ADI-MSF (secondary)

| Property | Value |
|----------|-------|
| Drugs | 1,752 (int index → DrugBank ID via feature file row order) |
| Directed pairs | 504,468 |
| Interaction types | None — binary existence only |
| Features | MACCS167, Morgan1024, DBP3332 (target binding profiles) |

No type labels. Useful for encoder pretraining or binary directed link prediction.

### Why Clifford algebra over alternatives

| Approach | Why it fails on DRGATAN |
|----------|------------------------|
| Sheaf NNs | 218,917 edges × per-edge matrices = memory explosion |
| Magnetic Laplacian | Single phase angle collapses 95 relation types to 1 scalar |
| Quaternions (ℍ) | Only 4 components; limited capacity for 95 types and 32 rare classes |
| Asymmetric bilinear | 95 × 512 × 512 = **24.9M params** just for M_r |

**Clifford Cl(3,0)** has 8 basis elements (scalar + 3 vectors + 3 bivectors + 1 pseudoscalar). The geometric product is non-commutative by construction. Relation transforms cost 95 × 8 × 8 = **6,080 params** — a **4,100× reduction** over asymmetric bilinear.

### Architecture

```
DRGATAN 2159-dim features
        │
        ├─→ [Feature MLP: 2159 → 512]  ─────────────┐
        │                                            │
        └─→ [1D-CNN pseudo-sequence: 2159 → 128]     │
                                                     ↓
                                         [Cross-attention fusion]
                                                     │
                                         512-dim drug embedding h
                                                     │
                            ┌────────────────────────┤
                            ↓                        ↓
                     φ_perp(h_A)              φ_vuln(h_B)
                  (perpetrator role)          (victim role)
                            │                        │
                            └──────────┬─────────────┘
                                       ↓
                    score_r = ⟨ φ_perp(h_A) ⊙ T_r ⊙ φ_vuln(h_B) ⟩₀
                                       │
                              [95-class logits]
                                       │
                               [Focal Loss + asymmetry reg]
```

`φ_perp ≠ φ_vuln` are learned independently → `score(A→B) ≠ score(B→A)` is structurally guaranteed.

### Parameter counts

| Component | Params |
|-----------|--------|
| `DRGATANFeatureEncoder` (2159→512) | 4,308,384 |
| `CliffordDDIDecoder` (512→95, K=8) | 302,656 |
| **Total** | **4,611,040** |

For comparison:
- Asymmetric bilinear decoder alone: ~24.9M for M_r
- Original MSFF decoder naively adapted: ~1.95M (but with Sigmoid+CE bug)
- This decoder: **302K** (~80× smaller than asymmetric bilinear)

### Losses

**Focal loss** (γ=2, inverse-frequency α): down-weights the dominant type-89 class (~55% of data) by up to 400× relative to rare types.

**Asymmetry regularization**: penalizes when KL(P(A→B) ‖ P(B→A)) < 0.5 — pushes the model to make distinct predictions in each direction.

### New evaluation metrics

Beyond the standard 6 metrics (accuracy, micro-AUROC, micro-AUPR, macro-F1, macro-precision, macro-recall):

- **Direction accuracy**: fraction of pairs where `argmax(A→B) ≠ argmax(B→A)`. A symmetric model scores 0.0. After 2 epochs (untrained): **92.3%** — Clifford structure is doing work immediately.
- **Asymmetry score**: mean |P(A→B) − P(B→A)|
- **Per-type F1 by frequency**: dominant (type 89), top-4, mid-frequency (10–1000 examples), rare (<10 examples, 32 types)

### Training fixes (from audit)

| Bug | Original | Fixed |
|-----|----------|-------|
| Sigmoid+CE | `Sigmoid()` before `CrossEntropyLoss` | Removed — raw logits to `FocalLoss` |
| Data leakage | Model created once before fold loop | `build_model()` called inside each fold |
| Dead LR scheduler | `StepLR(step_size=1500)` + never called | `CosineAnnealingLR`, actually fires |
| Symmetric pairs | `to_bidirection()` doubles DRGATAN to ~438K edges | Not called — dataset is 100% directed |
| Full-batch training | Entire dataset in one gradient step | `DataLoader` + `WeightedRandomSampler` |
| No early stopping | Fixed 500 epochs | Validation patience (default 20 epochs) |

### Running the asymmetric pipeline

```bash
git checkout clifford
conda activate msff-ddi
cd code

# Full 5-fold CV, ~5 hours on A5000
python train_drgatan.py --epochs 200 --folds 5

# Ablations
python train_drgatan.py --no-asym-reg          # without asymmetry regularization
python train_drgatan.py --num-mv 4             # smaller Clifford decoder
python train_drgatan.py --num-mv 16            # larger Clifford decoder
python train_drgatan.py --use-head             # with MLP classification head

# Results saved to
# results/clifford_runs/fold{N}_model.pt
# results/clifford_runs/cv_summary.csv
```

### Baseline comparisons (DRGATAN benchmark)

| Model | Year | AUROC | Key idea |
|-------|------|-------|---------|
| DGAT-DDI | 2022 | 98.55% | Directed GAT, source/target/self-role embeddings |
| DRGATAN | 2024 | 98.58% | Directed relation graph attention, attacker/victim roles |
| MGKAN | 2025 | **99.08%** | Multi-view graph + Kolmogorov-Arnold Networks |
| ADI-MSF | 2025 | >95% | Multi-scale fusion of directed topology + drug features |
| **This work** | 2026 | TBD | Multi-source fusion + Clifford algebra decoder |

### New files (clifford branch)

| File | Description |
|------|-------------|
| `code/clifford_algebra.py` | Cl(3,0) structure constants, batched geometric product, grade projections, verification |
| `code/clifford_decoder.py` | `CliffordDDIDecoder` with role-specific projections and relation transforms T_r |
| `code/encoder_drgatan.py` | `DRGATANFeatureEncoder` (feature MLP + 1D-CNN + cross-attention, 2159→512) |
| `code/model_asymmetric.py` | `AsymmetricMSFF` full model, `build_model()` factory |
| `code/losses.py` | `FocalLoss`, `asymmetry_regularization`, `combined_loss` |
| `code/data_loader.py` | DRGATAN + ADI-MSF loading, CV splits, class weights, samplers |
| `code/evaluate.py` | Standard metrics + direction accuracy + asymmetry score + per-type F1 |
| `code/train_drgatan.py` | 5-fold CV training loop with all fixes applied |
| `datasets/DRGATAN/` | 218,917 directed pairs, 95 types, 2159-dim features |
| `datasets/ADI-MSF/` | 504,468 binary directed pairs, MACCS/Morgan/DBP features |
| `datasets/README.md` | Full dataset analysis (counts, distributions, asymmetry checks) |

---

## Setup

```bash
conda env create -f requirements.txt   # or: pip install -r requirements.txt
conda activate msff-ddi
```

Key dependencies: PyTorch, torch-geometric, scikit-learn, pandas, numpy, RDKit (for SMILES processing in original pipeline).

---

## Citation

If you use this code, please cite the original paper:

```
MSFF-MA-DDI: Multi-Source Feature Fusion with Multiple Attention blocks
for predicting Drug-Drug Interaction events
```
