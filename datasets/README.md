# Asymmetric DDI Datasets

## Overview

| Dataset | Drugs | Directed Pairs | Types | Task | Features |
|---------|-------|---------------|-------|------|----------|
| **DRGATAN** | 1,876 | 218,917 | 95 | Multi-class (type prediction) | 2,159-dim continuous |
| **ADI-MSF** | 1,752 | 504,468 | — | Binary (existence) | MACCS167, Morgan1024, DBP3332 |

Both datasets are **100% unidirectional** — no pair (A→B) has a corresponding (B→A) entry.

---

## DRGATAN Dataset

**Source:** https://github.com/Wzew5Lp/DRGATAN/tree/master/my_dataset
**Derived from:** DrugBank, by parsing directional NL descriptions (e.g. *"Drug A decreases the efficacy of Drug B"*) into directed edges A→B.
**Used by:** DGAT-DDI (2022), DRGATAN (2024), MGKAN (2025)

### Files

| File | Format | Description |
|------|--------|-------------|
| `my_edge_list.csv` | TSV, 2 cols | 218,917 directed pairs as integer IDs (0–1875) |
| `my_edge_type.csv` | 1 col per line | Interaction type for each pair (0–94), aligned row-by-row with edge list |
| `my_drug_features.csv` | Space-delimited | 1,876 rows × 2,159 continuous features (range [0,1]) per drug |

### Statistics

- **Drugs:** 1,876 (integer IDs 0–1875)
- **Directed pairs:** 218,917
- **Interaction types:** 95 (labels 0–94)
- **Asymmetry:** 100% unidirectional (0 pairs appear in both directions)
- **Train/test split:** 95% / 5% (per `indices_fenge.py`)

### Type Distribution

| Metric | Value |
|--------|-------|
| Most common type | Type 89: 120,228 pairs (54.9%) |
| 2nd most common | Type 30: 29,444 pairs (13.4%) |
| Types with ≥1,000 pairs | 13 of 95 |
| Types with <100 pairs | 52 of 95 |
| Types with <10 pairs | 32 of 95 |
| Median pairs per type | 37 |

**Highly imbalanced.** Top 4 types account for ~82% of all pairs. 32 types have fewer than 10 examples — will need special handling (weighted loss, oversampling, or merge rare types).

---

## ADI-MSF Dataset

**Source:** https://github.com/FengxinHuang/ADI-MSF/tree/main/newdataset
**Used by:** ADI-MSF (2025)

### Files

| File | Format | Description |
|------|--------|-------------|
| `edgelist.txt` | Space-delimited, 2 cols | 504,468 directed pairs as integer IDs (0–1751) |
| `1752maccs167.csv` | CSV | 1,752 drugs × 167 MACCS fingerprint features (DrugBank IDs as index) |
| `1752morgan1024.csv` | CSV | 1,752 drugs × 1,024 Morgan fingerprint features |
| `1752DBP3332.csv` | CSV | 1,752 drugs × 3,332 Drug Binding Profile features (BindingDB target IDs as columns) |

### Statistics

- **Drugs:** 1,752 (integer IDs 0–1751 → DrugBank IDs in feature file row order)
- **Directed pairs:** 504,468
- **Interaction types:** None — binary existence only
- **Asymmetry:** 100% unidirectional (0 bidirectional pairs)
- **Drug ID mapping:** integer index N → row N in feature CSVs (e.g., 0→DB00006, 1→DB00014, ...)

### Notes

- No interaction type labels; edgelist encodes only directed existence (A interacts with B, not B with A)
- Feature files use DrugBank IDs as the index; the integer indices in edgelist correspond to the sorted row order in those files
- Useful for **binary directed DDI** prediction or as an additional pretraining source

---

## Comparison to MSFF-MA-DDI Original Data

| Property | MSFF-MA-DDI (`data/`) | DRGATAN | ADI-MSF |
|----------|-----------------------|---------|---------|
| Drugs | 572 | 1,876 | 1,752 |
| Pairs | 37,264 | 218,917 | 504,468 |
| Types | 65 | 95 | binary |
| Directed | No | Yes | Yes |
| Drug features | Drug similarity matrices | 2,159-dim continuous | MACCS/Morgan/DBP |
| Drug repr. | SMILES (sequence) | Features only | Features only |
