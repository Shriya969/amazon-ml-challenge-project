# Amazon ML Challenge — Business Entity Resolution Pipeline

A local, memory-efficient end-to-end entity resolution system for matching business entities across three data sources. Built for the Amazon ML Challenge.

---

## Project Overview

This pipeline resolves business entity records from **Source 1** against matching records in **Source 2** and **Source 3**. It handles the **one-to-many** nature of the problem — a single Source 1 entity may match multiple targets.

The system is designed to run **entirely locally** on a laptop (8 GB RAM) without any cloud infrastructure.

### Architecture

```
Source 1  ──►  Normalization  ──►  Blocking Indexes
                                        │
                              ┌─────────┼─────────┐
                              ▼         ▼         ▼
                           Exact     2-Token   3-Gram
                           Name      Tokens    Index
                              └─────────┼─────────┘
                                        ▼
                                 Candidate UNION
                                 (~4M pairs from
                                  2 billion possible)
                                        │
                                        ▼
                               Feature Engineering
                               (RapidFuzz, Jaccard,
                                char similarity)
                                        │
                                        ▼
                              Logistic Regression
                              (80/20 entity split)
                                        │
                                 threshold = 0.80
                                        │
                                        ▼
                             One-to-Many Predictions
```

---

## Results

### Blocking

| Source | Blocker | Candidates | Pair Recall | Complete Recall |
|--------|---------|----------:|------------:|----------------:|
| S1→S2 | Exact name | 5,460 | 21.03% | 10.61% |
| S1→S2 | 2-token | 118,854 | 56.22% | 40.10% |
| S1→S2 | 3-gram | 3,997,197 | 85.81% | 79.48% |
| S1→S2 | **Union** | **4,062,337** | **86.46%** | **80.21%** |
| S1→S3 | Exact name | 6,126 | 22.55% | 11.22% |
| S1→S3 | 2-token | 113,746 | 55.13% | 36.90% |
| S1→S3 | 3-gram | 3,997,398 | 87.12% | 80.15% |
| S1→S3 | **Union** | **4,061,058** | **88.70%** | **81.76%** |

### Model (Logistic Regression, validation set)

| Source | Threshold | Precision | Pair Recall | F1 | Complete Entity Recall |
|--------|----------:|----------:|------------:|----:|----------------------:|
| S1→S2 | 0.80 | 53.91% | **98.27%** | 0.6963 | **96.81%** |
| S1→S3 | 0.80 | 53.17% | **98.36%** | 0.6902 | **96.87%** |

> **Note:** Model recall is measured *within the blocked candidate set*. End-to-end recall is bounded by blocking recall (~86–89%).

---

## Installation

```bash
# Clone / navigate to project
cd "untitled folder"

# Install dependencies
python3 -m pip install pandas numpy scikit-learn rapidfuzz

# Verify data files exist
ls data/
# Expected: train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
```

---

## Data Layout

```
untitled folder/
├── data/
│   ├── train_source1.tsv          # 2.2M rows — query entities
│   ├── train_source2.tsv          # 5.0M rows — target entities (S2-)
│   ├── train_source3.tsv          # 5.3M rows — target entities (S3-)
│   ├── train_ground_truth.tsv     # 2.2M rows — matched_entity_ids column
│   └── subset/                    # Auto-created by create_subset.py
│       ├── train_source1_subset.tsv
│       ├── train_source2_subset.tsv
│       ├── train_source3_subset.tsv
│       └── train_ground_truth_subset.tsv
├── src/
│   ├── preprocessing.py           # Vectorized normalization
│   ├── matching.py                # Inverted index utilities
│   └── data_loader.py             # Loading helpers
├── scripts/
│   ├── create_subset.py           # Phase 1: reproducible subset creation
│   └── run_fast_pipeline.py       # Phases 2–7: blocking + model + evaluation
├── outputs/
│   ├── reports/
│   │   ├── FINAL_REPORT.md        # Full technical report
│   │   ├── blocking_comparison.csv
│   │   ├── model_results.csv
│   │   └── fast_pipeline_results.md
│   ├── predictions/
│   │   └── subset_predictions.csv # 1.6M scored candidate pairs
│   └── subset/
│       └── subset_statistics.md
└── README.md
```

---

## How to Run

### Step 1 — Create the Subset

```bash
python3 scripts/create_subset.py
```

This creates a reproducible 20K-entity subset with full ground-truth coverage. Reads full TSV files in streaming chunks — does not load all 12M rows into memory.

Options:
```bash
python3 scripts/create_subset.py --seed 42 --s1-size 20000
```

Expected output:
```
[1] Loading ground truth and Source 1 ...
[3] Sampled 10,000 matched + 10,000 unmatched S1 entities
[5] Source 2: 17,642 required + 82,358 extra = 100,000 total
✅ Subset creation complete in ~16s
```

### Step 2 — Run the Full Pipeline

```bash
python3 scripts/run_fast_pipeline.py
```

Runs all phases on the subset (default paths):
- Strategy 1, 2, 4 blocking
- Union candidate generation
- Feature engineering (RapidFuzz)
- Logistic Regression with threshold sweep
- One-to-many prediction

Expected runtime: **~4 minutes** on 8 GB RAM Mac.

### Run on Full Dataset

```bash
python3 scripts/run_fast_pipeline.py \
  --source1-path data/train_source1.tsv \
  --source2-path data/train_source2.tsv \
  --source3-path data/train_source3.tsv \
  --ground-truth-path data/train_ground_truth.tsv
```

### Pipeline Options

```bash
python3 scripts/run_fast_pipeline.py \
  --seed 42             # Random seed
  --ngram-n 3           # Character n-gram size (default: 3)
  --ngram-cap 200       # Max candidates per S1 from n-gram (default: 200)
  --s2-cap 500          # Max candidates per S1 from 2-token blocking (default: 500)
```

---

## Pipeline Architecture Detail

### Blocking (Candidate Generation)

The search space for the subset is **4 billion possible pairs** (20K × 100K × 2 sources). Blocking reduces this to ~4 million candidates using inverted indexes:

| Strategy | Key | Recall | Notes |
|----------|-----|--------|-------|
| Exact name | `(country, norm_name)` | 21% | High precision |
| 2-token | `(country, token1, token2)` | 56% | cap=500 |
| **3-gram** | `(country, char_3gram)` | **86%** | **Dominant** |
| Union | S1 ∪ S2 ∪ S4 | **87–89%** | Best recall |

Memory safety rules:
- No Cartesian products
- No `iterrows()`
- Index buckets capped at build time
- Candidates generated entity-by-entity via dict lookup
- Only candidates stored (not all pairs)

### Feature Engineering

10 features computed only on ~4M candidate pairs:

| Feature | Type |
|---------|------|
| `name_exact` | Binary |
| `ft_match`, `ft2_match` | Binary |
| `token_jaccard` | Float 0–1 |
| `char_sim` (bigram Dice) | Float 0–1 |
| `rf_ratio`, `rf_wratio` | Float 0–1 (RapidFuzz) |
| `len_diff` | Integer |
| `addr_jaccard` | Float 0–1 |
| `country_eq` | Binary |

### Model

- **Algorithm:** `LogisticRegression(class_weight="balanced")`
- **Split:** 80% / 20% **by Source 1 entity** (no leakage)
- **Positive class:** candidate pair is a ground-truth match
- **Prediction:** all candidates with score ≥ 0.80 are predicted matches
- **One-to-many:** multiple predictions per S1 entity are allowed and expected

---

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Pair Recall** | Fraction of true GT pairs retrieved |
| **Complete Entity Recall** | Fraction of S1 entities where ALL GT targets are retrieved |
| **Pair Precision** | Fraction of predicted pairs that are true matches |
| **F1** | Harmonic mean of precision and recall |

---

## Limitations

1. Experiments on 20K-entity subset only — full-dataset results not benchmarked
2. Blocking recall ceiling ~86–89%; remaining ~11–14% of true matches cannot be recovered by the model
3. N-gram bucket cap may exclude some true matches from large buckets
4. Precision ~54% — approximately half of predictions are false positives
5. Threshold 0.80 selected on subset validation; may need recalibration for full data

---

## Future Work

1. Phonetic/Soundex blocking for transliteration cases
2. LightGBM for non-linear feature interactions
3. Hard-negative mining for improved calibration
4. Raise n-gram caps for higher blocking recall
5. Bi-encoder (sentence transformer + FAISS) for scalable dense retrieval
6. Full 12M-row dataset evaluation

---

## File Reference

| File | Purpose |
|------|---------|
| [`scripts/create_subset.py`](scripts/create_subset.py) | Creates reproducible subset |
| [`scripts/run_fast_pipeline.py`](scripts/run_fast_pipeline.py) | Full pipeline (blocking + model) |
| [`src/preprocessing.py`](src/preprocessing.py) | Vectorized text normalization |
| [`src/matching.py`](src/matching.py) | Inverted index utilities |
| [`outputs/reports/FINAL_REPORT.md`](outputs/reports/FINAL_REPORT.md) | Full technical report |
| [`outputs/reports/blocking_comparison.csv`](outputs/reports/blocking_comparison.csv) | Blocking results |
| [`outputs/reports/model_results.csv`](outputs/reports/model_results.csv) | Model threshold sweep |
| [`outputs/predictions/subset_predictions.csv`](outputs/predictions/subset_predictions.csv) | 1.6M scored pairs |

---

*Amazon ML Challenge — Business Entity Resolution*  
*Experiments: 20K-entity subset, seed=42, local 8 GB RAM Mac, no cloud*
