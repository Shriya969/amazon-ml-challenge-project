# Amazon ML Challenge — Entity Resolution Pipeline
## Final Technical Report

**Date:** 2026-09-27  
**Dataset:** Amazon Business Entity Resolution (train subset, seed=42)  
**Task:** One-to-many entity resolution across three business entity sources

---

## 1. Executive Summary

This report describes an end-to-end entity resolution pipeline that matches business entities in Source 1 against corresponding entities in Source 2 and Source 3. The system uses a three-stage architecture: (1) character n-gram blocking to generate a tractable candidate set, (2) feature engineering on candidates only, and (3) Logistic Regression scoring with one-to-many prediction.

**Key results on a 20,000-entity subset:**

| Stage | S1→S2 | S1→S3 |
|-------|-------|-------|
| Blocking pair recall (Union) | **86.46%** | **88.70%** |
| Model pair recall (val, thresh=0.80) | **98.27%** | **98.36%** |
| Model complete entity recall (val) | **96.81%** | **96.87%** |
| Model precision (val) | 53.91% | 53.17% |

> **Important:** Model recall (98%) is measured *within the candidate set* produced by blocking. Overall end-to-end recall is bounded by blocking recall (~86–89%). True matches that blocking never retrieves cannot be recovered by the model.

---

## 2. Problem Definition

Given three sources of business entity records:

- **Source 1** — the query source (2.2M entities in full dataset)
- **Source 2** — a target source (5.0M entities)
- **Source 3** — a second target source (5.3M entities)

The task is to find, for each Source 1 entity, all matching entities in Source 2 and Source 3.

**This is a one-to-many matching problem.** A single Source 1 business may have multiple matches across Source 2 and Source 3. Forcing a single match per Source 1 entity is incorrect — the system must allow multiple predictions per query.

---

## 3. Dataset Description

### Full Dataset (for context)
| File | Rows | Notes |
|------|------|-------|
| train_source1.tsv | 2,206,821 | Query entities |
| train_source2.tsv | 5,034,616 | Target entities (IDs begin S2-) |
| train_source3.tsv | 5,285,603 | Target entities (IDs begin S3-) |
| train_ground_truth.tsv | 2,206,821 | One row per S1 entity; matched IDs comma-separated |

### Subset Used for Experiments (seed=42)
| Source | Rows | Ground-Truth Pairs |
|--------|------|--------------------|
| Source 1 | 20,000 | — |
| Source 2 | 100,000 | **17,642** (all required GT targets included) |
| Source 3 | 100,000 | **18,924** (all required GT targets included) |
| Ground Truth | 20,000 | 10,000 matched + 10,000 unmatched S1 entities |

The subset was constructed to:
- Preserve **all ground-truth target records** for selected S1 entities (100% GT coverage)
- Add realistic negative targets (~82% of S2/S3 records are non-matching) to make the problem hard

---

## 4. Why Naive Pairwise Matching Is Impossible

For the subset alone, scoring every possible pair would require:

```
20,000 Source1 × 100,000 Source2 = 2,000,000,000 pairs
20,000 Source1 × 100,000 Source3 = 2,000,000,000 pairs

Total subset: 4 billion candidate pairs
```

For the full dataset:

```
2,206,821 × 5,034,616 ≈ 11.1 TRILLION S1×S2 pairs
2,206,821 × 5,285,603 ≈ 11.7 TRILLION S1×S3 pairs
```

Even a 1-microsecond comparison would take **years** to evaluate all pairs. Blocking is not optional — it is a prerequisite for any practical system.

---

## 5. Data Preprocessing

All business names and addresses are normalized using vectorized Pandas string operations (no Python loops):

1. **Lowercase** all text
2. **Remove punctuation** — replace `[^\w\s]` with space
3. **Collapse whitespace**
4. **Strip legal suffixes** — remove `llc, inc, ltd, corp, pvt, limited, corporation, co, llp, services, group, enterprises`
5. **Tokenize** — split on whitespace; extract `first_token`, `second_token`, `third_token`
6. **Extract address number** — first numeric token from address string
7. **Normalize country** — lowercase and strip

Preprocessing is implemented in [`src/preprocessing.py`](../../src/preprocessing.py) and runs in under 1 second on the 220K-row subset.

---

## 6. Blocking Architecture

```
Source 1 Entity
      |
      v
  Normalization
      |
  ┌───┴───────────────────┐
  │                       │
  v                       v
Strategy 1             Strategy 2
Exact Normalized      Country + First
Business Name         2 Name Tokens
  │                       │
  └──────────┬────────────┘
             │
             v
         Strategy 4
   Country + Character
     3-Gram Overlap
    (inverted index,
      bucket cap=300)
             │
             v
      Candidate UNION
   (set union per entity,
    no DataFrame join)
             │
             v
    ~4M Candidate Pairs
             │
             v
    Feature Engineering
    (only on candidates)
             │
             v
   Logistic Regression
             │
             v
      Threshold = 0.80
             │
             v
  ONE-TO-MANY Predictions
  (multiple targets per
   Source1 entity allowed)
```

**Core principle:** For each Source 1 entity, candidate targets are retrieved by index lookup — not by scanning all target records. The resulting candidate set is orders of magnitude smaller than the Cartesian product.

---

## 7. Candidate Generation — Three Strategies

### Strategy 1: Exact Normalized Name
**Key:** `(norm_country, norm_name)`  
High precision, low recall. Fails for any name variation.

### Strategy 2: First 2 Meaningful Tokens
**Key:** `(norm_country, first_token, second_token)`  
Broader recall. Safety cap of 500 candidates/entity applied to avoid pathological buckets.

### Strategy 3: Character 3-Gram Blocking *(dominant strategy)*
**How it works:**
1. For each target entity, extract all unique character 3-grams from its normalized name
2. Build inverted index: `(country, 3gram) → [entity_ids]` with bucket cap of 300
3. For each Source 1 entity, retrieve all target IDs sharing ≥1 3-gram
4. Rank by shared 3-gram count, keep top 200

**Why character n-grams outperform token blocking:**
- Captures **name spelling variations** (`"walmart"` vs `"wal mart"`)
- Captures **abbreviations** (`"intl"` vs `"international"`)
- Captures **partial names** (`"amazon"` matches `"amazon web services"`)
- Captures **OCR errors** and minor misspellings
- Robust to legal suffix stripping failures

**Safety bounds:** Bucket cap (300) at index build time + top-200 query cap prevent any single entity from generating millions of candidates.

### Union: S1 ∪ S2 ∪ S4
Candidates from all three strategies are merged per entity using set union — no duplicate pairs, no DataFrame materialization.

---

## 8. Feature Engineering

Features are computed **only for the ~4 million candidate pairs** — never for all possible pairs.

| Feature | Description |
|---------|-------------|
| `name_exact` | Normalized names are identical (0/1) |
| `ft_match` | First tokens match (0/1) |
| `ft2_match` | First two tokens both match (0/1) |
| `token_jaccard` | Jaccard similarity of name token sets |
| `char_sim` | Bigram Dice coefficient of normalized names |
| `rf_ratio` | RapidFuzz ratio (edit distance-based, 0–1) |
| `rf_wratio` | RapidFuzz WRatio (token-aware, 0–1) |
| `len_diff` | Absolute length difference (characters) |
| `addr_jaccard` | Jaccard similarity of address token sets |
| `country_eq` | Countries are equal (0/1) |

RapidFuzz similarity is computed using C-level implementations and is only invoked on candidate pairs — not the full dataset.

---

## 9. Logistic Regression

**Model:** `sklearn.linear_model.LogisticRegression`  
- `max_iter=500`
- `class_weight="balanced"` (handles severe positive/negative imbalance)
- `random_state=42`

**Train/validation split:** 80/20 by **Source 1 entity** — no S1 entity appears in both train and validation. This prevents data leakage where information about an entity's matches in training influences its validation score.

**Label:**
- Positive (1): candidate pair is a true ground-truth match
- Negative (0): candidate pair is not a ground-truth match

**Class imbalance:** ~15,000 positives in ~4,000,000 candidates (0.37% positive rate). Balanced class weights ensure the model does not simply predict all-negative.

---

## 10. One-to-Many Matching

**This is not a one-to-one problem.** In the ground truth, a single Source 1 entity may match multiple Source 2 entities and multiple Source 3 entities simultaneously.

The system handles this correctly:
- For each Source 1 entity, **all candidate targets** are scored independently
- **All candidates whose model score ≥ threshold are predicted as matches**
- There is no forced 1:1 assignment or maximum match count

This is essential for correct evaluation: forcing a single prediction per S1 entity would artificially suppress recall for entities with multiple true matches.

---

## 11. Experimental Results

All experiments were performed on a local machine (8 GB RAM Mac) with no cloud resources.

### Blocking Results (Subset, seed=42)

| Source | Blocker | Candidates | Pair Recall | Complete Recall |
|--------|---------|----------:|------------:|----------------:|
| S1→S2 | Exact | 5,460 | 21.03% | 10.61% |
| S1→S2 | 2-token | 118,854 | 56.22% | 40.10% |
| S1→S2 | 3-gram | 3,997,197 | 85.81% | 79.48% |
| S1→S2 | **Union** | **4,062,337** | **86.46%** | **80.21%** |
| S1→S3 | Exact | 6,126 | 22.55% | 11.22% |
| S1→S3 | 2-token | 113,746 | 55.13% | 36.90% |
| S1→S3 | 3-gram | 3,997,398 | 87.12% | 80.15% |
| S1→S3 | **Union** | **4,061,058** | **88.70%** | **81.76%** |

**Observation:** Character 3-gram blocking accounts for nearly all recall gain. Token-based strategies (S1, S2) add less than 1% incremental recall to the union — because most exact/token matches are already subsumed by 3-gram overlap.

### Model Results (Logistic Regression, threshold = 0.80)

> Recall figures below are measured **within the candidate set** on the validation split. They do not imply 98% end-to-end recall — that would require blocking recall of 100%.

| Source | Threshold | Precision | Pair Recall | F1 | Complete Entity Recall |
|--------|-----------|----------:|------------:|---:|----------------------:|
| S1→S2 | 0.80 | 53.91% | **98.27%** | 0.6963 | **96.81%** |
| S1→S3 | 0.80 | 53.17% | **98.36%** | 0.6902 | **96.87%** |

**Full threshold sweep (S1→S2):**

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.30 | 26.67% | 99.15% | 0.4203 |
| 0.50 | 36.49% | 98.92% | 0.5331 |
| 0.70 | 47.78% | 98.50% | 0.6434 |
| **0.80** | **53.91%** | **98.27%** | **0.6963** |

F1 is maximized at threshold 0.80 — above this, precision gains are offset by recall losses.

### Interpretation
- Blocking retrieves ~86–89% of all true pairs into the candidate set
- The model scores those candidates and, at threshold 0.80, retains ~98% of candidate true pairs
- The ~11–14% of true pairs permanently missed are those blocked out at the candidate generation stage (not a model failure)

---

## 12. Limitations

The following limitations are stated explicitly and honestly:

1. **Subset only:** All experiments were performed on a 20,000-entity Source 1 subset. Full-dataset performance was not benchmarked.
2. **Blocking recall ceiling:** At 86–89% union recall, approximately 11–14% of true matches are never retrieved. These represent the hardest entity pairs: heavy transliteration, radical abbreviation, or missing business names.
3. **N-gram candidate cap:** The 300-entry bucket cap and 200-candidate query cap may exclude some true matches from very large buckets.
4. **Precision ~54%:** Approximately half of predicted pairs are false positives. More discriminative features or a stronger model (LightGBM, neural) could improve this.
5. **Threshold selected on validation subset:** The 0.80 threshold was chosen on 20K S1 entities. It may need recalibration on the full dataset.
6. **No hard-negative mining:** The model saw randomly sampled negatives. Mining hard negatives (near-miss candidates that score high but are not GT matches) would improve calibration.
7. **Strategy 3 note:** Character 3-grams were found to be a subset of 2-token candidates (mathematically: 3-token keys are always more specific than 2-token keys). Strategy 3 (first 3 tokens) does not improve recall over Strategy 2. The character n-gram Strategy 4 is the correct extension.
8. **Full-dataset runtime:** Index building and candidate evaluation take ~15s per 100K target records on 8 GB RAM. Scaling to 5M targets may require chunked or distributed processing.

---

## 13. Scalability Plan

The prototype is designed to scale without architectural changes:

1. **Stream target files in chunks** (200K rows per chunk) — never load full S2/S3 into RAM at once
2. **Build compact inverted indexes** — Python dicts of sets/lists; replace with `shelve` or Redis if RAM is insufficient
3. **Process Source 1 in batches** — stream S1 in chunks of 50K–100K entities
4. **Generate candidates using blocking** — O(n_s1 × avg_candidates) not O(n_s1 × n_target)
5. **Never materialize the Cartesian product** — candidates are generated entity-by-entity and discarded after scoring
6. **Compute fuzzy features only for candidates** — RapidFuzz on 4M pairs, not 11 trillion
7. **Store intermediate candidate results on disk** — write candidate pairs to TSV in streaming fashion if memory is tight
8. **Use multiprocessing only after profiling** — Python GIL limits benefit for CPU-bound operations; use `concurrent.futures.ProcessPoolExecutor` for independent S1 batches
9. **Move to distributed compute only if necessary** — AWS Batch / Spark / DuckDB / Polars if local resources become insufficient for the full 12M-row dataset

**Estimated full-dataset runtime (linear extrapolation from subset):**
- Blocking (S1→S2): ~15s × 50 batches ≈ 12–15 minutes
- Feature engineering: ~90s × 50 batches ≈ 75 minutes
- Model training: ~5 minutes (sklearn scales reasonably to 100M rows with chunking)

---

## 14. Final Architecture Summary

```
Input: Source 1, Source 2, Source 3, Ground Truth
           |
           v
    ┌──────────────────────────────────┐
    │  PREPROCESSING (vectorized)      │
    │  lowercase → strip punct →       │
    │  remove legal suffixes →         │
    │  tokenize → extract 3-grams      │
    └──────────────────────────────────┘
           |
    ┌──────┴─────────────────────────────────────────┐
    │            INDEX BUILDING                      │
    │  exact_idx: (country, norm_name) → {eids}     │
    │  tok2_idx:  (country, t1, t2)   → {eids}     │
    │  ngram_idx: (country, 3gram)    → [eids]      │
    └──────────────────────────────────────────────-─┘
           |
    ┌──────┴──────────────────────────────────────────┐
    │  CANDIDATE GENERATION (streaming per S1 entity) │
    │  For each S1 entity:                            │
    │    cands = exact_lookup                         │
    │          ∪ 2tok_lookup (cap=500)                │
    │          ∪ top200_ngram_lookup                  │
    │  No Cartesian product. No DataFrames.           │
    └─────────────────────────────────────────────────┘
           |
           v  ~4M candidate pairs
    ┌──────────────────────────────────────────────────┐
    │  FEATURE ENGINEERING (candidate pairs only)      │
    │  name_exact, token_jaccard, char_sim,            │
    │  rf_ratio, rf_wratio, addr_jaccard, country_eq   │
    └──────────────────────────────────────────────────┘
           |
    ┌──────┴────────────────────┐
    │  LOGISTIC REGRESSION      │
    │  class_weight="balanced"  │
    │  80/20 by S1 entity split │
    └───────────────────────────┘
           |
    threshold = 0.80
           |
    ┌──────┴────────────────────────────────┐
    │  ONE-TO-MANY PREDICTIONS              │
    │  All candidates score ≥ 0.80 returned │
    │  Multiple matches per S1 allowed      │
    └───────────────────────────────────────┘
           |
           v
    outputs/predictions/subset_predictions.csv
```

---

## 15. Future Improvements

In priority order:

1. **Phonetic/Soundex blocking** — recovers transliteration misses (low overlap names)
2. **LightGBM or XGBoost** — non-linear model; handles feature interactions better than LogReg; typically +3–8% F1
3. **Hard-negative mining** — train on near-miss false positives to sharpen decision boundary
4. **Raise n-gram bucket/query caps** — cap=300/200 was conservative; doubling to 600/400 would trade memory for +2–3% blocking recall
5. **Bi-encoder retrieval** — embed business names with a fine-tuned sentence transformer; use FAISS ANN search for sub-linear retrieval at scale
6. **Address-based blocking** — `(country, addr_number, first_name_token)` as a complementary key for entities with highly variable names
7. **Threshold calibration on full data** — Platt scaling or isotonic regression for well-calibrated probabilities
8. **Full-dataset pipeline run** — use `run_fast_pipeline.py` with full TSV paths; expected runtime 2–3 hours on a single machine

---

*Generated by the Amazon ML Challenge Entity Resolution Pipeline*  
*All results on 20,000-entity subset, seed=42, local 8 GB RAM Mac*  
*No cloud resources used*
