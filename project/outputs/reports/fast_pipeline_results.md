# Fast Pipeline Results
Generated: 2026-09-27 22:36:09  |  Total runtime: 251s

## Dataset (Subset, seed=42)
| Source | Rows | GT Pairs |
|--------|------|----------|
| Source 1 | 20,000 | — |
| Source 2 | 100,000 | 17,642 |
| Source 3 | 100,000 | 18,924 |

## Blocking Strategy Results

| Strategy | Target | Candidate Pairs | Pair Recall % | Complete Entity Recall % | Avg Cands | P95 Cands | Runtime (s) |
|---|---|---|---|---|---|---|---|
| Strategy 1: Exact Name | S1→S2 | 5,460 | 21.03 | 10.61 | 0.27 | 2.0 | 0.0 |
| Strategy 2: 2-Token | S1→S2 | 118,854 | 56.22 | 40.1 | 5.94 | 11.0 | 0.0 |
| Strategy 4: 3-gram | S1→S2 | 3,997,197 | 85.81 | 79.48 | 199.86 | 200.0 | 14.8 |
| Union: S1∪S2∪S4 | S1→S2 | 4,062,337 | 86.46 | 80.21 | 203.12 | 204.0 | 14.9 |
| Strategy 1: Exact Name | S1→S3 | 6,126 | 22.55 | 11.22 | 0.31 | 2.0 | 0.0 |
| Strategy 2: 2-Token | S1→S3 | 113,746 | 55.13 | 36.9 | 5.69 | 11.0 | 0.0 |
| Strategy 4: 3-gram | S1→S3 | 3,997,398 | 87.12 | 80.15 | 199.87 | 200.0 | 15.1 |
| Union: S1∪S2∪S4 | S1→S3 | 4,061,058 | 88.7 | 81.76 | 203.05 | 204.0 | 15.5 |

## Model Results (LogisticRegression, validation set, 80/20 entity split)

| Source | Threshold | Precision | Recall (pair) | F1 | Predicted Positives |
|---|---|---|---|---|---|
| S1→S2 | 0.3 | 0.2667 | 0.9915 | 0.4203 | 11366 |
| S1→S2 | 0.4 | 0.3158 | 0.9895 | 0.4788 | 9579 |
| S1→S2 | 0.5 | 0.3649 | 0.9892 | 0.5331 | 8288 |
| S1→S2 | 0.6 | 0.4174 | 0.9879 | 0.5869 | 7235 |
| S1→S2 | 0.7 | 0.4778 | 0.985 | 0.6434 | 6302 |
| S1→S2 | 0.8 | 0.5391 | 0.9827 | 0.6963 | 5572 |
| S1→S3 | 0.3 | 0.2524 | 0.9955 | 0.4027 | 13189 |
| S1→S3 | 0.4 | 0.3032 | 0.9934 | 0.4646 | 10955 |
| S1→S3 | 0.5 | 0.3516 | 0.9913 | 0.5191 | 9429 |
| S1→S3 | 0.6 | 0.403 | 0.9886 | 0.5726 | 8204 |
| S1→S3 | 0.7 | 0.4619 | 0.9862 | 0.6291 | 7140 |
| S1→S3 | 0.8 | 0.5317 | 0.9836 | 0.6902 | 6186 |

## Final Model Performance (Best Threshold = 0.80)

| Metric | S1→S2 | S1→S3 |
|--------|-------|-------|
| Pair Recall (val) | **98.27%** | **98.36%** |
| Complete Entity Recall (val) | **96.81%** | **96.87%** |
| Precision | 0.5391 | 0.5317 |
| F1 | 0.6963 | 0.6902 |

## Conclusions

### Key Findings
1. **Strategy 4 (character 3-gram) is the dominant blocker**: 85-87% pair recall vs 56% for token-based blocking. Name spelling variations, abbreviations, and partial names that defeat token matching are captured by shared n-grams.
2. **Union adds only ~1-2% over Strategy 4 alone** — the n-gram blocker subsumes most of what S1 and S2 find.
3. **LogisticRegression nearly perfectly re-ranks the 4M candidates**: 98.3% pair recall at threshold=0.80, with 54% precision — meaning ~half of predicted pairs are true matches.
4. **Candidate generation is the bottleneck**: at 86-89% union recall, ~11-14% of true matches are never retrieved. These represent the hardest cases (transliteration, heavy abbreviation, missing names).

### Why This Scales
- Inverted indexes + streaming S1 lookup: O(n × avg_cands) not O(n × m)
- Feature engineering only on ~4M candidate pairs, not 20K × 100K = 2B pairs
- No Cartesian product at any stage

### Recommended Next Steps
1. Raise n-gram bucket cap (300→500) to recover more recall
2. Add Soundex/phonetic key for transliteration cases
3. Replace LogReg with LightGBM for non-linear precision improvement
4. Run on full dataset: 

_Saved: outputs/reports/blocking_comparison.csv | outputs/reports/model_results.csv | outputs/predictions/subset_predictions.csv (1,623,273 rows)_
