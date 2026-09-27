# Subset Statistics
Generated with seed=42

| Dataset | Rows | Notes |
|---------|------|-------|
| Source 1 | 20,000 | 10,000 with matches + 10,000 without |
| Source 2 | 100,000 | 17,642 required GT targets + extras |
| Source 3 | 100,000 | 18,924 required GT targets + extras |
| Ground Truth | 20,000 | Matches only for selected S1 entities |

## Ground Truth Pair Counts (in subset)
- S1 → S2 GT pairs: 17,642
- S1 → S3 GT pairs: 18,924
- S1 matched entities: 10,000
- S1 unmatched entities: 10,000

## Coverage Check
- Required S2 targets in subset: 17,642 / 17,642 ✅
- Required S3 targets in subset: 18,924 / 18,924 ✅

Elapsed: 16.1s
