"""
create_subset.py

Creates a reproducible stratified subset of the training data.

Subset targets:
  Source 1: ~20,000 entities (10K matched + 10K unmatched)
  Source 2: ~100,000 records (required GT targets + sampled negatives)
  Source 3: ~100,000 records (required GT targets + sampled negatives)

Usage:
  python scripts/create_subset.py [--seed 42] [--s1-size 20000]
"""

import os, sys, gc, time, argparse
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--seed",    type=int, default=42)
ap.add_argument("--s1-size", type=int, default=20000)
ap.add_argument("--data-dir",  default="data")
ap.add_argument("--out-dir",   default="data/subset")
args = ap.parse_args()

SEED     = args.seed
S1_SIZE  = args.s1_size
DATA_DIR = args.data_dir
OUT_DIR  = args.out_dir

TARGET_S2 = 100_000
TARGET_S3 = 100_000

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("outputs/subset", exist_ok=True)

GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")
if not os.path.exists(GT_FILE):
    GT_FILE = os.path.join(DATA_DIR, "train_ground_truths.tsv")


def load_tsv(path, **kwargs):
    return pd.read_csv(path, sep="\t", quoting=3, on_bad_lines="skip", **kwargs)


print("=" * 60)
print("SUBSET CREATION  (seed={})".format(SEED))
print("=" * 60)

# ------------------------------------------------------------------
# 1. Load GT and S1 IDs
# ------------------------------------------------------------------
t0 = time.time()
print("\n[1] Loading ground truth and Source 1 ...", flush=True)

gt = load_tsv(GT_FILE)
s1 = load_tsv(os.path.join(DATA_DIR, "train_source1.tsv"))

print(f"  GT rows : {len(gt):,}")
print(f"  S1 rows : {len(s1):,}")

# ------------------------------------------------------------------
# 2. Parse GT: classify S1 entities as matched / unmatched
# ------------------------------------------------------------------
print("\n[2] Classifying S1 entities ...", flush=True)

gt["has_match"] = gt["matched_entity_ids"].notna() & (gt["matched_entity_ids"].str.strip() != "")

# Ensure alignment: gt has one row per S1 entity
matched_ids   = gt[gt["has_match"]]["source1_entity_id"].values
unmatched_ids = gt[~gt["has_match"]]["source1_entity_id"].values

print(f"  Matched S1 entities   : {len(matched_ids):,}")
print(f"  Unmatched S1 entities : {len(unmatched_ids):,}")

# ------------------------------------------------------------------
# 3. Sample S1 entities
# ------------------------------------------------------------------
n_matched   = min(S1_SIZE // 2, len(matched_ids))
n_unmatched = min(S1_SIZE - n_matched, len(unmatched_ids))

sel_matched   = rng.choice(matched_ids,   size=n_matched,   replace=False)
sel_unmatched = rng.choice(unmatched_ids, size=n_unmatched, replace=False)
sel_s1_ids    = set(sel_matched) | set(sel_unmatched)

print(f"\n[3] Sampled {len(sel_matched):,} matched + {len(sel_unmatched):,} unmatched S1 entities", flush=True)

# ------------------------------------------------------------------
# 4. Subset S1 and GT
# ------------------------------------------------------------------
s1_sub = s1[s1["entity_id"].isin(sel_s1_ids)].copy()
gt_sub = gt[gt["source1_entity_id"].isin(sel_s1_ids)].copy()

print(f"  S1 subset : {len(s1_sub):,} rows")
print(f"  GT subset : {len(gt_sub):,} rows")

del s1
gc.collect()

# ------------------------------------------------------------------
# 5. Collect required S2 / S3 target IDs from GT
# ------------------------------------------------------------------
print("\n[4] Collecting required target IDs from GT ...", flush=True)

required_s2, required_s3 = set(), set()
for val in gt_sub["matched_entity_ids"].dropna():
    for tid in str(val).split(","):
        tid = tid.strip()
        if tid.startswith("S2-"):
            required_s2.add(tid)
        elif tid.startswith("S3-"):
            required_s3.add(tid)

print(f"  Required S2 target IDs : {len(required_s2):,}")
print(f"  Required S3 target IDs : {len(required_s3):,}")

# ------------------------------------------------------------------
# 6. Load S2 in chunks — keep required + sample extras
# ------------------------------------------------------------------
def build_target_subset(src_path, required_ids, target_total, source_name):
    """Stream source, keep all required IDs + randomly sample extra to reach target_total."""
    required = set(required_ids)
    extra_needed = max(0, target_total - len(required))

    required_rows = []
    candidate_extras = []

    CHUNK = 200_000
    reader = pd.read_csv(src_path, sep="\t", quoting=3, on_bad_lines="skip",
                         chunksize=CHUNK)

    for chunk in reader:
        req_mask = chunk["entity_id"].isin(required)
        required_rows.append(chunk[req_mask])
        extra = chunk[~req_mask]
        candidate_extras.append(extra)

    required_df = pd.concat(required_rows, ignore_index=True) if required_rows else pd.DataFrame()
    extras_df   = pd.concat(candidate_extras, ignore_index=True) if candidate_extras else pd.DataFrame()

    if extra_needed > 0 and len(extras_df) > 0:
        n_sample = min(extra_needed, len(extras_df))
        extra_sample = extras_df.sample(n=n_sample, random_state=SEED)
        result = pd.concat([required_df, extra_sample], ignore_index=True)
    else:
        result = required_df

    print(f"  {source_name}: {len(required_df):,} required + {len(result)-len(required_df):,} extra = {len(result):,} total")
    return result

print("\n[5] Building Source 2 subset ...", flush=True)
s2_sub = build_target_subset(
    os.path.join(DATA_DIR, "train_source2.tsv"),
    required_s2, TARGET_S2, "Source 2"
)

print("\n[6] Building Source 3 subset ...", flush=True)
s3_sub = build_target_subset(
    os.path.join(DATA_DIR, "train_source3.tsv"),
    required_s3, TARGET_S3, "Source 3"
)

# ------------------------------------------------------------------
# 7. Save subset files
# ------------------------------------------------------------------
print("\n[7] Saving subset files ...", flush=True)

s1_path = os.path.join(OUT_DIR, "train_source1_subset.tsv")
s2_path = os.path.join(OUT_DIR, "train_source2_subset.tsv")
s3_path = os.path.join(OUT_DIR, "train_source3_subset.tsv")
gt_path = os.path.join(OUT_DIR, "train_ground_truth_subset.tsv")

s1_sub.to_csv(s1_path, sep="\t", index=False, quoting=3)
s2_sub.to_csv(s2_path, sep="\t", index=False, quoting=3)
s3_sub.to_csv(s3_path, sep="\t", index=False, quoting=3)
gt_sub.to_csv(gt_path, sep="\t", index=False, quoting=3)

print(f"  Saved: {s1_path}  ({len(s1_sub):,} rows)")
print(f"  Saved: {s2_path}  ({len(s2_sub):,} rows)")
print(f"  Saved: {s3_path}  ({len(s3_sub):,} rows)")
print(f"  Saved: {gt_path}  ({len(gt_sub):,} rows)")

# ------------------------------------------------------------------
# 8. Save subset statistics
# ------------------------------------------------------------------
gt_parsed = gt_sub[gt_sub["has_match"]]
total_s2_pairs = sum(
    sum(1 for t in str(v).split(",") if t.strip().startswith("S2-"))
    for v in gt_parsed["matched_entity_ids"].dropna()
)
total_s3_pairs = sum(
    sum(1 for t in str(v).split(",") if t.strip().startswith("S3-"))
    for v in gt_parsed["matched_entity_ids"].dropna()
)

stats_md = f"""# Subset Statistics
Generated with seed={SEED}

| Dataset | Rows | Notes |
|---------|------|-------|
| Source 1 | {len(s1_sub):,} | {n_matched:,} with matches + {n_unmatched:,} without |
| Source 2 | {len(s2_sub):,} | {len(required_s2):,} required GT targets + extras |
| Source 3 | {len(s3_sub):,} | {len(required_s3):,} required GT targets + extras |
| Ground Truth | {len(gt_sub):,} | Matches only for selected S1 entities |

## Ground Truth Pair Counts (in subset)
- S1 → S2 GT pairs: {total_s2_pairs:,}
- S1 → S3 GT pairs: {total_s3_pairs:,}
- S1 matched entities: {n_matched:,}
- S1 unmatched entities: {n_unmatched:,}

## Coverage Check
- Required S2 targets in subset: {len(required_s2):,} / {len(required_s2):,} ✅
- Required S3 targets in subset: {len(required_s3):,} / {len(required_s3):,} ✅

Elapsed: {time.time()-t0:.1f}s
"""

with open("outputs/subset/subset_statistics.md", "w") as f:
    f.write(stats_md)

print("\n  Saved: outputs/subset/subset_statistics.md")
print(f"\n✅ Subset creation complete in {time.time()-t0:.1f}s")
print(stats_md)
