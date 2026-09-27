"""
evaluate_strategy2.py

Strategy 2: Country + First 2 Meaningful Normalized Business-Name Tokens

Evaluates S1->S2 and S1->S3 separately using a streaming inverted-index
with a hard 500-candidate/entity safety cap.

Also evaluates Strategy 1 UNION Strategy 2 without materializing the full set.

All results are saved to:
  outputs/reports/blocking_results.csv
  outputs/reports/strategy1_vs_strategy2.md
"""

import os
import sys
import time
import gc
import tracemalloc
import statistics

import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# Allow running from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import preprocess_dataframe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR       = "data"
REPORTS_DIR    = "outputs/reports"
SAFETY_CAP     = 500
PROGRESS_EVERY = 200_000

GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")
if not os.path.exists(GT_FILE):
    GT_FILE = os.path.join(DATA_DIR, "train_ground_truths.tsv")

STRATEGY1_CSV = os.path.join(REPORTS_DIR, "strategy1_results.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tsv(path, name):
    print(f"  Loading {name} from {path} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", quoting=3, on_bad_lines="skip")
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
    return df


def parse_ground_truth(gt_df):
    """Return dicts: s1_id -> set of S2 IDs, s1_id -> set of S3 IDs."""
    gt_s2 = defaultdict(set)
    gt_s3 = defaultdict(set)
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        val   = row["matched_entity_ids"]
        if pd.notna(val) and str(val).strip():
            for tid in str(val).split(","):
                tid = tid.strip()
                if tid.startswith("S2-"):
                    gt_s2[s1_id].add(tid)
                elif tid.startswith("S3-"):
                    gt_s3[s1_id].add(tid)
    return dict(gt_s2), dict(gt_s3)


def build_2tok_index(target_df):
    """Build inverted index: (norm_country, first_token, second_token) -> set of entity_ids."""
    print("    Building 2-token inverted index ...", flush=True)
    t0 = time.time()
    idx = defaultdict(set)
    c_arr  = target_df["norm_country"].values
    eid    = target_df["entity_id"].values
    t1_arr = target_df["first_token"].values
    t2_arr = target_df["second_token"].values

    for c, e, t1, t2 in zip(c_arr, eid, t1_arr, t2_arr):
        if c and t1 and t2:
            idx[(c, t1, t2)].add(e)

    elapsed = time.time() - t0
    mem_mb  = sum(sys.getsizeof(v) for v in idx.values()) / 1e6
    print(f"    Index built: {len(idx):,} keys in {elapsed:.1f}s, ~{mem_mb:.0f} MB", flush=True)
    return idx, elapsed, mem_mb


def build_exact_index(target_df):
    """Build inverted index: (norm_country, norm_name) -> set of entity_ids  (Strategy 1)."""
    idx = defaultdict(set)
    c_arr = target_df["norm_country"].values
    eid   = target_df["entity_id"].values
    n_arr = target_df["norm_name"].values
    for c, e, n in zip(c_arr, eid, n_arr):
        if c and n:
            idx[(c, n)].add(e)
    return idx


def evaluate_streaming(
    s1_df,
    idx_2tok,
    idx_exact,          # for union evaluation
    gt_map,
    source_label,
    cap=SAFETY_CAP,
):
    """
    Stream over S1 entities, look up candidates from idx_2tok.
    Tracks cap impact meticulously.
    Returns metrics dict.
    """
    total_gt_pairs   = sum(len(v) for v in gt_map.values())
    n_s1             = len(s1_df)

    # S1 arrays
    s1_c   = s1_df["norm_country"].values
    s1_eid = s1_df["entity_id"].values
    s1_t1  = s1_df["first_token"].values
    s1_t2  = s1_df["second_token"].values
    # Also need exact name for union
    s1_n   = s1_df["norm_name"].values

    # ---- Accumulators: Strategy 2 only ----
    found_pairs_s2       = 0
    complete_ents_s2     = 0
    cand_counts_s2       = []

    cap_entities         = 0        # S1 entities where cap fired
    cap_removed_pairs    = 0        # candidate pairs removed by cap
    cap_gt_lost          = 0        # GT pairs potentially lost due to cap

    # ---- Accumulators: Union (S1 ∪ S2) ----
    found_pairs_union    = 0
    complete_ents_union  = 0
    cand_counts_union    = []

    t0    = time.time()
    start = t0

    for i, (c, eid, t1, t2, n) in enumerate(
        zip(s1_c, s1_eid, s1_t1, s1_t2, s1_n), start=1
    ):
        # --- Strategy 2 candidates ---
        raw_cands = set()
        if c and t1 and t2:
            raw_cands = idx_2tok.get((c, t1, t2), set())

        raw_count = len(raw_cands)

        if raw_count > cap:
            cap_entities      += 1
            cap_removed_pairs += (raw_count - cap)
            # Estimate GT pairs lost: how many GT targets are NOT in the cap'd subset
            gt_targets = gt_map.get(eid, set())
            if gt_targets:
                # After cap we can't know WHICH ids survive, worst case all GT lost
                # But we can check: are GT targets actually in the raw bucket?
                gt_in_bucket = gt_targets & raw_cands
                cap_gt_lost += len(gt_in_bucket)   # pessimistic: assume all lost by cap
            cands_s2 = set(list(raw_cands)[:cap])  # deterministic slice for repeatability
        else:
            cands_s2   = raw_cands
            gt_targets = gt_map.get(eid, set())

        nc_s2 = len(cands_s2)
        cand_counts_s2.append(nc_s2)

        gt_targets = gt_map.get(eid, set())
        if gt_targets:
            matched = len(gt_targets & cands_s2)
            found_pairs_s2 += matched
            if gt_targets.issubset(cands_s2):
                complete_ents_s2 += 1

        # --- Union candidates (Strategy 1 exact + Strategy 2 2-tok) ---
        exact_cands = idx_exact.get((c, n), set()) if (c and n) else set()
        union_cands = cands_s2 | exact_cands         # no extra memory explosion: both are small sets per entity
        nc_union    = len(union_cands)
        cand_counts_union.append(nc_union)

        if gt_targets:
            matched_u = len(gt_targets & union_cands)
            found_pairs_union += matched_u
            if gt_targets.issubset(union_cands):
                complete_ents_union += 1

        if i % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            pct     = 100 * i / n_s1
            eta     = (elapsed / i) * (n_s1 - i)
            print(
                f"  [{source_label}] {i:,}/{n_s1:,} ({pct:.1f}%)  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                f"cands_so_far={sum(cand_counts_s2):,}  cap_hits={cap_entities:,}",
                flush=True,
            )
            t0 = time.time()

    elapsed_total = time.time() - start

    # ---- Strategy 2 metrics ----
    tot_cands_s2 = sum(cand_counts_s2)
    pair_rec_s2  = (found_pairs_s2 / total_gt_pairs * 100) if total_gt_pairs else 0
    comp_rec_s2  = (complete_ents_s2 / len(gt_map) * 100) if gt_map else 0
    avg_s2       = np.mean(cand_counts_s2)
    med_s2       = np.median(cand_counts_s2)
    p95_s2       = np.percentile(cand_counts_s2, 95)
    max_s2       = int(np.max(cand_counts_s2))

    # ---- Cap impact ----
    cap_rec_impact = (cap_gt_lost / total_gt_pairs * 100) if total_gt_pairs else 0

    # ---- Union metrics ----
    tot_cands_u  = sum(cand_counts_union)
    pair_rec_u   = (found_pairs_union / total_gt_pairs * 100) if total_gt_pairs else 0
    comp_rec_u   = (complete_ents_union / len(gt_map) * 100) if gt_map else 0
    avg_u        = np.mean(cand_counts_union)
    med_u        = np.median(cand_counts_union)
    p95_u        = np.percentile(cand_counts_union, 95)
    max_u        = int(np.max(cand_counts_union))

    return {
        # identification
        "source_label":         source_label,
        "total_s1":             n_s1,
        "total_gt_pairs":       total_gt_pairs,
        "elapsed_s":            round(elapsed_total, 2),
        # Strategy 2
        "s2_candidate_pairs":   tot_cands_s2,
        "s2_pair_recall":       round(pair_rec_s2, 4),
        "s2_complete_recall":   round(comp_rec_s2, 4),
        "s2_avg_cands":         round(avg_s2, 3),
        "s2_median_cands":      round(med_s2, 3),
        "s2_p95_cands":         round(p95_s2, 1),
        "s2_max_cands":         max_s2,
        # Cap impact
        "cap_threshold":        cap,
        "cap_entities":         cap_entities,
        "cap_removed_pairs":    cap_removed_pairs,
        "cap_gt_lost_pessimistic": cap_gt_lost,
        "cap_recall_impact_pct": round(cap_rec_impact, 4),
        # Union
        "union_candidate_pairs":  tot_cands_u,
        "union_pair_recall":      round(pair_rec_u, 4),
        "union_complete_recall":  round(comp_rec_u, 4),
        "union_avg_cands":        round(avg_u, 3),
        "union_median_cands":     round(med_u, 3),
        "union_p95_cands":        round(p95_u, 1),
        "union_max_cands":        max_u,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Loading datasets ===", flush=True)
    s1 = load_tsv(os.path.join(DATA_DIR, "train_source1.tsv"), "Source 1")
    s2 = load_tsv(os.path.join(DATA_DIR, "train_source2.tsv"), "Source 2")
    s3 = load_tsv(os.path.join(DATA_DIR, "train_source3.tsv"), "Source 3")
    gt = load_tsv(GT_FILE, "Ground Truth")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("\n=== Preprocessing ===", flush=True)
    t0 = time.time()
    s1_p = preprocess_dataframe(s1)
    s2_p = preprocess_dataframe(s2)
    s3_p = preprocess_dataframe(s3)
    print(f"  Preprocessing done in {time.time()-t0:.1f}s", flush=True)
    del s1, s2, s3
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Parse ground truth
    # ------------------------------------------------------------------
    print("\n=== Parsing ground truth ===", flush=True)
    gt_s2, gt_s3 = parse_ground_truth(gt)
    del gt
    gc.collect()
    print(f"  GT pairs -> S2: {sum(len(v) for v in gt_s2.values()):,}  "
          f"S3: {sum(len(v) for v in gt_s3.values()):,}", flush=True)

    results = []

    # ==================================================================
    # SOURCE 1 → SOURCE 2
    # ==================================================================
    print("\n" + "="*60, flush=True)
    print("=== STRATEGY 2:  S1 → SOURCE 2 ===", flush=True)
    print("="*60, flush=True)

    print("\n  [S2] Building Strategy 2 (2-token) index ...", flush=True)
    idx_2tok_s2, _, idx_mem_s2 = build_2tok_index(s2_p)

    print("\n  [S2] Building Strategy 1 (exact-name) index for union ...", flush=True)
    idx_exact_s2 = build_exact_index(s2_p)

    print(f"\n  [S2] Streaming {len(s1_p):,} S1 entities ...", flush=True)
    res_s2 = evaluate_streaming(
        s1_df      = s1_p,
        idx_2tok   = idx_2tok_s2,
        idx_exact  = idx_exact_s2,
        gt_map     = gt_s2,
        source_label = "S1→S2",
        cap        = SAFETY_CAP,
    )
    res_s2["index_mem_mb"] = round(idx_mem_s2, 1)

    del idx_2tok_s2, idx_exact_s2
    gc.collect()

    # ------------------------------------------------------------------
    # Print S2 summary
    # ------------------------------------------------------------------
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  STRATEGY 2  (S1→S2)  Country + First 2 Tokens             │
  ├─────────────────────────────────────────────────────────────┤
  │  Candidate pairs       : {res_s2["s2_candidate_pairs"]:>15,}              │
  │  Pair recall           : {res_s2["s2_pair_recall"]:>14.2f}%              │
  │  Complete entity recall: {res_s2["s2_complete_recall"]:>14.2f}%              │
  │  Avg cands / S1 entity : {res_s2["s2_avg_cands"]:>15.2f}              │
  │  Median cands / entity : {res_s2["s2_median_cands"]:>15.2f}              │
  │  P95 cands / entity    : {res_s2["s2_p95_cands"]:>15.1f}              │
  │  Max cands / entity    : {res_s2["s2_max_cands"]:>15,}              │
  │  Runtime               : {res_s2["elapsed_s"]:>14.1f}s              │
  ├──────────────── 500-CAP IMPACT ──────────────────────────────┤
  │  S1 entities capped    : {res_s2["cap_entities"]:>15,}              │
  │  Pairs removed by cap  : {res_s2["cap_removed_pairs"]:>15,}              │
  │  GT pairs lost (pessim): {res_s2["cap_gt_lost_pessimistic"]:>15,}              │
  │  GT recall impact      : {res_s2["cap_recall_impact_pct"]:>14.4f}%              │
  ├──────────────── UNION S1+S2 ────────────────────────────────┤
  │  Union candidate pairs : {res_s2["union_candidate_pairs"]:>15,}              │
  │  Union pair recall     : {res_s2["union_pair_recall"]:>14.2f}%              │
  │  Union complete recall : {res_s2["union_complete_recall"]:>14.2f}%              │
  │  Union avg cands/entity: {res_s2["union_avg_cands"]:>15.2f}              │
  └─────────────────────────────────────────────────────────────┘
    """, flush=True)

    results.append(("Source 2", "Strategy 2: Country + First 2 Tokens", res_s2))

    # ==================================================================
    # SOURCE 1 → SOURCE 3
    # ==================================================================
    print("\n" + "="*60, flush=True)
    print("=== STRATEGY 2:  S1 → SOURCE 3 ===", flush=True)
    print("="*60, flush=True)

    print("\n  [S3] Building Strategy 2 (2-token) index ...", flush=True)
    idx_2tok_s3, _, idx_mem_s3 = build_2tok_index(s3_p)

    print("\n  [S3] Building Strategy 1 (exact-name) index for union ...", flush=True)
    idx_exact_s3 = build_exact_index(s3_p)

    print(f"\n  [S3] Streaming {len(s1_p):,} S1 entities ...", flush=True)
    res_s3 = evaluate_streaming(
        s1_df      = s1_p,
        idx_2tok   = idx_2tok_s3,
        idx_exact  = idx_exact_s3,
        gt_map     = gt_s3,
        source_label = "S1→S3",
        cap        = SAFETY_CAP,
    )
    res_s3["index_mem_mb"] = round(idx_mem_s3, 1)

    del idx_2tok_s3, idx_exact_s3
    gc.collect()

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  STRATEGY 2  (S1→S3)  Country + First 2 Tokens             │
  ├─────────────────────────────────────────────────────────────┤
  │  Candidate pairs       : {res_s3["s2_candidate_pairs"]:>15,}              │
  │  Pair recall           : {res_s3["s2_pair_recall"]:>14.2f}%              │
  │  Complete entity recall: {res_s3["s2_complete_recall"]:>14.2f}%              │
  │  Avg cands / S1 entity : {res_s3["s2_avg_cands"]:>15.2f}              │
  │  Median cands / entity : {res_s3["s2_median_cands"]:>15.2f}              │
  │  P95 cands / entity    : {res_s3["s2_p95_cands"]:>15.1f}              │
  │  Max cands / entity    : {res_s3["s2_max_cands"]:>15,}              │
  │  Runtime               : {res_s3["elapsed_s"]:>14.1f}s              │
  ├──────────────── 500-CAP IMPACT ──────────────────────────────┤
  │  S1 entities capped    : {res_s3["cap_entities"]:>15,}              │
  │  Pairs removed by cap  : {res_s3["cap_removed_pairs"]:>15,}              │
  │  GT pairs lost (pessim): {res_s3["cap_gt_lost_pessimistic"]:>15,}              │
  │  GT recall impact      : {res_s3["cap_recall_impact_pct"]:>14.4f}%              │
  ├──────────────── UNION S1+S3 ────────────────────────────────┤
  │  Union candidate pairs : {res_s3["union_candidate_pairs"]:>15,}              │
  │  Union pair recall     : {res_s3["union_pair_recall"]:>14.2f}%              │
  │  Union complete recall : {res_s3["union_complete_recall"]:>14.2f}%              │
  │  Union avg cands/entity: {res_s3["union_avg_cands"]:>15.2f}              │
  └─────────────────────────────────────────────────────────────┘
    """, flush=True)

    results.append(("Source 3", "Strategy 2: Country + First 2 Tokens", res_s3))

    # ==================================================================
    # Save blocking_results.csv  (append/replace Strategy 2 rows)
    # ==================================================================
    print("\n=== Saving results ===", flush=True)

    # Read existing Strategy 1 results
    s1_csv_rows = []
    if os.path.exists(STRATEGY1_CSV):
        s1_csv_rows = pd.read_csv(STRATEGY1_CSV).to_dict(orient="records")

    # Build new rows for blocking_results.csv
    new_rows = []

    # Include existing Strategy 1 rows
    for r in s1_csv_rows:
        new_rows.append(r)

    # Add Strategy 2 rows
    for (target_src, strat_name, res) in results:
        new_rows.append({
            "Target Source":             target_src,
            "Strategy":                  strat_name,
            "Elapsed Time (s)":          res["elapsed_s"],
            "Est Index Memory (MB)":     res["index_mem_mb"],
            "Candidate Pairs":           res["s2_candidate_pairs"],
            "Pair Recall %":             res["s2_pair_recall"],
            "Complete Entity Recall %":  res["s2_complete_recall"],
            "Avg Candidates/S1":         res["s2_avg_cands"],
            "Median Candidates/S1":      res["s2_median_cands"],
            "95th Percentile":           res["s2_p95_cands"],
            "Max Candidates/S1":         res["s2_max_cands"],
            "Cap Threshold":             res["cap_threshold"],
            "Cap Entities Hit":          res["cap_entities"],
            "Cap Pairs Removed":         res["cap_removed_pairs"],
            "Cap GT Lost (pessimistic)": res["cap_gt_lost_pessimistic"],
            "Cap Recall Impact %":       res["cap_recall_impact_pct"],
        })

    # Add Union rows
    for (target_src, _, res) in results:
        new_rows.append({
            "Target Source":             target_src,
            "Strategy":                  "Union: Strategy 1 + Strategy 2",
            "Elapsed Time (s)":          res["elapsed_s"],
            "Est Index Memory (MB)":     res["index_mem_mb"],
            "Candidate Pairs":           res["union_candidate_pairs"],
            "Pair Recall %":             res["union_pair_recall"],
            "Complete Entity Recall %":  res["union_complete_recall"],
            "Avg Candidates/S1":         res["union_avg_cands"],
            "Median Candidates/S1":      res["union_median_cands"],
            "95th Percentile":           res["union_p95_cands"],
            "Max Candidates/S1":         res["union_max_cands"],
            "Cap Threshold":             res["cap_threshold"],
            "Cap Entities Hit":          "",
            "Cap Pairs Removed":         "",
            "Cap GT Lost (pessimistic)": "",
            "Cap Recall Impact %":       "",
        })

    out_df = pd.DataFrame(new_rows)
    out_path = os.path.join(REPORTS_DIR, "blocking_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}", flush=True)

    # ==================================================================
    # Save strategy1_vs_strategy2.md
    # ==================================================================
    def fmt_row(d, key, fmt=","):
        v = d.get(key, "N/A")
        if v == "" or v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        if fmt == "," and isinstance(v, (int, float)):
            return f"{int(v):,}"
        if fmt == "%" and isinstance(v, (int, float)):
            return f"{v:.2f}%"
        return str(v)

    # Pull Strategy 1 numbers from CSV
    s1_rows = {r["Target Source"]: r for r in s1_csv_rows}
    r2_s2   = results[0][2]   # res_s2
    r2_s3   = results[1][2]   # res_s3

    def get_s1_val(src, key, default="N/A"):
        row = s1_rows.get(src, {})
        return row.get(key, default)

    md = f"""# Blocking Strategy Comparison Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## Strategy Definitions

| Strategy | Blocking Key |
|----------|-------------|
| **Strategy 1** | Country + Exact Normalized Business Name |
| **Strategy 2** | Country + First 2 Meaningful Normalized Business-Name Tokens |
| **Union S1∪S2** | Candidates from Strategy 1 OR Strategy 2 (per entity, streaming union) |

> **Safety cap**: Strategy 2 applies a **{SAFETY_CAP}-candidate/entity** hard cap.
> Pairs removed by cap are tracked and reported below.

---

## Source 1 → Source 2

### Strategy 2 vs Strategy 1

| Metric | Strategy 1 | Strategy 2 (cap={SAFETY_CAP}) | Union S1∪S2 |
|--------|-----------|------------------------------|-------------|
| Candidate Pairs | {get_s1_val('Source 2', 'Candidate Pairs', 'N/A')} | {r2_s2['s2_candidate_pairs']:,} | {r2_s2['union_candidate_pairs']:,} |
| Pair Recall % | {get_s1_val('Source 2', 'Pair Recall %', 'N/A')} | {r2_s2['s2_pair_recall']:.2f}% | {r2_s2['union_pair_recall']:.2f}% |
| Complete Entity Recall % | {get_s1_val('Source 2', 'Complete Entity Recall %', 'N/A')} | {r2_s2['s2_complete_recall']:.2f}% | {r2_s2['union_complete_recall']:.2f}% |
| Avg Cands / S1 | {get_s1_val('Source 2', 'Avg Candidates/S1', 'N/A')} | {r2_s2['s2_avg_cands']:.2f} | {r2_s2['union_avg_cands']:.2f} |
| Median Cands / S1 | N/A | {r2_s2['s2_median_cands']:.2f} | {r2_s2['union_median_cands']:.2f} |
| P95 Cands / S1 | {get_s1_val('Source 2', '95th Percentile', 'N/A')} | {r2_s2['s2_p95_cands']:.1f} | {r2_s2['union_p95_cands']:.1f} |
| Max Cands / S1 | {get_s1_val('Source 2', 'Max Candidates/S1', 'N/A')} | {r2_s2['s2_max_cands']:,} | {r2_s2['union_max_cands']:,} |
| Runtime (s) | {get_s1_val('Source 2', 'Elapsed Time (s)', 'N/A')} | {r2_s2['elapsed_s']:.1f} | (same run) |

### 500-Cap Impact (S1→S2)

| Metric | Value |
|--------|-------|
| S1 entities affected by cap | {r2_s2['cap_entities']:,} |
| Candidate pairs removed by cap | {r2_s2['cap_removed_pairs']:,} |
| GT pairs potentially lost (pessimistic) | {r2_s2['cap_gt_lost_pessimistic']:,} |
| GT recall impact (pessimistic upper bound) | {r2_s2['cap_recall_impact_pct']:.4f}% |

> **Interpretation**: The "GT pairs lost" figure is a **pessimistic upper bound** — it counts all GT pairs
> that were in an over-sized bucket, assuming the cap removed all of them. In practice, if GT targets
> appear early in the bucket, they may survive the cap. Zero cap impact means zero GT recall loss.

---

## Source 1 → Source 3

### Strategy 2 vs Strategy 1

| Metric | Strategy 1 | Strategy 2 (cap={SAFETY_CAP}) | Union S1∪S3 |
|--------|-----------|------------------------------|-------------|
| Candidate Pairs | {get_s1_val('Source 3', 'Candidate Pairs', 'N/A')} | {r2_s3['s2_candidate_pairs']:,} | {r2_s3['union_candidate_pairs']:,} |
| Pair Recall % | {get_s1_val('Source 3', 'Pair Recall %', 'N/A')} | {r2_s3['s2_pair_recall']:.2f}% | {r2_s3['union_pair_recall']:.2f}% |
| Complete Entity Recall % | {get_s1_val('Source 3', 'Complete Entity Recall %', 'N/A')} | {r2_s3['s2_complete_recall']:.2f}% | {r2_s3['union_complete_recall']:.2f}% |
| Avg Cands / S1 | {get_s1_val('Source 3', 'Avg Candidates/S1', 'N/A')} | {r2_s3['s2_avg_cands']:.2f} | {r2_s3['union_avg_cands']:.2f} |
| Median Cands / S1 | N/A | {r2_s3['s2_median_cands']:.2f} | {r2_s3['union_median_cands']:.2f} |
| P95 Cands / S1 | {get_s1_val('Source 3', '95th Percentile', 'N/A')} | {r2_s3['s2_p95_cands']:.1f} | {r2_s3['union_p95_cands']:.1f} |
| Max Cands / S1 | {get_s1_val('Source 3', 'Max Candidates/S1', 'N/A')} | {r2_s3['s2_max_cands']:,} | {r2_s3['union_max_cands']:,} |
| Runtime (s) | {get_s1_val('Source 3', 'Elapsed Time (s)', 'N/A')} | {r2_s3['elapsed_s']:.1f} | (same run) |

### 500-Cap Impact (S1→S3)

| Metric | Value |
|--------|-------|
| S1 entities affected by cap | {r2_s3['cap_entities']:,} |
| Candidate pairs removed by cap | {r2_s3['cap_removed_pairs']:,} |
| GT pairs potentially lost (pessimistic) | {r2_s3['cap_gt_lost_pessimistic']:,} |
| GT recall impact (pessimistic upper bound) | {r2_s3['cap_recall_impact_pct']:.4f}% |

---

## Overall Cap Verdict

{"✅ **The 500-candidate cap caused NO loss of ground-truth recall.**" if (r2_s2['cap_gt_lost_pessimistic'] == 0 and r2_s3['cap_gt_lost_pessimistic'] == 0) else
 f"⚠️ **The 500-candidate cap caused a pessimistic recall loss of up to {r2_s2['cap_recall_impact_pct']:.4f}% (S2) and {r2_s3['cap_recall_impact_pct']:.4f}% (S3).**"}

---

## Key Takeaways

- **Strategy 1** (exact name) is a high-precision, low-recall blocker — only captures entities with identical normalized names.
- **Strategy 2** (first 2 tokens) is a softer blocker — captures more candidates at the cost of more false positives.
- **Union** always has recall ≥ max(Strat1, Strat2) since it accumulates candidates from both strategies.
- Union candidate count is smaller than Strat1 + Strat2 due to set de-duplication per entity.

---

*Generated by evaluate_strategy2.py*
"""

    md_path = os.path.join(REPORTS_DIR, "strategy1_vs_strategy2.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Saved {md_path}", flush=True)

    print("\n=== ALL DONE ===", flush=True)
    print("Strategy 2 evaluation complete. No further strategies implemented.", flush=True)


if __name__ == "__main__":
    main()
