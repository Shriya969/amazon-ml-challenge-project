"""
evaluate_strategy3.py

Strategy 3: Country + First 3 Meaningful Normalized Business-Name Tokens

- Evaluates S1->S2 and S1->S3 independently
- Uses 500-candidate safety cap with full impact tracking
- Computes cumulative union: Strategy 1 ∪ Strategy 2 ∪ Strategy 3
- Reports incremental gain over the Strategy 1 ∪ Strategy 2 union
- Saves: strategy3_results.csv, blocking_results.csv, blocking_summary.md
"""

import os
import sys
import time
import gc

import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import preprocess_dataframe

# ---------------------------------------------------------------------------
DATA_DIR       = "data"
REPORTS_DIR    = "outputs/reports"
SAFETY_CAP     = 500
PROGRESS_EVERY = 200_000

GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")
if not os.path.exists(GT_FILE):
    GT_FILE = os.path.join(DATA_DIR, "train_ground_truths.tsv")

# Strategy 1+2 union recall from previous run (for incremental calculation)
# Populated later from blocking_results.csv
PREV_UNION_RECALL = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tsv(path, name):
    print(f"  Loading {name} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", quoting=3, on_bad_lines="skip")
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
    return df


def parse_ground_truth(gt_df):
    gt_s2, gt_s3 = defaultdict(set), defaultdict(set)
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


def build_3tok_index(target_df):
    """Inverted index: (country, t1, t2, t3) -> set of entity_ids."""
    print("    Building 3-token inverted index ...", flush=True)
    t0    = time.time()
    idx   = defaultdict(set)
    c_arr  = target_df["norm_country"].values
    eid    = target_df["entity_id"].values
    t1_arr = target_df["first_token"].values
    t2_arr = target_df["second_token"].values
    t3_arr = target_df["third_token"].values

    for c, e, t1, t2, t3 in zip(c_arr, eid, t1_arr, t2_arr, t3_arr):
        if c and t1 and t2 and t3:          # all three tokens must be non-empty
            idx[(c, t1, t2, t3)].add(e)

    elapsed = time.time() - t0
    mem_mb  = sum(sys.getsizeof(v) for v in idx.values()) / 1e6
    print(f"    Index built: {len(idx):,} keys in {elapsed:.1f}s, ~{mem_mb:.0f} MB", flush=True)
    return idx, elapsed, mem_mb


def build_2tok_index(target_df):
    """Inverted index: (country, t1, t2) -> set of entity_ids."""
    idx   = defaultdict(set)
    c_arr  = target_df["norm_country"].values
    eid    = target_df["entity_id"].values
    t1_arr = target_df["first_token"].values
    t2_arr = target_df["second_token"].values
    for c, e, t1, t2 in zip(c_arr, eid, t1_arr, t2_arr):
        if c and t1 and t2:
            idx[(c, t1, t2)].add(e)
    return idx


def build_exact_index(target_df):
    """Inverted index: (country, norm_name) -> set of entity_ids."""
    idx   = defaultdict(set)
    c_arr  = target_df["norm_country"].values
    eid    = target_df["entity_id"].values
    n_arr  = target_df["norm_name"].values
    for c, e, n in zip(c_arr, eid, n_arr):
        if c and n:
            idx[(c, n)].add(e)
    return idx


def evaluate_streaming(
    s1_df,
    idx_3tok,
    idx_2tok,
    idx_exact,
    gt_map,
    source_label,
    cap=SAFETY_CAP,
):
    """
    Stream over S1 entities; for each entity:
      - Look up Strategy 3 candidates (3-tok index, capped)
      - Look up Strategy 1+2 union (exact + 2-tok, uncapped — already evaluated before)
      - Compute cumulative union = exact ∪ 2tok ∪ 3tok(capped)
      - Track: Strategy 3 recall, union recall, cap impact, incremental new pairs
    """
    total_gt_pairs = sum(len(v) for v in gt_map.values())
    n_s1 = len(s1_df)

    s1_c   = s1_df["norm_country"].values
    s1_eid = s1_df["entity_id"].values
    s1_t1  = s1_df["first_token"].values
    s1_t2  = s1_df["second_token"].values
    s1_t3  = s1_df["third_token"].values
    s1_n   = s1_df["norm_name"].values

    # ---- Strategy 3 accumulators ----
    found_pairs_s3      = 0
    complete_ents_s3    = 0
    cand_counts_s3      = []
    cap_entities        = 0
    cap_removed_pairs   = 0
    cap_gt_lost         = 0

    # ---- Cumulative union accumulators ----
    found_pairs_union   = 0
    complete_ents_union = 0
    cand_counts_union   = []
    new_pairs_from_s3   = 0   # pairs in union_123 not already in union_12

    t0    = time.time()
    start = t0

    for i, (c, eid, t1, t2, t3, n) in enumerate(
        zip(s1_c, s1_eid, s1_t1, s1_t2, s1_t3, s1_n), start=1
    ):
        gt_targets = gt_map.get(eid, set())

        # --- Strategy 1+2 union (exact + 2-tok) ---
        exact_cands = idx_exact.get((c, n), set()) if (c and n) else set()
        tok2_cands  = idx_2tok.get((c, t1, t2), set()) if (c and t1 and t2) else set()
        union_12    = exact_cands | tok2_cands

        # --- Strategy 3 raw candidates ---
        raw_3tok = set()
        if c and t1 and t2 and t3:
            raw_3tok = idx_3tok.get((c, t1, t2, t3), set())

        raw_count = len(raw_3tok)

        if raw_count > cap:
            cap_entities      += 1
            cap_removed_pairs += (raw_count - cap)
            if gt_targets:
                gt_in_bucket = gt_targets & raw_3tok
                cap_gt_lost  += len(gt_in_bucket)
            cands_3tok = set(list(raw_3tok)[:cap])
        else:
            cands_3tok = raw_3tok

        # --- Strategy 3 only metrics ---
        nc_s3 = len(cands_3tok)
        cand_counts_s3.append(nc_s3)

        if gt_targets:
            matched_3 = len(gt_targets & cands_3tok)
            found_pairs_s3 += matched_3
            if gt_targets.issubset(cands_3tok):
                complete_ents_s3 += 1

        # --- Cumulative union ---
        union_123 = union_12 | cands_3tok
        nc_union  = len(union_123)
        cand_counts_union.append(nc_union)

        # Incremental new pairs: in union_123 but not in union_12
        new_pairs_from_s3 += max(0, nc_union - len(union_12))

        if gt_targets:
            matched_u = len(gt_targets & union_123)
            found_pairs_union += matched_u
            if gt_targets.issubset(union_123):
                complete_ents_union += 1

        if i % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            pct     = 100 * i / n_s1
            eta     = (elapsed / i) * (n_s1 - i)
            print(
                f"  [{source_label}] {i:,}/{n_s1:,} ({pct:.1f}%)  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                f"s3_cands={sum(cand_counts_s3):,}  cap_hits={cap_entities:,}  "
                f"union_cands={sum(cand_counts_union):,}",
                flush=True,
            )
            t0 = time.time()

    elapsed_total = time.time() - start

    # ---- Strategy 3 metrics ----
    tot_cands_s3 = sum(cand_counts_s3)
    pair_rec_s3  = (found_pairs_s3 / total_gt_pairs * 100) if total_gt_pairs else 0
    comp_rec_s3  = (complete_ents_s3 / len(gt_map) * 100) if gt_map else 0
    avg_s3       = float(np.mean(cand_counts_s3))
    med_s3       = float(np.median(cand_counts_s3))
    p95_s3       = float(np.percentile(cand_counts_s3, 95))
    max_s3       = int(np.max(cand_counts_s3))
    cap_recall_impact = (cap_gt_lost / total_gt_pairs * 100) if total_gt_pairs else 0

    # ---- Cumulative union metrics ----
    tot_cands_u  = sum(cand_counts_union)
    pair_rec_u   = (found_pairs_union / total_gt_pairs * 100) if total_gt_pairs else 0
    comp_rec_u   = (complete_ents_union / len(gt_map) * 100) if gt_map else 0
    avg_u        = float(np.mean(cand_counts_union))
    med_u        = float(np.median(cand_counts_union))
    p95_u        = float(np.percentile(cand_counts_union, 95))
    max_u        = int(np.max(cand_counts_union))

    return {
        "source_label":              source_label,
        "total_s1":                  n_s1,
        "total_gt_pairs":            total_gt_pairs,
        "elapsed_s":                 round(elapsed_total, 2),
        # Strategy 3
        "s3_candidate_pairs":        tot_cands_s3,
        "s3_pair_recall":            round(pair_rec_s3, 4),
        "s3_complete_recall":        round(comp_rec_s3, 4),
        "s3_avg_cands":              round(avg_s3, 3),
        "s3_median_cands":           round(med_s3, 3),
        "s3_p95_cands":              round(p95_s3, 1),
        "s3_max_cands":              max_s3,
        # Cap
        "cap_threshold":             cap,
        "cap_entities":              cap_entities,
        "cap_removed_pairs":         cap_removed_pairs,
        "cap_gt_lost_pessimistic":   cap_gt_lost,
        "cap_recall_impact_pct":     round(cap_recall_impact, 4),
        # Cumulative union
        "union_candidate_pairs":     tot_cands_u,
        "union_pair_recall":         round(pair_rec_u, 4),
        "union_complete_recall":     round(comp_rec_u, 4),
        "union_avg_cands":           round(avg_u, 3),
        "union_median_cands":        round(med_u, 3),
        "union_p95_cands":           round(p95_u, 1),
        "union_max_cands":           max_u,
        "new_pairs_from_s3":         new_pairs_from_s3,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # ------------------------------------------------------------------
    # Load previous union recall (Strategy 1+2) for incremental calc
    # ------------------------------------------------------------------
    prev_union = {}
    br_path = os.path.join(REPORTS_DIR, "blocking_results.csv")
    if os.path.exists(br_path):
        br_df = pd.read_csv(br_path)
        union_rows = br_df[br_df["Strategy"] == "Union: Strategy 1 + Strategy 2"]
        for _, row in union_rows.iterrows():
            src = row["Target Source"]
            prev_union[src] = {
                "pair_recall":      float(row["Pair Recall %"]),
                "complete_recall":  float(row["Complete Entity Recall %"]),
                "candidate_pairs":  int(row["Candidate Pairs"]),
            }
    print(f"\nPrevious Strategy 1+2 union recall loaded: {prev_union}", flush=True)

    # ------------------------------------------------------------------
    # 1. Load datasets
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
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    del s1, s2, s3
    gc.collect()

    # Verify third_token column exists
    assert "third_token" in s1_p.columns, "third_token missing from preprocessed S1!"
    print(f"  third_token column confirmed. Sample: {s1_p['third_token'].dropna().head(3).tolist()}", flush=True)

    # ------------------------------------------------------------------
    # 3. Parse ground truth
    # ------------------------------------------------------------------
    print("\n=== Parsing ground truth ===", flush=True)
    gt_s2, gt_s3 = parse_ground_truth(gt)
    del gt
    gc.collect()
    print(f"  GT pairs -> S2: {sum(len(v) for v in gt_s2.values()):,}  "
          f"S3: {sum(len(v) for v in gt_s3.values()):,}", flush=True)

    all_results = []

    # ==================================================================
    # STRATEGY 3: S1 → SOURCE 2
    # ==================================================================
    print("\n" + "="*60, flush=True)
    print("=== STRATEGY 3:  S1 → SOURCE 2 ===", flush=True)
    print("="*60, flush=True)

    print("\n  Building 3-token index (S2) ...", flush=True)
    idx_3tok_s2, _, idx_mem_s2 = build_3tok_index(s2_p)

    print("  Building 2-token index (S2) for union ...", flush=True)
    idx_2tok_s2 = build_2tok_index(s2_p)

    print("  Building exact-name index (S2) for union ...", flush=True)
    idx_exact_s2 = build_exact_index(s2_p)

    print(f"\n  Streaming {len(s1_p):,} S1 entities ...", flush=True)
    res_s2 = evaluate_streaming(
        s1_df        = s1_p,
        idx_3tok     = idx_3tok_s2,
        idx_2tok     = idx_2tok_s2,
        idx_exact    = idx_exact_s2,
        gt_map       = gt_s2,
        source_label = "S1→S2",
        cap          = SAFETY_CAP,
    )
    res_s2["index_mem_mb"] = round(idx_mem_s2, 1)

    # Incremental gain vs Strategy 1+2 union
    prev_s2 = prev_union.get("Source 2", {})
    incr_pair_rec_s2  = round(res_s2["union_pair_recall"] - prev_s2.get("pair_recall", 0), 4)
    incr_comp_rec_s2  = round(res_s2["union_complete_recall"] - prev_s2.get("complete_recall", 0), 4)
    res_s2["incremental_pair_recall"]     = incr_pair_rec_s2
    res_s2["incremental_complete_recall"] = incr_comp_rec_s2

    del idx_3tok_s2, idx_2tok_s2, idx_exact_s2
    gc.collect()

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  STRATEGY 3  (S1→S2)  Country + First 3 Tokens             │
  ├─────────────────────────────────────────────────────────────┤
  │  Candidate pairs       : {res_s2["s3_candidate_pairs"]:>15,}              │
  │  Pair recall           : {res_s2["s3_pair_recall"]:>14.2f}%              │
  │  Complete entity recall: {res_s2["s3_complete_recall"]:>14.2f}%              │
  │  Avg cands / S1        : {res_s2["s3_avg_cands"]:>15.2f}              │
  │  Median cands / S1     : {res_s2["s3_median_cands"]:>15.2f}              │
  │  P95 cands / S1        : {res_s2["s3_p95_cands"]:>15.1f}              │
  │  Max cands / S1        : {res_s2["s3_max_cands"]:>15,}              │
  │  Runtime               : {res_s2["elapsed_s"]:>14.1f}s              │
  │  Index memory          : {res_s2["index_mem_mb"]:>13.0f} MB              │
  ├──────────────── 500-CAP IMPACT ─────────────────────────────┤
  │  S1 entities capped    : {res_s2["cap_entities"]:>15,}              │
  │  Pairs removed by cap  : {res_s2["cap_removed_pairs"]:>15,}              │
  │  GT pairs lost (pessim): {res_s2["cap_gt_lost_pessimistic"]:>15,}              │
  │  GT recall impact      : {res_s2["cap_recall_impact_pct"]:>14.4f}%              │
  ├──────────────── CUMULATIVE UNION  S1∪S2∪S3 ────────────────┤
  │  Union candidate pairs : {res_s2["union_candidate_pairs"]:>15,}              │
  │  Union pair recall     : {res_s2["union_pair_recall"]:>14.2f}%              │
  │  Union complete recall : {res_s2["union_complete_recall"]:>14.2f}%              │
  │  Union avg cands/entity: {res_s2["union_avg_cands"]:>15.2f}              │
  │  New pairs from Strat3 : {res_s2["new_pairs_from_s3"]:>15,}              │
  ├──────────────── INCREMENTAL GAIN ───────────────────────────┤
  │  S1∪S2 pair recall     : {prev_s2.get("pair_recall", "N/A"):>14}%              │
  │  S1∪S2∪S3 pair recall  : {res_s2["union_pair_recall"]:>14.2f}%              │
  │  Incremental pair rec  : {incr_pair_rec_s2:>14.4f}%              │
  │  Incremental comp. rec : {incr_comp_rec_s2:>14.4f}%              │
  └─────────────────────────────────────────────────────────────┘
    """, flush=True)

    all_results.append(("Source 2", res_s2, prev_s2))

    # ==================================================================
    # STRATEGY 3: S1 → SOURCE 3
    # ==================================================================
    print("\n" + "="*60, flush=True)
    print("=== STRATEGY 3:  S1 → SOURCE 3 ===", flush=True)
    print("="*60, flush=True)

    print("\n  Building 3-token index (S3) ...", flush=True)
    idx_3tok_s3, _, idx_mem_s3 = build_3tok_index(s3_p)

    print("  Building 2-token index (S3) for union ...", flush=True)
    idx_2tok_s3 = build_2tok_index(s3_p)

    print("  Building exact-name index (S3) for union ...", flush=True)
    idx_exact_s3 = build_exact_index(s3_p)

    print(f"\n  Streaming {len(s1_p):,} S1 entities ...", flush=True)
    res_s3 = evaluate_streaming(
        s1_df        = s1_p,
        idx_3tok     = idx_3tok_s3,
        idx_2tok     = idx_2tok_s3,
        idx_exact    = idx_exact_s3,
        gt_map       = gt_s3,
        source_label = "S1→S3",
        cap          = SAFETY_CAP,
    )
    res_s3["index_mem_mb"] = round(idx_mem_s3, 1)

    prev_s3 = prev_union.get("Source 3", {})
    incr_pair_rec_s3  = round(res_s3["union_pair_recall"] - prev_s3.get("pair_recall", 0), 4)
    incr_comp_rec_s3  = round(res_s3["union_complete_recall"] - prev_s3.get("complete_recall", 0), 4)
    res_s3["incremental_pair_recall"]     = incr_pair_rec_s3
    res_s3["incremental_complete_recall"] = incr_comp_rec_s3

    del idx_3tok_s3, idx_2tok_s3, idx_exact_s3
    gc.collect()

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  STRATEGY 3  (S1→S3)  Country + First 3 Tokens             │
  ├─────────────────────────────────────────────────────────────┤
  │  Candidate pairs       : {res_s3["s3_candidate_pairs"]:>15,}              │
  │  Pair recall           : {res_s3["s3_pair_recall"]:>14.2f}%              │
  │  Complete entity recall: {res_s3["s3_complete_recall"]:>14.2f}%              │
  │  Avg cands / S1        : {res_s3["s3_avg_cands"]:>15.2f}              │
  │  Median cands / S1     : {res_s3["s3_median_cands"]:>15.2f}              │
  │  P95 cands / S1        : {res_s3["s3_p95_cands"]:>15.1f}              │
  │  Max cands / S1        : {res_s3["s3_max_cands"]:>15,}              │
  │  Runtime               : {res_s3["elapsed_s"]:>14.1f}s              │
  │  Index memory          : {res_s3["index_mem_mb"]:>13.0f} MB              │
  ├──────────────── 500-CAP IMPACT ─────────────────────────────┤
  │  S1 entities capped    : {res_s3["cap_entities"]:>15,}              │
  │  Pairs removed by cap  : {res_s3["cap_removed_pairs"]:>15,}              │
  │  GT pairs lost (pessim): {res_s3["cap_gt_lost_pessimistic"]:>15,}              │
  │  GT recall impact      : {res_s3["cap_recall_impact_pct"]:>14.4f}%              │
  ├──────────────── CUMULATIVE UNION  S1∪S2∪S3 ────────────────┤
  │  Union candidate pairs : {res_s3["union_candidate_pairs"]:>15,}              │
  │  Union pair recall     : {res_s3["union_pair_recall"]:>14.2f}%              │
  │  Union complete recall : {res_s3["union_complete_recall"]:>14.2f}%              │
  │  Union avg cands/entity: {res_s3["union_avg_cands"]:>15.2f}              │
  │  New pairs from Strat3 : {res_s3["new_pairs_from_s3"]:>15,}              │
  ├──────────────── INCREMENTAL GAIN ───────────────────────────┤
  │  S1∪S2 pair recall     : {prev_s3.get("pair_recall", "N/A"):>14}%              │
  │  S1∪S2∪S3 pair recall  : {res_s3["union_pair_recall"]:>14.2f}%              │
  │  Incremental pair rec  : {incr_pair_rec_s3:>14.4f}%              │
  │  Incremental comp. rec : {incr_comp_rec_s3:>14.4f}%              │
  └─────────────────────────────────────────────────────────────┘
    """, flush=True)

    all_results.append(("Source 3", res_s3, prev_s3))

    # ==================================================================
    # Save strategy3_results.csv
    # ==================================================================
    print("\n=== Saving results ===", flush=True)

    s3_rows = []
    for (target_src, res, _) in all_results:
        s3_rows.append({
            "Target Source":                  target_src,
            "Strategy":                       "Strategy 3: Country + First 3 Tokens",
            "Elapsed Time (s)":               res["elapsed_s"],
            "Est Index Memory (MB)":          res["index_mem_mb"],
            "Candidate Pairs":                res["s3_candidate_pairs"],
            "Pair Recall %":                  res["s3_pair_recall"],
            "Complete Entity Recall %":       res["s3_complete_recall"],
            "Avg Candidates/S1":              res["s3_avg_cands"],
            "Median Candidates/S1":           res["s3_median_cands"],
            "95th Percentile":                res["s3_p95_cands"],
            "Max Candidates/S1":              res["s3_max_cands"],
            "Cap Threshold":                  res["cap_threshold"],
            "Cap Entities Hit":               res["cap_entities"],
            "Cap Pairs Removed":              res["cap_removed_pairs"],
            "Cap GT Lost (pessimistic)":      res["cap_gt_lost_pessimistic"],
            "Cap Recall Impact %":            res["cap_recall_impact_pct"],
            "Union123 Candidate Pairs":       res["union_candidate_pairs"],
            "Union123 Pair Recall %":         res["union_pair_recall"],
            "Union123 Complete Recall %":     res["union_complete_recall"],
            "New Pairs from Strat3":          res["new_pairs_from_s3"],
            "Incremental Pair Recall %":      res["incremental_pair_recall"],
            "Incremental Complete Recall %":  res["incremental_complete_recall"],
        })

    s3_df = pd.DataFrame(s3_rows)
    s3_path = os.path.join(REPORTS_DIR, "strategy3_results.csv")
    s3_df.to_csv(s3_path, index=False)
    print(f"  Saved {s3_path}", flush=True)

    # ==================================================================
    # Update blocking_results.csv
    # ==================================================================
    existing_rows = []
    if os.path.exists(br_path):
        existing_df = pd.read_csv(br_path)
        # Drop any existing Strategy 3 rows to avoid duplicates
        existing_df = existing_df[~existing_df["Strategy"].str.contains("Strategy 3", na=False)]
        existing_df = existing_df[~existing_df["Strategy"].str.contains("Union: S1\\+S2\\+S3", na=False)]
        existing_rows = existing_df.to_dict(orient="records")

    new_rows = list(existing_rows)

    for (target_src, res, _) in all_results:
        # Strategy 3 standalone row
        new_rows.append({
            "Target Source":             target_src,
            "Strategy":                  "Strategy 3: Country + First 3 Tokens",
            "Elapsed Time (s)":          res["elapsed_s"],
            "Est Index Memory (MB)":     res["index_mem_mb"],
            "Candidate Pairs":           res["s3_candidate_pairs"],
            "Pair Recall %":             res["s3_pair_recall"],
            "Complete Entity Recall %":  res["s3_complete_recall"],
            "Avg Candidates/S1":         res["s3_avg_cands"],
            "Median Candidates/S1":      res["s3_median_cands"],
            "95th Percentile":           res["s3_p95_cands"],
            "Max Candidates/S1":         res["s3_max_cands"],
            "Cap Threshold":             res["cap_threshold"],
            "Cap Entities Hit":          res["cap_entities"],
            "Cap Pairs Removed":         res["cap_removed_pairs"],
            "Cap GT Lost (pessimistic)": res["cap_gt_lost_pessimistic"],
            "Cap Recall Impact %":       res["cap_recall_impact_pct"],
        })
        # Cumulative union row
        new_rows.append({
            "Target Source":             target_src,
            "Strategy":                  "Union: S1+S2+S3",
            "Elapsed Time (s)":          res["elapsed_s"],
            "Est Index Memory (MB)":     res["index_mem_mb"],
            "Candidate Pairs":           res["union_candidate_pairs"],
            "Pair Recall %":             res["union_pair_recall"],
            "Complete Entity Recall %":  res["union_complete_recall"],
            "Avg Candidates/S1":         res["union_avg_cands"],
            "Median Candidates/S1":      res["union_median_cands"],
            "95th Percentile":           res["union_p95_cands"],
            "Max Candidates/S1":         res["union_max_cands"],
            "Cap Threshold":             "",
            "Cap Entities Hit":          "",
            "Cap Pairs Removed":         "",
            "Cap GT Lost (pessimistic)": "",
            "Cap Recall Impact %":       "",
        })

    pd.DataFrame(new_rows).to_csv(br_path, index=False)
    print(f"  Updated {br_path}", flush=True)

    # ==================================================================
    # Save blocking_summary.md
    # ==================================================================
    r2 = all_results[0][1]
    r3 = all_results[1][1]
    p2 = all_results[0][2]
    p3 = all_results[1][2]

    md = f"""# Blocking Strategy Summary Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## Strategy Definitions

| # | Blocking Key | Notes |
|---|-------------|-------|
| **S1** | Country + Exact Normalized Business Name | High precision, lower recall |
| **S2** | Country + First 2 Meaningful Tokens | Broader match, cap=500 |
| **S3** | Country + First 3 Meaningful Tokens | Tighter match, cap=500 |
| **Union** | S1 ∪ S2 ∪ S3 | Cumulative, streaming set-union per entity |

> All token strategies strip legal suffixes (llc, inc, ltd, corp, pvt, limited, etc.) before tokenizing.
> Safety cap = **{SAFETY_CAP} candidates/entity** applied to Strategies 2 and 3.

---

## Source 1 → Source 2

### Individual Strategy Recall

| Strategy | Candidate Pairs | Pair Recall % | Complete Entity Recall % | Avg Cands | Median | P95 | Runtime |
|----------|----------------|--------------|--------------------------|-----------|--------|-----|---------|
| Strategy 1 (Exact) | ~10,344,168 | ~21.44% | ~10.78% | ~4.69 | — | ~30.0 | ~11s |
| Strategy 2 (2-tok) | 99,694,866 | 53.78% | 37.73% | 45.18 | 5.00 | 490.0 | ~20s |
| Strategy 3 (3-tok) | {r2["s3_candidate_pairs"]:,} | {r2["s3_pair_recall"]:.2f}% | {r2["s3_complete_recall"]:.2f}% | {r2["s3_avg_cands"]:.2f} | {r2["s3_median_cands"]:.2f} | {r2["s3_p95_cands"]:.1f} | {r2["elapsed_s"]:.1f}s |

### Cumulative Union Recall (S2)

| Strategies Combined | Candidate Pairs | Pair Recall % | Complete Entity Recall % | Avg Cands |
|---------------------|----------------|--------------|--------------------------|-----------|
| S1 ∪ S2 | 103,840,059 | 55.87% | 38.93% | 47.05 |
| **S1 ∪ S2 ∪ S3** | **{r2["union_candidate_pairs"]:,}** | **{r2["union_pair_recall"]:.2f}%** | **{r2["union_complete_recall"]:.2f}%** | **{r2["union_avg_cands"]:.2f}** |

### Strategy 3 Incremental Gain (S2)

| Metric | Value |
|--------|-------|
| New candidate pairs added by S3 | {r2["new_pairs_from_s3"]:,} |
| Incremental pair recall gain | **{r2["incremental_pair_recall"]:+.4f}%** |
| Incremental complete entity recall gain | **{r2["incremental_complete_recall"]:+.4f}%** |

### 500-Cap Impact — Strategy 3 (S2)

| Metric | Value |
|--------|-------|
| S1 entities capped | {r2["cap_entities"]:,} |
| Candidate pairs removed by cap | {r2["cap_removed_pairs"]:,} |
| GT pairs potentially lost (pessimistic) | {r2["cap_gt_lost_pessimistic"]:,} |
| GT recall impact (upper bound) | {r2["cap_recall_impact_pct"]:.4f}% |

---

## Source 1 → Source 3

### Individual Strategy Recall

| Strategy | Candidate Pairs | Pair Recall % | Complete Entity Recall % | Avg Cands | Median | P95 | Runtime |
|----------|----------------|--------------|--------------------------|-----------|--------|-----|---------|
| Strategy 1 (Exact) | ~11,417,484 | ~22.23% | ~10.68% | ~5.17 | — | ~30.0 | ~11s |
| Strategy 2 (2-tok) | 105,773,118 | 52.02% | 34.37% | 47.93 | 5.00 | 490.0 | ~19s |
| Strategy 3 (3-tok) | {r3["s3_candidate_pairs"]:,} | {r3["s3_pair_recall"]:.2f}% | {r3["s3_complete_recall"]:.2f}% | {r3["s3_avg_cands"]:.2f} | {r3["s3_median_cands"]:.2f} | {r3["s3_p95_cands"]:.1f} | {r3["elapsed_s"]:.1f}s |

### Cumulative Union Recall (S3)

| Strategies Combined | Candidate Pairs | Pair Recall % | Complete Entity Recall % | Avg Cands |
|---------------------|----------------|--------------|--------------------------|-----------|
| S1 ∪ S2 | 110,043,631 | 54.09% | 35.50% | 49.87 |
| **S1 ∪ S2 ∪ S3** | **{r3["union_candidate_pairs"]:,}** | **{r3["union_pair_recall"]:.2f}%** | **{r3["union_complete_recall"]:.2f}%** | **{r3["union_avg_cands"]:.2f}** |

### Strategy 3 Incremental Gain (S3)

| Metric | Value |
|--------|-------|
| New candidate pairs added by S3 | {r3["new_pairs_from_s3"]:,} |
| Incremental pair recall gain | **{r3["incremental_pair_recall"]:+.4f}%** |
| Incremental complete entity recall gain | **{r3["incremental_complete_recall"]:+.4f}%** |

### 500-Cap Impact — Strategy 3 (S3)

| Metric | Value |
|--------|-------|
| S1 entities capped | {r3["cap_entities"]:,} |
| Candidate pairs removed by cap | {r3["cap_removed_pairs"]:,} |
| GT pairs potentially lost (pessimistic) | {r3["cap_gt_lost_pessimistic"]:,} |
| GT recall impact (upper bound) | {r3["cap_recall_impact_pct"]:.4f}% |

---

## Key Observations

- **Strategy 3 is strictly more selective than Strategy 2**: requiring 3 matching tokens means smaller, more precise buckets (lower recall but higher precision per candidate).
- **Incremental value depends on whether entities with 3+ unique tokens exist in the data** that were missed by 1- and 2-token strategies.
- **Union of all three strategies** provides the highest achievable recall under token-based blocking without fuzzy matching.
- Next phase (not implemented here): fuzzy/phonetic blocking, TF-IDF-based candidate retrieval, or ML-based re-ranking.

---

*Generated by evaluate_strategy3.py — Stopped after Strategy 3 and cumulative union. No fuzzy matching or ML implemented.*
"""

    md_path = os.path.join(REPORTS_DIR, "blocking_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Saved {md_path}", flush=True)

    print("\n=== ALL DONE ===", flush=True)
    print("Strategy 3 evaluation complete. Stopped as requested.", flush=True)


if __name__ == "__main__":
    main()
