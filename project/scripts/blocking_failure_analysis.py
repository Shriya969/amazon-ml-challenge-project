"""
blocking_failure_analysis.py

Analyzes WHY Strategy 1 (exact name) + Strategy 2 (2-token, cap=500)
miss approximately 40-45% of ground-truth pairs.

Architecture:
  Pass 1 — stream S1, reservoir-sample up to 5000 missed pairs per source.
            Store only (s1_eid, s1_data, target_eid, cap_fired) per sample.
  Pass 2 — scan S2/S3 once to fetch details for sampled target entity IDs.
  Analysis — classify each missed pair, compute diagnostics.

Memory safety:
  No Cartesian product. No large DataFrames. No fuzzy matching on full data.
"""

import os
import sys
import time
import random
import math
import re

import numpy as np
import pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import preprocess_dataframe

# ---------------------------------------------------------------------------
DATA_DIR        = "data"
REPORTS_DIR     = "outputs/reports"
SAMPLE_SIZE     = 5000
RANDOM_SEED     = 42
SAFETY_CAP      = 500
PROGRESS_EVERY  = 200_000

GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")
if not os.path.exists(GT_FILE):
    GT_FILE = os.path.join(DATA_DIR, "train_ground_truths.tsv")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tsv(path, name, usecols=None):
    print(f"  Loading {name} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", quoting=3, on_bad_lines="skip",
                     usecols=usecols)
    print(f"    {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
    return df


def parse_gt_maps(gt_df):
    gt_s2, gt_s3 = defaultdict(set), defaultdict(set)
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        val   = row["matched_entity_ids"]
        if pd.notna(val) and str(val).strip():
            for tid in str(val).split(","):
                tid = tid.strip()
                if   tid.startswith("S2-"):
                    gt_s2[s1_id].add(tid)
                elif tid.startswith("S3-"):
                    gt_s3[s1_id].add(tid)
    return dict(gt_s2), dict(gt_s3)


def build_exact_index(df):
    idx = defaultdict(set)
    for c, e, n in zip(df["norm_country"].values,
                       df["entity_id"].values,
                       df["norm_name"].values):
        if c and n:
            idx[(c, n)].add(e)
    return idx


def build_2tok_index(df):
    idx = defaultdict(set)
    for c, e, t1, t2 in zip(df["norm_country"].values,
                              df["entity_id"].values,
                              df["first_token"].values,
                              df["second_token"].values):
        if c and t1 and t2:
            idx[(c, t1, t2)].add(e)
    return idx


def reservoir_update(reservoir, item, n_seen, k):
    """Update reservoir sample of size k. Returns updated reservoir."""
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        j = random.randint(0, n_seen - 1)
        if j < k:
            reservoir[j] = item
    return reservoir


def token_jaccard(a, b):
    ta = set(str(a).split()) if a else set()
    tb = set(str(b).split()) if b else set()
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


def char_overlap(a, b):
    """Character-level bigram overlap (Dice coefficient)."""
    def bigrams(s):
        s = str(s) if s else ""
        return set(s[i:i+2] for i in range(len(s)-1))
    ba, bb = bigrams(a), bigrams(b)
    if not ba and not bb:
        return 1.0
    inter = len(ba & bb)
    return 2 * inter / (len(ba) + len(bb)) if (ba or bb) else 0.0


def addr_token_overlap(a, b):
    ta = set(str(a).split()) if a else set()
    tb = set(str(b).split()) if b else set()
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


LEGAL = frozenset(["llc","inc","ltd","corp","pvt","limited","corporation",
                   "co","llp","services","group","enterprises","company"])

def strip_legal(tokens):
    return [t for t in tokens if t not in LEGAL]


def classify_miss(s1, tgt, raw_bucket_size, target_in_bucket):
    """
    Returns primary failure category string.
    s1, tgt: dicts with keys:
        norm_country, norm_name, norm_address,
        first_token, second_token, business_name, business_address
    raw_bucket_size: size of S2's 2-tok bucket for this S1 entity
    target_in_bucket: bool — was the target actually in the raw (uncapped) bucket?
    """
    s1_c  = s1.get("norm_country", "")
    t_c   = tgt.get("norm_country", "")
    s1_n  = str(s1.get("norm_name", "")).strip()
    t_n   = str(tgt.get("norm_name", "")).strip()
    s1_t1 = s1.get("first_token", "")
    s1_t2 = s1.get("second_token", "")

    # 1. Country mismatch
    if s1_c != t_c:
        return "Country mismatch"

    # 2. Missing name
    if not s1_n or not t_n:
        return "Missing business name"

    # 3. Cap caused the miss
    if target_in_bucket and raw_bucket_size > SAFETY_CAP:
        return "Strategy-2 bucket cap"

    # Tokenize both names (after legal stripping)
    s1_toks = strip_legal(s1_n.split())
    t_toks  = strip_legal(t_n.split())
    s1_set  = set(s1_toks)
    t_set   = set(t_toks)

    # 4. Word order difference (same tokens, different order)
    if s1_set == t_set and s1_n != t_n:
        return "Word order difference"

    # 5. Legal suffix only difference
    if s1_toks == t_toks and s1_n != t_n:
        return "Legal suffix difference"

    # 6. First token abbreviation
    t1_s = s1.get("first_token", "")
    t1_t = tgt.get("first_token", "")
    if t1_s and t1_t and t1_s != t1_t:
        if t1_t.startswith(t1_s) or t1_s.startswith(t1_t):
            return "Abbreviation"
        # also check if one is initials of the other
        if len(t1_s) <= 3 or len(t1_t) <= 3:
            return "Abbreviation"

    # 7. Word order (subset check)
    if s1_set and t_set and (s1_set.issubset(t_set) or t_set.issubset(s1_set)):
        if len(s1_set) != len(t_set):
            return "Partial business name"
        return "Word order difference"

    # 8. High character similarity but different tokens
    csim = char_overlap(s1_n, t_n)
    jaccard = token_jaccard(s1_n, t_n)

    if csim >= 0.7:
        return "Name spelling variation"

    if jaccard >= 0.3:
        return "Partial business name"

    # 9. Transliteration / very low overlap
    if csim < 0.3:
        return "Transliteration / script variation"

    # 10. Tokenization difference
    s1_squish = re.sub(r"\s+", "", s1_n)
    t_squish  = re.sub(r"\s+", "", t_n)
    if s1_squish == t_squish:
        return "Tokenization difference"

    return "Other"


# ---------------------------------------------------------------------------
# Pass 1: Stream S1, reservoir-sample missed pairs
# ---------------------------------------------------------------------------

def pass1_collect_misses(s1_p, idx_exact, idx_2tok, gt_map, source_label, k=SAMPLE_SIZE):
    """
    Returns:
        reservoir: list of dicts, each a sampled missed pair with s1 data
        stats: aggregate counts
    """
    n_s1      = len(s1_p)
    reservoir = []
    n_seen    = 0   # total missed pairs seen

    total_gt_pairs    = 0
    total_missed      = 0
    total_s1_with_gt  = 0
    cap_misses        = 0  # target was in raw bucket but bucket > cap

    s1_c   = s1_p["norm_country"].values
    s1_eid = s1_p["entity_id"].values
    s1_n   = s1_p["norm_name"].values
    s1_t1  = s1_p["first_token"].values
    s1_t2  = s1_p["second_token"].values
    s1_t3  = s1_p["third_token"].values
    s1_bn  = s1_p["business_name"].values
    s1_ba  = s1_p["business_address"].values
    s1_na  = s1_p["norm_address"].values
    s1_nc  = s1_p["norm_country"].values

    t0 = time.time()

    for i, (c, eid, n, t1, t2, t3, bn, ba, na) in enumerate(
        zip(s1_c, s1_eid, s1_n, s1_t1, s1_t2, s1_t3,
            s1_bn, s1_ba, s1_na), start=1
    ):
        gt_targets = gt_map.get(eid, set())
        if not gt_targets:
            continue

        total_s1_with_gt += 1
        total_gt_pairs   += len(gt_targets)

        # Compute union candidates (Strategy 1 + Strategy 2 capped)
        exact_cands = idx_exact.get((c, n), set()) if (c and n) else set()

        raw_bucket  = idx_2tok.get((c, t1, t2), set()) if (c and t1 and t2) else set()
        raw_size    = len(raw_bucket)
        if raw_size > SAFETY_CAP:
            tok2_cands = set(list(raw_bucket)[:SAFETY_CAP])
        else:
            tok2_cands = raw_bucket

        union_cands = exact_cands | tok2_cands
        missed      = gt_targets - union_cands

        total_missed += len(missed)

        for target_eid in missed:
            n_seen += 1

            # Check if cap caused this miss
            in_raw    = target_eid in raw_bucket
            cap_fired = in_raw and raw_size > SAFETY_CAP
            if cap_fired:
                cap_misses += 1

            item = {
                "s1_entity_id":   eid,
                "s1_business_name": str(bn),
                "s1_business_address": str(ba),
                "s1_norm_name":   str(n),
                "s1_norm_address": str(na),
                "s1_country":     str(c),
                "s1_first_token": str(t1),
                "s1_second_token": str(t2),
                "s1_third_token": str(t3),
                "target_entity_id": target_eid,
                "raw_bucket_size":  raw_size,
                "target_in_raw_bucket": in_raw,
                "cap_fired":      cap_fired,
            }
            reservoir = reservoir_update(reservoir, item, n_seen, k)

        if i % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            pct     = 100 * i / n_s1
            eta     = (elapsed / i) * (n_s1 - i)
            print(
                f"  [{source_label}] {i:,}/{n_s1:,} ({pct:.1f}%)  "
                f"missed_so_far={total_missed:,}  sample_size={len(reservoir):,}  "
                f"cap_misses={cap_misses:,}  ETA={eta:.0f}s",
                flush=True,
            )
            t0 = time.time()

    stats = {
        "total_gt_pairs":    total_gt_pairs,
        "total_missed":      total_missed,
        "total_s1_with_gt":  total_s1_with_gt,
        "cap_misses":        cap_misses,
        "n_sampled":         len(reservoir),
    }
    return reservoir, stats


# ---------------------------------------------------------------------------
# Pass 2: Fetch target entity details
# ---------------------------------------------------------------------------

def pass2_fetch_targets(target_df, needed_eids):
    """
    Scan target_df once. Return dict: entity_id -> data dict.
    Only keep rows whose entity_id is in needed_eids.
    """
    print(f"  Fetching details for {len(needed_eids):,} sampled target entities ...", flush=True)
    t0 = time.time()
    needed = set(needed_eids)
    result = {}

    eid_arr = target_df["entity_id"].values
    bn_arr  = target_df["business_name"].values
    ba_arr  = target_df["business_address"].values
    co_arr  = target_df["country"].values
    nn_arr  = target_df["norm_name"].values
    na_arr  = target_df["norm_address"].values
    nc_arr  = target_df["norm_country"].values
    t1_arr  = target_df["first_token"].values
    t2_arr  = target_df["second_token"].values
    t3_arr  = target_df["third_token"].values

    for eid, bn, ba, co, nn, na, nc, t1, t2, t3 in zip(
        eid_arr, bn_arr, ba_arr, co_arr, nn_arr, na_arr, nc_arr, t1_arr, t2_arr, t3_arr
    ):
        if eid in needed:
            result[eid] = {
                "entity_id":         eid,
                "business_name":     str(bn),
                "business_address":  str(ba),
                "country":           str(co),
                "norm_name":         str(nn),
                "norm_address":      str(na),
                "norm_country":      str(nc),
                "first_token":       str(t1),
                "second_token":      str(t2),
                "third_token":       str(t3),
            }
            if len(result) == len(needed):
                break

    print(f"  Fetched {len(result):,} target records in {time.time()-t0:.1f}s", flush=True)
    return result


# ---------------------------------------------------------------------------
# Analysis: classify + compute diagnostics
# ---------------------------------------------------------------------------

def analyze_sample(reservoir, target_lookup, source_label):
    """
    For each sampled pair, classify failure and compute diagnostics.
    Returns list of result dicts.
    """
    rows = []
    for item in reservoir:
        tgt = target_lookup.get(item["target_entity_id"])
        if tgt is None:
            # Entity not found in target source (edge case)
            tgt = {k: "" for k in ["entity_id","business_name","business_address",
                                   "country","norm_name","norm_address","norm_country",
                                   "first_token","second_token","third_token"]}
            tgt["entity_id"] = item["target_entity_id"]

        s1 = {
            "norm_country":  item["s1_country"],
            "norm_name":     item["s1_norm_name"],
            "norm_address":  item["s1_norm_address"],
            "first_token":   item["s1_first_token"],
            "second_token":  item["s1_second_token"],
            "business_name": item["s1_business_name"],
            "business_address": item["s1_business_address"],
        }

        category = classify_miss(
            s1,
            tgt,
            raw_bucket_size    = item["raw_bucket_size"],
            target_in_bucket   = item["target_in_raw_bucket"],
        )

        # Diagnostics
        s1_n  = item["s1_norm_name"]
        t_n   = tgt["norm_name"]
        s1_a  = item["s1_norm_address"]
        t_a   = tgt["norm_address"]

        name_exact        = int(s1_n == t_n)
        ft_equal          = int(item["s1_first_token"] == tgt["first_token"])
        ft2_equal         = int(item["s1_first_token"] == tgt["first_token"] and
                                item["s1_second_token"] == tgt["second_token"])
        ft3_equal         = int(item["s1_first_token"] == tgt["first_token"] and
                                item["s1_second_token"] == tgt["second_token"] and
                                item["s1_third_token"]  == tgt["third_token"])
        char_sim          = round(char_overlap(s1_n, t_n), 4)
        token_jacc        = round(token_jaccard(s1_n, t_n), 4)
        addr_overlap      = round(addr_token_overlap(s1_a, t_a), 4)
        country_eq        = int(item["s1_country"] == tgt["norm_country"])

        rows.append({
            "source":                   source_label,
            "s1_entity_id":             item["s1_entity_id"],
            "s1_business_name":         item["s1_business_name"],
            "s1_business_address":      item["s1_business_address"],
            "s1_country":               item["s1_country"],
            "s1_norm_name":             s1_n,
            "s1_norm_address":          s1_a,
            "s1_first_token":           item["s1_first_token"],
            "s1_second_token":          item["s1_second_token"],
            "target_entity_id":         item["target_entity_id"],
            "target_business_name":     tgt["business_name"],
            "target_business_address":  tgt["business_address"],
            "target_country":           tgt["country"],
            "target_norm_name":         t_n,
            "target_norm_address":      t_a,
            "target_first_token":       tgt["first_token"],
            "target_second_token":      tgt["second_token"],
            "failure_category":         category,
            "raw_bucket_size":          item["raw_bucket_size"],
            "target_in_raw_bucket":     item["target_in_raw_bucket"],
            "cap_fired":                item["cap_fired"],
            "name_exact_match":         name_exact,
            "first_token_equal":        ft_equal,
            "first_2tok_equal":         ft2_equal,
            "first_3tok_equal":         ft3_equal,
            "char_similarity":          char_sim,
            "token_jaccard":            token_jacc,
            "address_token_overlap":    addr_overlap,
            "country_equal":            country_eq,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print("\n=== Loading datasets ===", flush=True)
    s1 = load_tsv(os.path.join(DATA_DIR, "train_source1.tsv"), "Source 1")
    s2 = load_tsv(os.path.join(DATA_DIR, "train_source2.tsv"), "Source 2")
    s3 = load_tsv(os.path.join(DATA_DIR, "train_source3.tsv"), "Source 3")
    gt = load_tsv(GT_FILE, "Ground Truth")

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    print("\n=== Preprocessing ===", flush=True)
    t0 = time.time()
    s1_p = preprocess_dataframe(s1);  del s1
    s2_p = preprocess_dataframe(s2);  del s2
    s3_p = preprocess_dataframe(s3);  del s3
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Parse GT
    # ------------------------------------------------------------------
    print("\n=== Parsing ground truth ===", flush=True)
    gt_s2, gt_s3 = parse_gt_maps(gt);  del gt
    print(f"  S2 pairs: {sum(len(v) for v in gt_s2.values()):,}  "
          f"S3 pairs: {sum(len(v) for v in gt_s3.values()):,}", flush=True)

    all_analysis_rows = []

    for target_p, gt_map, src_label in [
        (s2_p, gt_s2, "S1→S2"),
        (s3_p, gt_s3, "S1→S3"),
    ]:
        print(f"\n{'='*60}", flush=True)
        print(f"=== FAILURE ANALYSIS: {src_label} ===", flush=True)
        print(f"{'='*60}", flush=True)

        # Build indexes
        print("  Building exact-name index ...", flush=True)
        idx_exact = build_exact_index(target_p)

        print("  Building 2-token index ...", flush=True)
        idx_2tok  = build_2tok_index(target_p)

        # PASS 1: collect missed pairs
        print(f"\n  PASS 1: Streaming {len(s1_p):,} S1 entities ...", flush=True)
        reservoir, stats = pass1_collect_misses(
            s1_p, idx_exact, idx_2tok, gt_map, src_label, k=SAMPLE_SIZE
        )

        miss_rate = stats["total_missed"] / stats["total_gt_pairs"] * 100 if stats["total_gt_pairs"] else 0
        cap_pct   = stats["cap_misses"] / stats["total_missed"] * 100 if stats["total_missed"] else 0

        print(f"""
  Pass 1 summary ({src_label}):
    Total GT pairs          : {stats["total_gt_pairs"]:,}
    Total missed pairs      : {stats["total_missed"]:,}  ({miss_rate:.2f}%)
    Cap-caused misses       : {stats["cap_misses"]:,}  ({cap_pct:.2f}% of missed)
    Sampled pairs           : {stats["n_sampled"]:,}
        """, flush=True)

        # PASS 2: fetch target details
        needed_eids = [item["target_entity_id"] for item in reservoir]
        target_lookup = pass2_fetch_targets(target_p, needed_eids)

        # Analyze
        print("  Classifying sampled missed pairs ...", flush=True)
        rows = analyze_sample(reservoir, target_lookup, src_label)
        all_analysis_rows.extend(rows)

        # Print category distribution
        categories = Counter(r["failure_category"] for r in rows)
        total = len(rows)
        print(f"\n  Failure category distribution ({src_label}):", flush=True)
        for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"    {cat:<40} {cnt:>5}  ({100*cnt/total:.1f}%)", flush=True)

        # Print diagnostics
        if rows:
            char_sims    = [r["char_similarity"] for r in rows]
            tok_jaccs    = [r["token_jaccard"] for r in rows]
            addr_ovlps   = [r["address_token_overlap"] for r in rows]
            cap_pct_s    = 100 * sum(1 for r in rows if r["cap_fired"]) / total
            name_exact_p = 100 * sum(r["name_exact_match"] for r in rows) / total
            ft_eq_p      = 100 * sum(r["first_token_equal"] for r in rows) / total
            ft2_eq_p     = 100 * sum(r["first_2tok_equal"] for r in rows) / total
            ft3_eq_p     = 100 * sum(r["first_3tok_equal"] for r in rows) / total
            high_csim    = 100 * sum(1 for x in char_sims if x >= 0.7) / total
            useful_addr  = 100 * sum(1 for x in addr_ovlps if x >= 0.3) / total

            print(f"""
  Diagnostic summary ({src_label}):
    Cap-fired pairs (in sample)     : {cap_pct_s:.2f}%
    Normalized name exact match     : {name_exact_p:.2f}%  (missed by S1 despite same name!)
    First token equal               : {ft_eq_p:.2f}%
    First 2 tokens equal            : {ft2_eq_p:.2f}%
    First 3 tokens equal            : {ft3_eq_p:.2f}%
    Char similarity >= 0.7          : {high_csim:.2f}%
    Mean char similarity            : {np.mean(char_sims):.3f}
    Mean token Jaccard              : {np.mean(tok_jaccs):.3f}
    Address overlap >= 0.3          : {useful_addr:.2f}%
    Mean address token overlap      : {np.mean(addr_ovlps):.3f}
            """, flush=True)

        # Free indexes before next source
        del idx_exact, idx_2tok, target_lookup
        import gc; gc.collect()

    # ------------------------------------------------------------------
    # Save blocking_failure_analysis.csv
    # ------------------------------------------------------------------
    print("\n=== Saving outputs ===", flush=True)
    analysis_df = pd.DataFrame(all_analysis_rows)
    analysis_path = os.path.join(REPORTS_DIR, "blocking_failure_analysis.csv")
    analysis_df.to_csv(analysis_path, index=False)
    print(f"  Saved {analysis_path}  ({len(analysis_df):,} rows)", flush=True)

    # ------------------------------------------------------------------
    # Save blocking_failure_examples.csv (100 representative examples)
    # ------------------------------------------------------------------
    # Pick ~10 per category per source — evenly distributed
    examples = []
    per_cat = max(1, 100 // len(analysis_df["failure_category"].unique()))
    for (src, cat), grp in analysis_df.groupby(["source", "failure_category"]):
        n = min(per_cat, len(grp))
        examples.append(grp.sample(n=n, random_state=RANDOM_SEED))

    examples_df = pd.concat(examples).head(100).reset_index(drop=True)
    ex_cols = ["source","s1_entity_id","s1_business_name","s1_business_address",
               "s1_country","s1_norm_name","target_entity_id","target_business_name",
               "target_business_address","target_country","target_norm_name",
               "failure_category","char_similarity","token_jaccard",
               "address_token_overlap","cap_fired","raw_bucket_size"]
    examples_df[[c for c in ex_cols if c in examples_df.columns]].to_csv(
        os.path.join(REPORTS_DIR, "blocking_failure_examples.csv"), index=False
    )
    print(f"  Saved blocking_failure_examples.csv  ({len(examples_df)} rows)", flush=True)

    # ------------------------------------------------------------------
    # Save blocking_failure_summary.md
    # ------------------------------------------------------------------
    # Aggregate per source
    def source_stats(df, src):
        sub = df[df["source"] == src]
        if sub.empty:
            return {}
        total   = len(sub)
        cats    = Counter(sub["failure_category"])
        cat_pct = {k: round(100*v/total, 2) for k,v in cats.most_common()}
        return {
            "n":        total,
            "cat_pct":  cat_pct,
            "cap_pct":  round(100 * sub["cap_fired"].sum() / total, 2),
            "name_pct": round(100 * sub["name_exact_match"].sum() / total, 2),
            "ft_pct":   round(100 * sub["first_token_equal"].sum() / total, 2),
            "ft2_pct":  round(100 * sub["first_2tok_equal"].sum() / total, 2),
            "high_csim":round(100 * (sub["char_similarity"] >= 0.7).sum() / total, 2),
            "addr_useful": round(100 * (sub["address_token_overlap"] >= 0.3).sum() / total, 2),
            "mean_csim": round(sub["char_similarity"].mean(), 3),
            "mean_jacc": round(sub["token_jaccard"].mean(), 3),
            "mean_addr": round(sub["address_token_overlap"].mean(), 3),
        }

    ss2 = source_stats(analysis_df, "S1→S2")
    ss3 = source_stats(analysis_df, "S1→S3")

    # Build category table
    all_cats = sorted(set(list(ss2.get("cat_pct",{}).keys()) +
                          list(ss3.get("cat_pct",{}).keys())))

    cat_table = "| Failure Category | S1→S2 % | S1→S3 % |\n|---|---|---|\n"
    for cat in sorted(all_cats,
                       key=lambda c: -max(ss2.get("cat_pct",{}).get(c,0),
                                          ss3.get("cat_pct",{}).get(c,0))):
        cat_table += f"| {cat} | {ss2.get('cat_pct',{}).get(cat,0.0):.1f}% | {ss3.get('cat_pct',{}).get(cat,0.0):.1f}% |\n"

    # Representative examples: 3 per top category
    top_cats = [k for k,_ in Counter(analysis_df["failure_category"]).most_common(6)]
    examples_md = ""
    for cat in top_cats:
        sub = analysis_df[analysis_df["failure_category"] == cat].head(3)
        examples_md += f"\n### {cat}\n\n"
        for _, r in sub.iterrows():
            examples_md += (
                f"- **S1**: `{r['s1_business_name']}` ({r['s1_country']}) "
                f"→ **Target**: `{r['target_business_name']}` ({r['target_country']})\n"
                f"  - Norm S1: `{r['s1_norm_name']}` | Norm Target: `{r['target_norm_name']}`\n"
                f"  - CharSim={r['char_similarity']:.2f}  TokenJacc={r['token_jaccard']:.2f}  "
                f"AddrOverlap={r['address_token_overlap']:.2f}  CapFired={r['cap_fired']}\n\n"
            )

    md = f"""# Blocking Failure Analysis Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overview

Analyzes why **Strategy 1 (Country + Exact Name) UNION Strategy 2 (Country + First 2 Tokens, cap=500)**
misses ~40-45% of ground-truth pairs.

| Metric | S1→S2 | S1→S3 |
|--------|-------|-------|
| Sampled missed pairs | {ss2.get("n","N/A")} | {ss3.get("n","N/A")} |
| Random seed | {RANDOM_SEED} | {RANDOM_SEED} |
| Safety cap | {SAFETY_CAP} | {SAFETY_CAP} |

---

## Failure Category Distribution

{cat_table}

---

## Key Diagnostic Metrics

| Metric | S1→S2 | S1→S3 |
|--------|-------|-------|
| Cap-fired misses | **{ss2.get("cap_pct","N/A")}%** | **{ss3.get("cap_pct","N/A")}%** |
| Norm name exact match (missed by S1!) | {ss2.get("name_pct","N/A")}% | {ss3.get("name_pct","N/A")}% |
| First token equal | {ss2.get("ft_pct","N/A")}% | {ss3.get("ft_pct","N/A")}% |
| First 2 tokens equal | {ss2.get("ft2_pct","N/A")}% | {ss3.get("ft2_pct","N/A")}% |
| Char similarity ≥ 0.70 | {ss2.get("high_csim","N/A")}% | {ss3.get("high_csim","N/A")}% |
| Mean char similarity | {ss2.get("mean_csim","N/A")} | {ss3.get("mean_csim","N/A")} |
| Mean token Jaccard | {ss2.get("mean_jacc","N/A")} | {ss3.get("mean_jacc","N/A")} |
| Address overlap ≥ 0.30 (potentially useful) | {ss2.get("addr_useful","N/A")}% | {ss3.get("addr_useful","N/A")}% |
| Mean address token overlap | {ss2.get("mean_addr","N/A")} | {ss3.get("mean_addr","N/A")} |

---

## Representative Examples by Category
{examples_md}

---

## Recommendations for Next Candidate-Generation Strategy

Based on the failure analysis above:

### Priority 1 — Address the Cap Problem
- **{ss2.get("cap_pct","?")}% of sampled misses (S2)** were caused by the 500-candidate cap.
- Consider raising the cap selectively for country+token keys that contain verified high-density legitimate business clusters (e.g., large US chains).
- Alternatively, use a **more selective blocking key** (e.g., country + first token + address number) for large-bucket entities.

### Priority 2 — Name Spelling & Abbreviation Variation
- A significant fraction of misses have high character-level similarity but different tokens.
- **Next strategy**: character-level n-gram indexing (bigram/trigram overlap ≥ threshold) or
  **edit-distance–based blocking** (fast BK-tree lookup) would capture these.

### Priority 3 — Word Order & Partial Name
- Some true matches differ only in token order or have one name as a prefix of another.
- **Next strategy**: token-set blocking (sorted token key, regardless of order) would capture word-order misses.

### Priority 4 — Address-Based Blocking
- ~{ss2.get("addr_useful","?")}% of missed pairs have ≥30% address token overlap.
- For entities with very different names, **address number + first business-name token** could be a complementary blocking key.

### Priority 5 — Transliteration / Script Variation
- Low character-similarity misses suggest different scripts or heavy transliteration.
- These likely require **phonetic keys** (Soundex / Double Metaphone) or embedding-based retrieval.

---

*Generated by blocking_failure_analysis.py — Stopped after analysis. No new blocking strategy implemented.*
"""

    md_path = os.path.join(REPORTS_DIR, "blocking_failure_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Saved {md_path}", flush=True)

    print("\n=== ALL DONE — Failure analysis complete. Stopped as requested. ===", flush=True)


if __name__ == "__main__":
    main()
