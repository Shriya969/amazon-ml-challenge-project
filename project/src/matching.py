"""
matching.py

Candidate generation and blocking strategies for entity resolution.
Implements memory-safe blocking indexes and evaluation metrics.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any


def parse_ground_truth(gt_df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Parses matched_entity_ids column into sets of target IDs.
    Returns:
        gt_s2: Dict[source1_entity_id -> set of S2 entity IDs]
        gt_s3: Dict[source1_entity_id -> set of S3 entity IDs]
        gt_union: Dict[source1_entity_id -> set of S2+S3 entity IDs]
    """
    gt_s2 = defaultdict(set)
    gt_s3 = defaultdict(set)
    gt_union = defaultdict(set)

    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        val = row["matched_entity_ids"]
        if pd.notna(val) and str(val).strip():
            targets = [t.strip() for t in str(val).split(",") if t.strip()]
            for tid in targets:
                gt_union[s1_id].add(tid)
                if tid.startswith("S2-"):
                    gt_s2[s1_id].add(tid)
                elif tid.startswith("S3-"):
                    gt_s3[s1_id].add(tid)
                    
    return dict(gt_s2), dict(gt_s3), dict(gt_union)


def build_blocking_indexes(target_df: pd.DataFrame) -> Dict[str, Dict[Any, Set[str]]]:
    """
    Builds inverted index dictionaries mapping blocking keys to target entity IDs.
    """
    c_arr = target_df["norm_country"].values
    eid_arr = target_df["entity_id"].values
    name_arr = target_df["norm_name"].values
    cname_arr = target_df["clean_name"].values
    t1_arr = target_df["first_token"].values
    t2_arr = target_df["second_token"].values
    num_arr = target_df["addr_num"].values

    idx_exact = defaultdict(set)
    idx_2tok = defaultdict(set)
    idx_tok = defaultdict(set)
    idx_pref = defaultdict(set)
    idx_tok_num = defaultdict(set)

    for c, eid, n, cn, t1, t2, num in zip(c_arr, eid_arr, name_arr, cname_arr, t1_arr, t2_arr, num_arr):
        if c and n:
            idx_exact[(c, n)].add(eid)
        if c and t1 and t2:
            idx_2tok[(c, t1, t2)].add(eid)
        if c and t1:
            idx_tok[(c, t1)].add(eid)
        if c and len(cn) >= 3:
            idx_pref[(c, cn[:4])].add(eid)
        if c and t1 and num:
            idx_tok_num[(c, t1, num)].add(eid)

    return {
        "exact_name": idx_exact,
        "first_2tok": idx_2tok,
        "first_tok": idx_tok,
        "name_pref": idx_pref,
        "tok_addr_num": idx_tok_num,
    }


def evaluate_blocking_strategy(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    indexes: Dict[str, Dict[Any, Set[str]]],
    gt_map: Dict[str, Set[str]],
    strategy_key: str,
    strategy_name: str,
    target_source_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates a specific blocking strategy for S1 -> target_df.
    Computes candidate pairs count, pair recall, complete entity recall, and candidate size statistics.
    """
    total_gt_pairs = sum(len(v) for v in gt_map.values())
    cartesian_size = len(s1_df) * len(target_df)

    s1_c = s1_df["norm_country"].values
    s1_eid = s1_df["entity_id"].values
    s1_n = s1_df["norm_name"].values
    s1_cn = s1_df["clean_name"].values
    s1_t1 = s1_df["first_token"].values
    s1_t2 = s1_df["second_token"].values
    s1_num = s1_df["addr_num"].values

    idx_exact = indexes["exact_name"]
    idx_2tok = indexes["first_2tok"]
    idx_tok = indexes["first_tok"]
    idx_pref = indexes["name_pref"]
    idx_tok_num = indexes["tok_addr_num"]

    found_pairs = 0
    complete_entities = 0
    cand_counts = []
    coverage = 0
    failed_examples = []

    for c, eid, n, cn, t1, t2, num in zip(s1_c, s1_eid, s1_n, s1_cn, s1_t1, s1_t2, s1_num):
        if strategy_key == "country_only":
            n_cands = len(target_df[target_df["norm_country"] == c])
            cands = set()
            gt_targets = gt_map.get(eid)
            if gt_targets:
                found_pairs += len(gt_targets)
                complete_entities += 1
        else:
            cands = set()
            if strategy_key in ["exact_name", "combined_union"]:
                if (c, n) in idx_exact:
                    cands.update(idx_exact[(c, n)])
            if strategy_key in ["first_2tok", "combined_union"]:
                if (c, t1, t2) in idx_2tok:
                    cands.update(idx_2tok[(c, t1, t2)])
            if strategy_key in ["first_tok"]:
                if (c, t1) in idx_tok:
                    cands.update(idx_tok[(c, t1)])
            if strategy_key in ["name_pref"]:
                if len(cn) >= 3 and (c, cn[:4]) in idx_pref:
                    cands.update(idx_pref[(c, cn[:4])])
            if strategy_key in ["tok_addr_num", "combined_union"]:
                if t1 and num and (c, t1, num) in idx_tok_num:
                    cands.update(idx_tok_num[(c, t1, num)])

            n_cands = len(cands)
            gt_targets = gt_map.get(eid)
            if gt_targets:
                matched = len(gt_targets.intersection(cands))
                found_pairs += matched
                if gt_targets.issubset(cands):
                    complete_entities += 1
                elif strategy_key == "combined_union" and len(failed_examples) < 20:
                    missed = gt_targets - cands
                    failed_examples.append({
                        "s1_entity_id": eid,
                        "target_source": target_source_name,
                        "missed_target_ids": ",".join(list(missed)),
                        "s1_business_name": s1_df.loc[s1_df["entity_id"] == eid, "business_name"].values[0] if len(s1_df.loc[s1_df["entity_id"] == eid]) else "",
                        "s1_address": s1_df.loc[s1_df["entity_id"] == eid, "business_address"].values[0] if len(s1_df.loc[s1_df["entity_id"] == eid]) else "",
                        "s1_country": c
                    })

        cand_counts.append(n_cands)
        if n_cands > 0:
            coverage += 1

    tot_cands = sum(cand_counts)
    pair_rec = (found_pairs / total_gt_pairs) * 100 if total_gt_pairs > 0 else 0
    comp_rec = (complete_entities / len(gt_map)) * 100 if len(gt_map) > 0 else 0
    cov_pct = (coverage / len(s1_df)) * 100
    avg_cands = np.mean(cand_counts)
    p95_cands = np.percentile(cand_counts, 95)
    max_cands = np.max(cand_counts)
    reduction = (1 - (tot_cands / cartesian_size)) * 100

    metrics = {
        "Target Source": target_source_name,
        "Strategy": strategy_name,
        "Candidate Pairs": tot_cands,
        "Cartesian Reduction %": round(reduction, 6),
        "S1 Coverage %": round(cov_pct, 2),
        "Pair Recall %": round(pair_rec, 2),
        "Complete Entity Recall %": round(comp_rec, 2),
        "Avg Candidates/S1": round(avg_cands, 2),
        "95th Percentile": round(p95_cands, 1),
        "Max Candidates/S1": int(max_cands),
    }

    return metrics, failed_examples
