"""
data_loader.py

Utility functions for loading entity resolution datasets and ground truth.
Designed for memory-efficient loading and inspection.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any


def load_raw_datasets(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all four TSV datasets from the specified data directory.
    
    Returns:
        s1 (pd.DataFrame): Source 1 dataset
        s2 (pd.DataFrame): Source 2 dataset
        s3 (pd.DataFrame): Source 3 dataset
        gt (pd.DataFrame): Ground truth dataset
    """
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(data_dir, "train_ground_truths.tsv")

    s1 = pd.read_csv(s1_path, sep="\t", quoting=3, on_bad_lines="skip")
    s2 = pd.read_csv(s2_path, sep="\t", quoting=3, on_bad_lines="skip")
    s3 = pd.read_csv(s3_path, sep="\t", quoting=3, on_bad_lines="skip")
    gt = pd.read_csv(gt_path, sep="\t", quoting=3, on_bad_lines="skip")

    return s1, s2, s3, gt


def get_dataset_info(data_dir: str = "data") -> Dict[str, Any]:
    """
    Computes dataset sizes, line counts, column schemas, and estimated memory usage.
    """
    files = {
        "train_source1.tsv": os.path.join(data_dir, "train_source1.tsv"),
        "train_source2.tsv": os.path.join(data_dir, "train_source2.tsv"),
        "train_source3.tsv": os.path.join(data_dir, "train_source3.tsv"),
        "train_ground_truth.tsv": os.path.join(data_dir, "train_ground_truth.tsv" if os.path.exists(os.path.join(data_dir, "train_ground_truth.tsv")) else "train_ground_truths.tsv")
    }

    info = {}
    for fname, path in files.items():
        if os.path.exists(path):
            size_bytes = os.path.getsize(path)
            size_mb = size_bytes / (1024 * 1024)
            sample_df = pd.read_csv(path, sep="\t", nrows=10000)
            cols = sample_df.columns.tolist()
            
            # Count total lines without loading entire file
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                total_lines = sum(1 for _ in f)
            data_rows = total_lines - 1
            
            sample_mem = sample_df.memory_usage(deep=True).sum() / (1024 * 1024)
            est_mem_mb = (sample_mem / 10000) * data_rows
            
            info[fname] = {
                "file_size_mb": round(size_mb, 2),
                "num_rows": data_rows,
                "num_cols": len(cols),
                "columns": cols,
                "est_ram_mb": round(est_mem_mb, 2)
            }
    return info
