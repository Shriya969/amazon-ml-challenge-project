"""
preprocessing.py

Data cleaning and string normalization utilities for business entity resolution.
Optimized using C-level vectorized Pandas operations.
"""

import pandas as pd
import numpy as np


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies vectorized string preprocessing to business_name, business_address, and country.
    Creates norm_name, clean_name, first_token, second_token, name_prefix3, addr_num, norm_country.
    """
    df_clean = df.copy()
    
    if "business_name" in df_clean.columns:
        bname = df_clean["business_name"].fillna("").astype(str).str.lower()
        # Remove non-alphanumeric except space
        bname_clean = bname.str.replace(r"[^\w\s]", " ", regex=True).str.strip().str.replace(r"\s+", " ", regex=True)
        
        # Remove common generic legal suffixes if needed
        # (llc, inc, ltd, corp, pvt, limited, corporation, co)
        legal_pattern = r"\b(llc|inc|ltd|corp|pvt|limited|corporation|co|llp|services|group|enterprises)\b"
        bname_no_legal = bname_clean.str.replace(legal_pattern, "", regex=True).str.strip().str.replace(r"\s+", " ", regex=True)
        
        tokens = bname_no_legal.str.split()
        
        df_clean["norm_name"] = bname_clean
        df_clean["clean_name"] = bname_no_legal
        df_clean["first_token"] = tokens.str[0].fillna("")
        df_clean["second_token"] = tokens.str[1].fillna("")
        df_clean["third_token"] = tokens.str[2].fillna("")
        df_clean["name_prefix3"] = bname_clean.str[:3].fillna("")
        
    if "business_address" in df_clean.columns:
        baddr = df_clean["business_address"].fillna("").astype(str).str.lower()
        baddr_clean = baddr.str.replace(r"[^\w\s]", " ", regex=True).str.strip().str.replace(r"\s+", " ", regex=True)
        df_clean["norm_address"] = baddr_clean
        df_clean["addr_num"] = baddr_clean.str.extract(r"(\d+)", expand=False).fillna("")
        
    if "country" in df_clean.columns:
        df_clean["norm_country"] = df_clean["country"].fillna("").astype(str).str.lower().str.strip()
        
    return df_clean
