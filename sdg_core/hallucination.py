"""
Hallucination metrics — binning + hashing + set membership.
==========================================================

Ported verbatim from sdpype's hallucination_eval_cli.py (working-tree version,
which is already sdpype-free: pure numpy/pandas, no DuckDB, no SQL). Only the
nine functions the pipeline's hallucination_evaluation task actually calls are
kept — the argparse CLI, the rich console `display_*` helpers, and the
closest-pair / column-mismatch diagnostics are dropped. CSV loading is done by
sdg_core.metadata, not the copy that lived in the CLI.

Metrics (all as a fraction of total synthetic rows):
- TotalFR     : synthetic records found in population
- NovelFR     : in population BUT NOT in training  (creative & real — the ideal)
- MemorizedFR : in population AND in training       (real but memorised)
- HR          : NOT in population                   (hallucinated)

Relationship: TotalFR = NovelFR + MemorizedFR, and TotalFR + HR = 1.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def get_column_types(metadata_path: Path) -> Dict[str, str]:
    """Map column name -> sdtype ('numerical', 'categorical', ...) from metadata JSON."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    columns = metadata.get('columns', {})
    return {col: info.get('sdtype', 'unknown') for col, info in columns.items()}


def compute_bin_boundaries(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    num_bins: int = 20,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute (min, max) per numerical column from a DataFrame (typically population).
    num_bins is unused here — kept for signature parity with the caller.
    """
    boundaries = {}

    for col, sdtype in column_types.items():
        if col not in df.columns:
            continue

        if sdtype == 'numerical':
            col_data = df[col].dropna()
            if len(col_data) > 0:
                boundaries[col] = (float(col_data.min()), float(col_data.max()))
            else:
                boundaries[col] = (0.0, 1.0)  # Fallback for all-null columns

    return boundaries


def bin_value(val: Any, min_val: float, max_val: float, num_bins: int) -> str:
    """Bin one numeric value: "Missing" for NaN, else "1".."num_bins" (1-indexed, clamped)."""
    if pd.isna(val):
        return "Missing"

    # Handle edge case: min == max
    if max_val == min_val:
        return "1"

    # Compute bin number (1-indexed)
    bin_num = int(np.floor((val - min_val) / (max_val - min_val) * num_bins)) + 1

    # Clamp to valid range [1, num_bins]
    bin_num = max(1, min(bin_num, num_bins))

    return str(bin_num)


def _to_str(val) -> str:
    """
    Canonical string for hashing.

    - NaN / None                   -> "Missing"
    - bool                         -> "True" / "False"  (checked before float/int)
    - float that is a whole number -> str(int(val))     e.g. 35.0 -> "35"
    - everything else              -> str(val)
    """
    if pd.isna(val):
        return "Missing"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def bin_dataframe(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    boundaries: Dict[str, Tuple[float, float]],
    num_bins: int = 20,
) -> pd.DataFrame:
    """
    All columns -> strings. Numerical columns with a boundary -> "Missing"/"1".."num_bins";
    everything else -> canonical string via _to_str().
    """
    binned_data = {}

    for col in df.columns:
        if col not in column_types:
            # Unknown column - treat as categorical
            binned_data[col] = df[col].apply(_to_str)
            continue

        sdtype = column_types[col]

        if sdtype == 'numerical' and col in boundaries:
            min_val, max_val = boundaries[col]
            binned_data[col] = df[col].apply(
                lambda v: bin_value(v, min_val, max_val, num_bins)
            )
        else:
            # Categorical or boolean - use canonical string conversion
            binned_data[col] = df[col].apply(_to_str)

    return pd.DataFrame(binned_data)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_record_hash(row: pd.Series) -> str:
    """MD5 of the '|'-joined row values (speed, not security)."""
    record_str = '|'.join(str(v) for v in row.values)
    return hashlib.md5(record_str.encode()).hexdigest()


def compute_record_hashes(df: pd.DataFrame) -> pd.Series:
    """Per-row hash Series."""
    return df.apply(compute_record_hash, axis=1)


def get_unique_hashes(hashes: pd.Series) -> Set[str]:
    """Set of unique hashes."""
    return set(hashes.unique())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_hallucination_metrics(
    population_hashes: Set[str],
    training_hashes: Set[str],
    synthetic_hashes: pd.Series,
) -> Tuple[Dict[str, Any], pd.Series, pd.Series]:
    """
    TotalFR / NovelFR / MemorizedFR / HR over the synthetic hash Series.
    Returns (metrics_dict, novel_mask, hallucinated_mask); the masks are for
    downstream record inspection and are discarded by the pipeline task.
    """
    total = len(synthetic_hashes)

    if total == 0:
        empty_mask = pd.Series([], dtype=bool)
        return {
            "TotalFR": {"count": 0, "rate": 0.0, "rate_pct": 0.0},
            "NovelFR": {"count": 0, "rate": 0.0, "rate_pct": 0.0},
            "MemorizedFR": {"count": 0, "rate": 0.0, "rate_pct": 0.0},
            "HR": {"count": 0, "rate": 0.0, "rate_pct": 0.0},
            "total_records": 0
        }, empty_mask, empty_mask

    # Count factual records (found in population)
    in_population = synthetic_hashes.isin(population_hashes)
    in_training = synthetic_hashes.isin(training_hashes)

    # TotalFR: in population
    factual_count = in_population.sum()

    # NovelFR: in population AND NOT in training
    novel_mask = in_population & ~in_training
    novel_count = novel_mask.sum()

    # MemorizedFR: in population AND in training
    memorized_mask = in_population & in_training
    memorized_count = memorized_mask.sum()

    # HR: NOT in population
    hallucinated_mask = ~in_population
    hallucinated_count = total - factual_count

    total_fr = factual_count / total
    novel_fr = novel_count / total
    memorized_fr = memorized_count / total
    hr = hallucinated_count / total

    metrics = {
        "TotalFR": {
            "count": int(factual_count),
            "rate": float(total_fr),
            "rate_pct": float(total_fr * 100)
        },
        "NovelFR": {
            "count": int(novel_count),
            "rate": float(novel_fr),
            "rate_pct": float(novel_fr * 100)
        },
        "MemorizedFR": {
            "count": int(memorized_count),
            "rate": float(memorized_fr),
            "rate_pct": float(memorized_fr * 100)
        },
        "HR": {
            "count": int(hallucinated_count),
            "rate": float(hr),
            "rate_pct": float(hr * 100)
        },
        "total_records": int(total)
    }

    return metrics, novel_mask, hallucinated_mask
