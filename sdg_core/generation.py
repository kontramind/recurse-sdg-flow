"""
Post-processing orchestration for synthetic data generation, ported from
sdpype/generation.py's _apply_post_processing (renamed to a public
apply_post_processing — it's now a real sdg_core API, not a private helper).

Flattened from `cfg: DictConfig` (which did
`cfg.post_processing.fix_invalid_categories.{enabled,method,knn_neighbors,
distance_metric,fallback}`) to plain args. `method="none"` is the
disable-switch — fix_invalid_categories() already treats it as a first-class
no-op, so no separate "enabled" flag is needed.

Also takes `sdtypes: dict` directly instead of an RDTDatasetEncoder object —
the original only ever used `encoder.sdtypes` off it, so this decouples
sdg_core.generation from sdg_core.encoding entirely.
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from sdg_core.post_processing import fix_invalid_categories, get_categorical_columns


def apply_post_processing(
    synthetic_decoded: pd.DataFrame,
    training_data: pd.DataFrame,
    sdtypes: Dict[str, str],
    method: str = "knn",
    knn_neighbors: int = 5,
    distance_metric: str = "hamming",
    fallback: str = "weighted",
) -> Tuple[pd.DataFrame, Optional[Dict[str, int]]]:
    """
    Fix invalid categories in synthetic data and normalize categorical dtypes.

    Args:
        synthetic_decoded: Decoded synthetic data
        training_data: Training data (source of valid categories)
        sdtypes: Column name -> sdtype mapping (e.g. encoder.sdtypes)
        method: 'knn' | 'weighted' | 'random' | 'none' (disables fixing)
        knn_neighbors: Number of neighbors for the 'knn' method
        distance_metric: Distance metric for the 'knn' method
        fallback: Fallback method used when a column is 100% invalid

    Returns:
        Tuple of (processed_dataframe, fix_metrics). fix_metrics is None
        when there were no categorical columns to check.
    """
    categorical_columns = get_categorical_columns(sdtypes)

    if not categorical_columns:
        print("ℹ️  No categorical columns to fix")
        return synthetic_decoded, None

    print(f"\n🔧 Post-processing: Fixing invalid categories")
    print(f"   Method: {method}")
    print(f"   Categorical columns: {len(categorical_columns)}")

    fixed_df, fix_metrics = fix_invalid_categories(
        synthetic_df=synthetic_decoded,
        reference_df=training_data,  # Training data = valid categories for this generation
        categorical_columns=categorical_columns,
        method=method,
        knn_neighbors=knn_neighbors,
        distance_metric=distance_metric,
        fallback=fallback,
    )

    # CRITICAL: Convert ALL categorical columns to strings
    # This ensures consistent dtype for categorical comparisons downstream
    # (RDT encoders expect strings for categorical columns, not int64/float64)
    print(f"\n🔄 Normalizing categorical dtypes to strings...")
    for col in categorical_columns:
        if col in fixed_df.columns:
            current_dtype = fixed_df[col].dtype

            if current_dtype != 'object':
                fixed_df[col] = fixed_df[col].astype(str)
                print(f"   ✓ Converted {col}: {current_dtype} → object (string)")
            else:
                print(f"   ✓ {col}: already object (string)")

    if fix_metrics:
        total_fixed = sum(fix_metrics.values())
        print(f"   ✓ Fixed {total_fixed} invalid values across {len(fix_metrics)} columns:")
        for col, count in fix_metrics.items():
            print(f"      - {col}: {count} values")
    else:
        print(f"   ✓ No invalid categories found")

    return fixed_df, fix_metrics
