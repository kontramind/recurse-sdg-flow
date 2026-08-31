"""
Statistical / detection evaluation — shared column-selection helpers.
====================================================================

Seeded with the helpers that live at the top of sdpype/evaluation/statistical.py
and are imported by sdpype/evaluation/detection.py: ensure_json_serializable,
get_columns_by_sdtype, get_encoded_numeric_columns, log_column_selection.
Ported verbatim. The statistical_similarity metric classes + dispatch will be
appended to this module when that stage is ported.

Uses the real sdv.metadata.SingleTableMetadata.
"""

from typing import Any

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata


def ensure_json_serializable(obj: Any) -> Any:
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj


def get_columns_by_sdtype(metadata: dict, sdtypes: list) -> list:
    """
    Extract column names that match specified sdtypes from metadata.

    Args:
        metadata: SDV SingleTableMetadata object or dict with column definitions
        sdtypes: List of sdtypes to filter (e.g., ['numerical', 'categorical'])

    Returns:
        List of column names matching the specified sdtypes
    """
    # Handle SDV metadata object
    if hasattr(metadata, 'columns'):
        metadata_dict = metadata.to_dict()
    else:
        metadata_dict = metadata

    if not metadata_dict or 'columns' not in metadata_dict:
        return []

    columns = []
    for col_name, col_info in metadata_dict['columns'].items():
        if col_info.get('sdtype') in sdtypes:
            columns.append(col_name)

    return columns


def get_encoded_numeric_columns(encoding_config: dict, encoded_df: pd.DataFrame,
                                metadata: SingleTableMetadata, exclude_ids: bool = True) -> list:
    """
    Get columns that are numeric in the encoded DataFrame based on encoding config.

    This is the SOURCE OF TRUTH for which columns should be used in encoded data metrics.
    All transformers in the encoding config produce numeric output, so any column that
    was encoded will be numeric in the resulting DataFrame.

    Handles:
    - Direct column transformations (e.g., age -> numeric)
    - OneHotEncoder expansion (e.g., city -> city.NYC, city.LA, city.SF)
    - ID column exclusion (IDs shouldn't be in distance metrics)

    Args:
        encoding_config: Dict from load_encoding_config() with 'transformers' key
        encoded_df: The encoded DataFrame (to get actual column names after OneHot expansion)
        metadata: SDV SingleTableMetadata (to identify ID columns)
        exclude_ids: Whether to exclude ID columns from metrics (default: True)

    Returns:
        List of column names that are numeric and suitable for encoded data metrics

    Example:
        >>> config = load_encoding_config('encoding.yaml')
        >>> numeric_cols = get_encoded_numeric_columns(config, encoded_df, metadata)
        >>> # Returns: ['age', 'income', 'date_encoded', 'city.NYC', 'city.LA']
    """
    # Get base column names from transformers config (these were encoded)
    encoded_base_cols = set(encoding_config.get('transformers', {}).keys())

    # Get ID columns to exclude
    id_cols = set(get_columns_by_sdtype(metadata, ['id'])) if exclude_ids else set()

    # Include columns that either:
    # 1. Are directly in transformers config (e.g., 'age')
    # 2. Start with a base column name (for OneHot expansion like 'city.NYC' -> 'city')
    numeric_cols = []
    for col in encoded_df.columns:
        # Handle OneHot expansion: city.NYC -> city
        base_col = col.split('.')[0]

        # Skip ID columns
        if col in id_cols or base_col in id_cols:
            continue

        # Include if the column or its base was encoded
        if col in encoded_base_cols or base_col in encoded_base_cols:
            numeric_cols.append(col)

    return numeric_cols


def log_column_selection(metric_name: str, encoding_config: dict, encoded_df: pd.DataFrame,
                         metadata: SingleTableMetadata, usable_cols: list, exclude_ids: bool = True):
    """
    Log detailed information about column selection for metrics.

    Args:
        metric_name: Name of the metric being evaluated
        encoding_config: Encoding configuration dict
        encoded_df: The encoded DataFrame
        metadata: SDV SingleTableMetadata
        usable_cols: List of columns that will be used for evaluation
        exclude_ids: Whether ID columns were excluded
    """
    from rich.panel import Panel
    from rich.text import Text

    # Get information about columns
    all_encoded_cols = set(encoding_config.get('transformers', {}).keys())
    id_cols = set(get_columns_by_sdtype(metadata, ['id'])) if exclude_ids else set()
    datetime_cols = set(get_columns_by_sdtype(metadata, ['datetime']))

    # Identify datetime columns that are in usable_cols
    datetime_in_use = [col for col in usable_cols if col.split('.')[0] in datetime_cols]

    # Identify one-hot encoded columns (contain a dot)
    onehot_cols = [col for col in usable_cols if '.' in col]
    onehot_base = set([col.split('.')[0] for col in onehot_cols])

    # Build info text
    info_lines = []
    info_lines.append(f"[bold cyan]📊 {metric_name} - Column Selection[/bold cyan]")
    info_lines.append(f"[green]✓[/green] Total encoded columns from config: [bold]{len(all_encoded_cols)}[/bold]")

    if id_cols:
        id_list = ', '.join(sorted(id_cols))
        info_lines.append(f"[yellow]⊘[/yellow] Excluded ID columns ({len(id_cols)}): {id_list}")

    if datetime_in_use:
        dt_list = ', '.join(sorted(set([col.split('.')[0] for col in datetime_in_use])))
        info_lines.append(f"[blue]◷[/blue] Including datetime columns ({len(set([col.split('.')[0] for col in datetime_in_use]))}): {dt_list}")

    if onehot_base:
        onehot_list = ', '.join(sorted(onehot_base))
        info_lines.append(f"[magenta]⊕[/magenta] One-hot encoded features ({len(onehot_base)}): {onehot_list}")
        info_lines.append(f"    → Expanded to {len(onehot_cols)} binary columns")

    info_lines.append(f"[bold green]→[/bold green] Final evaluation set: [bold]{len(usable_cols)}[/bold] columns")

    # Show first few column names if reasonable
    if len(usable_cols) <= 10:
        col_preview = ', '.join(usable_cols)
        info_lines.append(f"    Columns: {col_preview}")
    else:
        col_preview = ', '.join(usable_cols[:5])
        info_lines.append(f"    Columns (first 5): {col_preview}, ...")

    # Print with Rich
    print("\n".join(info_lines))
    print()
