"""
Ported from sdpype/metadata.py — keeps only load_csv_with_metadata and the
dtype map it depends on. detect_metadata()/_display_metadata_summary() are
dropped: both are CLI-only (import sdv.metadata.SingleTableMetadata + typer),
unused by the encode/train/generate task chain, and prepare_step7.py already
covers metadata-generation for this repo.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Mapping from SDV computer_representation to pandas dtype
NUMERIC_DTYPE_MAP = {
    "Int8": "Int8",
    "Int16": "Int16",
    "Int32": "Int32",
    "Int64": "Int64",
    "UInt8": "UInt8",
    "UInt16": "UInt16",
    "UInt32": "UInt32",
    "UInt64": "UInt64",
    "Float": "float64",
}


def load_csv_with_metadata(
    csv_path: Path,
    metadata_path: Path,
    **read_csv_kwargs
) -> pd.DataFrame:
    """
    Load CSV with proper dtypes enforced from SDV-style metadata.

    This function ensures type consistency across multiple CSV files by using
    metadata as the single source of truth for column types. This is critical
    for avoiding string/number mismatches when comparing datasets.

    Handles all SDV sdtypes:
    - categorical → str (ensures exact value matching, avoids "591" vs 591)
    - numerical → Int64, Float, etc. (from computer_representation)
    - boolean → bool
    - datetime → parsed with datetime_format
    - id, PII types (email, phone, etc.) → str (for exact comparison)

    Args:
        csv_path: Path to CSV file to load
        metadata_path: Path to metadata JSON file (REQUIRED)
        **read_csv_kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        DataFrame with properly typed columns according to metadata

    Raises:
        FileNotFoundError: If csv_path or metadata_path doesn't exist
        ValueError: If metadata format is invalid
    """
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)

    if "columns" not in metadata_dict:
        raise ValueError(f"Metadata file {metadata_path} missing 'columns' key")

    columns_meta = metadata_dict["columns"]

    dtype_map: Dict[str, Any] = {}
    parse_dates_list = []
    date_format_dict: Dict[str, str] = {}
    boolean_columns = []

    for col_name, col_meta in columns_meta.items():
        sdtype = col_meta.get("sdtype")

        if sdtype == "categorical":
            # Force categorical to string to avoid type mismatches
            # (e.g., Region ID could be "591" in one file, 591 in another)
            dtype_map[col_name] = str

        elif sdtype == "numerical":
            comp_repr = col_meta.get("computer_representation", "Float")
            dtype_map[col_name] = NUMERIC_DTYPE_MAP.get(comp_repr, "float64")

        elif sdtype == "boolean":
            # Mark for post-processing (pandas may read as string)
            boolean_columns.append(col_name)
            # Don't set dtype - let pandas infer, then convert

        elif sdtype == "datetime":
            parse_dates_list.append(col_name)
            datetime_fmt = col_meta.get("datetime_format")
            if datetime_fmt:
                date_format_dict[col_name] = datetime_fmt

        elif sdtype == "id":
            # IDs should be treated as strings for exact comparison
            dtype_map[col_name] = str

        else:
            # All other sdtypes (PII types like email, phone, etc.)
            # treat as string for exact comparison purposes
            dtype_map[col_name] = str

    csv_kwargs = read_csv_kwargs.copy()

    if dtype_map:
        csv_kwargs["dtype"] = dtype_map

    if parse_dates_list:
        csv_kwargs["parse_dates"] = parse_dates_list

    if date_format_dict:
        csv_kwargs["date_format"] = date_format_dict

    df = pd.read_csv(csv_path, **csv_kwargs)

    # Post-process boolean columns
    # (pandas might read "True"/"False" strings, need explicit conversion)
    for bool_col in boolean_columns:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].map({
                'True': True, 'true': True, 'TRUE': True, '1': True, 1: True,
                'False': False, 'false': False, 'FALSE': False, '0': False, 0: False,
                True: True, False: False,
                # Keep NaN as NaN
            })

    return df
