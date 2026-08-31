"""
Statistical similarity evaluation — metric classes + dispatch + report.
======================================================================

Ported from sdpype/evaluation/statistical.py.

Top section: the shared column-selection helpers that sdpype/evaluation/
detection.py also imports (ensure_json_serializable, get_columns_by_sdtype,
get_encoded_numeric_columns, log_column_selection) — verbatim.

Lower section: the statistical_similarity metric implementations. All 15
sub-metrics are ported (they were added one metric per commit); each has a
`get_metric_evaluator` case. Listed below in sd-lake JSON order:
  table_structure    -> TableStructureMetric     (sdmetrics.single_table.TableStructure + pandas-dtype table)
  semantic_structure -> SemanticStructureMetric  (pure-python metadata sdtype compare)
  boundary_adherence -> BoundaryAdherenceMetric  (sdmetrics.single_column.BoundaryAdherence)
  category_adherence -> CategoryAdherenceMetric  (sdmetrics.single_column.CategoryAdherence)
  alpha_precision    -> AlphaPrecisionMetric     (synthcity.metrics.eval_statistical.AlphaPrecision)
  prdc_score         -> PRDCScoreMetric          (synthcity.metrics.eval_statistical.PRDCScore)
  wasserstein_distance -> WassersteinDistanceMetric  (custom CPU Sinkhorn OT via
    geomloss.SamplesLoss on MinMax-scaled tensors — NOT synthcity's WassersteinDistance)
  maximum_mean_discrepancy -> MaximumMeanDiscrepancyMetric  (corrected MMD, NOT synthcity's
    — see the module docstring)
  new_row_synthesis  -> NewRowSynthesisMetric  (sdmetrics.single_table.NewRowSynthesis
    compute_breakdown; synthetic_sample_size stays null — what made the ground truth)
  jensenshannon_synthcity -> JensenShannonSynthcityMetric  (synthcity JensenShannonDistance)
  jensenshannon_syndat -> JensenShannonSyndatMetric  (syndat.metrics.jensen_shannon_distance)
    -- the strictest JSD variant; the one the paper reports.
  jensenshannon_nannyml   -> JensenShannonNannyMLMetric    (scipy Doane-binned JSD; the
    module-level `from nannyml...` import in sdpype is dead code and is NOT carried)
  ks_complement      -> KSComplementMetric       (sdmetrics.single_column.KSComplement)
  tv_complement      -> TVComplementMetric       (sdmetrics.single_column.TVComplement)
  sdmetrics_quality  -> SDMetricsQualityMetric  (sdmetrics.reports.single_table.
    QualityReport with the working-tree SpearmanColumnPairTrends(ColumnPairTrends)
    override — Column-Pair-Trends CorrelationSimilarity forced to Spearman via the
    private _properties/_compute_average/_compute_pair_score API — plus custom
    Spearman / Cramér-V / correlation-ratio matrices. This is why sdmetrics==0.28.0
    is pinned.)
`evaluate_statistical_metrics` (generic router) and
`generate_statistical_report` (per-metric guarded) are ported whole and
handle every metric name; `get_metric_evaluator`'s `case _` still raises
(an unknown/unported metric fails loudly rather than degrading silently).

Differences from sdpype: the 2 `OmegaConf.is_config(parameters)` lines in
`evaluate_statistical_metrics` are dropped (no Hydra here); the module-level
`from rich import print` is not carried (same choice as the rest of the port
— builtin print, rich markup rendered literally where sdpype emits it).

`maximum_mean_discrepancy` is DELIBERATELY NOT synthcity's implementation.
synthcity's `MaximumMeanDiscrepancy` hardcodes gamma=1.0 on RDT-native-unit
data, so the RBF kernel underflows and the biased estimator collapses to
2/n ≈ 0.0002 for every generator (see sdpype's recompute_mmd_corrected.py).
`MaximumMeanDiscrepancyMetric` here is that script's corrected computation
inlined as a per-run metric: z-score every column on the REAL reference,
a frozen per-variant RBF gamma (median heuristic on the standardized real
reference, pooled over the variant's dseeds — Step7pfa 0.02524003218131145,
Step7pfp 0.048527854564490464; supply via the `gamma` parameter), and the
UNBIASED MMD^2 estimator (k(x,x) diagonals dropped). This does NOT match
sd-lake's stored `maximum_mean_discrepancy` block — it matches
metrics_long_<variant>_mmd_corrected.csv.

Uses the real sdv.metadata.SingleTableMetadata.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata

from geomloss import SamplesLoss
from scipy.stats import chi2_contingency, spearmanr
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table._properties.column_pair_trends import ColumnPairTrends
from sdmetrics.single_column import (
    BoundaryAdherence,
    CategoryAdherence,
    KSComplement,
    TVComplement,
)
from sdmetrics.single_table import TableStructure
from sdmetrics.single_table.new_row_synthesis import NewRowSynthesis
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.preprocessing import MinMaxScaler
from syndat.metrics import jensen_shannon_distance as syndat_jsd
from synthcity.metrics.eval_statistical import (
    AlphaPrecision,
    JensenShannonDistance,
    PRDCScore,
)
from synthcity.plugins.core.dataloader import GenericDataLoader


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


# ===========================================================================
# Statistical similarity metric implementations
# ===========================================================================


class KSComplementMetric:
    """KSComplement metric implementation for column-wise distribution similarity"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.target_columns = parameters.get("target_columns", None)  # None = all numerical/datetime

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate KSComplement metric across compatible columns"""
        start_time = time.time()

        try:
            # Identify compatible columns (numerical and datetime)
            compatible_columns = self._get_compatible_columns(metadata)

            if self.target_columns:
                # Filter to user-specified columns
                compatible_columns = [col for col in compatible_columns if col in self.target_columns]

            if not compatible_columns:
                # No compatible columns is not an error - just return empty results
                return {
                    "aggregate_score": None,  # Use None to indicate N/A
                    "column_scores": {},
                    "compatible_columns": [],
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": "No compatible numerical/datetime columns found"
                }

            column_scores = {}
            failed_columns = []
            for column in compatible_columns:
                try:
                    # Convert datetime columns to datetime64 dtype if needed
                    # SDMetrics KSComplement expects datetime64, not object/string
                    orig_col = original[column]
                    synth_col = synthetic[column]

                    # Check if column should be datetime based on metadata
                    col_sdtype = metadata.columns[column].get('sdtype') if column in metadata.columns else None
                    if col_sdtype == 'datetime' and orig_col.dtype == 'object':
                        # Convert string datetime to datetime64
                        orig_col = pd.to_datetime(orig_col, errors='coerce')
                        synth_col = pd.to_datetime(synth_col, errors='coerce')

                    score = KSComplement.compute(
                        real_data=orig_col,
                        synthetic_data=synth_col
                    )
                    column_scores[column] = float(score)
                except Exception as e:
                    # Handle individual column failures - use None instead of 0.0
                    column_scores[column] = None
                    failed_columns.append(column)
                    print(f"Warning: KSComplement failed for column '{column}': {e}")

            # Calculate aggregate score - exclude failed (None) columns
            successful_scores = [score for score in column_scores.values() if score is not None]
            aggregate_score = float(np.mean(successful_scores)) if successful_scores else None

            return {
                "aggregate_score": aggregate_score,
                "column_scores": column_scores,
                "compatible_columns": compatible_columns,
                "failed_columns": failed_columns,
                "successful_columns": len(successful_scores),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "aggregate_score": 0.0,
                "column_scores": {},
                "compatible_columns": [],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }

    def _get_compatible_columns(self, metadata: SingleTableMetadata) -> List[str]:
        """Get columns compatible with KSComplement (numerical and datetime) from SDV metadata"""

        compatible_columns = []
        for column_name, column_info in metadata.columns.items():
            sdtype = column_info.get('sdtype', 'unknown')
            if sdtype in ['numerical', 'datetime']:
                compatible_columns.append(column_name)

        return compatible_columns


class TVComplementMetric:
    """TVComplement metric implementation for column-wise categorical distribution similarity"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.target_columns = parameters.get("target_columns", None)  # None = all categorical/boolean

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate TVComplement metric across compatible columns"""
        start_time = time.time()

        try:
            # Identify compatible columns (categorical and boolean)
            compatible_columns = self._get_compatible_columns(metadata)

            if self.target_columns:
                # Filter to user-specified columns
                compatible_columns = [col for col in compatible_columns if col in self.target_columns]

            if not compatible_columns:
                # No compatible columns is not an error - just return empty results
                return {
                    "aggregate_score": None,  # Use None to indicate N/A
                    "column_scores": {},
                    "compatible_columns": [],
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": "No compatible categorical/boolean columns found"
                }

            column_scores = {}
            failed_columns = []
            for column in compatible_columns:
                try:
                    score = TVComplement.compute(
                        real_data=original[column],
                        synthetic_data=synthetic[column]
                    )
                    column_scores[column] = float(score)
                except Exception as e:
                    # Handle individual column failures - use None instead of 0.0
                    column_scores[column] = None
                    failed_columns.append(column)
                    print(f"Warning: TVComplement failed for column '{column}': {e}")

            # Calculate aggregate score - exclude failed (None) columns
            successful_scores = [score for score in column_scores.values() if score is not None]
            aggregate_score = float(np.mean(successful_scores)) if successful_scores else None

            return {
                "aggregate_score": aggregate_score,
                "column_scores": column_scores,
                "compatible_columns": compatible_columns,
                "failed_columns": failed_columns,
                "successful_columns": len(successful_scores),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "aggregate_score": 0.0,
                "column_scores": {},
                "compatible_columns": [],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }

    def _get_compatible_columns(self, metadata: SingleTableMetadata) -> List[str]:
        """Get columns compatible with TVComplement (categorical and boolean) from SDV metadata"""

        compatible_columns = []
        for column_name, column_info in metadata.columns.items():
            sdtype = column_info.get('sdtype', 'unknown')
            if sdtype in ['categorical', 'boolean']:
                compatible_columns.append(column_name)

        return compatible_columns


class BoundaryAdherenceMetric:
    """BoundaryAdherence metric implementation for column-wise boundary validation"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.target_columns = parameters.get("target_columns", None)  # None = all numerical/datetime

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate BoundaryAdherence metric across compatible columns"""
        start_time = time.time()

        try:
            # Identify compatible columns (numerical and datetime)
            compatible_columns = self._get_compatible_columns(metadata)

            if self.target_columns:
                # Filter to only requested columns that are also compatible
                target_set = set(self.target_columns)
                compatible_set = set(compatible_columns)
                valid_targets = list(target_set.intersection(compatible_set))
                invalid_targets = list(target_set - compatible_set)

                if invalid_targets:
                    print(f"Warning: These target columns are not compatible with BoundaryAdherence: {invalid_targets}")

                if not valid_targets:
                    return {
                        "aggregate_score": None,
                        "column_scores": {},
                        "compatible_columns": compatible_columns,
                        "parameters": self.parameters,
                        "execution_time": time.time() - start_time,
                        "status": "success",
                        "message": f"No compatible columns found from target list: {self.target_columns}"
                    }

                columns_to_evaluate = valid_targets
            else:
                columns_to_evaluate = compatible_columns

            if not columns_to_evaluate:
                return {
                    "aggregate_score": None,
                    "column_scores": {},
                    "compatible_columns": compatible_columns,
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": "No compatible numerical/datetime columns found in the dataset"
                }

            # Calculate BoundaryAdherence for each compatible column
            column_scores = {}
            failed_columns = []
            for column in columns_to_evaluate:
                try:
                    # Convert datetime columns to datetime64 dtype if needed
                    # SDMetrics BoundaryAdherence expects datetime64, not object/string
                    orig_col = original[column]
                    synth_col = synthetic[column]

                    # Check if column should be datetime based on metadata
                    col_sdtype = metadata.columns[column].get('sdtype') if column in metadata.columns else None
                    if col_sdtype == 'datetime' and orig_col.dtype == 'object':
                        # Convert string datetime to datetime64
                        orig_col = pd.to_datetime(orig_col, errors='coerce')
                        synth_col = pd.to_datetime(synth_col, errors='coerce')

                    score = BoundaryAdherence.compute(
                        real_data=orig_col,
                        synthetic_data=synth_col
                    )
                    column_scores[column] = float(score)
                except Exception as e:
                    print(f"Error computing BoundaryAdherence for column {column}: {e}")
                    column_scores[column] = None
                    failed_columns.append(column)

            # Extract successful scores (exclude None values from failures)
            successful_scores = [score for score in column_scores.values() if score is not None]

            if not successful_scores:
                return {
                    "aggregate_score": None,
                    "column_scores": column_scores,
                    "compatible_columns": compatible_columns,
                    "failed_columns": failed_columns,
                    "successful_columns": 0,
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": "All column evaluations failed"
                }

            # Calculate aggregate score as mean of successful column scores only
            aggregate_score = float(np.mean(successful_scores))

            return {
                "aggregate_score": aggregate_score,
                "column_scores": column_scores,
                "compatible_columns": compatible_columns,
                "failed_columns": failed_columns,
                "successful_columns": len(successful_scores),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "aggregate_score": 0.0,
                "column_scores": {},
                "compatible_columns": [],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }

    def _get_compatible_columns(self, metadata: SingleTableMetadata) -> List[str]:
        """Get columns compatible with BoundaryAdherence (numerical and datetime) from SDV metadata"""

        compatible_columns = []
        for column_name, column_info in metadata.columns.items():
            sdtype = column_info.get('sdtype', 'unknown')
            if sdtype in ['numerical', 'datetime']:
                compatible_columns.append(column_name)

        return compatible_columns


class CategoryAdherenceMetric:
    """CategoryAdherence metric implementation for categorical/boolean column validation"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.target_columns = parameters.get("target_columns", None)  # None = all categorical/boolean

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate CategoryAdherence metric across compatible columns"""
        start_time = time.time()

        try:
            # Identify compatible columns (categorical and boolean)
            compatible_columns = self._get_compatible_columns(metadata)

            if self.target_columns:
                # Filter to only requested columns that are also compatible
                target_set = set(self.target_columns)
                compatible_set = set(compatible_columns)
                valid_targets = list(target_set.intersection(compatible_set))

                # Warn about invalid target columns
                invalid_targets = target_set - compatible_set
                if invalid_targets:
                    print(f"Warning: Columns {invalid_targets} are not compatible with CategoryAdherence (not categorical/boolean)")

                columns_to_evaluate = valid_targets
            else:
                columns_to_evaluate = compatible_columns

            if not columns_to_evaluate:
                return {
                    "aggregate_score": None,
                    "column_scores": {},
                    "compatible_columns": compatible_columns,
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": f"No compatible categorical/boolean columns found for evaluation"
                }

            # Evaluate each column
            column_scores = {}
            failed_columns = []
            for column in columns_to_evaluate:
                try:
                    score = CategoryAdherence.compute(
                        real_data=original[column],
                        synthetic_data=synthetic[column]
                    )
                    column_scores[column] = float(score)
                except Exception as e:
                    print(f"Warning: Failed to compute CategoryAdherence for column '{column}': {str(e)}")
                    column_scores[column] = None
                    failed_columns.append(column)

            # Calculate aggregate score - exclude failed (None) columns
            successful_scores = [score for score in column_scores.values() if score is not None]
            aggregate_score = float(np.mean(successful_scores)) if successful_scores else None

            return {
                "aggregate_score": aggregate_score,
                "column_scores": column_scores,
                "compatible_columns": compatible_columns,
                "failed_columns": failed_columns,
                "successful_columns": len(successful_scores),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "aggregate_score": 0.0,
                "column_scores": {},
                "compatible_columns": [],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }

    def _get_compatible_columns(self, metadata: SingleTableMetadata) -> List[str]:
        """Get columns compatible with CategoryAdherence (categorical and boolean) from SDV metadata"""

        compatible_columns = []
        for column_name, column_info in metadata.columns.items():
            sdtype = column_info.get('sdtype', 'unknown')
            if sdtype in ['categorical', 'boolean']:
                compatible_columns.append(column_name)

        return compatible_columns


class TableStructureMetric:
    """TableStructure metric implementation for table structure validation"""

    def __init__(self, **parameters):
        self.parameters = parameters

    def _build_column_comparison(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """
        Build detailed column comparison between original and synthetic data.

        Following SDV's TableStructure logic:
        - Numerator: columns with same name AND same pandas dtype
        - Denominator: all combinations of (column name, dtype) across both datasets

        Args:
            original: Original DataFrame
            synthetic: Synthetic DataFrame

        Returns:
            Dict with comparison details including counts and per-column status
        """
        # Get column information
        real_cols = set(original.columns)
        synth_cols = set(synthetic.columns)

        # Initialize counters
        n_matching = 0
        n_dtype_mismatch = 0
        n_missing = 0
        n_extra = 0

        # Build detailed comparison
        column_details = {}
        comparison_table = []

        # Columns in both datasets
        common_cols = real_cols & synth_cols
        for col in sorted(common_cols):
            real_dtype = str(original[col].dtype)
            synth_dtype = str(synthetic[col].dtype)

            if real_dtype == synth_dtype:
                status = "match"
                n_matching += 1
            else:
                status = "dtype_mismatch"
                n_dtype_mismatch += 1

            column_details[col] = {
                "real_dtype": real_dtype,
                "synthetic_dtype": synth_dtype,
                "status": status
            }
            comparison_table.append({
                "column": col,
                "real_dtype": real_dtype,
                "synthetic_dtype": synth_dtype,
                "status": status
            })

        # Columns only in real data (missing in synthetic)
        missing_cols = real_cols - synth_cols
        for col in sorted(missing_cols):
            real_dtype = str(original[col].dtype)
            n_missing += 1

            column_details[col] = {
                "real_dtype": real_dtype,
                "synthetic_dtype": None,
                "status": "missing_in_synthetic"
            }
            comparison_table.append({
                "column": col,
                "real_dtype": real_dtype,
                "synthetic_dtype": None,
                "status": "missing_in_synthetic"
            })

        # Columns only in synthetic data (extra/unexpected)
        extra_cols = synth_cols - real_cols
        for col in sorted(extra_cols):
            synth_dtype = str(synthetic[col].dtype)
            n_extra += 1

            column_details[col] = {
                "real_dtype": None,
                "synthetic_dtype": synth_dtype,
                "status": "only_in_synthetic"
            }
            comparison_table.append({
                "column": col,
                "real_dtype": None,
                "synthetic_dtype": synth_dtype,
                "status": "only_in_synthetic"
            })

        return {
            "summary": {
                "total_real_columns": len(real_cols),
                "total_synthetic_columns": len(synth_cols),
                "matching_columns": n_matching,
                "dtype_mismatches": n_dtype_mismatch,
                "missing_in_synthetic": n_missing,
                "only_in_synthetic": n_extra
            },
            "column_details": column_details,
            "comparison_table": comparison_table
        }

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate TableStructure metric"""
        start_time = time.time()

        try:
            # Run evaluation using SDMetrics
            score = TableStructure.compute(
                real_data=original,
                synthetic_data=synthetic
            )

            # Build detailed column comparison
            comparison_data = self._build_column_comparison(original, synthetic)

            return {
                "score": float(score),
                "summary": comparison_data["summary"],
                "column_details": comparison_data["column_details"],
                "comparison_table": comparison_data["comparison_table"],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "score": 0.0,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class SemanticStructureMetric:
    """SemanticStructure metric implementation using SDV metadata sdtypes instead of pandas dtypes"""

    def __init__(self, **parameters):
        self.parameters = parameters

    def _build_semantic_comparison(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                                   original_metadata: SingleTableMetadata,
                                   synthetic_metadata: SingleTableMetadata) -> Dict[str, Any]:
        """
        Build detailed column comparison based on SDV semantic types (sdtypes).

        Compares columns based on their semantic meaning (numerical, categorical, datetime, etc.)
        rather than strict pandas dtypes (int64, float64, object, etc.).

        Args:
            original: Original DataFrame
            synthetic: Synthetic DataFrame
            original_metadata: SDV metadata for original data
            synthetic_metadata: SDV metadata for synthetic data

        Returns:
            Dict with comparison details including counts and per-column status
        """
        # Get column information
        real_cols = set(original.columns)
        synth_cols = set(synthetic.columns)

        # Initialize counters
        n_matching = 0
        n_sdtype_mismatch = 0
        n_missing = 0
        n_extra = 0

        # Build detailed comparison
        column_details = {}
        comparison_table = []

        # Columns in both datasets
        common_cols = real_cols & synth_cols
        for col in sorted(common_cols):
            # Get sdtypes from metadata
            real_sdtype = original_metadata.columns.get(col, {}).get('sdtype', 'unknown')
            synth_sdtype = synthetic_metadata.columns.get(col, {}).get('sdtype', 'unknown')

            if real_sdtype == synth_sdtype:
                status = "match"
                n_matching += 1
            else:
                status = "sdtype_mismatch"
                n_sdtype_mismatch += 1

            column_details[col] = {
                "real_sdtype": real_sdtype,
                "synthetic_sdtype": synth_sdtype,
                "status": status
            }
            comparison_table.append({
                "column": col,
                "real_sdtype": real_sdtype,
                "synthetic_sdtype": synth_sdtype,
                "status": status
            })

        # Columns only in real data (missing in synthetic)
        missing_cols = real_cols - synth_cols
        for col in sorted(missing_cols):
            real_sdtype = original_metadata.columns.get(col, {}).get('sdtype', 'unknown')
            n_missing += 1

            column_details[col] = {
                "real_sdtype": real_sdtype,
                "synthetic_sdtype": None,
                "status": "missing_in_synthetic"
            }
            comparison_table.append({
                "column": col,
                "real_sdtype": real_sdtype,
                "synthetic_sdtype": None,
                "status": "missing_in_synthetic"
            })

        # Columns only in synthetic data (extra/unexpected)
        extra_cols = synth_cols - real_cols
        for col in sorted(extra_cols):
            synth_sdtype = synthetic_metadata.columns.get(col, {}).get('sdtype', 'unknown')
            n_extra += 1

            column_details[col] = {
                "real_sdtype": None,
                "synthetic_sdtype": synth_sdtype,
                "status": "only_in_synthetic"
            }
            comparison_table.append({
                "column": col,
                "real_sdtype": None,
                "synthetic_sdtype": synth_sdtype,
                "status": "only_in_synthetic"
            })

        # Calculate score following SDV's TableStructure logic:
        # Score = matching_columns / total_unique_combinations
        total_combinations = n_matching + n_sdtype_mismatch + n_missing + n_extra
        score = n_matching / total_combinations if total_combinations > 0 else 0.0

        return {
            "score": score,
            "summary": {
                "total_real_columns": len(real_cols),
                "total_synthetic_columns": len(synth_cols),
                "matching_columns": n_matching,
                "sdtype_mismatches": n_sdtype_mismatch,
                "missing_in_synthetic": n_missing,
                "only_in_synthetic": n_extra
            },
            "column_details": column_details,
            "comparison_table": comparison_table
        }

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate SemanticStructure metric using SDV metadata sdtypes"""
        start_time = time.time()

        try:
            # Use declared metadata for both — data is loaded with metadata-driven
            # types via load_csv_with_metadata, so the schema is the ground truth.
            original_metadata = metadata
            synthetic_metadata = metadata

            # Build detailed semantic comparison
            comparison_data = self._build_semantic_comparison(
                original, synthetic,
                original_metadata, synthetic_metadata
            )

            return {
                "score": float(comparison_data["score"]),
                "summary": comparison_data["summary"],
                "column_details": comparison_data["column_details"],
                "comparison_table": comparison_data["comparison_table"],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "score": 0.0,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class AlphaPrecisionMetric:
    """Alpha Precision metric implementation.

    NOTE on reproducibility: the ``*_naive`` scores are deterministic
    (MinMaxScaler + kNN) and reproduce sd-lake exactly. The ``*_OC``
    (optimally-corrected) scores go through synthcity's OneClassLayer NN
    embedding, whose ``fit()`` (synthcity 0.2.12) is unseeded — so the OC
    scores vary run to run (synthcity disk-caches the trained model per
    real-data hash under ``./workspace/``, making them stable *within* an
    environment after the first call, but not equal to sd-lake's stored
    values). Same class of upstream non-determinism as MMD; the port runs
    the identical synthcity code.
    """

    def __init__(self, **parameters):
        self.evaluator = AlphaPrecision()
        self.parameters = parameters  # Store for reporting

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate Alpha Precision metric"""

        start_time = time.time()

        try:
            # Use encoding config as source of truth for which columns are numeric
            if encoding_config:
                usable_cols = get_encoded_numeric_columns(encoding_config, original, metadata)
                log_column_selection("Alpha Precision", encoding_config, original, metadata, usable_cols)
            else:
                # Fallback to old logic if no encoding config provided
                numeric_cols = get_columns_by_sdtype(metadata, ['numerical'])
                datetime_cols = get_columns_by_sdtype(metadata, ['datetime'])
                categorical_cols = get_columns_by_sdtype(metadata, ['categorical'])
                numeric_categorical_cols = [
                    col for col in categorical_cols
                    if pd.api.types.is_numeric_dtype(original[col])
                ]
                usable_cols = numeric_cols + datetime_cols + numeric_categorical_cols
                print(f"  Alpha Precision using {len(usable_cols)} columns (fallback mode)")

            if not usable_cols:
                raise ValueError("No numeric columns found for Alpha Precision metric")

            # Select only usable columns
            original_numeric = original[usable_cols].copy()
            synthetic_numeric = synthetic[usable_cols].copy()

            # Create data loaders
            real_loader = GenericDataLoader(original_numeric)
            synth_loader = GenericDataLoader(synthetic_numeric)

            # Run evaluation
            result = self.evaluator.evaluate(real_loader, synth_loader)
            scores = {
                "delta_precision_alpha_OC": float(result.get("delta_precision_alpha_OC", 0.0)),
                "delta_coverage_beta_OC": float(result.get("delta_coverage_beta_OC", 0.0)),
                "authenticity_OC": float(result.get("authenticity_OC", 0.0)),
                "delta_precision_alpha_naive": float(result.get("delta_precision_alpha_naive", 0.0)),
                "delta_coverage_beta_naive": float(result.get("delta_coverage_beta_naive", 0.0)),
                "authenticity_naive": float(result.get("authenticity_naive", 0.0))
            }

            return {
                "scores": scores,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "scores": {
                    "delta_precision_alpha_OC": 0.0,
                    "delta_coverage_beta_OC": 0.0,
                    "authenticity_OC": 0.0,
                    "delta_precision_alpha_naive": 0.0,
                    "delta_coverage_beta_naive": 0.0,
                    "authenticity_naive": 0.0
                },
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class PRDCScoreMetric:
    """PRDC Score metric implementation"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.evaluator = PRDCScore(nearest_k=parameters.get("nearest_k", 5))

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate PRDC Score metric"""
        start_time = time.time()

        try:
            # Use encoding config as source of truth for which columns are numeric
            if encoding_config:
                usable_cols = get_encoded_numeric_columns(encoding_config, original, metadata)
                log_column_selection("PRDC Score", encoding_config, original, metadata, usable_cols)
            else:
                # Fallback to old logic if no encoding config provided
                numeric_cols = get_columns_by_sdtype(metadata, ['numerical'])
                datetime_cols = get_columns_by_sdtype(metadata, ['datetime'])
                categorical_cols = get_columns_by_sdtype(metadata, ['categorical'])
                numeric_categorical_cols = [
                    col for col in categorical_cols
                    if pd.api.types.is_numeric_dtype(original[col])
                ]
                usable_cols = numeric_cols + datetime_cols + numeric_categorical_cols
                print(f"  PRDC Score using {len(usable_cols)} columns (fallback mode)")

            if not usable_cols:
                raise ValueError("No numeric columns found for PRDC Score metric")

            # Select only usable columns
            original_numeric = original[usable_cols].copy()
            synthetic_numeric = synthetic[usable_cols].copy()

            # Create data loaders
            real_loader = GenericDataLoader(original_numeric)
            synth_loader = GenericDataLoader(synthetic_numeric)

            # Run evaluation
            result = self.evaluator.evaluate(real_loader, synth_loader)

            # PRDC returns a dict with precision, recall, density, coverage
            return {
                "precision": float(result.get("precision", 0.0)),
                "recall": float(result.get("recall", 0.0)),
                "density": float(result.get("density", 0.0)),
                "coverage": float(result.get("coverage", 0.0)),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "density": 0.0,
                "coverage": 0.0,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class JensenShannonSynthcityMetric:
    """Jensen-Shannon Distance metric implementation (Synthcity)"""

    def __init__(self, **parameters):
        self.parameters = parameters
        normalize = parameters.get("normalize", True)
        n_histogram_bins = parameters.get("n_histogram_bins", 10)
        self.evaluator = JensenShannonDistance(normalize=normalize, n_histogram_bins=n_histogram_bins)

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate Jensen-Shannon Distance metric using Synthcity"""
        start_time = time.time()

        try:
            # Use encoding config as source of truth for which columns are numeric
            if encoding_config:
                usable_cols = get_encoded_numeric_columns(encoding_config, original, metadata)
                log_column_selection("Jensen-Shannon Distance (Synthcity)", encoding_config, original, metadata, usable_cols)
            else:
                # Fallback to old logic if no encoding config provided
                numeric_cols = get_columns_by_sdtype(metadata, ['numerical'])
                datetime_cols = get_columns_by_sdtype(metadata, ['datetime'])
                categorical_cols = get_columns_by_sdtype(metadata, ['categorical'])
                numeric_categorical_cols = [
                    col for col in categorical_cols
                    if pd.api.types.is_numeric_dtype(original[col])
                ]
                usable_cols = numeric_cols + datetime_cols + numeric_categorical_cols
                print(f"  Jensen-Shannon Distance (Synthcity) using {len(usable_cols)} columns (fallback mode)")

            if not usable_cols:
                raise ValueError("No numeric columns found for Jensen-Shannon Distance metric")

            # Select only usable columns
            original_numeric = original[usable_cols].copy()
            synthetic_numeric = synthetic[usable_cols].copy()

            # Create data loaders
            real_loader = GenericDataLoader(original_numeric)
            synth_loader = GenericDataLoader(synthetic_numeric)

            # Run evaluation
            result = self.evaluator.evaluate(real_loader, synth_loader)

            # JSD returns raw distance (0-1, lower=better)
            raw_distance = float(result.get("marginal", 0.0))

            return {
                "distance_score": raw_distance,
                "normalize": self.parameters.get("normalize", True),
                "n_histogram_bins": self.parameters.get("n_histogram_bins", 10),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "distance_score": 1.0,
                "normalize": self.parameters.get("normalize", True),
                "n_histogram_bins": self.parameters.get("n_histogram_bins", 10),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class JensenShannonSyndatMetric:
    """Jensen-Shannon Distance metric implementation (SYNDAT)"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.n_unique_threshold = parameters.get("n_unique_threshold", 10)

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate Jensen-Shannon Distance metric using SYNDAT"""
        start_time = time.time()

        try:
            # SYNDAT works directly with DataFrames
            # Returns per-column JSD scores
            jsd_per_column = syndat_jsd(original, synthetic, n_unique_threshold=self.n_unique_threshold)

            # Calculate aggregate distance (mean across columns), lower=better
            raw_distance = float(np.mean(list(jsd_per_column.values())))

            return {
                "distance_score": raw_distance,
                "n_unique_threshold": self.n_unique_threshold,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "distance_score": 1.0,
                "n_unique_threshold": self.n_unique_threshold,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class JensenShannonNannyMLMetric:
    """Jensen-Shannon Distance metric implementation (NannyML)"""

    def __init__(self, **parameters):
        self.parameters = parameters

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate Jensen-Shannon Distance metric using NannyML's binning methodology"""
        start_time = time.time()

        try:
            column_scores = {}

            # Use encoding config as source of truth for which columns to evaluate
            if encoding_config:
                # Get columns that were encoded (all are numeric after encoding)
                usable_cols = get_encoded_numeric_columns(encoding_config, original, metadata)
                log_column_selection("Jensen-Shannon Distance (NannyML)", encoding_config, original, metadata, usable_cols)

                # All encoded columns are numeric - use Doane's binning for all
                for column in usable_cols:
                    try:
                        # Continuous feature - use Doane's formula for adaptive binning
                        n_bins = int(1 + np.log2(len(original)) + np.log2(1 + np.abs(original[column].skew()) / np.sqrt(6 * (len(original) - 2) / ((len(original) + 1) * (len(original) + 3)))))
                        n_bins = max(10, min(n_bins, 50))  # Bound bins between 10-50

                        # Create bins from original (reference)
                        bins = np.histogram_bin_edges(original[column].dropna(), bins=n_bins)

                        # Calculate histograms
                        orig_hist, _ = np.histogram(original[column].dropna(), bins=bins)
                        synth_hist, _ = np.histogram(synthetic[column].dropna(), bins=bins)

                        # Normalize to get probabilities
                        orig_prob = orig_hist / orig_hist.sum() if orig_hist.sum() > 0 else orig_hist
                        synth_prob = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist

                        # Compute JSD
                        from scipy.spatial.distance import jensenshannon
                        jsd = jensenshannon(orig_prob, synth_prob)

                        column_scores[column] = float(jsd)
                    except Exception as e:
                        # Skip column on failure but continue
                        continue
            else:
                # Fallback mode: iterate all columns and determine type at runtime
                print(f"  Jensen-Shannon Distance (NannyML) using fallback mode (no encoding config)")
                for column in original.columns:
                    try:
                        # Determine if column is continuous or categorical
                        if pd.api.types.is_numeric_dtype(original[column]):
                            # Continuous feature - use Doane's formula for binning
                            n_bins = int(1 + np.log2(len(original)) + np.log2(1 + np.abs(original[column].skew()) / np.sqrt(6 * (len(original) - 2) / ((len(original) + 1) * (len(original) + 3)))))
                            n_bins = max(10, min(n_bins, 50))  # Bound bins between 10-50

                            # Create bins from original (reference)
                            bins = np.histogram_bin_edges(original[column].dropna(), bins=n_bins)

                            # Calculate histograms
                            orig_hist, _ = np.histogram(original[column].dropna(), bins=bins)
                            synth_hist, _ = np.histogram(synthetic[column].dropna(), bins=bins)

                            # Normalize to get probabilities
                            orig_prob = orig_hist / orig_hist.sum() if orig_hist.sum() > 0 else orig_hist
                            synth_prob = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist

                            # Compute JSD
                            from scipy.spatial.distance import jensenshannon
                            jsd = jensenshannon(orig_prob, synth_prob)

                        else:
                            # Categorical feature - use frequency counts
                            orig_counts = original[column].value_counts(normalize=True)
                            synth_counts = synthetic[column].value_counts(normalize=True)

                            # Align categories
                            all_cats = orig_counts.index.union(synth_counts.index)
                            orig_prob = orig_counts.reindex(all_cats, fill_value=0).values
                            synth_prob = synth_counts.reindex(all_cats, fill_value=0).values

                            from scipy.spatial.distance import jensenshannon
                            jsd = jensenshannon(orig_prob, synth_prob)

                        column_scores[column] = float(jsd)

                    except Exception as e:
                        # Skip column on failure but continue
                        continue

            # Calculate aggregate distance, lower=better
            if column_scores:
                raw_distance = float(np.mean(list(column_scores.values())))
            else:
                raw_distance = 1.0

            return {
                "distance_score": raw_distance,
                "n_columns_evaluated": len(column_scores),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            # print(f"\n🔴 NannyML ERROR: {e}")
            # import traceback
            # traceback.print_exc()

            return {
                "distance_score": 1.0,
                "n_columns_evaluated": 0,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


# Chunk size for the pairwise RBF kernel sum; matches recompute_mmd_corrected.py
# so the chunked float summation order (and thus the last ULPs) is identical.
_MMD_CHUNK = 2000


def _mmd_kernel_sum(A: np.ndarray, B: np.ndarray, gamma: float) -> float:
    """Sum of the RBF kernel over all pairs, chunked to bound peak memory."""
    total = 0.0
    for i in range(0, len(A), _MMD_CHUNK):
        total += float(rbf_kernel(A[i:i + _MMD_CHUNK], B, gamma).sum())
    return total


def _mmd_median_sqdist(X: np.ndarray, sample: int = 2000, seed: int = 0) -> float:
    """Median pairwise squared distance on a subsample (median heuristic)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    S = X[idx]
    d2 = euclidean_distances(S, S, squared=True)
    off = d2[~np.eye(len(S), dtype=bool)]
    return float(np.median(off))


class MaximumMeanDiscrepancyMetric:
    """Maximum Mean Discrepancy — the CORRECTED computation (not synthcity's).

    synthcity's MaximumMeanDiscrepancy hardcodes gamma=1.0 on RDT-native-unit
    data: the RBF kernel underflows and the biased estimator collapses to
    2/n ≈ 0.0002 for every generator. This reimplements sdpype's
    recompute_mmd_corrected.py as a per-run metric:

      1. z-score every column using the REAL reference mean/sd (ddof=0);
         constant columns (sd == 0) are left uncentred by scale (sd -> 1).
      2. RBF gamma: the ``gamma`` parameter when given (the paper freezes a
         per-variant value — Step7pfa 0.02524003218131145,
         Step7pfp 0.048527854564490464 — a median heuristic on the
         standardized real reference pooled over the variant's dseeds). With
         no ``gamma`` it falls back to the median heuristic on THIS run's
         standardized reference (sample=2000, seed=0) — close but not the
         frozen value.
      3. the UNBIASED MMD^2 estimator: the k(x,x) = 1 diagonals are dropped
         from XX and YY, so the 2/n floor disappears and an exact match
         scores ~0.

    Runs on ENCODED data (native units — the standardization is what makes
    the kernel well-scaled). Output ``joint_distance`` is the corrected MMD^2
    and will NOT match sd-lake's stored block; it matches
    metrics_long_<variant>_mmd_corrected.csv.
    """

    def __init__(self, **parameters):
        self.parameters = parameters
        self.kernel = parameters.get("kernel", "rbf")
        self.gamma = parameters.get("gamma", None)

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate the corrected Maximum Mean Discrepancy."""
        start_time = time.time()

        try:
            if self.kernel != "rbf":
                raise ValueError(f"Only the RBF kernel is supported, got {self.kernel!r}")

            cols = list(original.columns)
            if list(synthetic.columns) != cols:
                synthetic = synthetic[cols]

            A = original.to_numpy(dtype=float)
            B = synthetic.to_numpy(dtype=float)

            # Standardize on the REAL reference (never synthetic stats).
            mu = A.mean(axis=0)
            sd = A.std(axis=0)
            sd[sd == 0] = 1.0
            As = (A - mu) / sd
            Bs = (B - mu) / sd

            gamma = self.gamma
            gamma_source = "parameter"
            if gamma is None:
                gamma = 1.0 / _mmd_median_sqdist(As)
                gamma_source = "median_heuristic_this_run"
            gamma = float(gamma)

            n, m = len(As), len(Bs)
            xx = _mmd_kernel_sum(As, As, gamma)
            yy = _mmd_kernel_sum(Bs, Bs, gamma)
            xy = _mmd_kernel_sum(As, Bs, gamma)

            # Drop the k(x, x) = 1 diagonals: n of them in XX, m in YY.
            xx_u = (xx - n) / (n * (n - 1))
            yy_u = (yy - m) / (m * (m - 1))
            xy_mean = xy / (n * m)
            mmd = xx_u + yy_u - 2 * xy_mean

            return {
                "joint_distance": float(mmd),
                "kernel": self.kernel,
                "gamma": gamma,
                "gamma_source": gamma_source,
                "estimator": "unbiased",
                "standardization": "z-score on real reference",
                "terms": {
                    "xx_unbiased": float(xx_u),
                    "yy_unbiased": float(yy_u),
                    "xy_mean": float(xy_mean),
                },
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "joint_distance": 1.0,
                "kernel": self.kernel,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


# --- SIMPLIFIED CPU WASSERSTEIN IMPLEMENTATION ---
def evaluate_wasserstein(
    original_encoded_df: pd.DataFrame,
    synthetic_encoded_df: pd.DataFrame,
) -> float:
    """
    Compute Wasserstein distance using a CPU-only, tensorized Sinkhorn OT backend.

    Assumes input DataFrames are already encoded (all numeric).
    """
    X = original_encoded_df.to_numpy(dtype=np.float32)
    X_syn = synthetic_encoded_df.to_numpy(dtype=np.float32)

    # 1) If synthetic has fewer rows, pad it with zeros
    if X.shape[0] > X_syn.shape[0]:
        pad_rows = X.shape[0] - X_syn.shape[0]
        X_syn = np.concatenate(
            [X_syn, np.zeros((pad_rows, X.shape[1]), dtype=X_syn.dtype)],
            axis=0,
        )
    # If original has fewer rows, pad it with zeros (unlikely but safe)
    elif X_syn.shape[0] > X.shape[0]:
        pad_rows = X_syn.shape[0] - X.shape[0]
        X = np.concatenate(
            [X, np.zeros((pad_rows, X.shape[1]), dtype=X.dtype)],
            axis=0,
        )

    # 2) MinMax-scale features using real data statistics
    scaler = MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_syn_scaled = scaler.transform(X_syn)

    # 3) Convert to CPU tensors
    X_ten = torch.from_numpy(X_scaled)
    X_syn_ten = torch.from_numpy(X_syn_scaled)

    # 4) Sinkhorn OT distance, tensorized backend (CPU)
    OT_solver = SamplesLoss(
        loss="sinkhorn",
        backend="tensorized",
    )

    with torch.no_grad():
        dist = OT_solver(X_ten, X_syn_ten).cpu().numpy().item()

    return float(dist)
# --- END SIMPLIFIED CPU WASSERSTEIN IMPLEMENTATION ---


class WassersteinDistanceMetric:
    """Wasserstein Distance metric implementation (now CPU-based)"""

    def __init__(self, **parameters):
        self.parameters = parameters
        # Removed: self.evaluator = WassersteinDistance()

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate Wasserstein Distance metric"""
        start_time = time.time()

        try:
            # 1. Use encoding config to identify the correct subset of columns
            if encoding_config:
                usable_cols = get_encoded_numeric_columns(encoding_config, original, metadata)
                log_column_selection("Wasserstein Distance", encoding_config, original, metadata, usable_cols)
            else:
                # Fallback logic (unchanged)
                numeric_cols = get_columns_by_sdtype(metadata, ['numerical'])
                datetime_cols = get_columns_by_sdtype(metadata, ['datetime'])
                categorical_cols = get_columns_by_sdtype(metadata, ['categorical'])
                numeric_categorical_cols = [
                    col for col in categorical_cols
                    if pd.api.types.is_numeric_dtype(original[col])
                ]
                usable_cols = numeric_cols + datetime_cols + numeric_categorical_cols
                print(f"  Wasserstein Distance using {len(usable_cols)} columns (fallback mode)")

            if not usable_cols:
                raise ValueError("No numeric columns found for Wasserstein Distance metric")

            # 2. Select only usable columns (which are already encoded/numeric)
            original_numeric = original[usable_cols].copy()
            synthetic_numeric = synthetic[usable_cols].copy()

            # 3. Run evaluation using the CPU function
            # NOTE: We pass the two pre-processed DataFrames directly
            joint_distance = evaluate_wasserstein(original_numeric, synthetic_numeric)

            # ... (rest of the return dict is unchanged) ...
            return {
                "joint_distance": float(joint_distance),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            # ... (error handling is unchanged) ...
            return {
                "joint_distance": 0.0,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class NewRowSynthesisMetric:
    """NewRowSynthesis metric implementation"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.numerical_match_tolerance = parameters.get("numerical_match_tolerance", 0.01)
        self.synthetic_sample_size = parameters.get("synthetic_sample_size", None)

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata, encoding_config: dict = None) -> Dict[str, Any]:
        """Evaluate NewRowSynthesis metric"""
        start_time = time.time()

        try:
            # Convert the SingleTableMetadata object to a dictionary as required by SDMetrics
            metadata_dict = metadata.to_dict()

            # Run evaluation using SDMetrics
            result = NewRowSynthesis.compute_breakdown(
                real_data=original,
                synthetic_data=synthetic,
                metadata=metadata_dict,
                numerical_match_tolerance=self.numerical_match_tolerance,
                synthetic_sample_size=self.synthetic_sample_size
            )

            return {
                "score": float(result.get("score", 0.0)),
                "num_new_rows": int(result.get("num_new_rows", 0)),
                "num_matched_rows": int(result.get("num_matched_rows", 0)),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "score": 0.0,
                "num_new_rows": 0,
                "num_matched_rows": len(synthetic) if synthetic is not None else 0,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class SpearmanColumnPairTrends(ColumnPairTrends):
    def _compute_pair_score(self, metric, col_real, col_synthetic, metric_params):
        if metric.__name__ == 'CorrelationSimilarity':
            metric_params = dict(metric_params)
            metric_params['coefficient'] = 'Spearman'
        return super()._compute_pair_score(metric, col_real, col_synthetic, metric_params)


class SDMetricsQualityMetric:
    """
    SDMetrics comprehensive quality analysis with correlation matrices.

    Evaluates:
    - Column Shapes (distribution similarity using KS test, TV distance, etc.)
    - Column Pair Trends (correlation preservation across column pairs)

    Generates three correlation matrices:
    1. Real data correlations (with pairwise deletion and confidence metrics)
    2. Synthetic data correlations (complete data analysis)
    3. Quality scores (SDMetrics similarity assessment)

    Uses pairwise deletion to handle missing values in real data.
    Supports mixed data types (numerical, categorical, boolean).
    """

    def __init__(self, max_display_cols: int = 10, **parameters):
        """
        Initialize SDMetrics Quality metric.

        Args:
            max_display_cols: Maximum number of columns to display in matrices
            **parameters: Additional parameters (reserved for future use)
        """
        self.max_display_cols = max_display_cols
        self.parameters = parameters

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Cramér's V statistic for categorical-categorical association."""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1

        if min_dim == 0:
            return 0.0

        return np.sqrt(chi2 / (n * min_dim)) if n > 0 else 0.0

    def _correlation_ratio(self, categories: pd.Series, values: pd.Series) -> float:
        """Calculate correlation ratio (eta) for categorical-numerical association."""
        categories = categories.fillna('Missing')
        values = values.fillna(values.mean())

        categories = pd.Categorical(categories)
        grouped_mean = values.groupby(categories, observed=True).mean()
        overall_mean = values.mean()
        group_counts = values.groupby(categories, observed=True).size()

        ss_between = ((grouped_mean - overall_mean) ** 2 * group_counts).sum()
        ss_total = ((values - overall_mean) ** 2).sum()

        if ss_total == 0:
            return 0.0

        return np.sqrt(ss_between / ss_total)

    def _compute_correlation_matrix(
        self,
        data: pd.DataFrame,
        metadata: dict,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute correlation matrix using pairwise deletion.

        Args:
            data: DataFrame to analyze
            metadata: SDV metadata dictionary

        Returns:
            Tuple of (correlation_matrix, sample_size_matrix)
        """
        columns = list(data.columns)
        corr_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
        sample_matrix = pd.DataFrame(index=columns, columns=columns, dtype=int)

        sdtypes = {col: metadata['columns'][col]['sdtype'] for col in columns}

        for i, col1 in enumerate(columns):
            for col2 in columns[i:]:
                if col1 == col2:
                    corr_matrix.loc[col1, col2] = 1.0
                    sample_matrix.loc[col1, col2] = len(data)
                    continue

                sdtype1 = sdtypes[col1].lower()
                sdtype2 = sdtypes[col2].lower()

                # Skip datetime columns
                if sdtype1 == 'datetime' or sdtype2 == 'datetime':
                    corr_matrix.loc[col1, col2] = np.nan
                    corr_matrix.loc[col2, col1] = np.nan
                    sample_matrix.loc[col1, col2] = 0
                    sample_matrix.loc[col2, col1] = 0
                    continue

                try:
                    # Pairwise deletion: only rows where BOTH columns are not null
                    valid_mask = data[[col1, col2]].notna().all(axis=1)
                    n_valid = valid_mask.sum()

                    if n_valid < 2:  # Need at least 2 samples
                        corr_matrix.loc[col1, col2] = np.nan
                        corr_matrix.loc[col2, col1] = np.nan
                        sample_matrix.loc[col1, col2] = n_valid
                        sample_matrix.loc[col2, col1] = n_valid
                        continue

                    data1 = data.loc[valid_mask, col1]
                    data2 = data.loc[valid_mask, col2]

                    # Both numerical - Spearman
                    if sdtype1 == 'numerical' and sdtype2 == 'numerical':
                        corr, _ = spearmanr(data1, data2)
                        corr_matrix.loc[col1, col2] = abs(corr)
                        corr_matrix.loc[col2, col1] = abs(corr)

                    # Both categorical - Cramér's V
                    elif sdtype1 in ['categorical', 'boolean'] and sdtype2 in ['categorical', 'boolean']:
                        v = self._cramers_v(data1, data2)
                        corr_matrix.loc[col1, col2] = v
                        corr_matrix.loc[col2, col1] = v

                    # Mixed - Correlation ratio
                    else:
                        if sdtype1 in ['categorical', 'boolean']:
                            eta = self._correlation_ratio(data1, data2)
                        else:
                            eta = self._correlation_ratio(data2, data1)

                        corr_matrix.loc[col1, col2] = eta
                        corr_matrix.loc[col2, col1] = eta

                    sample_matrix.loc[col1, col2] = n_valid
                    sample_matrix.loc[col2, col1] = n_valid

                except Exception:
                    corr_matrix.loc[col1, col2] = np.nan
                    corr_matrix.loc[col2, col1] = np.nan
                    sample_matrix.loc[col1, col2] = 0
                    sample_matrix.loc[col2, col1] = 0

        return corr_matrix, sample_matrix

    def _build_quality_pair_matrix(self, report: QualityReport) -> pd.DataFrame:
        """Extract pairwise quality scores from QualityReport."""
        details = report.get_details('Column Pair Trends')

        all_cols = sorted(set(details['Column 1'].unique()) | set(details['Column 2'].unique()))
        matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)

        for _, row in details.iterrows():
            col1, col2 = row['Column 1'], row['Column 2']
            score = row['Score']
            matrix.loc[col1, col2] = score
            matrix.loc[col2, col1] = score

        for col in all_cols:
            matrix.loc[col, col] = 1.0

        return matrix

    def evaluate(self,
                 original: pd.DataFrame,
                 synthetic: pd.DataFrame,
                 metadata: SingleTableMetadata,
                 encoding_config: dict = None) -> Dict[str, Any]:
        """
        Evaluate using SDMetrics QualityReport.

        Args:
            original: Original/reference dataset (decoded)
            synthetic: Synthetic dataset (decoded)
            metadata: SDV metadata object
            encoding_config: Optional encoding configuration (unused)

        Returns:
            Dictionary with:
            - status: "success" or "error"
            - score: Overall quality score (0-1)
            - property_scores: Column Shapes and Column Pair Trends scores
            - matrices: Real/synthetic correlations and quality scores
            - diagnostics: Pairwise diagnostic information
            - null_values: Null count information
            - column_shapes_details: Per-column distribution scores
            - execution_time: Time taken in seconds
        """
        start_time = time.time()

        try:
            metadata_dict = metadata.to_dict()

            # Generate SDMetrics QualityReport (Column Pair Trends uses Spearman)
            report = QualityReport()
            report._properties['Column Pair Trends'] = SpearmanColumnPairTrends()
            report.generate(original, synthetic, metadata_dict, verbose=False)

            # Compute correlation matrices
            real_matrix, real_samples = self._compute_correlation_matrix(original, metadata_dict)
            synth_matrix, synth_samples = self._compute_correlation_matrix(synthetic, metadata_dict)
            quality_matrix = self._build_quality_pair_matrix(report)

            # Get property scores
            column_shapes_score = report._properties["Column Shapes"]._compute_average()
            column_pair_trends_score = report._properties["Column Pair Trends"]._compute_average()
            overall_score = report.get_score()

            # Get column shapes details
            column_shapes_details = report.get_details('Column Shapes').to_dict(orient='records')

            # Build diagnostics for each pair
            cols = sorted(real_matrix.columns)
            diagnostics = {}

            for i, col1 in enumerate(cols):
                for col2 in cols[i+1:]:
                    pair_key = f"{col1}×{col2}"

                    real_corr = float(real_matrix.loc[col1, col2]) if not pd.isna(real_matrix.loc[col1, col2]) else None
                    n_samples = int(real_samples.loc[col1, col2])
                    synth_corr = float(synth_matrix.loc[col1, col2]) if not pd.isna(synth_matrix.loc[col1, col2]) else None
                    quality_score = float(quality_matrix.loc[col1, col2]) if not pd.isna(quality_matrix.loc[col1, col2]) else None

                    if all(x is not None for x in [real_corr, synth_corr, quality_score]):
                        diagnostics[pair_key] = {
                            "column_1": col1,
                            "column_2": col2,
                            "real_correlation": real_corr,
                            "real_n_samples": n_samples,
                            "synthetic_correlation": synth_corr,
                            "quality_score": quality_score,
                        }

            return {
                "status": "success",
                "score": float(overall_score),
                "property_scores": {
                    "column_shapes": float(column_shapes_score),
                    "column_pair_trends": float(column_pair_trends_score),
                },
                "matrices": {
                    "real_correlations": real_matrix.to_dict(),
                    "real_sample_sizes": real_samples.to_dict(),
                    "synthetic_correlations": synth_matrix.to_dict(),
                    "quality_scores": quality_matrix.to_dict(),
                },
                "diagnostics": diagnostics,
                "null_values": {
                    "real_data": original.isnull().sum().to_dict(),
                    "synthetic_data": synthetic.isnull().sum().to_dict(),
                },
                "column_shapes_details": column_shapes_details,
                "parameters": {"max_display_cols": self.max_display_cols, **self.parameters},
                "execution_time": time.time() - start_time,
            }

        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "parameters": {"max_display_cols": self.max_display_cols, **self.parameters},
                "execution_time": time.time() - start_time,
            }


# ===========================================================================
# Dispatch
# ===========================================================================


def evaluate_statistical_metrics(original: pd.DataFrame,
                                synthetic: pd.DataFrame,
                                metrics_config: list,
                                experiment_name: str,
                                metadata: SingleTableMetadata,
                                reference_data_decoded: pd.DataFrame = None,
                                synthetic_data_decoded: pd.DataFrame = None,
                                reference_data_encoded: pd.DataFrame = None,
                                synthetic_data_encoded: pd.DataFrame = None,
                                encoded_metrics: set = None,
                                decoded_metrics: set = None,
                                encoding_config: dict = None) -> Dict[str, Any]:
    """
    Evaluate configured statistical metrics with data format routing

    Args:
        original: Original dataset (legacy parameter, may be encoded or decoded)
        synthetic: Synthetic dataset (legacy parameter, may be encoded or decoded)
        metrics_config: List of metric configurations
        experiment_name: Experiment identifier
        metadata: SDV metadata
        reference_data_decoded: Decoded reference data (for SDV metrics)
        synthetic_data_decoded: Decoded synthetic data (for SDV metrics)
        reference_data_encoded: Encoded reference data (for synthcity metrics)
        synthetic_data_encoded: Encoded synthetic data (for synthcity metrics)
        encoded_metrics: Set of metric names that need encoded data
        decoded_metrics: Set of metric names that need decoded data
        encoding_config: Encoding configuration dict (from load_encoding_config) used to
                        determine which columns are numeric in encoded data

    Returns:
        Complete statistical metrics results
    """

    print(f"Evaluating statistical metrics for experiment: {experiment_name}")
    print(f"Original shape: {original.shape}, Synthetic shape: {synthetic.shape}")

    results = {
        "metadata": {
            "experiment_name": experiment_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "original_shape": list(original.shape),
            "synthetic_shape": list(synthetic.shape),
            "evaluation_type": "statistical_metrics"
        },
        "metrics": {}
    }

    # Run each configured metric
    metric_scores = []

    for metric_config in metrics_config:
        metric_name = metric_config.get("name")
        parameters = metric_config.get("parameters", {})

        # Route to correct data format
        if encoded_metrics and metric_name in encoded_metrics:
            # Use encoded data for synthcity metrics
            if reference_data_encoded is None or synthetic_data_encoded is None:
                print(f"⚠️  Metric {metric_name} needs encoded data but not available, using default")
                ref_data = original
                syn_data = synthetic
            else:
                print(f"📊 Routing {metric_name} to ENCODED data")
                ref_data = reference_data_encoded
                syn_data = synthetic_data_encoded
        elif decoded_metrics and metric_name in decoded_metrics:
            # Use decoded data for SDV metrics
            if reference_data_decoded is None or synthetic_data_decoded is None:
                print(f"⚠️  Metric {metric_name} needs decoded data but not available, using default")
                ref_data = original
                syn_data = synthetic
            else:
                print(f"📊 Routing {metric_name} to DECODED data")
                ref_data = reference_data_decoded
                syn_data = synthetic_data_decoded
        else:
            # Use default data (backward compatibility)
            ref_data = original
            syn_data = synthetic

        print(f"Running {metric_name} metric...")

        try:
            evaluator = get_metric_evaluator(metric_name, parameters)
            metric_result = evaluator.evaluate(ref_data, syn_data, metadata, encoding_config=encoding_config)
            results["metrics"][metric_name] = metric_result

            # Collect scores for overall calculation
            if metric_result["status"] == "success":
                match metric_name:
                    case "alpha_precision" | "prdc_score" | "new_row_synthesis" | "ks_complement" | "tv_complement":
                        # Individual scores handled in report - no aggregation
                        pass
                    case _:
                        # Future metrics that might need aggregation
                        pass

        except Exception as e:
            results["metrics"][metric_name] = {
                "status": "error",
                "error_message": str(e),
                "parameters": parameters
            }

    print("Individual statistical metrics completed - see detailed scores")

    # Ensure all results are JSON serializable
    results = ensure_json_serializable(results)

    return results


def get_metric_evaluator(metric_name: str, parameters: Dict[str, Any]):
    """Factory function to create metric evaluators.

    Ported incrementally — a `case` is added here together with its metric
    class. Unported names raise (the port asserts explicit coverage rather
    than silently degrading), so a `statistical_similarity` run configured
    with a metric that is not yet ported fails loudly on that entry.
    """

    match metric_name:
        case "ks_complement":
            return KSComplementMetric(**parameters)
        case "tv_complement":
            return TVComplementMetric(**parameters)
        case "boundary_adherence":
            return BoundaryAdherenceMetric(**parameters)
        case "category_adherence":
            return CategoryAdherenceMetric(**parameters)
        case "table_structure":
            return TableStructureMetric(**parameters)
        case "semantic_structure":
            return SemanticStructureMetric(**parameters)
        case "alpha_precision":
            return AlphaPrecisionMetric(**parameters)
        case "prdc_score":
            return PRDCScoreMetric(**parameters)
        case "jensenshannon_synthcity":
            return JensenShannonSynthcityMetric(**parameters)
        case "jensenshannon_syndat":
            return JensenShannonSyndatMetric(**parameters)
        case "jensenshannon_nannyml":
            return JensenShannonNannyMLMetric(**parameters)
        case "maximum_mean_discrepancy":
            return MaximumMeanDiscrepancyMetric(**parameters)
        case "new_row_synthesis":
            return NewRowSynthesisMetric(**parameters)
        case "wasserstein_distance":
            return WassersteinDistanceMetric(**parameters)
        case "sdmetrics_quality":
            return SDMetricsQualityMetric(**parameters)
        case _:
            raise ValueError(f"Unknown metric: {metric_name}")


# ===========================================================================
# Report
# ===========================================================================


def generate_statistical_report(results: Dict[str, Any]) -> str:
    """Generate a human-readable statistical metrics report"""

    report = f"""
Statistical Metrics Evaluation Report
=======================================

Experiment: {results['metadata']['experiment_name']}
Timestamp: {results['metadata']['evaluation_timestamp']}
Dataset Shapes: Original {tuple(results['metadata']['original_shape'])}, Synthetic {tuple(results['metadata']['synthetic_shape'])}

Note: Individual metric scores provided

Metrics Results
---------------
"""
    metrics = results.get("metrics", {})

    # Alpha Precision results
    if "alpha_precision" in metrics:
        alpha_result = metrics["alpha_precision"]
        if alpha_result["status"] == "success":
            scores = alpha_result['scores']
            report += f"""Alpha Precision Results:
  Parameters: {alpha_result['parameters'] if alpha_result['parameters'] else 'none'}
  Execution time: {alpha_result['execution_time']:.2f}s

  Individual Scores:
    Optimally-Corrected (OC) Variant:
      → Delta Precision Alpha: {scores['delta_precision_alpha_OC']:.3f}
      → Delta Coverage Beta:   {scores['delta_coverage_beta_OC']:.3f}
      → Authenticity:          {scores['authenticity_OC']:.3f}

    Naive Variant:
      → Delta Precision Alpha: {scores['delta_precision_alpha_naive']:.3f}
      → Delta Coverage Beta:   {scores['delta_coverage_beta_naive']:.3f}
      → Authenticity:          {scores['authenticity_naive']:.3f}
"""
        else:
            report += f"""Alpha Precision: ERROR
  Error: {alpha_result.get('error_message', 'Unknown error')}
"""

    # PRDC Score results
    if "prdc_score" in metrics:
        prdc_result = metrics["prdc_score"]
        if prdc_result["status"] == "success":
            report += f"""PRDC Score Results:
  Parameters: {prdc_result['parameters'] if prdc_result['parameters'] else 'default settings'}
  Execution time: {prdc_result['execution_time']:.2f}s

  Individual Scores:
  Precision: {prdc_result['precision']:.3f}
  Recall: {prdc_result['recall']:.3f}
  Density: {prdc_result['density']:.3f}
  Coverage: {prdc_result['coverage']:.3f}
"""

        else:
            report += f"""PRDC Score: ERROR
  Error: {prdc_result.get('error_message', 'Unknown error')}
"""

    # Wasserstein Distance results
    if "wasserstein_distance" in metrics:
        wd_result = metrics["wasserstein_distance"]
        if wd_result["status"] == "success":
            report += f"""Wasserstein Distance Results:
  Parameters: {wd_result['parameters'] if wd_result['parameters'] else 'default settings'}
  Execution time: {wd_result['execution_time']:.2f}s

  Distance Score:
  → Joint Distance: {wd_result['joint_distance']:.6f}

  Note: Lower values indicate more similar distributions (0 = identical)
"""
        else:
            report += f"""Wasserstein Distance: ERROR
  Error: {wd_result.get('error_message', 'Unknown error')}
"""

    # Maximum Mean Discrepancy results
    if "maximum_mean_discrepancy" in metrics:
        mmd_result = metrics["maximum_mean_discrepancy"]
        if mmd_result["status"] == "success":
            report += f"""Maximum Mean Discrepancy Results:
  Parameters: kernel={mmd_result['kernel']}
  Execution time: {mmd_result['execution_time']:.2f}s

  Distance Score:
  → Joint Distance: {mmd_result['joint_distance']:.6f}

  Note: Lower values indicate more similar distributions (0 = identical)
"""
        else:
            report += f"""Maximum Mean Discrepancy: ERROR
  Error: {mmd_result.get('error_message', 'Unknown error')}
"""

    # Jensen-Shannon Distance (Synthcity) results
    if "jensenshannon_synthcity" in metrics:
        jsd_sc_result = metrics["jensenshannon_synthcity"]
        if jsd_sc_result["status"] == "success":
            report += f"""Jensen-Shannon Distance (Synthcity) Results:
  Parameters: normalize={jsd_sc_result.get('normalize', True)}, n_histogram_bins={jsd_sc_result.get('n_histogram_bins', 10)}
  Execution time: {jsd_sc_result['execution_time']:.2f}s

  Distance Score:
  → Joint Distance: {jsd_sc_result['distance_score']:.6f}

  Note: Lower values indicate more similar distributions (0 = identical)
"""
        else:
            report += f"""Jensen-Shannon Distance (Synthcity): ERROR
  Error: {jsd_sc_result.get('error_message', 'Unknown error')}
"""

    # Jensen-Shannon Distance (SYNDAT) results
    if "jensenshannon_syndat" in metrics:
        jsd_sd_result = metrics["jensenshannon_syndat"]
        if jsd_sd_result["status"] == "success":
            report += f"""Jensen-Shannon Distance (SYNDAT) Results:
  Parameters: n_unique_threshold={jsd_sd_result.get('n_unique_threshold', 10)}
  Execution time: {jsd_sd_result['execution_time']:.2f}s

  Distance Score:
  → Joint Distance: {jsd_sd_result['distance_score']:.6f}

  Note: Lower values indicate more similar distributions (0 = identical)
"""
        else:
            report += f"""Jensen-Shannon Distance (SYNDAT): ERROR
  Error: {jsd_sd_result.get('error_message', 'Unknown error')}
"""

    # Jensen-Shannon Distance (NannyML) results
    if "jensenshannon_nannyml" in metrics:
        jsd_nm_result = metrics["jensenshannon_nannyml"]
        if jsd_nm_result["status"] == "success":
            report += f"""Jensen-Shannon Distance (NannyML) Results:
  Parameters: {jsd_nm_result['parameters'] if jsd_nm_result['parameters'] else 'default settings'}
  Execution time: {jsd_nm_result['execution_time']:.2f}s

  Distance Score:
  → Joint Distance: {jsd_nm_result['distance_score']:.6f}
  → Columns Evaluated: {jsd_nm_result['n_columns_evaluated']}

  Note: Lower values indicate more similar distributions (0 = identical)
"""
        else:
            report += f"""Jensen-Shannon Distance (NannyML): ERROR
  Error: {jsd_nm_result.get('error_message', 'Unknown error')}
"""

    # NewRowSynthesis results
    if "new_row_synthesis" in metrics:
        nrs_result = metrics["new_row_synthesis"]
        if nrs_result["status"] == "success":
            params_info = nrs_result["parameters"]
            tolerance = params_info.get("numerical_match_tolerance", 0.01)
            sample_size = params_info.get("synthetic_sample_size", "all rows")

            report += f"""NewRowSynthesis Results:
  Parameters: tolerance={tolerance}, sample_size={sample_size}
  Execution time: {nrs_result['execution_time']:.2f}s

  Synthesis Quality:
  → New Row Score:     {nrs_result['score']:.3f}
  → New Rows:          {nrs_result['num_new_rows']:,}
  → Matched Rows:      {nrs_result['num_matched_rows']:,}
"""

    # KSComplement results
    if "ks_complement" in metrics:
        ks_result = metrics["ks_complement"]
        if ks_result["status"] == "success":
            params_info = ks_result["parameters"]
            target_cols = params_info.get("target_columns", "all numerical/datetime")

            if ks_result.get("message"):
                # Handle case where no compatible columns found
                report += f"""KSComplement Results:
  Parameters: target_columns={target_cols}
  Execution time: {ks_result['execution_time']:.2f}s
  Status: {ks_result['message']}
"""
            else:
                # Format aggregate score (handle None for failed columns)
                agg_score_str = f"{ks_result['aggregate_score']:.3f}" if ks_result['aggregate_score'] is not None else "N/A (all columns failed)"

                report += f"""KSComplement Results:
  Parameters: target_columns={target_cols}
  Execution time: {ks_result['execution_time']:.2f}s

  Distribution Similarity:
  → Aggregate Score:   {agg_score_str}
  → Columns Evaluated: {len(ks_result['compatible_columns'])}
  → Successful:        {ks_result.get('successful_columns', 'unknown')}
  → Failed:            {len(ks_result.get('failed_columns', []))}

  Individual Column Scores:"""
                for col, score in ks_result['column_scores'].items():
                    score_str = f"{score:.3f}" if score is not None else "FAILED"
                    report += f"""
    → {col}: {score_str}"""
                report += "\n"
        else:
            report += f"""KSComplement: ERROR
  Error: {ks_result.get('error_message', 'Unknown error')}
"""

    # TVComplement results
    if "tv_complement" in metrics:
        tv_result = metrics["tv_complement"]
        if tv_result["status"] == "success":
            params_info = tv_result["parameters"]
            target_cols = params_info.get("target_columns", "all categorical/boolean")

            if tv_result.get("message"):
                # Handle case where no compatible columns found
                report += f"""TVComplement Results:
  Parameters: target_columns={target_cols}
  Execution time: {tv_result['execution_time']:.2f}s
  Status: {tv_result['message']}
"""
            else:
                # Format aggregate score (handle None for failed columns)
                agg_score_str = f"{tv_result['aggregate_score']:.3f}" if tv_result['aggregate_score'] is not None else "N/A (all columns failed)"

                report += f"""TVComplement Results:
  Parameters: target_columns={target_cols}
  Execution time: {tv_result['execution_time']:.2f}s

  Categorical Distribution Similarity:
  → Aggregate Score:   {agg_score_str}
  → Columns Evaluated: {len(tv_result['compatible_columns'])}
  → Successful:        {tv_result.get('successful_columns', 'unknown')}
  → Failed:            {len(tv_result.get('failed_columns', []))}

  Individual Column Scores:"""
                for col, score in tv_result['column_scores'].items():
                    score_str = f"{score:.3f}" if score is not None else "FAILED"
                    report += f"""
    → {col}: {score_str}"""
                report += "\n"
        else:
            report += f"""TVComplement: ERROR
  Error: {tv_result.get('error_message', 'Unknown error')}
"""

    # TableStructure results
    if "table_structure" in metrics:
        ts_result = metrics["table_structure"]
        if ts_result["status"] == "success":
            report += f"""TableStructure Results:
  Parameters: {ts_result['parameters'] if ts_result['parameters'] else 'none'}
  Execution time: {ts_result['execution_time']:.2f}s

  Structure Similarity:
  → Table Structure Score: {ts_result['score']:.3f}
"""
            # Add summary if available (backward compatible)
            if 'summary' in ts_result:
                summary = ts_result['summary']
                report += f"""
  Summary:
  → Total columns (real/synthetic): {summary['total_real_columns']}/{summary['total_synthetic_columns']}
  → Matching columns (name + dtype): {summary['matching_columns']}
  → Dtype mismatches: {summary['dtype_mismatches']}
  → Missing in synthetic: {summary['missing_in_synthetic']}
  → Only in synthetic: {summary['only_in_synthetic']}
"""

            # Add column-by-column comparison table if available
            if 'comparison_table' in ts_result and ts_result['comparison_table']:
                report += f"""
  Column-by-Column Comparison:
  {'─' * 80}
  {"Column":<25} {"Real dtype":<20} {"Synthetic dtype":<20} {"Status":<15}
  {'─' * 80}
"""
                # Status symbols for better readability
                status_symbols = {
                    "match": "✓ Match",
                    "dtype_mismatch": "⚠ Dtype mismatch",
                    "missing_in_synthetic": "✗ Missing in synth",
                    "only_in_synthetic": "⚠ Only in synth"
                }

                for item in ts_result['comparison_table']:
                    col = item['column'][:24]  # Truncate long column names
                    real_dtype = item['real_dtype'] if item['real_dtype'] else '-'
                    synth_dtype = item['synthetic_dtype'] if item['synthetic_dtype'] else '-'
                    status = status_symbols.get(item['status'], item['status'])

                    report += f"  {col:<25} {real_dtype:<20} {synth_dtype:<20} {status:<15}\n"

                report += f"  {'─' * 80}\n"
        else:
            report += f"""TableStructure: ERROR
  Error: {ts_result.get('error_message', 'Unknown error')}
"""

    # SemanticStructure results
    if "semantic_structure" in metrics:
        ss_result = metrics["semantic_structure"]
        if ss_result["status"] == "success":
            report += f"""SemanticStructure Results:
  Parameters: {ss_result['parameters'] if ss_result['parameters'] else 'none'}
  Execution time: {ss_result['execution_time']:.2f}s

  Semantic Structure Similarity:
  → Semantic Structure Score: {ss_result['score']:.3f}
"""
            # Add summary if available
            if 'summary' in ss_result:
                summary = ss_result['summary']
                report += f"""
  Summary:
  → Total columns (real/synthetic): {summary['total_real_columns']}/{summary['total_synthetic_columns']}
  → Matching columns (name + sdtype): {summary['matching_columns']}
  → Sdtype mismatches: {summary['sdtype_mismatches']}
  → Missing in synthetic: {summary['missing_in_synthetic']}
  → Only in synthetic: {summary['only_in_synthetic']}
"""

            # Add column-by-column comparison table if available
            if 'comparison_table' in ss_result and ss_result['comparison_table']:
                report += f"""
  Column-by-Column Comparison (Semantic Types):
  {'─' * 80}
  {"Column":<25} {"Real sdtype":<20} {"Synthetic sdtype":<20} {"Status":<15}
  {'─' * 80}
"""
                # Status symbols for better readability
                status_symbols = {
                    "match": "✓ Match",
                    "sdtype_mismatch": "⚠ Sdtype mismatch",
                    "missing_in_synthetic": "✗ Missing in synth",
                    "only_in_synthetic": "⚠ Only in synth"
                }

                for item in ss_result['comparison_table']:
                    col = item['column'][:24]  # Truncate long column names
                    real_sdtype = item['real_sdtype'] if item['real_sdtype'] else '-'
                    synth_sdtype = item['synthetic_sdtype'] if item['synthetic_sdtype'] else '-'
                    status = status_symbols.get(item['status'], item['status'])

                    report += f"  {col:<25} {real_sdtype:<20} {synth_sdtype:<20} {status:<15}\n"

                report += f"  {'─' * 80}\n"
        else:
            report += f"""SemanticStructure: ERROR
  Error: {ss_result.get('error_message', 'Unknown error')}
"""

    # BoundaryAdherence results
    if "boundary_adherence" in metrics:
        ba_result = metrics["boundary_adherence"]
        if ba_result["status"] == "success":
            params_info = ba_result["parameters"]
            target_cols = params_info.get("target_columns", "all numerical/datetime")

            if ba_result.get("message"):
                # Handle case where no compatible columns found
                report += f"""BoundaryAdherence Results:
  Parameters: target_columns={target_cols}
  Execution time: {ba_result['execution_time']:.2f}s
  Status: {ba_result['message']}
"""
            else:
                # Format aggregate score (handle None for failed columns)
                agg_score_str = f"{ba_result['aggregate_score']:.3f}" if ba_result['aggregate_score'] is not None else "N/A (all columns failed)"

                report += f"""BoundaryAdherence Results:
  Parameters: target_columns={target_cols}
  Execution time: {ba_result['execution_time']:.2f}s

  Boundary Compliance:
  → Aggregate Score:   {agg_score_str}
  → Columns Evaluated: {len(ba_result['compatible_columns'])}
  → Successful:        {ba_result.get('successful_columns', 'unknown')}
  → Failed:            {len(ba_result.get('failed_columns', []))}

  Individual Column Scores:"""
                for col, score in ba_result['column_scores'].items():
                    score_str = f"{score:.3f}" if score is not None else "FAILED"
                    report += f"""
    → {col}: {score_str}"""
                report += "\n"
        else:
            report += f"""BoundaryAdherence: ERROR
  Error: {ba_result.get('error_message', 'Unknown error')}
"""

    # CategoryAdherence results
    if "category_adherence" in metrics:
        ca_result = metrics["category_adherence"]
        if ca_result["status"] == "success":
            params_info = ca_result["parameters"]
            target_cols = params_info.get("target_columns", "all categorical/boolean")

            if ca_result.get("message"):
                # Handle case where no compatible columns found
                report += f"""CategoryAdherence Results:
  Parameters: target_columns={target_cols}
  Execution time: {ca_result['execution_time']:.2f}s
  Status: {ca_result['message']}
"""
            else:
                # Format aggregate score (handle None for failed columns)
                agg_score_str = f"{ca_result['aggregate_score']:.3f}" if ca_result['aggregate_score'] is not None else "N/A (all columns failed)"

                report += f"""CategoryAdherence Results:
  Parameters: target_columns={target_cols}
  Execution time: {ca_result['execution_time']:.2f}s

  Category Compliance:
  → Aggregate Score:   {agg_score_str}
  → Columns Evaluated: {len(ca_result['compatible_columns'])}
  → Successful:        {ca_result.get('successful_columns', 'unknown')}
  → Failed:            {len(ca_result.get('failed_columns', []))}

  Individual Column Scores:"""
                for col, score in ca_result['column_scores'].items():
                    score_str = f"{score:.3f}" if score is not None else "FAILED"
                    report += f"""
    → {col}: {score_str}"""
                report += "\n"
        else:
            report += f"""CategoryAdherence: ERROR
  Error: {ca_result.get('error_message', 'Unknown error')}
"""

    # SDMetrics Quality Report results
    if "sdmetrics_quality" in metrics:
        sq_result = metrics["sdmetrics_quality"]
        if sq_result["status"] == "success":
            params_info = sq_result["parameters"]
            max_display = params_info.get("max_display_cols", 10)

            property_scores = sq_result.get("property_scores", {})
            overall_score = sq_result.get("score", 0.0)

            report += f"""SDMetrics Quality Report:
  Parameters: max_display_cols={max_display}, pairwise deletion for real data
  Execution time: {sq_result['execution_time']:.2f}s

  Quality Scores:
  → Overall Quality:       {overall_score:.2%}
  → Column Shapes:         {property_scores.get('column_shapes', 0.0):.2%}
  → Column Pair Trends:    {property_scores.get('column_pair_trends', 0.0):.2%}
"""

            # Add null value summary
            null_values = sq_result.get("null_values", {})
            if null_values:
                real_nulls = null_values.get("real_data", {})
                synth_nulls = null_values.get("synthetic_data", {})

                total_real = sum(real_nulls.values()) if real_nulls else 0
                total_synth = sum(synth_nulls.values()) if synth_nulls else 0

                report += f"""
  Null Values:
  → Real data:       {total_real:,} null values
  → Synthetic data:  {total_synth:,} null values
"""

            # Add diagnostics for worst pairs
            diagnostics = sq_result.get("diagnostics", {})
            if diagnostics:
                pairs_list = [(k, v) for k, v in diagnostics.items()]
                pairs_sorted = sorted(pairs_list, key=lambda x: x[1].get("quality_score", 1.0))

                if len(pairs_sorted) >= 5:
                    report += f"""
  Lowest Quality Pairs (Bottom 5):"""
                    for pair_key, pair_data in pairs_sorted[:5]:
                        quality = pair_data.get("quality_score", 0.0)
                        real_corr = pair_data.get("real_correlation", 0.0)
                        synth_corr = pair_data.get("synthetic_correlation", 0.0)
                        report += f"""
    → {pair_key}: Quality={quality:.3f}, Real={real_corr:.3f}, Synth={synth_corr:.3f}"""
                    report += "\n"
        else:
            report += f"""SDMetrics Quality Report: ERROR
  Error: {sq_result.get('error_message', 'Unknown error')}
"""

    # Individual metric insights
    insights = []

    if "alpha_precision" in metrics and metrics["alpha_precision"]["status"] == "success":
        auth_oc = metrics["alpha_precision"]["scores"]["authenticity_OC"]
        if auth_oc >= 0.8:
            insights.append("Strong authenticity (OC)")
        elif auth_oc >= 0.6:
            insights.append("Moderate authenticity (OC)")
        else:
            insights.append("Low authenticity (OC)")

    if "prdc_score" in metrics and metrics["prdc_score"]["status"] == "success":
        prdc = metrics["prdc_score"]
        precision = prdc["precision"]
        recall = prdc["recall"]
        coverage = prdc["coverage"]

        prdc_avg = (precision + recall + coverage) / 3  # Density often lower, so exclude
        if prdc_avg >= 0.8:
            insights.append("Strong PRDC performance")
        elif prdc_avg >= 0.6:
            insights.append("Moderate PRDC performance")
        else:
            insights.append("Low PRDC performance")

    if "wasserstein_distance" in metrics and metrics["wasserstein_distance"]["status"] == "success":
        wd_distance = metrics["wasserstein_distance"]["joint_distance"]
        if wd_distance < 0.01:
            insights.append("Excellent distributional similarity (Wasserstein)")
        elif wd_distance < 0.05:
            insights.append("Good distributional similarity (Wasserstein)")
        elif wd_distance < 0.1:
            insights.append("Moderate distributional similarity (Wasserstein)")
        else:
            insights.append("Poor distributional similarity (Wasserstein)")

    if "maximum_mean_discrepancy" in metrics and metrics["maximum_mean_discrepancy"]["status"] == "success":
        mmd_distance = metrics["maximum_mean_discrepancy"]["joint_distance"]
        if mmd_distance < 0.001:
            insights.append("Excellent distributional similarity (MMD)")
        elif mmd_distance < 0.01:
            insights.append("Good distributional similarity (MMD)")
        elif mmd_distance < 0.1:
            insights.append("Moderate distributional similarity (MMD)")
        else:
            insights.append("Poor distributional similarity (MMD)")

    if "jensenshannon_synthcity" in metrics and metrics["jensenshannon_synthcity"]["status"] == "success":
        jsd_distance = metrics["jensenshannon_synthcity"]["distance_score"]
        if jsd_distance < 0.01:
            insights.append("Excellent distributional similarity (JS-Synthcity)")
        elif jsd_distance < 0.05:
            insights.append("Good distributional similarity (JS-Synthcity)")
        elif jsd_distance < 0.1:
            insights.append("Moderate distributional similarity (JS-Synthcity)")
        else:
            insights.append("Poor distributional similarity (JS-Synthcity)")

    if "jensenshannon_syndat" in metrics and metrics["jensenshannon_syndat"]["status"] == "success":
        jsd_distance = metrics["jensenshannon_syndat"]["distance_score"]
        if jsd_distance < 0.01:
            insights.append("Excellent distributional similarity (JS-SYNDAT)")
        elif jsd_distance < 0.05:
            insights.append("Good distributional similarity (JS-SYNDAT)")
        elif jsd_distance < 0.1:
            insights.append("Moderate distributional similarity (JS-SYNDAT)")
        else:
            insights.append("Poor distributional similarity (JS-SYNDAT)")

    if "jensenshannon_nannyml" in metrics and metrics["jensenshannon_nannyml"]["status"] == "success":
        jsd_distance = metrics["jensenshannon_nannyml"]["distance_score"]
        if jsd_distance < 0.01:
            insights.append("Excellent distributional similarity (JS-NannyML)")
        elif jsd_distance < 0.05:
            insights.append("Good distributional similarity (JS-NannyML)")
        elif jsd_distance < 0.1:
            insights.append("Moderate distributional similarity (JS-NannyML)")
        else:
            insights.append("Poor distributional similarity (JS-NannyML)")

    if "new_row_synthesis" in metrics and metrics["new_row_synthesis"]["status"] == "success":
        nrs_score = metrics["new_row_synthesis"]["score"]
        if nrs_score >= 0.9:
            insights.append("Excellent synthesis novelty")
        elif nrs_score >= 0.7:
            insights.append("Good synthesis novelty")
        else:
            insights.append("Low synthesis novelty")

    if "ks_complement" in metrics and metrics["ks_complement"]["status"] == "success":
        ks_score = metrics["ks_complement"]["aggregate_score"]
        if ks_score is not None:  # Only evaluate if we have compatible columns
            if ks_score >= 0.9:
                insights.append("Excellent distribution similarity")
            elif ks_score >= 0.7:
                insights.append("Good distribution similarity")
            else:
                insights.append("Poor distribution similarity")
        # If ks_score is None, we simply don't add any insight (no numerical columns to evaluate)

    if "tv_complement" in metrics and metrics["tv_complement"]["status"] == "success":
        tv_score = metrics["tv_complement"]["aggregate_score"]
        if tv_score is not None:  # Only evaluate if we have compatible columns
            if tv_score >= 0.9:
                insights.append("Excellent categorical similarity")
            elif tv_score >= 0.7:
                insights.append("Good categorical similarity")
            else:
                insights.append("Poor categorical similarity")
        # If tv_score is None, we simply don't add any insight (no categorical columns to evaluate)

    if "table_structure" in metrics and metrics["table_structure"]["status"] == "success":
        ts_score = metrics["table_structure"]["score"]
        if ts_score >= 0.95:
            insights.append("Perfect table structure match")
        elif ts_score >= 0.8:
            insights.append("Good table structure match")
        else:
            insights.append("Poor table structure match")

    if "semantic_structure" in metrics and metrics["semantic_structure"]["status"] == "success":
        ss_score = metrics["semantic_structure"]["score"]
        if ss_score >= 0.95:
            insights.append("Perfect semantic structure match")
        elif ss_score >= 0.8:
            insights.append("Good semantic structure match")
        else:
            insights.append("Poor semantic structure match")

    if "boundary_adherence" in metrics and metrics["boundary_adherence"]["status"] == "success":
        ba_score = metrics["boundary_adherence"]["aggregate_score"]
        if ba_score is not None:  # Only evaluate if we have compatible columns
            if ba_score >= 0.95:
                insights.append("Excellent boundary adherence")
            elif ba_score >= 0.8:
                insights.append("Good boundary adherence")
            else:
                insights.append("Poor boundary adherence")
        # If ba_score is None, we simply don't add any insight (no numerical/datetime columns to evaluate)

    if "category_adherence" in metrics and metrics["category_adherence"]["status"] == "success":
        ca_score = metrics["category_adherence"]["aggregate_score"]
        if ca_score is not None:  # Only evaluate if we have compatible columns
            if ca_score >= 0.95:
                insights.append("Excellent category adherence")
            elif ca_score >= 0.8:
                insights.append("Good category adherence")
            else:
                insights.append("Poor category adherence")
        # If ca_score is None, we simply don't add any insight (no categorical/boolean columns to evaluate)

    if "sdmetrics_quality" in metrics and metrics["sdmetrics_quality"]["status"] == "success":
        overall_quality = metrics["sdmetrics_quality"]["score"]
        if overall_quality >= 0.8:
            insights.append("Excellent overall quality (SDMetrics)")
        elif overall_quality >= 0.6:
            insights.append("Good overall quality (SDMetrics)")
        elif overall_quality >= 0.4:
            insights.append("Fair overall quality (SDMetrics)")
        else:
            insights.append("Poor overall quality (SDMetrics)")

    assessment = ", ".join(insights) if insights else "No successful metrics"

    report += f"""
Assessment: {assessment}
"""

    return report
