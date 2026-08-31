"""
Privacy metrics — k-anonymisation.
=================================

Ported from sdpype/evaluation/statistical.py (the privacy slice: the
KAnonymizationMetric class, evaluate_privacy_metrics, generate_privacy_report,
ensure_json_serializable). Verbatim except:
  - the two `OmegaConf.is_config(...)` defensive lines are dropped (parameters
    are always plain dicts here);
  - the metric factory is a local k-anonymisation-only dispatch — sdpype's
    dcr_baseline_protection is not in the Step7 eval config (its standalone
    dcr_*.json in sd-lake come from a separate script), so it isn't ported.

Uses the real sdv.metadata.SingleTableMetadata (see pyproject note) — k-anon
reads metadata.columns[col]["sdtype"] to decide which QI columns to label-encode.
"""

import time
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sdv.metadata import SingleTableMetadata
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_privacy import kAnonymization

import numpy as np


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


class KAnonymizationMetric:
    """K-Anonymization privacy metric using synthcity's kAnonymization"""

    def __init__(self, qi_columns: List[str], **parameters):
        """
        Initialize K-Anonymization metric.

        Args:
            qi_columns: List of quasi-identifier column names
            **parameters: Additional parameters (reserved for future use)
        """
        if not qi_columns:
            raise ValueError("qi_columns parameter is required and cannot be empty")
        self.qi_columns = qi_columns
        self.parameters = parameters

    def _apply_label_encoding(self, datasets_dict: Dict[str, pd.DataFrame], metadata: SingleTableMetadata) -> tuple[Dict[str, pd.DataFrame], Dict[str, LabelEncoder], List[str]]:
        """
        Apply label encoding to categorical QI columns across all datasets.

        Uses metadata as single source of truth to determine which columns are categorical.

        Args:
            datasets_dict: Dictionary of {dataset_name: dataframe} with QI columns only
            metadata: SDV metadata object (single source of truth for column types)

        Returns:
            Tuple of (encoded_datasets_dict, label_encoders, categorical_columns)
        """
        categorical_cols = []
        label_encoders = {}
        encoded_datasets = {name: df.copy() for name, df in datasets_dict.items()}

        for col in self.qi_columns:
            # Use metadata as single source of truth for categorical detection
            if col not in metadata.columns:
                # Column not in metadata - skip with warning
                print(f"⚠️  Warning: QI column '{col}' not found in metadata, skipping encoding")
                continue

            col_meta = metadata.columns[col]
            is_categorical = col_meta.get("sdtype") == "categorical"

            if is_categorical:
                categorical_cols.append(col)

                # Fit encoder on combined data from all datasets
                le = LabelEncoder()
                combined_values = pd.concat([
                    df[col] for df in datasets_dict.values()
                    if col in df.columns
                ]).astype(str)
                le.fit(combined_values)

                # Transform all datasets
                for name in encoded_datasets.keys():
                    if col in encoded_datasets[name].columns:
                        encoded_datasets[name][col] = le.transform(encoded_datasets[name][col].astype(str))

                label_encoders[col] = le

        return encoded_datasets, label_encoders, categorical_cols

    def _interpret_k_value(self, k: int) -> str:
        """Interpret k-anonymity value."""
        if k >= 10:
            return "Excellent"
        elif k >= 5:
            return "Good"
        elif k >= 3:
            return "Moderate"
        else:
            return "Poor"

    def _interpret_ratio(self, ratio: float) -> str:
        """Interpret k-ratio value."""
        if ratio > 1.5:
            return "Much better privacy"
        elif ratio > 1.1:
            return "Better privacy"
        elif ratio > 0.9:
            return "Similar privacy"
        elif ratio > 0.7:
            return "Worse privacy"
        else:
            return "Much worse privacy"

    def evaluate(self,
                 original: pd.DataFrame,
                 synthetic: pd.DataFrame,
                 metadata: SingleTableMetadata,
                 encoding_config: dict = None,
                 population: pd.DataFrame = None,
                 training: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Evaluate K-Anonymization metric across multiple datasets.

        Args:
            original: Reference dataset (required)
            synthetic: Synthetic dataset (required)
            metadata: SDV metadata
            encoding_config: Encoding configuration (not used, k-anon uses decoded data)
            population: Population dataset (optional)
            training: Training dataset (optional)

        Returns:
            Dictionary with k-anonymity values and ratios
        """
        start_time = time.time()

        try:
            # Validate QI columns exist in reference and synthetic
            missing_cols_ref = [c for c in self.qi_columns if c not in original.columns]
            missing_cols_syn = [c for c in self.qi_columns if c not in synthetic.columns]

            if missing_cols_ref:
                raise ValueError(f"QI columns missing from reference dataset: {missing_cols_ref}")
            if missing_cols_syn:
                raise ValueError(f"QI columns missing from synthetic dataset: {missing_cols_syn}")

            # Collect all provided datasets
            datasets = {
                "reference": original,
                "synthetic": synthetic
            }

            if population is not None:
                # Validate QI columns exist
                missing_cols_pop = [c for c in self.qi_columns if c not in population.columns]
                if missing_cols_pop:
                    print(f"⚠️ Warning: QI columns missing from population dataset: {missing_cols_pop}, skipping population")
                else:
                    datasets["population"] = population

            if training is not None:
                # Validate QI columns exist
                missing_cols_train = [c for c in self.qi_columns if c not in training.columns]
                if missing_cols_train:
                    print(f"⚠️ Warning: QI columns missing from training dataset: {missing_cols_train}, skipping training")
                else:
                    datasets["training"] = training

            print(f"🔒 Computing k-anonymity for datasets: {', '.join(datasets.keys())}")
            print(f"   QI columns: {self.qi_columns}")

            # Filter to only QI columns
            datasets_qi = {name: df[self.qi_columns].copy() for name, df in datasets.items()}

            # Apply label encoding for categorical columns (using metadata as single source of truth)
            encoded_datasets, label_encoders, categorical_cols = self._apply_label_encoding(datasets_qi, metadata)

            if categorical_cols:
                print(f"   Encoded {len(categorical_cols)} categorical column(s): {categorical_cols}")

            # Compute k-anonymity for each dataset using synthcity
            k_values = {}
            k_anon_metric = kAnonymization()

            for name, df_qi in encoded_datasets.items():
                loader = GenericDataLoader(df_qi)
                k = k_anon_metric.evaluate_data(loader)
                k_values[name] = int(k)
                print(f"   {name}: k = {k}")

            # Compute k-ratios for relevant pairs
            k_ratios = {}
            ratio_pairs = [
                ("reference", "population", "Reference / Population"),
                ("synthetic", "population", "Synthetic / Population"),
                ("synthetic", "reference", "Synthetic / Reference"),
                ("synthetic", "training", "Synthetic / Training"),
            ]

            for numerator, denominator, label in ratio_pairs:
                if numerator in k_values and denominator in k_values:
                    if k_values[denominator] > 0:
                        ratio = k_values[numerator] / k_values[denominator]
                        k_ratios[label] = ratio

            # Prepare results
            result = {
                "k_values": {
                    name: {
                        "k": int(k),
                        "interpretation": self._interpret_k_value(k)
                    }
                    for name, k in k_values.items()
                },
                "k_ratios": {
                    label: {
                        "ratio": float(ratio),
                        "interpretation": self._interpret_ratio(ratio)
                    }
                    for label, ratio in k_ratios.items()
                },
                "qi_columns": self.qi_columns,
                "categorical_columns": categorical_cols,
                "encoded_categories": {
                    col: list(label_encoders[col].classes_)
                    for col in categorical_cols
                },
                "datasets_evaluated": list(k_values.keys()),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }

            return result

        except Exception as e:
            return {
                "k_values": {},
                "k_ratios": {},
                "qi_columns": self.qi_columns,
                "categorical_columns": [],
                "encoded_categories": {},
                "datasets_evaluated": [],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


def _get_privacy_evaluator(metric_name: str, parameters: Dict[str, Any]):
    """Factory for privacy metric evaluators (k-anonymisation only in this port)."""
    if metric_name == "k_anonymization":
        return KAnonymizationMetric(**parameters)
    raise ValueError(f"Unknown privacy metric: {metric_name}")


def evaluate_privacy_metrics(original: pd.DataFrame,
                            synthetic: pd.DataFrame,
                            metrics_config: list,
                            experiment_name: str,
                            metadata: SingleTableMetadata,
                            reference_data_decoded: pd.DataFrame = None,
                            synthetic_data_decoded: pd.DataFrame = None,
                            reference_data_encoded: pd.DataFrame = None,
                            synthetic_data_encoded: pd.DataFrame = None,
                            encoding_config: dict = None,
                            population_data: pd.DataFrame = None,
                            training_data: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Evaluate configured privacy metrics (k-anonymisation).

    Ported from sdpype evaluate_privacy_metrics — dcr_baseline_protection routing
    kept structurally (harmless: never matched here) but its evaluator is not
    ported; the OmegaConf coercion is dropped.
    """

    print(f"Evaluating privacy metrics for experiment: {experiment_name}")
    print(f"Original shape: {original.shape}, Synthetic shape: {synthetic.shape}")

    results = {
        "metadata": {
            "experiment_name": experiment_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "original_shape": list(original.shape),
            "synthetic_shape": list(synthetic.shape),
            "evaluation_type": "privacy_metrics"
        },
        "metrics": {}
    }

    # Privacy metrics that need encoded data (distance-based)
    ENCODED_PRIVACY_METRICS = {'dcr_baseline_protection'}

    # Privacy metrics that need decoded data
    DECODED_PRIVACY_METRICS = {'k_anonymization'}

    # Privacy metrics that support multi-dataset evaluation
    MULTIDATASET_PRIVACY_METRICS = {'k_anonymization'}

    # Run each configured metric
    for metric_config in metrics_config:
        metric_name = metric_config.get("name")
        parameters = metric_config.get("parameters", {})

        # Route to correct data format
        if metric_name in ENCODED_PRIVACY_METRICS:
            # Use encoded data for distance-based privacy metrics
            if reference_data_encoded is None or synthetic_data_encoded is None:
                print(f"⚠️  Metric {metric_name} needs encoded data but not available, using default")
                ref_data = original
                syn_data = synthetic
            else:
                print(f"📊 Routing {metric_name} to ENCODED data")
                ref_data = reference_data_encoded
                syn_data = synthetic_data_encoded
        elif metric_name in DECODED_PRIVACY_METRICS:
            # Use decoded data for other privacy metrics
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

        print(f"Running {metric_name} privacy metric...")

        try:
            evaluator = _get_privacy_evaluator(metric_name, parameters)

            # K-anonymization needs multi-dataset support
            if metric_name in MULTIDATASET_PRIVACY_METRICS:
                metric_result = evaluator.evaluate(
                    ref_data,
                    syn_data,
                    metadata,
                    encoding_config=encoding_config,
                    population=population_data,
                    training=training_data
                )
            else:
                metric_result = evaluator.evaluate(ref_data, syn_data, metadata, encoding_config=encoding_config)

            results["metrics"][metric_name] = metric_result

        except Exception as e:
            results["metrics"][metric_name] = {
                "status": "error",
                "error_message": str(e),
                "parameters": parameters
            }

    print("Privacy metrics evaluation completed")

    # Ensure all results are JSON serializable
    results = ensure_json_serializable(results)

    return results


def generate_privacy_report(results: Dict[str, Any]) -> str:
    """Generate a human-readable privacy metrics report"""

    report = f"""
Privacy Metrics Evaluation Report
===================================

Experiment: {results['metadata']['experiment_name']}
Timestamp: {results['metadata']['evaluation_timestamp']}
Dataset Shapes: Original {tuple(results['metadata']['original_shape'])}, Synthetic {tuple(results['metadata']['synthetic_shape'])}

Privacy Metrics Results
-----------------------
"""
    metrics = results.get("metrics", {})

    # K-Anonymization results
    if "k_anonymization" in metrics:
        k_anon_result = metrics["k_anonymization"]
        if k_anon_result["status"] == "success":
            params_info = k_anon_result["parameters"]
            params_display = str(params_info) if params_info else "default settings"

            report += f"""K-Anonymization Results:
  Parameters: {params_display}
  Execution time: {k_anon_result['execution_time']:.2f}s
  QI Columns: {', '.join(k_anon_result['qi_columns'])}
  Datasets Evaluated: {', '.join(k_anon_result['datasets_evaluated'])}
"""

            if k_anon_result.get('categorical_columns'):
                report += f"  Categorical Columns (Label Encoded): {', '.join(k_anon_result['categorical_columns'])}\n"

            report += "\n  K-Anonymity Values:\n"
            for dataset_name, k_data in k_anon_result['k_values'].items():
                k_val = k_data['k']
                interp = k_data['interpretation']
                report += f"  → {dataset_name.capitalize():12} k = {k_val:4d}  ({interp})\n"

            if k_anon_result.get('k_ratios'):
                report += "\n  K-Anonymity Ratios:\n"
                for ratio_name, ratio_data in k_anon_result['k_ratios'].items():
                    ratio_val = ratio_data['ratio']
                    interp = ratio_data['interpretation']
                    report += f"  → {ratio_name:30} = {ratio_val:.4f}  ({interp})\n"

            report += """
  Note: Higher k values indicate better privacy protection
        k ≥ 10: Excellent, k ≥ 5: Good, k ≥ 3: Moderate, k < 3: Poor
        Synthcity uses clustering-based approach (not traditional equivalence classes)
"""
        else:
            report += f"""K-Anonymization: ERROR
  Error: {k_anon_result.get('error_message', 'Unknown error')}
"""

    # Assessment
    insights = []

    if "k_anonymization" in metrics and metrics["k_anonymization"]["status"] == "success":
        k_vals = metrics["k_anonymization"]["k_values"]
        if "synthetic" in k_vals:
            k_syn = k_vals["synthetic"]["k"]
            if k_syn >= 10:
                insights.append("Excellent k-anonymity for synthetic data")
            elif k_syn >= 5:
                insights.append("Good k-anonymity for synthetic data")
            elif k_syn >= 3:
                insights.append("Moderate k-anonymity for synthetic data")
            else:
                insights.append("Poor k-anonymity for synthetic data")

    assessment = ", ".join(insights) if insights else "No successful privacy metrics"

    report += f"""
Assessment: {assessment}
"""

    return report
