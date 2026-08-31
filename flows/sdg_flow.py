import os
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

"""
SDG Pipeline — Prefect flow (encode → train → generate → evaluate)
=================================================================

Ported from sdpype's flows/sdg_flow.py. resolve_config is replaced entirely
by argparse (no Hydra/YAML template resolution); fit_encoder, encode_data,
train_sdg and generate_synthetic are the core encode → train → generate
chain (arf/ctgan/ddpm/rtvae/nflow, synthcity-only).

The evaluation stages from sdpype's own "# Evaluation tasks" marker onward
(encode_evaluation, then statistical / privacy / detection / hallucination /
TSTR + report) are being ported incrementally, one stage per commit, each
verified against real gen-0 artifacts in ../sd-lake/. Currently wired:
encode_evaluation, hallucination_evaluation, tstr_evaluation,
privacy_evaluation, detection_evaluation, statistical_similarity
(table_structure, semantic_structure, boundary_adherence,
category_adherence, alpha_precision, prdc_score, the three
jensenshannon_* variants, maximum_mean_discrepancy (corrected),
new_row_synthesis, wasserstein_distance, ks_complement, tv_complement,
sdmetrics_quality — all 15 statistical sub-metrics now ported).

Usage:
    python flows/sdg_flow.py \\
        --training-file   path/to/..._training.csv \\
        --population-file path/to/..._population.csv \\
        --metadata-file   path/to/..._metadata.json \\
        --encoding-config path/to/..._encoding.yaml \\
        --model-type arf --seed 28657
"""

import argparse
import json
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from prefect import flow, task
from sdv.metadata import SingleTableMetadata

from flows.lgbm_cv_flow import encode_features, evaluate_on_test, train_final_model
from sdg_core.detection import evaluate_detection_metrics, generate_detection_report
from sdg_core.downstream import LGBMBayesianTuner
from sdg_core.encoding import RDTDatasetEncoder, load_encoding_config
from sdg_core.generation import apply_post_processing
from sdg_core.hallucination import (
    bin_dataframe,
    compute_bin_boundaries,
    compute_hallucination_metrics,
    compute_record_hashes,
    get_column_types,
    get_unique_hashes,
)
from sdg_core.hashing import calculate_file_hash
from sdg_core.metadata import load_csv_with_metadata
from sdg_core.privacy import evaluate_privacy_metrics, generate_privacy_report
from sdg_core.serialization import create_model_metadata, load_model, save_model
from sdg_core.statistical import (
    ensure_json_serializable,
    evaluate_statistical_metrics,
    generate_statistical_report,
)
from sdg_core.training import create_experiment_hash, create_synthcity_model


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="fit-encoder", log_prints=True)
def fit_encoder(
    population_file: str,
    metadata_file: str,
    encoding_config_file: str,
    encoder_dir: str,
) -> dict:
    """
    Fit RDT encoder on POPULATION data and save it.

    The population encoder is keyed on the population file hash alone — not
    on any experiment/seed — so it is reused across every run that shares
    the same population file. If the file already exists it is loaded, not
    re-fitted.

    Categories are domain knowledge (all valid codes, admission types, etc.),
    not statistical signal, so fitting on the full population gives the
    encoder a complete category space regardless of which patients landed in
    any particular training split. Numerical transformers (FloatFormatter
    min/max) also use population statistics — acceptable since population is
    the ground-truth schema.
    """
    population_file = Path(population_file)
    metadata_file = Path(metadata_file)
    encoding_config_path = Path(encoding_config_file)

    for p in (population_file, metadata_file, encoding_config_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    encoder_dir = Path(encoder_dir)
    encoder_dir.mkdir(parents=True, exist_ok=True)

    population_hash = calculate_file_hash(str(population_file))
    encoder_path = encoder_dir / f"population_encoder_{population_hash}.pkl"

    if encoder_path.exists():
        print(f"Reusing existing population encoder: {encoder_path}")
        return {"encoder_path": str(encoder_path), "population_hash": population_hash}

    print(f"Loading encoding config: {encoding_config_path}")
    encoding_config = load_encoding_config(encoding_config_path)

    print(f"Loading population data: {population_file}")
    population_data = load_csv_with_metadata(population_file, metadata_file)
    print(f"  Population shape: {population_data.shape}")

    print("Fitting encoder on population data ...")
    encoder = RDTDatasetEncoder(encoding_config)
    encoder.fit(population_data)

    encoder.save(encoder_path)
    print(f"Saved population encoder → {encoder_path}")

    return {"encoder_path": str(encoder_path), "population_hash": population_hash}


@task(name="encode-data", retries=1, log_prints=True)
def encode_data(
    training_file: str,
    metadata_file: str,
    encoder_path: str,
    base_name: str,
    output_dir: str,
) -> dict:
    """
    Load the pre-fitted population encoder and encode the training data.

    The encoder is shared across runs; only the transform target
    (training_file) differs between dseeds/generations.
    """
    start_time = time.time()
    print(f"Base name: {base_name}")

    training_file = Path(training_file)
    metadata_file = Path(metadata_file)

    for p in (training_file, metadata_file):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    output_dir = Path(output_dir)
    encoded_dir = output_dir / "data" / "encoded"
    decoded_dir = output_dir / "data" / "decoded"
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    for d in (encoded_dir, decoded_dir, models_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading encoder: {encoder_path}")
    encoder = RDTDatasetEncoder.load(encoder_path)

    print(f"Loading training data: {training_file}")
    training_data = load_csv_with_metadata(training_file, metadata_file)
    print(f"  Training shape: {training_data.shape}")

    print("Transforming training data ...")
    encoded_training = encoder.transform(training_data)
    print(f"  {training_data.shape[1]} columns → {encoded_training.shape[1]}")

    decoded_training = encoder.reverse_transform(encoded_training)

    encoded_path = encoded_dir / f"training_{base_name}.csv"
    decoded_path = decoded_dir / f"training_{base_name}.csv"
    # Copy encoder to a per-run path (mirrors production's per-experiment copy)
    exp_encoder_path = models_dir / f"training_encoder_{base_name}.pkl"

    encoded_training.to_csv(encoded_path, index=False)
    decoded_training.to_csv(decoded_path, index=False)
    shutil.copy2(encoder_path, exp_encoder_path)

    print(f"Saved encoded → {encoded_path}")
    print(f"Saved decoded → {decoded_path}")
    print(f"Saved encoder → {exp_encoder_path}")

    elapsed = round(time.time() - start_time, 2)
    metrics = {
        "encoding_type": "training",
        "encoding_version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "base_name": base_name,
        "fitted_on": "population_data",
        "population_encoder": str(encoder_path),
        "input_shapes": {"training": list(training_data.shape)},
        "output_shapes": {
            "encoded_training": list(encoded_training.shape),
            "decoded_training": list(decoded_training.shape),
        },
        "transformers": {
            col: type(trans).__name__ for col, trans in encoder.transformers.items()
        },
        "encoding_time_seconds": elapsed,
        "outputs": {
            "encoded_training": str(encoded_path),
            "decoded_training": str(decoded_path),
            "training_encoder": str(exp_encoder_path),
        },
    }

    metrics_path = metrics_dir / f"encoding_{base_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")
    print(f"encode_data done in {elapsed}s")

    return {
        "encoded_training": str(encoded_path),
        "decoded_training": str(decoded_path),
        "training_encoder": str(exp_encoder_path),
        "metrics": str(metrics_path),
        "base_name": base_name,
    }


@task(name="train-sdg", log_prints=True)
def train_sdg(
    encoded_training_path: str,
    training_file: str,
    model_type: str,
    library: str,
    seed: int,
    parameters: dict,
    run_params: dict,
    base_name: str,
    output_dir: str,
) -> dict:
    """
    Train a synthcity SDG model on the encoded training data.

    Only library="synthcity" is implemented (this repo's locked scope —
    see the port plan). `library` is still a real parameter, not hardcoded,
    so adding another library later is a contained change.
    """
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    for d in (models_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"sdg_model_{base_name}.pkl"
    metrics_path = metrics_dir / f"training_{base_name}.json"

    if model_path.exists():
        print(f"Reusing existing SDG model: {model_path}")
        return {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "base_name": base_name,
            "library": library,
            "model_type": model_type,
        }

    print(f"Library    : {library}")
    print(f"Model type : {model_type}")
    print(f"Seed       : {seed}")

    if library != "synthcity":
        raise ValueError(f"Unsupported library: {library!r} (only 'synthcity' is implemented)")

    training_data = pd.read_csv(encoded_training_path)
    print(f"Training data (encoded): {training_data.shape}")
    model = create_synthcity_model(model_type, parameters, seed)

    print(f"Training {library} {model_type} ...")
    start_time = time.time()
    model.fit(training_data)
    training_time = round(time.time() - start_time, 2)
    print(f"Training completed in {training_time}s")

    sdg_params = {"library": library, "model_type": model_type, "parameters": parameters}
    experiment_hash = create_experiment_hash(sdg_params, seed, training_file)
    experiment_id = f"{model_type}_{seed}_{experiment_hash}"

    model_metadata = create_model_metadata(
        model_type=model_type,
        library=library,
        seed=seed,
        training_time=training_time,
        data=training_data,
        experiment_id=experiment_id,
        experiment_hash=experiment_hash,
        parameters=parameters,
        run_params=run_params,
    )
    saved_model_path = save_model(model, model_metadata, library, model_path)
    print(f"Saved model → {saved_model_path}")

    metrics = {
        "experiment_id": experiment_id,
        "experiment_hash": experiment_hash,
        "seed": seed,
        "library": library,
        "model_type": model_type,
        "training_time": training_time,
        "training_rows": len(training_data),
        "training_columns": len(training_data.columns),
        "timestamp": datetime.now().isoformat(),
        "data_source": str(encoded_training_path),
        "model_output": saved_model_path,
        "model_parameters": parameters or {},
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    return {
        "model_path": saved_model_path,
        "metrics_path": str(metrics_path),
        "base_name": base_name,
        "library": library,
        "model_type": model_type,
    }


@task(name="generate-synthetic", log_prints=True)
def generate_synthetic(
    model_path: str,
    training_encoder_path: str,
    training_file: str,
    metadata_file: str,
    reference_file: str,
    encoded_training_path: str,
    model_type: str,
    library: str,
    seed: int,
    base_name: str,
    output_dir: str,
    n_samples: Optional[int] = None,
    post_process_method: str = "knn",
    knn_neighbors: int = 5,
    distance_metric: str = "hamming",
    fallback: str = "weighted",
    force_generate: bool = False,
) -> dict:
    """
    Generate synthetic data from a trained synthcity model.

    synthcity models output encoded data directly, which is reverse-
    transformed to decoded (dual pipeline). Only library="synthcity" is
    implemented (this repo's locked scope).
    """
    output_dir = Path(output_dir)
    synthetic_dir = output_dir / "data" / "synthetic"
    metrics_dir = output_dir / "metrics"
    for d in (synthetic_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    encoded_path = synthetic_dir / f"synthetic_data_{base_name}_encoded.csv"
    decoded_path = synthetic_dir / f"synthetic_data_{base_name}_decoded.csv"
    metrics_path = metrics_dir / f"generation_{base_name}.json"

    if not force_generate and decoded_path.exists():
        print(f"Reusing existing synthetic data: {decoded_path}")
        return {
            "encoded_path": str(encoded_path),
            "decoded_path": str(decoded_path),
            "metrics_path": str(metrics_path),
            "n_samples": len(pd.read_csv(decoded_path)),
            "base_name": base_name,
        }

    if library != "synthcity":
        raise ValueError(f"Unsupported library: {library!r} (only 'synthcity' is implemented)")

    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    print(f"Loading model: {model_path}")
    model, _model_meta = load_model(model_path)

    encoder = RDTDatasetEncoder.load(training_encoder_path)
    training_data = load_csv_with_metadata(Path(training_file), Path(metadata_file))
    print(f"Training data loaded: {training_data.shape}")

    n_samples_config = n_samples
    if n_samples is None:
        reference_data = load_csv_with_metadata(Path(reference_file), Path(metadata_file))
        n_samples = len(reference_data)
        print(f"Auto n_samples from reference dataset: {n_samples}")
    print(f"Generating {n_samples} samples ({library} {model_type}) ...")

    start_time = time.time()
    synthetic_encoded = model.generate(count=n_samples).dataframe()
    generation_time = round(time.time() - start_time, 2)
    print(f"Generated {len(synthetic_encoded)} samples in {generation_time}s")

    if len(synthetic_encoded) == 0:
        raise ValueError("Generated dataset is empty")

    # Column validation
    expected_cols = pd.read_csv(encoded_training_path, nrows=0).columns.tolist()
    missing = set(expected_cols) - set(synthetic_encoded.columns)
    if missing:
        raise ValueError(
            f"Synthcity {model_type} generated incomplete data: "
            f"missing {len(missing)} columns: {sorted(missing)}"
        )

    synthetic_decoded = encoder.reverse_transform(synthetic_encoded)
    synthetic_decoded, fix_metrics = apply_post_processing(
        synthetic_decoded,
        training_data,
        encoder.sdtypes,
        method=post_process_method,
        knn_neighbors=knn_neighbors,
        distance_metric=distance_metric,
        fallback=fallback,
    )

    print(f"Encoded shape : {synthetic_encoded.shape}")
    print(f"Decoded shape : {synthetic_decoded.shape}")

    synthetic_encoded.to_csv(encoded_path, index=False)
    synthetic_decoded.to_csv(decoded_path, index=False)
    print(f"Saved encoded  → {encoded_path}")
    print(f"Saved decoded  → {decoded_path}")

    metrics = {
        "seed": seed,
        "library": library,
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "n_samples_config": n_samples_config,
        "n_samples_auto_determined": n_samples_config is None,
        "samples_generated": len(synthetic_decoded),
        "samples_requested": n_samples,
        "generation_time_seconds": generation_time,
        "post_processing": {"fix_metrics": fix_metrics} if fix_metrics else {},
        "outputs": {"encoded": str(encoded_path), "decoded": str(decoded_path)},
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics  → {metrics_path}")

    return {
        "encoded_path": str(encoded_path),
        "decoded_path": str(decoded_path),
        "metrics_path": str(metrics_path),
        "n_samples": len(synthetic_decoded),
        "base_name": base_name,
    }


# ---------------------------------------------------------------------------
# Evaluation tasks
# ---------------------------------------------------------------------------

@task(name="encode-evaluation", log_prints=True)
def encode_evaluation(
    reference_file: str,
    metadata_file: str,
    encoding_config_file: str,
    synthetic_decoded_path: str,
    base_name: str,
    output_dir: str,
    force: bool = False,
) -> dict:
    """
    Fit a fresh RDT encoder on the REFERENCE data and push both reference and
    synthetic data through it (dual pipeline: encoded + reverse-decoded).

    Unlike encode_data — which loads the shared population encoder — the
    evaluation encoder is keyed on the reference (real) data, so it captures
    exactly the categories present in the real comparison set. This is the
    input stage every downstream statistical / privacy / detection metric
    consumes. Ported near-verbatim from sdpype flows/sdg_flow.py::encode_evaluation
    (the only changes: flat args instead of a cfg dict, and the sd-lake
    data/{encoded,decoded} + models + metrics layout under output_dir).
    """
    start_time = time.time()
    print(f"Base name: {base_name}")

    reference_file = Path(reference_file)
    metadata_file = Path(metadata_file)
    encoding_config_path = Path(encoding_config_file)
    synthetic_decoded_path = Path(synthetic_decoded_path)

    for p in (reference_file, metadata_file, encoding_config_path, synthetic_decoded_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    output_dir = Path(output_dir)
    encoded_dir = output_dir / "data" / "encoded"
    decoded_dir = output_dir / "data" / "decoded"
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    for d in (encoded_dir, decoded_dir, models_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    encoded_reference_path = encoded_dir / f"reference_{base_name}.csv"
    encoded_synthetic_path = encoded_dir / f"synthetic_{base_name}.csv"
    decoded_reference_path = decoded_dir / f"reference_{base_name}.csv"
    decoded_synthetic_path = decoded_dir / f"synthetic_{base_name}_decoded.csv"
    eval_encoder_path = models_dir / f"evaluation_encoder_{base_name}.pkl"
    metrics_path = metrics_dir / f"encoding_evaluation_{base_name}.json"

    outputs = {
        "encoded_reference": str(encoded_reference_path),
        "encoded_synthetic": str(encoded_synthetic_path),
        "decoded_reference": str(decoded_reference_path),
        "decoded_synthetic": str(decoded_synthetic_path),
        "eval_encoder": str(eval_encoder_path),
        "metrics_path": str(metrics_path),
        "base_name": base_name,
    }

    if not force and decoded_synthetic_path.exists():
        print(f"Reusing existing evaluation-encoded data: {decoded_synthetic_path}")
        return outputs

    encoding_config = load_encoding_config(encoding_config_path)

    print(f"Loading reference data: {reference_file}")
    reference_data = load_csv_with_metadata(reference_file, metadata_file)
    print(f"  Reference shape: {reference_data.shape}")

    print(f"Loading synthetic decoded data: {synthetic_decoded_path}")
    synthetic_data = load_csv_with_metadata(synthetic_decoded_path, metadata_file)
    print(f"  Synthetic shape: {synthetic_data.shape}")

    print("Fitting evaluation encoder on reference data ...")
    encoder = RDTDatasetEncoder(encoding_config)
    encoder.fit(reference_data)

    encoded_reference = encoder.transform(reference_data)
    encoded_synthetic = encoder.transform(synthetic_data)
    decoded_reference = encoder.reverse_transform(encoded_reference)
    decoded_synthetic = encoder.reverse_transform(encoded_synthetic)

    print(f"Encoded reference: {encoded_reference.shape}")
    print(f"Encoded synthetic: {encoded_synthetic.shape}")

    encoded_reference.to_csv(encoded_reference_path, index=False)
    encoded_synthetic.to_csv(encoded_synthetic_path, index=False)
    decoded_reference.to_csv(decoded_reference_path, index=False)
    decoded_synthetic.to_csv(decoded_synthetic_path, index=False)
    encoder.save(eval_encoder_path)

    print(f"Saved encoded reference  → {encoded_reference_path}")
    print(f"Saved encoded synthetic  → {encoded_synthetic_path}")
    print(f"Saved decoded reference  → {decoded_reference_path}")
    print(f"Saved decoded synthetic  → {decoded_synthetic_path}")
    print(f"Saved eval encoder       → {eval_encoder_path}")

    elapsed = round(time.time() - start_time, 2)
    metrics = {
        "encoding_type": "evaluation",
        "encoding_version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "base_name": base_name,
        "fitted_on": "reference_data",
        "input_shapes": {
            "reference": list(reference_data.shape),
            "synthetic": list(synthetic_data.shape),
        },
        "output_shapes": {
            "encoded_reference": list(encoded_reference.shape),
            "encoded_synthetic": list(encoded_synthetic.shape),
            "decoded_reference": list(decoded_reference.shape),
            "decoded_synthetic": list(decoded_synthetic.shape),
        },
        "transformers": {
            col: type(trans).__name__ for col, trans in encoder.transformers.items()
        },
        "encoding_time_seconds": elapsed,
        "outputs": {
            "encoded_reference": str(encoded_reference_path),
            "encoded_synthetic": str(encoded_synthetic_path),
            "decoded_reference": str(decoded_reference_path),
            "decoded_synthetic": str(decoded_synthetic_path),
            "evaluation_encoder": str(eval_encoder_path),
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")
    print(f"encode_evaluation done in {elapsed}s")

    return outputs


@task(name="hallucination-evaluation", log_prints=True)
def hallucination_evaluation(
    population_file: str,
    training_file: str,
    synthetic_decoded_path: str,
    metadata_file: str,
    base_name: str,
    output_dir: str,
    num_bins: int = 20,
    force: bool = False,
) -> dict:
    """
    Hallucination metrics (TotalFR / NovelFR / MemorizedFR / HR) via the
    SQL-free binning + hashing approach: bin numerical columns against
    population-derived boundaries, hash each row, do set-membership checks
    against population and training. No DuckDB, no query file.

    Consumes the evaluation-encoded *decoded* synthetic (encode_evaluation's
    output), matching sdpype. Ported from sdpype flows/sdg_flow.py::
    hallucination_evaluation — flat args instead of a cfg dict, sd-lake
    metrics/ layout, base_name in place of experiment_name/seed/config_hash.
    """
    population_file = Path(population_file)
    training_file = Path(training_file)
    synthetic_decoded_path = Path(synthetic_decoded_path)
    metadata_file = Path(metadata_file)

    for p in (population_file, training_file, synthetic_decoded_path, metadata_file):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"hallucination_{base_name}.json"
    report_path = metrics_dir / f"hallucination_report_{base_name}.txt"

    result = {"metrics_path": str(metrics_path), "report_path": str(report_path)}
    if not force and metrics_path.exists():
        print(f"Reusing existing hallucination metrics: {metrics_path}")
        return result

    start_time = time.time()
    print(f"Loading population : {population_file}")
    population_df = load_csv_with_metadata(population_file, metadata_file, low_memory=False)
    print(f"Loading training   : {training_file}")
    training_df = load_csv_with_metadata(training_file, metadata_file, low_memory=False)
    print(f"Loading synthetic  : {synthetic_decoded_path}")
    synthetic_df = load_csv_with_metadata(synthetic_decoded_path, metadata_file, low_memory=False)
    print(f"  population {population_df.shape}, training {training_df.shape}, synthetic {synthetic_df.shape}")

    print(f"Computing bin boundaries from population (num_bins={num_bins}) ...")
    column_types = get_column_types(metadata_file)
    boundaries = compute_bin_boundaries(population_df, column_types, num_bins)
    print(f"  Boundaries for {len(boundaries)} numerical columns")

    print("Binning datasets ...")
    binned_population = bin_dataframe(population_df, column_types, boundaries, num_bins)
    binned_training = bin_dataframe(training_df, column_types, boundaries, num_bins)
    binned_synthetic = bin_dataframe(synthetic_df, column_types, boundaries, num_bins)

    print("Hashing and computing metrics ...")
    population_hashes = compute_record_hashes(binned_population)
    training_hashes = compute_record_hashes(binned_training)
    synthetic_hashes = compute_record_hashes(binned_synthetic)

    population_unique = get_unique_hashes(population_hashes)
    training_unique = get_unique_hashes(training_hashes)
    print(f"  Population unique: {len(population_unique):,}  Training unique: {len(training_unique):,}")

    metrics, _, _ = compute_hallucination_metrics(
        population_unique, training_unique, synthetic_hashes
    )

    m = metrics
    print(f"  TotalFR    : {m['TotalFR']['rate_pct']:.2f}%")
    print(f"  NovelFR    : {m['NovelFR']['rate_pct']:.2f}%")
    print(f"  MemorizedFR: {m['MemorizedFR']['rate_pct']:.2f}%")
    print(f"  HR         : {m['HR']['rate_pct']:.2f}%")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "base_name": base_name,
            "population_file": str(population_file),
            "training_file": str(training_file),
            "synthetic_file": str(synthetic_decoded_path),
            "num_bins": num_bins,
        },
        "dataset_statistics": {
            "population": {"rows": population_df.shape[0], "columns": population_df.shape[1]},
            "training": {"rows": training_df.shape[0], "columns": training_df.shape[1]},
            "synthetic": {"rows": synthetic_df.shape[0], "columns": synthetic_df.shape[1]},
        },
        "binning": {
            "num_bins": num_bins,
            "boundaries_source": "population",
            "numerical_columns": list(boundaries.keys()),
        },
        "metrics": metrics,
        "execution_time": time.time() - start_time,
    }

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    report_lines = [
        "Hallucination Evaluation Report",
        "=" * 40,
        f"Base name : {base_name}",
        f"Num bins  : {num_bins}",
        "",
        f"TotalFR     (in population)            : {m['TotalFR']['rate_pct']:.2f}%  ({m['TotalFR']['count']:,} records)",
        f"NovelFR     (in population, not in trn): {m['NovelFR']['rate_pct']:.2f}%  ({m['NovelFR']['count']:,} records)",
        f"MemorizedFR (in population and in trn) : {m['MemorizedFR']['rate_pct']:.2f}%  ({m['MemorizedFR']['count']:,} records)",
        f"HR          (not in population)        : {m['HR']['rate_pct']:.2f}%  ({m['HR']['count']:,} records)",
        f"Total                                  : {m['total_records']:,} records",
    ]
    report_path.write_text("\n".join(report_lines))

    print(f"Saved hallucination metrics → {metrics_path}")
    print(f"Saved hallucination report  → {report_path}")
    return result


@task(name="tstr-evaluation", log_prints=True)
def tstr_evaluation(
    dseed_dir: str,
    synth_decoded_path: str,
    base_name: str,
    output_dir: str,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """
    Train on Synthetic, Test on Real. Trains LightGBM on the decoded synthetic
    data using the *frozen* best_params + decision threshold from the dseed
    folder's lgbm_cv_*.json (the TRTR baseline), evaluates on that folder's
    real held-out test set, and reports the utility gap.

    Freezing the params/threshold (rather than re-tuning per run) is deliberate
    — it keeps every generation on the same operating point so a utility drop
    reflects data quality, not optimizer drift.

    Ported from sdpype flows/sdg_flow.py::tstr_evaluation — flat args instead of
    a cfg dict, base_name passed in, sd-lake metrics/ layout. Reuses Step 3's
    ported LGBMBayesianTuner + lgbm_cv_flow tasks (encode_features /
    train_final_model / evaluate_on_test).

    dseed_dir: the folder holding lgbm_cv_*.json + the one *test*.csv — i.e.
    Path(training_file).parent. (Once recursion exists this must stay pinned to
    the original dseed folder, not a per-generation synthetic path.)
    """
    start_time = time.time()
    dseed_dir = Path(dseed_dir)

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"tstr_{base_name}.json"

    if not force and metrics_path.exists():
        print(f"Reusing existing TSTR metrics: {metrics_path}")
        payload = json.loads(metrics_path.read_text())
        return {**payload, "metrics_path": str(metrics_path)}

    # --- Auto-discover lgbm_cv JSON (latest by filename timestamp) ---
    lgbm_candidates = sorted(dseed_dir.glob("lgbm_cv_*.json"))
    if not lgbm_candidates:
        raise FileNotFoundError(
            f"No lgbm_cv_*.json found in dseed folder: {dseed_dir}\n"
            f"Run flows/lgbm_cv_flow.py with --output-dir {dseed_dir} first."
        )
    tstr_params_file = lgbm_candidates[-1]
    print(f"Auto-discovered lgbm_cv JSON: {tstr_params_file}")

    with open(tstr_params_file) as f:
        lgbm_payload = json.load(f)
    best_params = lgbm_payload["best_params"]
    trtr_metrics = lgbm_payload["test_metrics"]
    target_col = lgbm_payload["data_info"]["target"]
    print(f"Target column: {target_col}  |  TRTR baseline AUROC: {trtr_metrics['auroc']:.4f}")

    # --- Auto-discover the one real test file ---
    candidates = sorted(dseed_dir.glob("*test*.csv"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No *test*.csv found in dseed folder: {dseed_dir}")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple *test*.csv files found in {dseed_dir}: {[c.name for c in candidates]}. "
            f"Cannot auto-discover — ensure only one test file exists."
        )
    test_path = candidates[0]
    print(f"Auto-discovered real test file: {test_path}")

    synth_df = pd.read_csv(synth_decoded_path)
    test_df = pd.read_csv(test_path)

    for name, df in (("synthetic", synth_df), ("test", test_df)):
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' (from lgbm_cv JSON) not found in {name} data. "
                f"Columns: {list(df.columns)}"
            )

    X_synth = synth_df.drop(columns=[target_col])
    y_synth = synth_df[target_col].astype(int)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)
    print(f"Synth train: {len(X_synth):,} rows  |  Real test: {len(X_test):,} rows")

    # --- Guard: synthetic data must have both classes ---
    if y_synth.nunique() < 2:
        present = y_synth.unique().tolist()
        msg = (
            f"TSTR skipped — synthetic training data contains only class {present} "
            f"(model collapsed to single class). Classification not possible."
        )
        print(msg)
        return {
            "base_name": base_name,
            "status": "skipped",
            "skip_reason": msg,
            "tstr_metrics": None,
            "trtr_metrics": trtr_metrics,
            "utility_gap": None,
            "metrics_path": None,
        }

    # --- Categorical encoding (fit on synth, apply to test) ---
    X_synth_enc, X_test_enc = encode_features(X_synth, X_test)

    # --- Train on synthetic with the frozen real-data best_params ---
    tuner = LGBMBayesianTuner(
        X_train=X_synth_enc,
        y_train=y_synth,
        n_folds=3,
        n_trials=1,
        random_state=seed,
    )
    tuner.best_params = best_params

    # Evaluate every generation at the same operating point (real-data threshold).
    fixed_threshold = trtr_metrics["threshold"]

    try:
        model, _synth_threshold, target_encoder, calibrator = train_final_model(
            X_synth_enc, y_synth, best_params, tuner, seed=seed, quiet=True,
        )
    except ValueError as e:
        msg = f"TSTR skipped — training failed (likely single-class split in synthetic data): {e}"
        print(msg)
        return {
            "base_name": base_name,
            "status": "skipped",
            "skip_reason": msg,
            "tstr_metrics": None,
            "trtr_metrics": trtr_metrics,
            "utility_gap": None,
            "metrics_path": None,
        }

    tstr_metrics = evaluate_on_test(
        model, X_test_enc, y_test, fixed_threshold, target_encoder, calibrator
    )

    utility_gap = trtr_metrics["auroc"] - tstr_metrics["auroc"]
    print(
        f"TSTR AUROC: {tstr_metrics['auroc']:.4f}  |  TRTR AUROC: {trtr_metrics['auroc']:.4f}  |  "
        f"Utility gap: {utility_gap:+.4f}"
    )

    payload = {
        "base_name": base_name,
        "tstr_params_source": str(tstr_params_file),
        "train_rows_synth": len(X_synth),
        "test_rows_real": len(X_test),
        "trtr_metrics": trtr_metrics,
        "tstr_metrics": tstr_metrics,
        "utility_gap": utility_gap,
        "execution_time": time.time() - start_time,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved TSTR metrics → {metrics_path}")

    return {**payload, "metrics_path": str(metrics_path)}


@task(name="privacy-evaluation", log_prints=True)
def privacy_evaluation(
    metadata_file: str,
    population_file: str,
    training_file: str,
    reference_decoded_path: str,
    synthetic_decoded_path: str,
    base_name: str,
    output_dir: str,
    qi_columns: list,
    force: bool = False,
) -> dict:
    """
    Privacy metrics — k-anonymisation of the quasi-identifier columns, computed
    on the decoded reference / synthetic / population / training data via
    synthcity's kAnonymization, plus k-ratios between them.

    Ported from sdpype flows/sdg_flow.py::privacy_evaluation — flat args instead
    of a cfg dict, base_name in place of experiment_name/seed, sd-lake metrics/
    layout. Only k_anonymization is wired (the sole metric in the Step7 privacy
    config); dcr_baseline_protection isn't ported (see sdg_core/privacy.py).
    Consumes encode_evaluation's decoded reference + synthetic.
    """
    metadata_file = Path(metadata_file)
    population_file = Path(population_file)
    training_file = Path(training_file)
    reference_decoded_path = Path(reference_decoded_path)
    synthetic_decoded_path = Path(synthetic_decoded_path)

    for p in (metadata_file, population_file, training_file, reference_decoded_path, synthetic_decoded_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"privacy_{base_name}.json"
    report_path = metrics_dir / f"privacy_report_{base_name}.txt"

    result = {"metrics_path": str(metrics_path), "report_path": str(report_path)}
    if not force and metrics_path.exists():
        print(f"Reusing existing privacy metrics: {metrics_path}")
        return result

    metrics_config = [{"name": "k_anonymization", "parameters": {"qi_columns": list(qi_columns)}}]

    metadata = SingleTableMetadata.load_from_json(str(metadata_file))
    reference_decoded = load_csv_with_metadata(reference_decoded_path, metadata_file)
    synthetic_decoded = load_csv_with_metadata(synthetic_decoded_path, metadata_file)
    population_data = load_csv_with_metadata(population_file, metadata_file)
    training_data = load_csv_with_metadata(training_file, metadata_file)

    print(f"Running {len(metrics_config)} privacy metric(s) ...  QI: {list(qi_columns)}")
    results = evaluate_privacy_metrics(
        reference_decoded,
        synthetic_decoded,
        metrics_config,
        experiment_name=base_name,
        metadata=metadata,
        reference_data_decoded=reference_decoded,
        synthetic_data_decoded=synthetic_decoded,
        population_data=population_data,
        training_data=training_data,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    report_path.write_text(generate_privacy_report(results), encoding="utf-8")

    print(f"Saved privacy metrics → {metrics_path}")
    print(f"Saved privacy report  → {report_path}")
    return result


@task(name="detection-evaluation", log_prints=True)
def detection_evaluation(
    metadata_file: str,
    encoding_config_file: str,
    encoded_reference_path: str,
    encoded_synthetic_path: str,
    base_name: str,
    output_dir: str,
    methods: list,
    common_params: dict,
    force: bool = False,
) -> dict:
    """
    Detection metrics — synthcity's real-vs-synthetic C2ST classifiers
    (GMM / XGB / MLP / Linear), each a k-fold AUC, plus a mean-of-4 ensemble.
    Runs on ENCODED (all-numeric) data with ID columns excluded.

    Ported from sdpype flows/sdg_flow.py::detection_evaluation — flat args
    instead of a cfg dict, base_name for experiment_name, sd-lake metrics/
    layout. Consumes encode_evaluation's encoded reference + synthetic.
    """
    metadata_file = Path(metadata_file)
    encoding_config_path = Path(encoding_config_file)
    encoded_reference_path = Path(encoded_reference_path)
    encoded_synthetic_path = Path(encoded_synthetic_path)

    for p in (metadata_file, encoded_reference_path, encoded_synthetic_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"detection_evaluation_{base_name}.json"
    report_path = metrics_dir / f"detection_report_{base_name}.txt"

    result = {"metrics_path": str(metrics_path), "report_path": str(report_path)}
    if not force and metrics_path.exists():
        print(f"Reusing existing detection metrics: {metrics_path}")
        return result

    metadata = SingleTableMetadata.load_from_json(str(metadata_file))
    reference_data = pd.read_csv(encoded_reference_path)
    synthetic_data = pd.read_csv(encoded_synthetic_path)
    print(f"Loaded encoded reference {reference_data.shape}, synthetic {synthetic_data.shape}")

    encoding_config = None
    if encoding_config_path.exists():
        encoding_config = load_encoding_config(encoding_config_path)

    print(f"Running {len(methods)} detection method(s) ...  common_params={common_params}")
    results = evaluate_detection_metrics(
        reference_data, synthetic_data, metadata, methods, common_params,
        base_name, encoding_config,
    )
    results = ensure_json_serializable(results)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    report_path.write_text(generate_detection_report(results), encoding="utf-8")

    print(f"Saved detection metrics → {metrics_path}")
    print(f"Saved detection report  → {report_path}")
    return result


# Statistical-similarity data-format routing — copied verbatim from sdpype
# flows/sdg_flow.py::statistical_similarity. A name in ENCODED_METRICS is fed
# the RDT-encoded (all-numeric) frames; DECODED_METRICS gets the reverse-
# transformed frames; anything else falls through to `original`/`synthetic`.
# The full sets are kept even while only a subset of metrics is ported — they
# just decide which frames get loaded and passed through.
_STAT_ENCODED_METRICS = {
    "alpha_precision", "prdc_score", "jensenshannon_synthcity",
    "jensenshannon_syndat", "jensenshannon_nannyml", "wasserstein_distance",
    "maximum_mean_discrepancy", "ks_complement",
}
_STAT_DECODED_METRICS = {
    "tv_complement", "table_structure", "semantic_structure",
    "boundary_adherence", "category_adherence", "new_row_synthesis",
    "sdmetrics_quality", "k_anonymization",
}


@task(name="statistical-similarity", log_prints=True)
def statistical_similarity(
    metadata_file: str,
    encoding_config_file: str,
    encoded_reference_path: str,
    encoded_synthetic_path: str,
    decoded_reference_path: str,
    decoded_synthetic_path: str,
    base_name: str,
    output_dir: str,
    metrics_config: list,
    force: bool = False,
) -> dict:
    """
    Statistical similarity metrics between reference and synthetic data, with
    per-metric encoded-vs-decoded data routing (see _STAT_ENCODED_METRICS /
    _STAT_DECODED_METRICS). Each metric block carries its own per-column scores
    plus an aggregate.

    Ported from sdpype flows/sdg_flow.py::statistical_similarity — flat args
    instead of a cfg dict, base_name in place of experiment_name/seed, sd-lake
    metrics/ layout. Consumes encode_evaluation's encoded + decoded reference
    and synthetic. `metrics_config` is the evaluation.statistical_similarity.
    metrics list from the config (Step7 default supplied by the flow).
    """
    metadata_file = Path(metadata_file)
    encoding_config_path = Path(encoding_config_file)

    if not metadata_file.exists():
        raise FileNotFoundError(f"Required file not found: {metadata_file}")

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"statistical_similarity_{base_name}.json"
    report_path = metrics_dir / f"statistical_report_{base_name}.txt"

    result = {"metrics_path": str(metrics_path), "report_path": str(report_path)}
    if not force and metrics_path.exists():
        print(f"Reusing existing statistical metrics: {metrics_path}")
        return result

    if not metrics_config:
        print("No statistical metrics configured, skipping")
        empty = {"metadata": {"status": "skipped"}, "metrics": {}}
        with open(metrics_path, "w") as f:
            json.dump(empty, f, indent=2)
        report_path.write_text("Statistical Similarity Report\nStatus: Skipped\n")
        return result

    needs_encoded = any(m.get("name") in _STAT_ENCODED_METRICS for m in metrics_config)
    needs_decoded = any(m.get("name") in _STAT_DECODED_METRICS for m in metrics_config)

    metadata = SingleTableMetadata.load_from_json(str(metadata_file))

    encoding_config = None
    if needs_encoded and encoding_config_path.exists():
        encoding_config = load_encoding_config(encoding_config_path)

    reference_data_encoded = synthetic_data_encoded = None
    reference_data_decoded = synthetic_data_decoded = None

    if needs_encoded:
        reference_data_encoded = pd.read_csv(encoded_reference_path)
        synthetic_data_encoded = pd.read_csv(encoded_synthetic_path)
        print(f"Loaded encoded reference {reference_data_encoded.shape}, synthetic {synthetic_data_encoded.shape}")

    if needs_decoded:
        reference_data_decoded = load_csv_with_metadata(Path(decoded_reference_path), metadata_file)
        synthetic_data_decoded = load_csv_with_metadata(Path(decoded_synthetic_path), metadata_file)
        print(f"Loaded decoded reference {reference_data_decoded.shape}, synthetic {synthetic_data_decoded.shape}")

    print(f"Running {len(metrics_config)} statistical metric(s) ...")
    results = evaluate_statistical_metrics(
        reference_data_encoded if needs_encoded else reference_data_decoded,
        synthetic_data_encoded if needs_encoded else synthetic_data_decoded,
        metrics_config,
        experiment_name=base_name,
        metadata=metadata,
        reference_data_decoded=reference_data_decoded,
        synthetic_data_decoded=synthetic_data_decoded,
        reference_data_encoded=reference_data_encoded,
        synthetic_data_encoded=synthetic_data_encoded,
        encoded_metrics=_STAT_ENCODED_METRICS,
        decoded_metrics=_STAT_DECODED_METRICS,
        encoding_config=encoding_config,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    report_path.write_text(generate_statistical_report(results), encoding="utf-8")

    print(f"Saved statistical metrics → {metrics_path}")
    print(f"Saved statistical report  → {report_path}")
    return result


# ---------------------------------------------------------------------------
# Flow — encode → train → generate → evaluation-encode → hallucination → TSTR → privacy → detection
# ---------------------------------------------------------------------------

def _run_name(training_file: str, library: str, model_type: str, seed: int,
              tag: Optional[str] = None) -> str:
    """Assemble the run-directory name.

    sd-lake nests each run as
        <lake>/<experiment>/<model_type>/<tag>_<dseed>_<library>_<model>_mseed<seed>/
    e.g. Step7pfp_dseed1597_synthcity_arf_mseed987 — where <dseed> is the
    dseedNNN token from the training-file path (falls back to the training
    stem when absent). Without a tag the leading segment is dropped.
    """
    m = re.search(r"dseed\d+", Path(training_file).stem)
    dseed = m.group(0) if m else Path(training_file).stem
    stem = f"{dseed}_{library}_{model_type}_mseed{seed}"
    return f"{tag}_{stem}" if tag else stem


@flow(name="sdg-pipeline")
def sdg_pipeline(
    training_file: str,
    population_file: str,
    metadata_file: str,
    encoding_config_file: str,
    model_type: str,
    library: str = "synthcity",
    seed: int = 42,
    parameters: dict | None = None,
    reference_file: Optional[str] = None,
    n_samples: Optional[int] = None,
    post_process_method: str = "knn",
    knn_neighbors: int = 5,
    distance_metric: str = "hamming",
    fallback: str = "weighted",
    force_generate: bool = False,
    hallucination_num_bins: int = 20,
    privacy_qi_columns: Optional[list] = None,
    detection_methods: Optional[list] = None,
    detection_common_params: Optional[dict] = None,
    statistical_metrics: Optional[list] = None,
    experiment_tag: Optional[str] = None,
    run_name: Optional[str] = None,
    encoder_dir: str = "outputs/sdg_runs/encoders",
    output_dir: str = "outputs/sdg_runs",
) -> dict:
    # Production configs set reference_file == training_file (confirmed in
    # params_step7_pf_pilgram.yaml) — defaulting here matches real behavior,
    # not a shortcut.
    if reference_file is None:
        reference_file = training_file
    # Step7 pf_all quasi-identifier set (pf_pilgram drops "gender"); override
    # via --privacy-qi-columns or evaluation.privacy.metrics[0].parameters.
    if privacy_qi_columns is None:
        privacy_qi_columns = ["age", "gender", "ethnicity", "admission_type"]
    # Step7 detection config (evaluation.detection_evaluation in the yaml).
    if detection_methods is None:
        detection_methods = [
            {"name": n, "parameters": {}}
            for n in ("detection_gmm", "detection_xgb", "detection_mlp", "detection_linear")
        ]
    if detection_common_params is None:
        detection_common_params = {"n_folds": 5, "random_state": 987, "reduction": "max"}
    # evaluation.statistical_similarity.metrics in the yaml. This built-in
    # default is the full Step7 set — all 15 sub-metrics, in the same order as
    # sd-lake's stored statistical_similarity_*.json — with the Step7 parameter
    # values baked in. A --config run passes whatever the yaml lists instead
    # (configs/step7_pf_*.yaml carry the same 15 in the same order, and supply
    # the frozen per-variant MMD gamma the built-in default leaves as None).
    if statistical_metrics is None:
        statistical_metrics = [
            {"name": "table_structure", "parameters": {}},
            {"name": "semantic_structure", "parameters": {}},
            {"name": "boundary_adherence", "parameters": {"target_columns": None}},
            {"name": "category_adherence", "parameters": {"target_columns": None}},
            {"name": "alpha_precision", "parameters": {}},
            {"name": "prdc_score", "parameters": {"nearest_k": 5}},
            # Custom CPU Sinkhorn OT (geomloss), NOT synthcity's WassersteinDistance.
            {"name": "wasserstein_distance", "parameters": {}},
            # Corrected MMD (not synthcity's degenerate one). gamma=None here ->
            # the metric falls back to a per-run median-heuristic gamma; a
            # --config run passes the frozen per-variant value.
            {"name": "maximum_mean_discrepancy", "parameters": {"kernel": "rbf", "gamma": None}},
            # synthetic_sample_size stays None — that is what produced the sd-lake
            # ground truth (and it is ~67s/model at that setting).
            {"name": "new_row_synthesis", "parameters": {"numerical_match_tolerance": 0.01, "synthetic_sample_size": None}},
            {"name": "jensenshannon_synthcity", "parameters": {"normalize": True, "n_histogram_bins": 10}},
            {"name": "jensenshannon_syndat", "parameters": {"n_unique_threshold": 10}},
            {"name": "jensenshannon_nannyml", "parameters": {}},
            {"name": "ks_complement", "parameters": {"target_columns": None}},
            {"name": "tv_complement", "parameters": {"target_columns": None}},
            # QualityReport + working-tree SpearmanColumnPairTrends override. The
            # config carries max_display_cols as a sibling of `name` (not under
            # parameters), so it never reaches the constructor -> the block always
            # reports max_display_cols=10, matching sd-lake.
            {"name": "sdmetrics_quality", "parameters": {}},
        ]

    # Run-directory wrapper: every run nests under output_dir/<run_name>/ so
    # the tree maps 1:1 onto a sd-lake run dir (data/ models/ metrics/ under a
    # <tag>_<dseed>_<library>_<model>_mseed<seed> folder). run_name is taken
    # verbatim when given, else assembled from experiment_tag (+ the data
    # identifiers).
    if run_name is None:
        run_name = _run_name(training_file, library, model_type, seed, tag=experiment_tag)
    output_dir = str(Path(output_dir) / run_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {output_dir}")

    run_params = {
        "training_file": training_file,
        "population_file": population_file,
        "metadata_file": metadata_file,
        "encoding_config_file": encoding_config_file,
        "model_type": model_type,
        "library": library,
        "seed": seed,
        "parameters": parameters,
        "reference_file": reference_file,
        "n_samples": n_samples,
        "post_process_method": post_process_method,
        "knn_neighbors": knn_neighbors,
        "distance_metric": distance_metric,
        "fallback": fallback,
    }

    fit_out = fit_encoder(
        population_file=population_file,
        metadata_file=metadata_file,
        encoding_config_file=encoding_config_file,
        encoder_dir=encoder_dir,
    )

    base_name = f"{model_type}_{Path(training_file).stem}_{seed}"
    encode_out = encode_data(
        training_file=training_file,
        metadata_file=metadata_file,
        encoder_path=fit_out["encoder_path"],
        base_name=base_name,
        output_dir=output_dir,
    )

    train_out = train_sdg(
        encoded_training_path=encode_out["encoded_training"],
        training_file=training_file,
        model_type=model_type,
        library=library,
        seed=seed,
        parameters=parameters,
        run_params=run_params,
        base_name=base_name,
        output_dir=output_dir,
    )

    generate_out = generate_synthetic(
        model_path=train_out["model_path"],
        training_encoder_path=encode_out["training_encoder"],
        training_file=training_file,
        metadata_file=metadata_file,
        reference_file=reference_file,
        encoded_training_path=encode_out["encoded_training"],
        model_type=model_type,
        library=library,
        seed=seed,
        base_name=base_name,
        output_dir=output_dir,
        n_samples=n_samples,
        post_process_method=post_process_method,
        knn_neighbors=knn_neighbors,
        distance_metric=distance_metric,
        fallback=fallback,
        force_generate=force_generate,
    )

    eval_encode_out = encode_evaluation(
        reference_file=reference_file,
        metadata_file=metadata_file,
        encoding_config_file=encoding_config_file,
        synthetic_decoded_path=generate_out["decoded_path"],
        base_name=base_name,
        output_dir=output_dir,
        force=force_generate,
    )

    halluc_out = hallucination_evaluation(
        population_file=population_file,
        training_file=training_file,
        synthetic_decoded_path=eval_encode_out["decoded_synthetic"],
        metadata_file=metadata_file,
        base_name=base_name,
        output_dir=output_dir,
        num_bins=hallucination_num_bins,
        force=force_generate,
    )

    # TSTR needs the dseed folder (lgbm_cv_*.json + the one *test*.csv). Today
    # that is simply the training file's parent; a future recursive flow must
    # keep this pinned to the *original* dseed folder, not gen-N's synthetic.
    tstr_out = tstr_evaluation(
        dseed_dir=str(Path(training_file).parent),
        synth_decoded_path=generate_out["decoded_path"],
        base_name=base_name,
        output_dir=output_dir,
        seed=seed,
        force=force_generate,
    )

    privacy_out = privacy_evaluation(
        metadata_file=metadata_file,
        population_file=population_file,
        training_file=training_file,
        reference_decoded_path=eval_encode_out["decoded_reference"],
        synthetic_decoded_path=eval_encode_out["decoded_synthetic"],
        base_name=base_name,
        output_dir=output_dir,
        qi_columns=privacy_qi_columns,
        force=force_generate,
    )

    detection_out = detection_evaluation(
        metadata_file=metadata_file,
        encoding_config_file=encoding_config_file,
        encoded_reference_path=eval_encode_out["encoded_reference"],
        encoded_synthetic_path=eval_encode_out["encoded_synthetic"],
        base_name=base_name,
        output_dir=output_dir,
        methods=detection_methods,
        common_params=detection_common_params,
        force=force_generate,
    )

    statistical_out = statistical_similarity(
        metadata_file=metadata_file,
        encoding_config_file=encoding_config_file,
        encoded_reference_path=eval_encode_out["encoded_reference"],
        encoded_synthetic_path=eval_encode_out["encoded_synthetic"],
        decoded_reference_path=eval_encode_out["decoded_reference"],
        decoded_synthetic_path=eval_encode_out["decoded_synthetic"],
        base_name=base_name,
        output_dir=output_dir,
        metrics_config=statistical_metrics,
        force=force_generate,
    )

    return {
        "run_dir": output_dir,
        "fit_encoder": fit_out,
        "encode_data": encode_out,
        "train_sdg": train_out,
        "generate_synthetic": generate_out,
        "encode_evaluation": eval_encode_out,
        "hallucination": halluc_out,
        "tstr": tstr_out,
        "privacy": privacy_out,
        "detection": detection_out,
        "statistical": statistical_out,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
#
# Every setting can come from either a CLI flag or an optional --config YAML
# file; a CLI flag always wins when both are given. The config file mirrors
# sdpype's params_step7_*.yaml nested layout (experiment / sdg / data /
# encoding / generation / post_processing / evaluation) — see
# configs/step7_pf_all.yaml. It is a plain yaml.safe_load: no OmegaConf, no
# interpolation/templating, no config-group composition (that machinery is
# what the scaffolding step dropped along with Hydra); the Hydra-only
# experiment.name / tags templates are omitted. The evaluation block
# (a 15-entry statistical-metrics list, per-metric parameters, ...) has no
# sane CLI-flag shape, which is why the file exists — but every flag below
# still works standalone with zero config file.

VALID_MODEL_TYPES = {"arf", "ctgan", "ddpm", "rtvae", "nflow"}
VALID_POST_PROCESS_METHODS = {"knn", "weighted", "random", "none"}
VALID_FALLBACKS = {"knn", "weighted", "random"}

# Hardcoded fallback defaults, used only when a setting is given by neither
# the CLI nor --config.
_DEFAULTS = {
    "library": "synthcity",
    "seed": 42,
    "params": {},
    "post_process_method": "knn",
    "knn_neighbors": 5,
    "distance_metric": "hamming",
    "fallback": "weighted",
    "force_generate": False,
    "hallucination_num_bins": 20,
    "encoder_dir": "outputs/sdg_runs/encoders",
    "output_dir": "outputs/sdg_runs",
}


def _load_config(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


_MISSING = object()


def _cfg_get(config: dict, dotted: str, default=_MISSING):
    """Walk a dotted path into a nested config dict (mirrors sdpype's
    params.yaml layout, e.g. 'data.training_file', 'sdg.model_type',
    'evaluation.hallucination.num_bins'). Returns default if any level is
    absent; a key that is present but explicitly null returns None."""
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _pick(cli_value, config: dict, dotted: str, default=None):
    """CLI value wins if given; else the config's dotted-path value if the
    path exists (even if null); else default."""
    if cli_value is not None:
        return cli_value
    val = _cfg_get(config, dotted, _MISSING)
    return default if val is _MISSING else val


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SDG pipeline (encode -> train -> generate -> encode_evaluation -> hallucination -> tstr -> privacy -> detection)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Optional path to a plain YAML config file; CLI flags below override it")
    parser.add_argument("--training-file", default=None, help="Path to training CSV")
    parser.add_argument("--population-file", default=None, help="Path to population CSV")
    parser.add_argument("--metadata-file", default=None, help="Path to metadata JSON")
    parser.add_argument("--encoding-config", default=None, help="Path to encoding YAML")
    parser.add_argument("--model-type", default=None, choices=sorted(VALID_MODEL_TYPES), help="Synthcity generator to train")
    parser.add_argument("--library", default=None, help="Generator library (only 'synthcity' is implemented; kept as a real seam for future libraries)")
    parser.add_argument("--seed", type=int, default=None, help="Model seed")
    parser.add_argument("--params", default=None, help="Inline JSON string of model hyperparameter overrides")
    parser.add_argument("--reference-file", default=None, help="Reference CSV for auto-sizing --n-samples (default: same as --training-file, matching production configs)")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of synthetic rows to generate (default: auto-size from --reference-file)")
    parser.add_argument("--post-process-method", default=None, choices=sorted(VALID_POST_PROCESS_METHODS), help="Invalid-category fixing method ('none' disables post-processing)")
    parser.add_argument("--knn-neighbors", type=int, default=None, help="Neighbors for the 'knn' post-process method")
    parser.add_argument("--distance-metric", default=None, help="Distance metric for the 'knn' post-process method")
    parser.add_argument("--fallback", default=None, choices=sorted(VALID_FALLBACKS), help="Fallback method when a column is 100%% invalid")
    parser.add_argument("--force-generate", action="store_true", default=None, help="Regenerate synthetic data even if cached output exists")
    parser.add_argument("--hallucination-num-bins", type=int, default=None, help="Bins for the hallucination metric's numerical quantisation (or evaluation.hallucination.num_bins in --config)")
    parser.add_argument("--privacy-qi-columns", default=None, help="Comma-separated quasi-identifier columns for k-anonymity (or evaluation.privacy.metrics[0].parameters.qi_columns in --config; default: pf_all set)")
    parser.add_argument("--encoder-dir", default=None, help="Shared population-encoder cache dir")
    parser.add_argument("--output-dir", default=None, help="Root output directory; every run nests under it as <run-name>/")
    parser.add_argument("--run-name", default=None, help="Run-directory name under --output-dir (default: <tag>_<dseed>_<library>_<model>_mseed<seed>, or the same without the <tag> segment when experiment.tag is unset)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = _load_config(args.config)

    # Nested config paths mirror sdpype's params_step7_*.yaml exactly; every
    # CLI flag still overrides the matching key.
    training_file = _pick(args.training_file, config, "data.training_file")
    population_file = _pick(args.population_file, config, "data.population_file")
    metadata_file = _pick(args.metadata_file, config, "data.metadata_file")
    encoding_config_file = _pick(args.encoding_config, config, "encoding.config_file")
    model_type = _pick(args.model_type, config, "sdg.model_type")
    library = _pick(args.library, config, "sdg.library", _DEFAULTS["library"])
    seed = _pick(args.seed, config, "experiment.seed", _DEFAULTS["seed"])
    reference_file = _pick(args.reference_file, config, "data.reference_file")
    n_samples = _pick(args.n_samples, config, "generation.n_samples")

    _pp = "post_processing.fix_invalid_categories."
    post_process_method = _pick(args.post_process_method, config, _pp + "method", _DEFAULTS["post_process_method"])
    # `enabled: false` in the config is shorthand for method "none" (only when
    # --post-process-method wasn't given on the CLI).
    if args.post_process_method is None and _cfg_get(config, _pp + "enabled", True) is False:
        post_process_method = "none"
    knn_neighbors = _pick(args.knn_neighbors, config, _pp + "knn_neighbors", _DEFAULTS["knn_neighbors"])
    distance_metric = _pick(args.distance_metric, config, _pp + "distance_metric", _DEFAULTS["distance_metric"])
    fallback = _pick(args.fallback, config, _pp + "fallback", _DEFAULTS["fallback"])

    force_generate = _pick(args.force_generate, config, "force_generate", _DEFAULTS["force_generate"])
    encoder_dir = _pick(args.encoder_dir, config, "encoder_dir", _DEFAULTS["encoder_dir"])
    output_dir = _pick(args.output_dir, config, "output_dir", _DEFAULTS["output_dir"])
    # experiment.tag (Step7pfa / Step7pfp) drives the sd-lake-style run-dir
    # wrapper; --run-name overrides the assembled name outright.
    experiment_tag = _cfg_get(config, "experiment.tag", None)
    run_name = args.run_name

    if args.hallucination_num_bins is not None:
        hallucination_num_bins = args.hallucination_num_bins
    else:
        hallucination_num_bins = _cfg_get(
            config, "evaluation.hallucination.num_bins", _DEFAULTS["hallucination_num_bins"]
        )

    if args.privacy_qi_columns is not None:
        privacy_qi_columns = [c.strip() for c in args.privacy_qi_columns.split(",") if c.strip()]
    else:
        _priv_metrics = _cfg_get(config, "evaluation.privacy.metrics", None) or []
        privacy_qi_columns = (
            (_priv_metrics[0].get("parameters", {}) or {}).get("qi_columns")
            if _priv_metrics else None
        )  # None -> sdg_pipeline() falls back to the pf_all QI set

    # Detection is config-only (no flat-CLI shape); None -> sdg_pipeline()
    # falls back to the Step7 methods / common_params.
    detection_methods = _cfg_get(config, "evaluation.detection_evaluation.methods", None)
    detection_common_params = _cfg_get(config, "evaluation.detection_evaluation.common_params", None)

    # Statistical similarity is config-only too; None -> sdg_pipeline() falls
    # back to its built-in list (the full Step7 set of 15 sub-metrics).
    statistical_metrics = _cfg_get(config, "evaluation.statistical_similarity.metrics", None)

    # --params is a JSON string on the CLI; sdg.parameters in the config is
    # already a native mapping (YAML parses nested dicts directly).
    if args.params is not None:
        parameters = json.loads(args.params)
    else:
        _p = _cfg_get(config, "sdg.parameters", None)
        parameters = _DEFAULTS["params"] if _p is None else _p

    missing = [
        name for name, value in [
            ("--training-file", training_file),
            ("--population-file", population_file),
            ("--metadata-file", metadata_file),
            ("--encoding-config", encoding_config_file),
            ("--model-type", model_type),
        ] if value is None
    ]
    if missing:
        raise SystemExit(
            f"Missing required setting(s) (pass via CLI flag or --config): {', '.join(missing)}"
        )
    if model_type not in VALID_MODEL_TYPES:
        raise SystemExit(f"Invalid model_type {model_type!r} (choices: {sorted(VALID_MODEL_TYPES)})")
    if post_process_method not in VALID_POST_PROCESS_METHODS:
        raise SystemExit(f"Invalid post_process_method {post_process_method!r} (choices: {sorted(VALID_POST_PROCESS_METHODS)})")
    if fallback not in VALID_FALLBACKS:
        raise SystemExit(f"Invalid fallback {fallback!r} (choices: {sorted(VALID_FALLBACKS)})")

    sdg_pipeline(
        training_file=training_file,
        population_file=population_file,
        metadata_file=metadata_file,
        encoding_config_file=encoding_config_file,
        model_type=model_type,
        library=library,
        seed=seed,
        parameters=parameters,
        reference_file=reference_file,
        n_samples=n_samples,
        post_process_method=post_process_method,
        knn_neighbors=knn_neighbors,
        distance_metric=distance_metric,
        fallback=fallback,
        force_generate=bool(force_generate),
        hallucination_num_bins=hallucination_num_bins,
        privacy_qi_columns=privacy_qi_columns,
        detection_methods=detection_methods,
        detection_common_params=detection_common_params,
        statistical_metrics=statistical_metrics,
        experiment_tag=experiment_tag,
        run_name=run_name,
        encoder_dir=encoder_dir,
        output_dir=output_dir,
    )
