import os
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

"""
SDG Pipeline — Prefect flow (encode → train → generate)
=========================================================

Ported from sdpype's flows/sdg_flow.py. Only the first 5 of its 13 tasks are
in scope for this repo: resolve_config (replaced entirely by argparse — no
Hydra/YAML template resolution needed), fit_encoder, encode_data, train_sdg,
generate_synthetic. Everything from sdg_flow.py's own "# Evaluation tasks"
marker onward (statistical/privacy/detection/hallucination/TSTR + report) is
a separate, later step — not ported here.

This checkpoint adds generate_synthetic on top of the already-verified
fit_encoder + encode_data + train_sdg — the full encode → train → generate
chain (arf/ctgan/ddpm/rtvae/nflow, synthcity-only) is now wired.

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
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from prefect import flow, task

from sdg_core.encoding import RDTDatasetEncoder, load_encoding_config
from sdg_core.generation import apply_post_processing
from sdg_core.hashing import calculate_file_hash
from sdg_core.metadata import load_csv_with_metadata
from sdg_core.serialization import create_model_metadata, load_model, save_model
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
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading encoder: {encoder_path}")
    encoder = RDTDatasetEncoder.load(encoder_path)

    print(f"Loading training data: {training_file}")
    training_data = load_csv_with_metadata(training_file, metadata_file)
    print(f"  Training shape: {training_data.shape}")

    print("Transforming training data ...")
    encoded_training = encoder.transform(training_data)
    print(f"  {training_data.shape[1]} columns → {encoded_training.shape[1]}")

    decoded_training = encoder.reverse_transform(encoded_training)

    encoded_path = output_dir / f"encoded_training_{base_name}.csv"
    decoded_path = output_dir / f"decoded_training_{base_name}.csv"
    # Copy encoder to a per-run path (mirrors production's per-experiment copy)
    exp_encoder_path = output_dir / f"training_encoder_{base_name}.pkl"

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

    metrics_path = output_dir / f"metrics_encoding_{base_name}.json"
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
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"sdg_model_{base_name}.pkl"
    metrics_path = output_dir / f"metrics_training_{base_name}.json"

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
    output_dir.mkdir(parents=True, exist_ok=True)

    encoded_path = output_dir / f"synthetic_{base_name}_encoded.csv"
    decoded_path = output_dir / f"synthetic_{base_name}_decoded.csv"
    metrics_path = output_dir / f"metrics_generation_{base_name}.json"

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
# Flow — full encode → train → generate chain
# ---------------------------------------------------------------------------

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
    encoder_dir: str = "outputs/sdg_runs/encoders",
    output_dir: str = "outputs/sdg_runs",
) -> dict:
    # Production configs set reference_file == training_file (confirmed in
    # params_step7_pf_pilgram.yaml) — defaulting here matches real behavior,
    # not a shortcut.
    if reference_file is None:
        reference_file = training_file

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

    return {
        "fit_encoder": fit_out,
        "encode_data": encode_out,
        "train_sdg": train_out,
        "generate_synthetic": generate_out,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SDG pipeline (fit_encoder + encode_data + train_sdg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training-file", required=True, help="Path to training CSV")
    parser.add_argument("--population-file", required=True, help="Path to population CSV")
    parser.add_argument("--metadata-file", required=True, help="Path to metadata JSON")
    parser.add_argument("--encoding-config", required=True, help="Path to encoding YAML")
    parser.add_argument("--model-type", required=True, choices=["arf", "ctgan", "ddpm", "rtvae", "nflow"], help="Synthcity generator to train")
    parser.add_argument("--library", default="synthcity", help="Generator library (only 'synthcity' is implemented; kept as a real seam for future libraries)")
    parser.add_argument("--seed", type=int, default=42, help="Model seed")
    parser.add_argument("--params", default="{}", help="Inline JSON string of model hyperparameter overrides")
    parser.add_argument("--reference-file", default=None, help="Reference CSV for auto-sizing --n-samples (default: same as --training-file, matching production configs)")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of synthetic rows to generate (default: auto-size from --reference-file)")
    parser.add_argument("--post-process-method", default="knn", choices=["knn", "weighted", "random", "none"], help="Invalid-category fixing method ('none' disables post-processing)")
    parser.add_argument("--knn-neighbors", type=int, default=5, help="Neighbors for the 'knn' post-process method")
    parser.add_argument("--distance-metric", default="hamming", help="Distance metric for the 'knn' post-process method")
    parser.add_argument("--fallback", default="weighted", choices=["knn", "weighted", "random"], help="Fallback method when a column is 100% invalid")
    parser.add_argument("--force-generate", action="store_true", help="Regenerate synthetic data even if cached output exists")
    parser.add_argument("--encoder-dir", default="outputs/sdg_runs/encoders", help="Shared population-encoder cache dir")
    parser.add_argument("--output-dir", default="outputs/sdg_runs", help="Per-run output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    parameters = json.loads(args.params)

    sdg_pipeline(
        training_file=args.training_file,
        population_file=args.population_file,
        metadata_file=args.metadata_file,
        encoding_config_file=args.encoding_config,
        model_type=args.model_type,
        library=args.library,
        seed=args.seed,
        parameters=parameters,
        reference_file=args.reference_file,
        n_samples=args.n_samples,
        post_process_method=args.post_process_method,
        knn_neighbors=args.knn_neighbors,
        distance_metric=args.distance_metric,
        fallback=args.fallback,
        force_generate=args.force_generate,
        encoder_dir=args.encoder_dir,
        output_dir=args.output_dir,
    )
