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

This checkpoint wires only fit_encoder + encode_data (the deterministic
encode stage) so it can be verified byte-exact against real production
output before train_sdg/generate_synthetic (which involve stochastic model
training) are added.

Usage:
    python flows/sdg_flow.py \\
        --training-file   path/to/..._training.csv \\
        --population-file path/to/..._population.csv \\
        --metadata-file   path/to/..._metadata.json \\
        --encoding-config path/to/..._encoding.yaml \\
        --seed 28657
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from prefect import flow, task

from sdg_core.encoding import RDTDatasetEncoder, load_encoding_config
from sdg_core.hashing import calculate_file_hash
from sdg_core.metadata import load_csv_with_metadata


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


# ---------------------------------------------------------------------------
# Flow (encode-only checkpoint — train_sdg/generate_synthetic added next)
# ---------------------------------------------------------------------------

@flow(name="sdg-pipeline")
def sdg_pipeline(
    training_file: str,
    population_file: str,
    metadata_file: str,
    encoding_config_file: str,
    seed: int = 42,
    encoder_dir: str = "outputs/sdg_runs/encoders",
    output_dir: str = "outputs/sdg_runs",
) -> dict:
    fit_out = fit_encoder(
        population_file=population_file,
        metadata_file=metadata_file,
        encoding_config_file=encoding_config_file,
        encoder_dir=encoder_dir,
    )

    base_name = f"{Path(training_file).stem}_{seed}"
    encode_out = encode_data(
        training_file=training_file,
        metadata_file=metadata_file,
        encoder_path=fit_out["encoder_path"],
        base_name=base_name,
        output_dir=output_dir,
    )

    return {"fit_encoder": fit_out, "encode_data": encode_out}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SDG encode stage (fit_encoder + encode_data)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training-file", required=True, help="Path to training CSV")
    parser.add_argument("--population-file", required=True, help="Path to population CSV")
    parser.add_argument("--metadata-file", required=True, help="Path to metadata JSON")
    parser.add_argument("--encoding-config", required=True, help="Path to encoding YAML")
    parser.add_argument("--seed", type=int, default=42, help="Model seed (used in output naming)")
    parser.add_argument("--encoder-dir", default="outputs/sdg_runs/encoders", help="Shared population-encoder cache dir")
    parser.add_argument("--output-dir", default="outputs/sdg_runs", help="Per-run output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    sdg_pipeline(
        training_file=args.training_file,
        population_file=args.population_file,
        metadata_file=args.metadata_file,
        encoding_config_file=args.encoding_config,
        seed=args.seed,
        encoder_dir=args.encoder_dir,
        output_dir=args.output_dir,
    )
