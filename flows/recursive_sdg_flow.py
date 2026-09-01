"""
Recursive SDG pipeline — drives N sequential synthetic-data generations.

Each generation trains on the previous generation's decoded synthetic output;
the population, reference, metadata and encoding config stay fixed from gen 0.

Ported from sdpype's flows/recursive_sdg_flow.py, trimmed to the core loop:
  * recursive_sdg_pipeline — the loop over sdg_pipeline().
  * recursive_sdg_sweep    — a sequential Cartesian sweep over model types x
                             model seeds (sdpype also swept dseed-dirs; here
                             the dataset comes from --config, so a multi-dataset
                             sweep is a shell loop over configs, exactly as
                             sdpype's own pilgram handover script did it).
  * _adapt_hallucination_json — injects sd-lake's recursion-layer blocks
                             (unified_metrics / complexity_metrics) into each
                             generation's hallucination JSON.
sdpype's publish_to_sdlake + trace_chain machinery is out of scope — the port
matches sdpype's experiments/ flow output, not sd-lake's promoted layout.

Usage:
    uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml
    uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml --n-generations 20
    uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml --mseed 28657
    uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml \\
        --model-type arf --model-type ddpm --model-type ctgan --model-type rtvae \\
        --mseed 987 --mseed 28657 --mseed 5702887 --mseed 24157817 --mseed 39088169 \\
        --n-generations 20
    uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml --no-report
    uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml --force-generate
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from prefect import flow

# Ensure the repo root is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from flows.sdg_flow import (  # noqa: E402
    VALID_MODEL_TYPES,
    _load_config,
    _run_name,
    resolve_pipeline_settings,
    sdg_pipeline,
    validate_pipeline_settings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adapt_hallucination_json(path: Path) -> bool:
    """Add sd-lake's recursion-layer blocks to a hallucination JSON, in place.

    sdpype's sdg_flow.py hallucination task (and the port's) writes only the
    raw block — metadata / dataset_statistics / binning / metrics /
    execution_time. sd-lake's published hallucination JSONs carry two extra
    blocks, injected by recursive_sdg_flow._adapt_hallucination_json during
    sd-lake promotion. The port has no promote step, so the recursive flow
    adds them directly to each generation's hallucination_<base>.json — a
    standalone sdg_flow.py run still omits them, exactly as sdpype's does.

    Both blocks are row-count proxies:
      unified_metrics    — keeps rate_pct per FR/HR key.
      complexity_metrics — population / training / synthetic row counts
                           (reference == training; the Prefect format doesn't
                           track reference separately).

    Returns True if the file was rewritten, False if missing / malformed /
    already adapted.
    """
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    if "unified_metrics" in data and "complexity_metrics" in data:
        return False
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return False

    unified = {}
    for key in ("TotalFR", "NovelFR", "MemorizedFR", "HR"):
        if isinstance(metrics.get(key), dict):
            unified[key] = {"rate_pct": metrics[key].get("rate_pct", 0.0)}

    stats = data.get("dataset_statistics", {})
    pop_rows = stats.get("population", {}).get("rows", 0)
    trn_rows = stats.get("training", {}).get("rows", 0)
    syn_rows = stats.get("synthetic", {}).get("rows", 0)
    ref_rows = trn_rows  # reference not separately tracked in the Prefect format

    complexity = {
        "population": {"total_complexity": pop_rows},
        "training": {"total_complexity": trn_rows},
        "reference": {"total_complexity": ref_rows},
        "synthetic": {"total_complexity": syn_rows},
        "comparisons": {
            "synthetic_vs_population_ratio": syn_rows / pop_rows if pop_rows else 0,
            "synthetic_vs_training_ratio": syn_rows / trn_rows if trn_rows else 0,
            "synthetic_vs_reference_ratio": syn_rows / ref_rows if ref_rows else 0,
        },
    }

    data["unified_metrics"] = unified
    data["complexity_metrics"] = complexity
    path.write_text(json.dumps(data, indent=2))
    return True


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------

@flow(name="recursive-sdg-pipeline", log_prints=True)
def recursive_sdg_pipeline(settings: dict, n_generations: int = 3) -> list:
    """Run sdg_pipeline() N times, feeding each generation's decoded synthetic
    CSV as the next generation's training data.

    `settings` is a resolve_pipeline_settings() dict. Population, reference,
    metadata and encoding config are pinned from gen 0 for the whole chain;
    only training_file advances. Every generation writes into one flat run dir
    (mirroring sd-lake — one chain folder, gen_<k> in every base_name).

    Returns a list of per-generation sdg_pipeline() result dicts.
    """
    gen0_training_file = settings["training_file"]
    gen0_stem = Path(gen0_training_file).stem
    library = settings["library"]
    model_type = settings["model_type"]
    seed = settings["seed"]
    tag = settings.get("experiment_tag")

    # Fixed for the whole chain.
    reference_file = settings.get("reference_file") or gen0_training_file
    chain_run_name = _run_name(gen0_training_file, library, model_type, seed, tag=tag)
    # TSTR always resolves lgbm_cv_*.json + the one *test*.csv against the
    # ORIGINAL dseed folder, never gen-N's synthetic CSV parent.
    tstr_dseed_dir = str(Path(gen0_training_file).parent)

    print(f"\nRecursive chain: {chain_run_name}")
    print(f"  model={model_type}  seed={seed}  generations={n_generations}")
    print(f"  gen 0 training file: {gen0_training_file}")

    results = []
    training_file = gen0_training_file
    for gen_idx in range(n_generations):
        print(f"\n{'=' * 60}")
        print(f"  GENERATION {gen_idx} / {n_generations - 1}")
        print(f"{'=' * 60}\n")

        gen_settings = {
            **settings,
            "training_file": training_file,
            "reference_file": reference_file,
            "run_name": chain_run_name,
            "base_name": f"{model_type}_{gen0_stem}_gen_{gen_idx}_{seed}",
            "tstr_dseed_dir": tstr_dseed_dir,
        }
        result = sdg_pipeline(**gen_settings)
        results.append(result)

        # sd-lake parity: give this generation's hallucination JSON the
        # unified_metrics / complexity_metrics blocks.
        halluc_path = (result.get("hallucination") or {}).get("metrics_path")
        if halluc_path:
            _adapt_hallucination_json(Path(halluc_path))

        # Next generation trains on this generation's decoded synthetic data —
        # the raw generate output, NOT encode_evaluation's decoded_synthetic
        # (matches sdpype).
        training_file = result["generate_synthetic"]["decoded_path"]

    print(f"\n{'=' * 60}")
    print(f"  RECURSIVE CHAIN COMPLETE — {n_generations} generation(s), "
          f"model={model_type}, seed={seed}")
    for i, r in enumerate(results):
        print(f"  gen {i}: {r['generate_synthetic']['decoded_path']}")
    print(f"{'=' * 60}\n")

    return results


@flow(name="recursive-sdg-sweep", log_prints=True)
def recursive_sdg_sweep(
    settings: dict,
    n_generations: int = 3,
    model_types: Optional[list] = None,
    mseeds: Optional[list] = None,
) -> list:
    """Cartesian sweep over model types x model seeds — each combination runs
    a full N-generation recursive chain, one after another.

    model_types / mseeds each default to the single value already in
    `settings`. Returns a flat list of
    {"model_type", "seed", "generations": [...]} dicts.
    """
    models = model_types or [settings["model_type"]]
    seeds = mseeds or [settings["seed"]]
    total = len(models) * len(seeds)

    sweep_results = []
    combo = 0
    for mt in models:
        for s in seeds:
            combo += 1
            print(f"\n{'#' * 60}")
            print(f"  CHAIN {combo}/{total} — model={mt}, seed={s}, "
                  f"{n_generations} generation(s)")
            print(f"{'#' * 60}\n")
            chain_settings = {**settings, "model_type": mt, "seed": s}
            gens = recursive_sdg_pipeline(chain_settings, n_generations=n_generations)
            sweep_results.append({"model_type": mt, "seed": s, "generations": gens})

    print(f"\n{'#' * 60}")
    print(f"  SWEEP COMPLETE — {len(models)} model(s) x {len(seeds)} seed(s) "
          f"x {n_generations} generation(s)")
    for r in sweep_results:
        print(f"  [{r['model_type']}, seed={r['seed']}]: {len(r['generations'])} generation(s)")
    print(f"{'#' * 60}\n")
    return sweep_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recursive SDG pipeline — N sequential generations, each "
                    "trained on the previous generation's synthetic output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Plain YAML config (same shape as sdg_flow.py's --config): "
                        "dataset paths + evaluation config")
    p.add_argument("--n-generations", type=int, default=3,
                   help="Recursive generations per chain")
    p.add_argument("--model-type", action="append", dest="model_types", metavar="MODEL",
                   choices=sorted(VALID_MODEL_TYPES),
                   help="SDG model type; repeat for a Cartesian sweep "
                        "(--model-type arf --model-type ddpm). Default: sdg.model_type from --config")
    p.add_argument("--mseed", type=int, action="append", dest="mseeds", metavar="SEED",
                   help="Model seed; repeat for a Cartesian sweep "
                        "(--mseed 987 --mseed 28657). Default: experiment.seed from --config")
    p.add_argument("--seed", type=int, default=None,
                   help="Single model seed (shorthand for one --mseed); ignored when --mseed is given")
    p.add_argument("--output-dir", default=None,
                   help="Root output directory (default: outputs/recursive_runs, "
                        "or output_dir from --config)")
    p.add_argument("--force-generate", action="store_true", default=None,
                   help="Regenerate synthetic data even if cached output exists")
    p.add_argument("--no-report", action="store_true", default=False,
                   help="Run evaluation but suppress the per-generation rich report")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = _load_config(args.config)

    model_types = args.model_types or None
    mseeds = args.mseeds or None
    if args.seed is not None and not mseeds:
        mseeds = [args.seed]

    # Fold a lone --model-type / --mseed into the resolved settings so both the
    # single-chain path and up-front validation work without a --config that
    # already names the model / seed. A multi-value sweep still overrides
    # model_type / seed per combination inside recursive_sdg_sweep.
    cli = {
        "model_type": model_types[0] if model_types else None,
        "seed": mseeds[0] if mseeds else None,
        "force_generate": args.force_generate,
        "report": not args.no_report,
        "output_dir": args.output_dir or "outputs/recursive_runs",
    }
    settings = resolve_pipeline_settings(config, cli)
    validate_pipeline_settings(settings)

    is_sweep = (model_types and len(model_types) > 1) or (mseeds and len(mseeds) > 1)
    if is_sweep:
        results = recursive_sdg_sweep(
            settings,
            n_generations=args.n_generations,
            model_types=model_types,
            mseeds=mseeds,
        )
        print("\nSweep outputs:")
        for r in results:
            print(f"  [{r['model_type']}, seed={r['seed']}]")
            for i, gen in enumerate(r["generations"]):
                print(f"    gen {i}: {gen['generate_synthetic']['decoded_path']}")
    else:
        results = recursive_sdg_pipeline(settings, n_generations=args.n_generations)
        print("\nGeneration outputs:")
        for i, gen in enumerate(results):
            print(f"  gen {i}: {gen['generate_synthetic']['decoded_path']}")
