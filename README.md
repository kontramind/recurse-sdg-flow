# recurse-sdg-flow

Recursive synthetic data generation pipeline, built with Prefect.

This is a clean, standalone port of the SDG pipeline used in [paper reference
TBD]. It is being assembled incrementally:

1. ✅ Project scaffolding
2. ✅ Data-prep CLI (`prepare_step7.py`) — builds train/test/population splits
   + encoding config from a source population file
3. ✅ TRTR baseline (`flows/lgbm_cv_flow.py`) — LightGBM + Optuna HPO on real
   data, the reference point TSTR is later measured against
4. ⏳ Minimal Prefect pipeline: encode → train → generate
5. ⏳ Recursive multi-generation loop
6. ⏳ Evaluation stages (statistical, privacy, detection, hallucination, TSTR)

## Setup

```bash
uv sync
```

## Usage

### Data prep (`prepare_step7.py`)

Builds a `train`/`test`/`population` CSV split, an RDT encoding config, and an
SDV-style metadata JSON from a source population file, for one or more data
seeds (`--dseeds`). Two column-config revisions are available:

- `pf_all` — 23 columns, the full feature set
- `pf_pilgram` — 13-column clinical-core subset

```bash
# Prepare a single dseed, pf_all revision, default sample size (10,000/split)
uv run python3 prepare_step7.py --revision pf_all --dseeds 1597

# Multiple dseeds in one run (Cartesian over --dseeds is not needed — just list them)
uv run python3 prepare_step7.py --revision pf_all --dseeds 1597 196418 14930352

# pf_pilgram revision, smaller sample size, custom output root
uv run python3 prepare_step7.py --revision pf_pilgram --dseeds 1597 --sample 500 --output /tmp/out/

# Point at your own population file instead of the default ../rd-lake/population_fixed.xlsx
uv run python3 prepare_step7.py --revision pf_all --dseeds 1597 --source /path/to/your_population.xlsx
```

Output lands under `<output>/Step7/dseed<seed>_rev_<revision>/`, e.g.:

```
Step7/dseed1597_rev_pf_all/
├── data_sample10000_dseed1597_rev_pf_all_training.csv
├── data_sample10000_dseed1597_rev_pf_all_test.csv
├── data_sample10000_dseed1597_rev_pf_all_population.csv
├── data_sample10000_dseed1597_rev_pf_all_encoding.yaml
└── data_sample10000_dseed1597_rev_pf_all_metadata.json
```

The command is idempotent per dseed/revision — it skips (with a `[SKIP]`
message) if the target folder already exists, rather than overwriting it.

### TRTR baseline (`flows/lgbm_cv_flow.py`)

TRTR ("Train Real, Test Real") is the real-data reference point that a later
TSTR ("Train Synthetic, Test Real") evaluation is measured against. It's
produced by running the LightGBM + Optuna binary-classification flow directly
on a dseed's real `training`/`test` CSVs (the output of `prepare_step7.py`
above): Optuna runs Bayesian HPO (default 50 trials, 5-fold inner CV,
optimizing AUROC) to pick hyperparameters once, trains a final model, and
evaluates it on the held-out real test set.

```bash
uv run python3 flows/lgbm_cv_flow.py \
  --dataset      ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_training.csv \
  --test-dataset ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_test.csv \
  --target readmission --n-trials 50 --n-folds 5 --seed 42 \
  --output-dir ../rd-lake/Step7/dseed1597_rev_pf_all/
```

This writes `lgbm_cv_<timestamp>.{json,md,pkl}` plus
`shap_importance_<timestamp>.{csv,png}` into `--output-dir`, and — since
`--test-dataset` was given — also copies the metrics JSON next to the test
CSV in the dseed folder (matching the layout a later TSTR stage will expect
to auto-discover it from). The JSON's `best_params` and decision `threshold`
are meant to be **frozen and reused** for every TSTR run on that dseed —
don't re-tune per generation, or you'd conflate synthetic-data quality
decline with the optimizer landing on different hyperparameters.

Fewer trials for a quick smoke test: `--n-trials 5` finishes in well under a
minute; the default `--n-trials 50` can take up to ~1 hour depending on
hardware, since Optuna trials run sequentially (each doing its own 5-fold
CV with early-stopped LightGBM fits).

**Reproducibility:** this was verified against `sdpype`'s original production
runs for dseed 1597 (both `pf_all` and `pf_pilgram`) and reproduced every
metric, the Optuna CV score, and all best hyperparameters as an exact match
down to full float precision — deterministic given identical library
versions (see `pyproject.toml`'s exact pins).
