# recurse-sdg-flow

Recursive synthetic data generation pipeline, built with Prefect.

This is a clean, standalone port of the SDG pipeline used in [paper reference
TBD]. It is being assembled incrementally:

1. ✅ Project scaffolding
2. ✅ Data-prep CLI (`prepare_step7.py`) — builds train/test/population splits
   + encoding config from a source population file
3. ✅ TRTR baseline (`flows/lgbm_cv_flow.py`) — LightGBM + Optuna HPO on real
   data, the reference point TSTR is later measured against
4. ✅ Minimal Prefect pipeline (`flows/sdg_flow.py`): encode → train → generate
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

### SDG pipeline (`flows/sdg_flow.py`)

Trains one of five synthcity-backed generators on a dseed's data and samples
synthetic data from it. Runs the same three stages the original DVC/Hydra
pipeline did — fit an encoder on the population data, encode the training
data, train, generate — now driven entirely by CLI flags instead of a
`params.yaml`.

```bash
uv run python3 flows/sdg_flow.py \
  --training-file   ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_training.csv \
  --population-file ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_population.csv \
  --metadata-file   ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_metadata.json \
  --encoding-config ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_encoding.yaml \
  --model-type arf --seed 28657 \
  --output-dir outputs/sdg_runs
```

Available generators (`--model-type`), all via synthcity's `Plugins()`:

| Model | Notes |
|---|---|
| `arf` | Adversarial Random Forests — fastest, tree-based |
| `ctgan` | Conditional GAN |
| `ddpm` | Diffusion model |
| `rtvae` | Robust-divergence VAE |
| `nflow` | Normalizing flow |

Only `library=synthcity` is implemented (`--library` defaults to it) — the
original pipeline also supported SDV- and synthpop-backed generators, but
neither was ever used to produce a reported result (confirmed by checking
every experiment folder in the production data lake), so they're out of
scope here. `--library` is kept as a real argument rather than hardcoded,
so adding another backend later is a contained change, not a rewrite.

Every setting above can also come from an optional `--config path/to/file.yaml`
instead of (or alongside) CLI flags — a CLI flag always wins when both are
given. This is a plain `yaml.safe_load`, not Hydra: no interpolation,
templating, or config-group composition, just a flat mapping of the same
option names (underscored: `training_file`, `model_type`, `post_process_method`,
etc.), e.g.:

```yaml
training_file: ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_training.csv
population_file: ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_population.csv
metadata_file: ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_metadata.json
encoding_config: ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_encoding.yaml
model_type: arf
seed: 28657
params: {}
```

```bash
uv run python3 flows/sdg_flow.py --config my_run.yaml
uv run python3 flows/sdg_flow.py --config my_run.yaml --seed 111   # CLI --seed wins over the file's
```

This exists for settings that don't have a sane CLI-flag shape — the
upcoming evaluation stages (statistical/privacy/detection metrics) have
deeply nested per-metric configuration that only really works as a config
file, not a wall of flags.

This writes, under `--output-dir`:

```
encoded_training_<base_name>.csv / decoded_training_<base_name>.csv     # dual-pipeline encode output
training_encoder_<base_name>.pkl                                        # per-run copy of the population encoder
sdg_model_<base_name>.pkl                                               # trained generator
synthetic_<base_name>_encoded.csv / synthetic_<base_name>_decoded.csv   # generated synthetic data
metrics_{encoding,training,generation}_<base_name>.json
```

where `<base_name>` is `<model-type>_<training-file-stem>_<seed>`. The
population encoder itself is cached separately under `--encoder-dir`
(default `outputs/sdg_runs/encoders/`), keyed by a hash of the population
file's content — fit once per population file and reused across every
dseed/model/seed that shares it, rather than refit per run.

Hyperparameters are passed as an inline JSON string via `--params` (default
`'{}'`, i.e. all model defaults) — e.g. `--params '{"n_iter": 500}'` for a
faster ctgan/ddpm/rtvae/nflow smoke test; `arf` has no iteration count to
shrink. `--n-samples` defaults to auto-sizing from `--reference-file`
(which itself defaults to `--training-file`, matching production configs);
pass an explicit count to override. `--post-process-method
{knn,weighted,random,none}` controls how out-of-vocabulary categorical
values a generator produces get fixed against the training data's real
categories (default `knn`, matching production; `none` disables this step).

**Reproducibility:** verified against `sdpype`'s original production runs
(dseed 1597 and 196418, `pf_all`, model seed 28657). The encode stage is
byte-exact, as expected (same deterministic RDT/scipy transforms as the
data-prep step). More surprisingly, **`arf`, `rtvae`, and `ctgan` all
reproduce their generated synthetic data byte-for-byte too** — only
`ddpm`'s diffusion training showed the stochastic drift you'd expect from
neural-net training in general (same shapes/columns/dtypes, distributions
in the same range, but not identical values). `nflow` has no production
run anywhere to compare against — it's smoke-tested only (runs cleanly,
correct output shape).
