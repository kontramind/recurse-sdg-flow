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

Trains one of five synthcity-backed generators on a dseed's data, samples
synthetic data from it, and runs the evaluation stages — fit an encoder on
the population data, encode the training data, train, generate, then
encode-for-evaluation and compute hallucination / TSTR / privacy (k-anon)
metrics. Same stages the original DVC/Hydra pipeline ran, now driven by a
plain config file and/or CLI flags instead of Hydra.

```bash
uv run python3 flows/sdg_flow.py --config configs/step7_pf_all.yaml
# or point at the pf_pilgram revision, and override any key on the CLI:
uv run python3 flows/sdg_flow.py --config configs/step7_pf_pilgram.yaml --model-type ddpm --seed 28657
```

`configs/step7_pf_{all,pilgram}.yaml` are committed and mirror sdpype's
`params_step7_pf_{all,pilgram}.yaml`. Everything can also be given purely as
flags, no config file:

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

The `--config` file mirrors sdpype's `params_step7_*.yaml` nested layout —
`experiment` / `sdg` / `data` / `encoding` / `generation` /
`post_processing` / `evaluation` — so it maps 1:1 onto the original
experiment configs. It is a plain `yaml.safe_load`: no Hydra interpolation,
`${...}` templating, or config-group composition (the Hydra-only
`experiment.name` / `tags` templates are omitted; `experiment.tag` is kept
— it names the run directory, see below). A CLI flag always wins over the
file. CLI-flag ↔ config-key mapping:

| flag | config key |
|---|---|
| `--training-file` / `--population-file` / `--metadata-file` | `data.training_file` / `data.population_file` / `data.metadata_file` |
| `--reference-file` | `data.reference_file` |
| `--encoding-config` | `encoding.config_file` |
| `--model-type` / `--library` / `--params` | `sdg.model_type` / `sdg.library` / `sdg.parameters` |
| `--seed` | `experiment.seed` |
| `--n-samples` | `generation.n_samples` |
| `--post-process-method` / `--knn-neighbors` / `--distance-metric` / `--fallback` | `post_processing.fix_invalid_categories.{method,knn_neighbors,distance_metric,fallback}` (`enabled: false` → method `none`) |
| `--hallucination-num-bins` | `evaluation.hallucination.num_bins` |
| `--privacy-qi-columns` | `evaluation.privacy.metrics[0].parameters.qi_columns` |

The `evaluation.statistical_similarity` / `detection_evaluation` blocks
(a 15-entry metrics list, per-metric `parameters`, ...) have no sane
CLI-flag shape and are read straight from the config, with the Step7 values
as built-in defaults.

When `experiment.tag` is set (or `--run-name` is given), everything nests
under `--output-dir/<run-name>/` in the same per-run tree layout the
production data lake (`sd-lake/<experiment>/<model>/<run>/`) uses — so port
output maps directly onto a real run folder for comparison. The run name is
`<tag>_<dseed>_<library>_<model>_mseed<seed>` (e.g.
`Step7pfp_dseed1597_synthcity_arf_mseed987`), matching sd-lake's run-dir
convention; `<dseed>` is the `dseedNNN` token from the training-file path.
Without a tag / `--run-name`, `--output-dir` itself is the run dir (flat).

```
<run-name>/
  data/encoded/training_<base_name>.csv        # dual-pipeline encode output
  data/decoded/training_<base_name>.csv
  data/synthetic/synthetic_data_<base_name>_encoded.csv    # generated synthetic data
  data/synthetic/synthetic_data_<base_name>_decoded.csv
  models/training_encoder_<base_name>.pkl      # per-run copy of the population encoder
  models/sdg_model_<base_name>.pkl             # trained generator
  metrics/{encoding,training,generation}_<base_name>.json
```

The remaining structural departure from `sd-lake` is `<base_name>` itself —
`<model-type>_<training-file-stem>_<seed>` here, versus the Hydra
`experiment_name` template (`synthcity_<model>_<3 data hashes>_gen_<N>_<tag>_<config
hash>_<mseed>`) in production, which was dropped along with Hydra in the
scaffolding step. File *contents* still match byte-for-byte. The
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
