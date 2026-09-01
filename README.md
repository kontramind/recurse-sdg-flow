# recurse-sdg-flow

A clean, standalone, Prefect-based reimplementation of the recursive
synthetic-data-generation pipeline used in [paper reference TBD].

It takes a real tabular dataset, fits a synthetic-data generator, samples
synthetic data, scores that data against the real data with a full
evaluation suite, then feeds the synthetic data back in as training input
and repeats — for *N* generations — to study how synthetic data degrades
under recursive training.

Extracted from a large research monorepo (`sdpype`): the config-framework
and build-orchestration layers are dropped, the ~90 one-off analysis
scripts are not carried, and two parallel pipeline implementations collapse
to one. Every stage is verified to reproduce the original's stored
production results — see **Reproducibility** at the end.

## Pipeline

| stage | entry point | what it does |
|---|---|---|
| Data prep | `prepare_step7.py` | `train`/`test`/`population` split + RDT encoding config + metadata JSON, from a source population file |
| TRTR baseline | `flows/lgbm_cv_flow.py` | LightGBM + Optuna HPO on the real data — the "Train Real, Test Real" reference point that TSTR is later measured against |
| SDG pipeline | `flows/sdg_flow.py` | one generation: encode → train → generate → evaluate (statistical similarity ×15, privacy, detection, hallucination, TSTR) + a rich console report |
| Recursive loop | `flows/recursive_sdg_flow.py` | runs the SDG pipeline for *N* generations, each trained on the previous generation's synthetic output |

The repo was built one stage at a time; the git history has the per-stage
detail.

## Setup

```bash
uv sync
```

Dependencies are pinned to exact versions (`pyproject.toml` + `uv.lock`),
matched to what the original project has installed — the GPU stack
(`torch==2.9.1+cu128`, `synthcity==0.2.12`) in particular is version-sensitive,
and several results turn out to be bit-reproducible only when the whole
transitive tree matches.

## Quick start

```bash
# 1. Build one data seed: train/test/population CSVs + encoding.yaml + metadata.json
uv run python3 prepare_step7.py --revision pf_all --dseeds 1597

# 2. TRTR baseline for that seed (use --n-trials 5 for a <1 min smoke test)
uv run python3 flows/lgbm_cv_flow.py \
  --dataset      ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_training.csv \
  --test-dataset ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_test.csv \
  --target readmission --n-trials 50 \
  --output-dir   ../rd-lake/Step7/dseed1597_rev_pf_all/

# 3. One SDG generation (arf), full evaluation
uv run python3 flows/sdg_flow.py --config configs/step7_pf_all.yaml --model-type arf --seed 28657

# 4. A recursive chain — 20 generations of arf trained on their own output
uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml \
  --model-type arf --mseed 28657 --n-generations 20
```

Steps 3–4 read the dataset paths and evaluation config from
`configs/step7_pf_all.yaml` (committed; mirrors the original's
`params_step7_pf_all.yaml`). `configs/step7_pf_pilgram.yaml` is the 13-column
variant.

---

## Usage

### Data prep (`prepare_step7.py`)

Builds a `train`/`test`/`population` CSV split, an RDT encoding config, and an
SDV-style metadata JSON from a source population file, for one or more data
seeds (`--dseeds`). Two column-config revisions are available:

- `pf_all` — 23 columns, the full feature set
- `pf_pilgram` — 13-column clinical-core subset

```bash
# Single dseed, pf_all revision, default sample size (10,000 / split)
uv run python3 prepare_step7.py --revision pf_all --dseeds 1597

# Several dseeds in one run (just list them)
uv run python3 prepare_step7.py --revision pf_all --dseeds 1597 196418 14930352

# pf_pilgram revision, smaller sample size, custom output root
uv run python3 prepare_step7.py --revision pf_pilgram --dseeds 1597 --sample 500 --output /tmp/out/

# Point at your own population file (default: ../rd-lake/population_fixed.xlsx)
uv run python3 prepare_step7.py --revision pf_all --dseeds 1597 --source /path/to/your_population.xlsx
```

Output lands under `<output>/Step7/dseed<seed>_rev_<revision>/`:

```
Step7/dseed1597_rev_pf_all/
├── data_sample10000_dseed1597_rev_pf_all_training.csv
├── data_sample10000_dseed1597_rev_pf_all_test.csv
├── data_sample10000_dseed1597_rev_pf_all_population.csv
├── data_sample10000_dseed1597_rev_pf_all_encoding.yaml
└── data_sample10000_dseed1597_rev_pf_all_metadata.json
```

The command is idempotent per dseed/revision — it skips (with a `[SKIP]`
message) if the target folder already exists, rather than overwriting.

### TRTR baseline (`flows/lgbm_cv_flow.py`)

TRTR ("Train Real, Test Real") is the real-data reference point a later
TSTR ("Train Synthetic, Test Real") evaluation is measured against. It runs
the LightGBM + Optuna binary-classification flow directly on a dseed's real
`training`/`test` CSVs: Optuna does Bayesian HPO (default 50 trials, 5-fold
inner CV, AUROC objective) to pick hyperparameters once, trains a final
model, and evaluates it on the held-out real test set.

```bash
uv run python3 flows/lgbm_cv_flow.py \
  --dataset      ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_training.csv \
  --test-dataset ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_test.csv \
  --target readmission --n-trials 50 --n-folds 5 --seed 42 \
  --output-dir   ../rd-lake/Step7/dseed1597_rev_pf_all/
```

Writes `lgbm_cv_<timestamp>.{json,md,pkl}` + `shap_importance_<timestamp>.{csv,png}`
into `--output-dir`, and — because `--test-dataset` was given — also copies
the metrics JSON next to the test CSV, where the SDG pipeline's TSTR stage
auto-discovers it. The JSON's `best_params` and decision `threshold` are
**frozen and reused** for every TSTR run on that dseed — re-tuning per
generation would conflate synthetic-data-quality decline with the optimizer
landing on different hyperparameters.

`--n-trials 5` finishes in well under a minute (smoke test); the default
`--n-trials 50` can take up to ~1 hour depending on hardware — Optuna trials
run sequentially, each doing its own 5-fold CV with early-stopped LightGBM
fits.

### SDG pipeline (`flows/sdg_flow.py`)

One generation. Trains one of five synthcity generators on a dseed's data,
samples synthetic data, and runs the full evaluation suite: fit an encoder
on the population data, encode the training data, train, generate, then
encode-for-evaluation and compute

- **statistical similarity** — all 15 sub-metrics (`table_structure`,
  `semantic_structure`, `boundary_adherence`, `category_adherence`,
  `alpha_precision`, `prdc_score`, `wasserstein_distance`,
  `maximum_mean_discrepancy`, `new_row_synthesis`, `jensenshannon_synthcity`
  / `_syndat` / `_nannyml`, `ks_complement`, `tv_complement`,
  `sdmetrics_quality`)
- **privacy** — k-anonymity over a quasi-identifier set
- **detection** — synthcity real-vs-synthetic classifier two-sample test
- **hallucination** — TotalFR / NovelFR / MemorizedFR / HR against the
  population
- **TSTR** — train LightGBM on the synthetic data, test on the real held-out
  set, using the frozen TRTR params from the previous stage

Each stage writes `metrics/<stage>_<base_name>.json` (+ a `*_report.txt`),
then a rich console report prints every stage as a table plus a per-stage
timing summary (`--no-report` skips the report).

```bash
uv run python3 flows/sdg_flow.py --config configs/step7_pf_all.yaml
# pick a different revision / generator / seed, or override any key:
uv run python3 flows/sdg_flow.py --config configs/step7_pf_pilgram.yaml --model-type ddpm --seed 28657
```

Everything can also be given purely as flags, with no config file:

```bash
uv run python3 flows/sdg_flow.py \
  --training-file   ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_training.csv \
  --population-file ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_population.csv \
  --metadata-file   ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_metadata.json \
  --encoding-config ../rd-lake/Step7/dseed1597_rev_pf_all/data_sample10000_dseed1597_rev_pf_all_encoding.yaml \
  --model-type arf --seed 28657
```

#### Generators (`--model-type`)

| model | notes |
|---|---|
| `arf` | Adversarial Random Forests — fastest, tree-based, deterministic |
| `ctgan` | Conditional GAN |
| `ddpm` | Diffusion model |
| `rtvae` | Robust-divergence VAE |
| `nflow` | Normalizing flow |

All five are synthcity generators, loaded via synthcity's `Plugins()`.
`--library` defaults to `synthcity`, the only backend implemented.

Hyperparameters go in as an inline JSON string via `--params` (default
`'{}'` — all model defaults), e.g. `--params '{"n_iter": 500}'` for a faster
ctgan/ddpm/rtvae/nflow smoke test (`arf` has no iteration count to shrink).
`--n-samples` defaults to auto-sizing from `--reference-file` (which itself
defaults to `--training-file`, matching the production configs).
`--post-process-method {knn,weighted,random,none}` controls how
out-of-vocabulary categorical values a generator emits get repaired against
the training data's real categories (default `knn`; `none` disables the step).

#### Config file

`--config` mirrors `sdpype`'s `params_step7_*.yaml` nested layout —
`experiment` / `sdg` / `data` / `encoding` / `generation` /
`post_processing` / `evaluation` — so it maps 1:1 onto the original configs.
It is read with a plain `yaml.safe_load`: literal values only, no
interpolation, `${...}` templating, or config-group composition. The
original's computed `experiment.name` / `tags` fields are omitted;
`experiment.tag` is kept — it names the run directory. A CLI flag always
wins over the file.

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

The `evaluation.statistical_similarity` / `detection_evaluation` blocks (a
15-entry metrics list, per-metric `parameters`, …) have no sane CLI-flag
shape and are read straight from the config, with the Step7 values as
built-in defaults.

#### Output layout

Every run nests under `--output-dir/<run-name>/` (default
`outputs/sdg_runs/`), in the same per-run tree the production data lake
(`sd-lake/<experiment>/<model>/<run>/`) uses:

```
<run-name>/
  data/encoded/training_<base_name>.csv                  # dual-pipeline encode output
  data/decoded/training_<base_name>.csv
  data/synthetic/synthetic_data_<base_name>_encoded.csv  # generated synthetic data
  data/synthetic/synthetic_data_<base_name>_decoded.csv
  models/training_encoder_<base_name>.pkl                # per-run copy of the population encoder
  models/sdg_model_<base_name>.pkl                       # trained generator
  models/evaluation_encoder_<base_name>.pkl              # encoder refit on the reference data
  metrics/{encoding,training,generation,encoding_evaluation}_<base_name>.json
  metrics/{statistical_similarity,privacy,detection_evaluation,hallucination,tstr}_<base_name>.json
  metrics/{statistical,privacy,detection,hallucination}_report_<base_name>.txt
```

`<run-name>` is `<tag>_<dseed>_<library>_<model>_mseed<seed>` (e.g.
`Step7pfa_dseed1597_synthcity_arf_mseed28657`) — `<dseed>` is the `dseedNNN`
token from the training-file path, `<tag>` comes from `experiment.tag`
(dropped from the name when unset); `--run-name` overrides it. The
population encoder is cached separately under `--encoder-dir` (default
`outputs/sdg_runs/encoders/`), keyed by a content hash of the population
file — fit once and reused across every run that shares it.

The one structural departure from `sd-lake` is `<base_name>` itself:
`<model-type>_<training-file-stem>_<seed>` here, versus the original's
computed `experiment_name`
(`synthcity_<model>_<3 data hashes>_gen_<N>_<tag>_<config hash>_<mseed>`).
File *contents* still match byte-for-byte (see **Reproducibility**).

### Recursive loop (`flows/recursive_sdg_flow.py`)

Runs `sdg_flow.py` for *N* generations. Generation 0 trains on the config's
training file; generation *k* trains on generation *k−1*'s **decoded
synthetic** output. The population, reference, metadata and encoding config
stay fixed from generation 0 for the whole chain.

```bash
# One 20-generation chain (dataset + eval config come from --config)
uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml --n-generations 20

# Suppress the per-generation report / force regeneration
uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml --n-generations 20 --no-report
uv run python3 flows/recursive_sdg_flow.py --config configs/step7_pf_all.yaml --n-generations 20 --force-generate
```

**Sweeps.** Repeat `--model-type` and/or `--mseed` for a sequential
Cartesian sweep — each combination runs its own full *N*-generation chain.
This is the grid the paper's Step7 runs used:

```bash
uv run python3 flows/recursive_sdg_flow.py \
  --config configs/step7_pf_all.yaml \
  --model-type arf --model-type ddpm --model-type ctgan --model-type rtvae \
  --mseed 987 --mseed 28657 --mseed 5702887 --mseed 24157817 --mseed 39088169 \
  --n-generations 20
# 4 models × 5 seeds = 20 chains, one after another
```

For more than one dataset, loop over configs in the shell — `configs/` is
how this port selects a dataset, so `sdpype`'s original `--dseed-dir` sweep
dimension collapses to this:

```bash
for cfg in configs/step7_pf_all.yaml configs/step7_pf_pilgram.yaml; do
  uv run python3 flows/recursive_sdg_flow.py --config "$cfg" \
    --model-type arf --model-type ddpm --mseed 987 --mseed 28657 --n-generations 20
done
```

**Layout.** All generations of one chain share a single flat run directory
(`outputs/recursive_runs/<tag>_<dseed>_<library>_<model>_mseed<seed>/`),
told apart only by a `gen_<k>` token in every `<base_name>` — the same shape
`sd-lake` uses. Re-running the same command resumes: a generation whose
artifacts already exist is skipped (per-stage cache guards);
`--force-generate` overrides. Each generation's
`hallucination_<base_name>.json` additionally gets the `unified_metrics` /
`complexity_metrics` blocks that `sd-lake` carries (row-count proxies,
injected during sd-lake promotion in the original; a standalone `sdg_flow.py`
run omits them, as `sdpype`'s does).

`sdpype`'s `publish_to_sdlake` / `trace_chain` promotion machinery is out of
scope — this port matches `sdpype`'s `experiments/` flow output, not
`sd-lake`'s promoted/filtered layout.

## Reproducibility

This is a faithful port: it computes the same values the original `sdpype`
pipeline does, verified stage by stage against its stored production runs
(the git history has the per-stage detail).

- **The deterministic path reproduces exactly.** Data prep, the TRTR
  LightGBM baseline, and every `arf` stage are byte-for-byte identical to
  the production runs — including a cold 20-generation `arf` recursive chain
  rebuilt from the population file, where every generation's decoded
  synthetic CSV and every metric block matches the stored
  `Step7pfa_dseed1597_synthcity_arf_mseed28657` chain. `rtvae` and `ctgan`
  also reproduce their synthetic output exactly; `nflow` has no production
  run to compare against (smoke-tested only).
- **Stochastic generators match in distribution and trajectory, not
  bit-for-bit.** `ddpm` diverges from the stored output after generation 0.
  This is the original's own behaviour, not the port's — a fresh `sdpype`
  DDPM run diverges from `sdpype`'s stored output by the same margin
  (~0.05 σ on gen-0 column means). synthcity's DDPM training is not fully
  deterministic on GPU even with fixed seeds; the other generators are.

Two metric fields deliberately differ from the stored JSON:

- `maximum_mean_discrepancy` — the in-pipeline synthcity metric hardcodes
  `gamma = 1.0` on unit-scaled data and collapses to a `2/n` floor
  (`≈ 0.0002` for every generator). This port computes the **corrected** MMD
  (z-score on the real reference, a frozen per-variant RBF gamma, unbiased
  estimator), matching the paper's post-hoc recomputation.
- `alpha_precision`'s `*_OC` fields depend on an unseeded one-class network
  in synthcity 0.2.12 and are not bit-reproducible across processes; the
  `*_naive` variant and `prdc_score` are exact.

Run-directory layout follows `sdpype`'s `experiments/` flow output, not the
promoted `sd-lake` layout, so a `diff -r` against the data lake shows
folder-shape and `<base_name>` differences — the metric *contents* match.
