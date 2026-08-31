# recurse-sdg-flow

Recursive synthetic data generation pipeline, built with Prefect.

This is a clean, standalone port of the SDG pipeline used in [paper reference
TBD]. It is being assembled incrementally:

1. ✅ Project scaffolding
2. ✅ Data-prep CLI (`prepare_step7.py`) — builds train/test/population splits
   + encoding config from a source population file
3. ⏳ Minimal Prefect pipeline: encode → train → generate
4. ⏳ Recursive multi-generation loop
5. ⏳ Evaluation stages (statistical, privacy, detection, hallucination, TSTR)

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
