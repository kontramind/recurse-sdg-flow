#!/usr/bin/env python3
"""
prepare_step7.py

Prepare Step7 data for a given revision:
  - pf_all      (population_fixed, all features — 23 columns)
  - pf_pilgram  (population_fixed, 13-column pilgram clinical core)

Produces train/test/population splits + encoding YAML + metadata JSON under
  <output>/Step7/dseed{seed}_rev_{revision}/
for each supplied dseed.

Ported from sdpype's prepare_step7_rev_pf_all.py / prepare_step7_rev_pf_pilgram.py,
which were identical except for their REVISION constant and COLUMN_CONFIG list.
Merged here into one module with a --revision flag; all other logic (ColCfg,
build_encoding, build_metadata, stratified_split, run) is unchanged.

Usage:
    uv run python3 prepare_step7.py --revision pf_all --dseeds 1597 196418 14930352
    uv run python3 prepare_step7.py --revision pf_pilgram --dseeds 1597 --sample 500 --output /tmp/out/
    uv run python3 prepare_step7.py --revision pf_all --dseeds 1597 --source /path/to/population_fixed.xlsx
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# ── Default source ───────────────────────────────────────────────────────────
# Assumes this script sits alongside sdpype/rd-lake under the same parent
# directory (as in the original). Override with --source for a different layout.
DEFAULT_SOURCE = Path(__file__).parent / "../rd-lake/population_fixed.xlsx"

# ── Column config ──────────────────────────────────────────────────────────────

_ENCODER_FOR_SDTYPE = {
    "categorical": "UniformEncoder",
    "boolean":     "BinaryEncoder",
    "numerical":   "FloatFormatter",
}


@dataclass
class ColCfg:
    source_name: str
    sdtype:      str               # 'numerical' | 'categorical' | 'boolean'
    repr_:       Optional[str] = None  # 'Int8' | 'Int16' | 'Float' | None
    is_target:   bool = False
    ordered:     bool = False      # use OrderedLabelEncoder instead of UniformEncoder
    order:       Optional[list] = None  # explicit category order, e.g. [-1, 0, 1, 2]

    @property
    def encoder(self) -> str:
        if self.ordered:
            return "OrderedLabelEncoder"
        return _ENCODER_FOR_SDTYPE[self.sdtype]


COLUMN_CONFIGS: dict[str, list[ColCfg]] = {
    "pf_all": [
        # Target
        ColCfg("readmission",      sdtype="boolean",                             is_target=True),
        # Booleans
        ColCfg("gender",           sdtype="boolean"),
        ColCfg("prior_icu",        sdtype="boolean"),
        # Numericals
        ColCfg("age",              sdtype="numerical", repr_="Int8"),
        ColCfg("heartrate",        sdtype="numerical", repr_="Int16"),
        ColCfg("systolic_bp",      sdtype="numerical", repr_="Int16"),
        ColCfg("diastolic_bp",     sdtype="numerical", repr_="Int16"),
        ColCfg("respiratory_rate", sdtype="numerical", repr_="Int8"),
        ColCfg("spo2",             sdtype="numerical", repr_="Int8"),
        ColCfg("glucose",          sdtype="numerical", repr_="Int16"),
        ColCfg("sodium",           sdtype="numerical", repr_="Int16"),
        ColCfg("blood_urea_nitro", sdtype="numerical", repr_="Int16"),
        ColCfg("creatinine",       sdtype="numerical", repr_="Float"),
        ColCfg("potassium",        sdtype="numerical", repr_="Float"),
        ColCfg("hemoglobin",       sdtype="numerical", repr_="Float"),
        # Categoricals
        ColCfg("admission_type",   sdtype="categorical"),
        ColCfg("first_careunit",   sdtype="categorical"),
        ColCfg("ethnicity",        sdtype="categorical"),
        ColCfg("icd9",             sdtype="categorical"),
        # Ordered categoricals (binned/sparse labs, -1 = missing)
        ColCfg("bmi",              sdtype="categorical", ordered=True, order=[-1, 0, 1, 2, 3, 4]),
        ColCfg("nt-probnp",        sdtype="categorical", ordered=True, order=[-1, 0, 1, 2]),
        ColCfg("cholesterol",      sdtype="categorical", ordered=True, order=[-1, 0, 1, 2]),
        ColCfg("albumin",          sdtype="categorical", ordered=True, order=[-1, 0, 1, 2]),
    ],
    "pf_pilgram": [
        # Order mirrors population_fixed.xlsx column order
        ColCfg("readmission",      sdtype="boolean",                             is_target=True),
        ColCfg("age",              sdtype="numerical", repr_="Int8"),
        ColCfg("admission_type",   sdtype="categorical"),
        ColCfg("ethnicity",        sdtype="categorical"),
        ColCfg("nt-probnp",        sdtype="categorical", ordered=True, order=[-1, 0, 1, 2]),
        ColCfg("cholesterol",      sdtype="categorical", ordered=True, order=[-1, 0, 1, 2]),
        ColCfg("respiratory_rate", sdtype="numerical", repr_="Int8"),
        ColCfg("heartrate",        sdtype="numerical", repr_="Int16"),
        ColCfg("systolic_bp",      sdtype="numerical", repr_="Int16"),
        ColCfg("diastolic_bp",     sdtype="numerical", repr_="Int16"),
        ColCfg("blood_urea_nitro", sdtype="numerical", repr_="Int16"),
        ColCfg("creatinine",       sdtype="numerical", repr_="Float"),
        ColCfg("potassium",        sdtype="numerical", repr_="Float"),
    ],
}

_NO_IMPUTE = {"missing_value_replacement": None, "missing_value_generation": None}

# ── Encoding / metadata builders ───────────────────────────────────────────────

def build_encoding(configs: list[ColCfg]) -> dict:
    sdtypes = {c.source_name: c.sdtype for c in configs}
    transformers: dict = {}
    for c in configs:
        if c.sdtype == "numerical":
            params: dict = {
                "computer_representation": c.repr_ or "Float",
                "enforce_min_max_values": True,
                "learn_rounding_scheme": True,
                **_NO_IMPUTE,
            }
        elif c.ordered:
            params = {"order": [str(v) for v in c.order], **_NO_IMPUTE}
        else:
            params = dict(_NO_IMPUTE)
        transformers[c.source_name] = {"type": c.encoder, "params": params}
    return {"sdtypes": sdtypes, "transformers": transformers}


def build_metadata(configs: list[ColCfg]) -> dict:
    columns: dict = {}
    for c in configs:
        entry: dict = {"sdtype": c.sdtype}
        if c.sdtype == "numerical" and c.repr_:
            entry["computer_representation"] = c.repr_
        columns[c.source_name] = entry
    return {"METADATA_SPEC_VERSION": "SINGLE_TABLE_V1", "columns": columns}


# ── Split ──────────────────────────────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame, sample: int, seed: int, target_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample 2*N rows stratified by target, then 50/50 stratified split."""
    cohort, _ = train_test_split(
        df, train_size=sample * 2,
        stratify=df[target_col], random_state=seed,
    )
    train, test = train_test_split(
        cohort, test_size=0.5,
        stratify=cohort[target_col], random_state=seed,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(
    revision: str, dseeds: list[int], sample: int, output_root: Path, source: Path
) -> None:
    column_config = COLUMN_CONFIGS[revision]
    target_col = next(c.source_name for c in column_config if c.is_target)

    src = source.resolve()
    if not src.exists():
        sys.exit(f"[ERROR] Source not found: {src}")

    print(f"Loading {src} …")
    df = pd.read_excel(src)

    cols = [c.source_name for c in column_config]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Columns missing from source: {missing}")

    pop_df   = df[cols].copy()
    encoding = build_encoding(column_config)
    metadata = build_metadata(column_config)

    pop_rate = pop_df[target_col].mean()
    print(f"Population: {len(pop_df):,} rows  {target_col} rate={pop_rate:.4f}  cols={len(cols)}")

    for seed in dseeds:
        folder = output_root / "Step7" / f"dseed{seed}_rev_{revision}"
        if folder.exists():
            print(f"[SKIP] Already exists: {folder}")
            continue
        folder.mkdir(parents=True)

        base = f"data_sample{sample}_dseed{seed}_rev_{revision}"
        train_df, test_df = stratified_split(pop_df, sample, seed, target_col)

        train_df.to_csv(folder / f"{base}_training.csv",   index=False)
        test_df.to_csv( folder / f"{base}_test.csv",       index=False)
        pop_df.to_csv(  folder / f"{base}_population.csv", index=False)

        with open(folder / f"{base}_encoding.yaml", "w") as f:
            yaml.dump(encoding, f, default_flow_style=False, sort_keys=False)
        with open(folder / f"{base}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"  dseed={seed:>10}  train={len(train_df):,}  test={len(test_df):,}"
            f"  {target_col}: train={train_df[target_col].mean():.4f}"
            f"  test={test_df[target_col].mean():.4f}"
            f"  → {folder}"
        )

    print("Done.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Step7 data")
    parser.add_argument("--revision", choices=sorted(COLUMN_CONFIGS), required=True,
                        help="Which column-config revision to prepare")
    parser.add_argument("--dseeds", nargs="+", type=int, required=True,
                        metavar="SEED", help="One or more data seeds")
    parser.add_argument("--sample", type=int, default=10000,
                        help="Rows per split half (default: 10000)")
    parser.add_argument("--output", type=Path, default=Path("../rd-lake/"),
                        help="Root output directory (Step7/ created inside, default: ../rd-lake/)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"Path to the source population .xlsx file (default: {DEFAULT_SOURCE})")
    args = parser.parse_args()
    run(args.revision, args.dseeds, args.sample, args.output, args.source)


if __name__ == "__main__":
    main()
