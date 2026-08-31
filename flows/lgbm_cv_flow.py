import os
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

"""
LightGBM Binary Classification Flow
=====================================
General-purpose Prefect flow for binary classification with LightGBM.

Standard mode (single model):
    Runs Bayesian HPO (Optuna) + k-fold inner CV, trains one final model,
    evaluates on a held-out test set.

    python flows/lgbm_cv_flow.py --dataset data.csv --target LABEL

Two-file input:
    python flows/lgbm_cv_flow.py --dataset train.csv --test-dataset test.csv --target LABEL

Full options:
    python flows/lgbm_cv_flow.py \\
        --dataset data.csv \\
        --target LABEL \\
        --n-trials 50 \\
        --n-folds 5 \\
        --test-size 0.2 \\
        --seed 42 \\
        --timeout 600 \\
        --output-dir ./outputs/lgbm_runs

Ported from sdpype's flows/lgbm_cv_flow.py — standard mode only. The nested
(double) CV mode (lgbm_nested_cv_flow, --nested) was not ported: confirmed
unused anywhere in the actual pipeline (only appeared in its own docstring
example there), so it's out of scope for this port.
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from pandas.api.types import CategoricalDtype
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
import plotly.express as px
from sklearn.model_selection import train_test_split

from sdg_core.downstream import LGBMBayesianTuner, evaluate_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype == object:
        lower = s.str.strip().str.lower()
        if lower.isin({'true', 'false'}).all():
            return (lower == 'true').astype(int)
    return pd.to_numeric(s, errors='raise').astype(int)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="load-data", description="Load CSV(s) and produce train/test splits")
def load_data(
    dataset_path: str,
    target_col: str,
    test_dataset_path: Optional[str] = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Load data from one or two CSV files.

    - One file  → stratified train/test split using *test_size*.
    - Two files → dataset_path is train, test_dataset_path is test.

    Returns (X_train, X_test, y_train, y_test, data_info).
    """
    logger = get_run_logger()

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset: {dataset_path} ({len(df):,} rows)")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {dataset_path}")

    if test_dataset_path:
        test_df = pd.read_csv(test_dataset_path)
        logger.info(f"Loaded test dataset: {test_dataset_path} ({len(test_df):,} rows)")
        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {test_dataset_path}")
        X_train = df.drop(columns=[target_col])
        y_train = df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        split_mode = "explicit train/test files"
    else:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        split_mode = f"internal {int((1-test_size)*100)}/{int(test_size*100)} split"
        logger.info(f"Applied stratified split: {len(X_train):,} train / {len(X_test):,} test")

    y_train = _to_binary(y_train)
    y_test = _to_binary(y_test)

    class_counts = dict(y_train.value_counts().sort_index())
    n_total = len(y_train)
    data_info = {
        "split_mode": split_mode,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "target": target_col,
        "class_distribution_train": {str(k): int(v) for k, v in class_counts.items()},
    }

    table = Table(title="[bold]Data Loaded", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Split mode", split_mode)
    table.add_row("Train rows", f"{len(X_train):,}")
    table.add_row("Test rows", f"{len(X_test):,}")
    table.add_row("Features", str(X_train.shape[1]))
    table.add_row("Target", f"[cyan]{target_col}[/cyan]")
    for cls, cnt in class_counts.items():
        pct = cnt / n_total * 100
        table.add_row(f"  Class {cls}", f"{cnt:,}  ({pct:.1f}%)")
    console.print(table)

    return X_train, X_test, y_train, y_test, data_info


@task(name="encode-features", description="Cast categorical columns to pd.Categorical for native LightGBM handling")
def encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """
    Cast object/category columns to pd.Categorical using train-fitted categories.
    LightGBM auto-detects pd.Categorical dtype and uses its native Fisher-split algorithm.
    Unseen categories in test become NaN, which LightGBM handles as missing values.

    Returns (X_train_enc, X_test_enc).
    """
    logger = get_run_logger()

    X_train = X_train.copy()
    X_test = X_test.copy()

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        console.print("[dim]encode-features:[/dim] no categorical columns — skipping")
        logger.info("No categorical columns found — skipping encoding")
        return X_train, X_test

    for col in categorical_cols:
        dtype = CategoricalDtype(categories=sorted(X_train[col].dropna().unique()))
        X_train[col] = X_train[col].astype(dtype)
        X_test[col] = X_test[col].astype(dtype)

    console.print(
        f"[dim]encode-features:[/dim] cast [bold]{len(categorical_cols)}[/bold] "
        f"column(s) → pd.Categorical  [dim]{categorical_cols}[/dim]"
    )
    logger.info(f"Cast {len(categorical_cols)} column(s) to pd.Categorical: {categorical_cols}")
    return X_train, X_test


@task(
    name="tune-hyperparameters",
    description="Bayesian HPO with Optuna over stratified CV",
)
def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int = 5,
    n_trials: int = 50,
    seed: int = 42,
    timeout: Optional[int] = None,
    n_jobs: int = -1,
    quiet: bool = False,
) -> tuple:
    """
    Run Optuna TPE search. Each trial evaluates mean AUROC over *n_folds* CV folds.

    Returns (best_params, cv_score, tuner).
    """
    logger = get_run_logger()
    logger.info(f"Starting Optuna HPO: {n_trials} trials, {n_folds}-fold CV")

    # Parallel Optuna trials (n_workers > 1) call the progress callback from multiple threads
    # simultaneously — Rich's Live display is not thread-safe, so suppress it.
    tuner = LGBMBayesianTuner(
        X_train=X_train,
        y_train=y_train,
        n_folds=n_folds,
        n_trials=n_trials,
        random_state=seed,
        n_jobs=n_jobs,
    )

    best_so_far = [0.0]

    if quiet:
        def _optuna_callback(study, trial):
            if trial.value is not None and trial.value > best_so_far[0]:
                best_so_far[0] = trial.value

        best_params = tuner.tune(timeout=timeout, callbacks=[_optuna_callback])
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]HPO[/bold cyan]"),
            BarColumn(bar_width=36),
            MofNCompleteColumn(),
            TextColumn("[green]best AUROC:[/green] {task.fields[best]}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            hpo_task = progress.add_task("HPO", total=n_trials, best="—")

            def _optuna_callback(study, trial):
                if trial.value is not None and trial.value > best_so_far[0]:
                    best_so_far[0] = trial.value
                progress.update(
                    hpo_task,
                    advance=1,
                    best=f"{best_so_far[0]:.4f}" if best_so_far[0] > 0 else "—",
                )

            best_params = tuner.tune(timeout=timeout, callbacks=[_optuna_callback])

    cv_score = tuner.best_score

    # Top-5 trials summary
    trials = tuner.study.trials
    completed = sorted(
        [t for t in trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )[:5]

    top_table = Table(title="[bold]Top-5 HPO Trials", show_header=True)
    top_table.add_column("Rank", justify="right", style="dim")
    top_table.add_column("Trial", justify="right")
    top_table.add_column("CV AUROC", justify="right", style="bold green")
    top_table.add_column("boosting", style="cyan")
    top_table.add_column("leaves", justify="right")
    top_table.add_column("lr", justify="right")
    top_table.add_column("target_enc", justify="center")
    top_table.add_column("calib", justify="center")
    for rank, t in enumerate(completed, 1):
        p = t.params
        top_table.add_row(
            str(rank),
            str(t.number),
            f"{t.value:.4f}",
            p.get("boosting_type", "—"),
            str(p.get("num_leaves", "—")),
            f"{p.get('learning_rate', 0):.4f}",
            "✓" if p.get("use_target_encoding") else "✗",
            "✓" if p.get("use_calibration") else "✗",
        )
    console.print(top_table)

    logger.info(f"HPO complete — best CV AUROC: {cv_score:.4f}")
    logger.info(f"Best params: {best_params}")
    return best_params, cv_score, tuner


@task(name="train-final-model", description="Train LightGBM with best params")
def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    tuner: LGBMBayesianTuner,
    seed: int = 42,
    val_split: float = 0.2,
    n_jobs: int = -1,
    quiet: bool = False,
) -> tuple:
    """
    Train a final model using the best hyperparameters found by Optuna.
    Applies target encoding and beta calibration if selected during HPO.
    Uses a stratified validation split for early stopping and threshold tuning.

    Returns (model, optimal_threshold, target_encoder, calibrator).
    """
    logger = get_run_logger()

    def _train_body():
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=val_split, random_state=seed, stratify=y_train
        )

        tuner.best_params = best_params
        tuner.n_jobs = n_jobs
        model, target_encoder, calibrator = tuner.train_final_model(X_tr, y_tr, X_val, y_val)

        if target_encoder:
            X_val_for_threshold = X_val.copy()
            for col, encoder in target_encoder.items():
                means = encoder["means"]
                global_mean = encoder["global_mean"]
                X_val_for_threshold[col] = X_val_for_threshold[col].astype(object).map(means).fillna(global_mean).astype(float)
        else:
            X_val_for_threshold = X_val

        threshold = tuner.find_optimal_threshold(model, X_val_for_threshold, y_val, metric="youden", calibrator=calibrator)
        return model, target_encoder, calibrator, threshold

    if quiet:
        model, target_encoder, calibrator, threshold = _train_body()
    else:
        with console.status("[bold cyan]Training final model…[/bold cyan]", spinner="dots"):
            model, target_encoder, calibrator, threshold = _train_body()

    if not quiet:
        console.print(
            f"[dim]train-final-model:[/dim]  "
            f"trees [bold]{model.num_trees()}[/bold]  "
            f"best iter [bold]{model.best_iteration}[/bold]  "
            f"threshold [bold]{threshold:.3f}[/bold]  "
            f"target_enc [bold]{'on' if target_encoder else 'off'}[/bold]  "
            f"calibration [bold]{'on' if calibrator else 'off'}[/bold]"
        )
    logger.info(f"Model trained — best iteration: {model.best_iteration}, trees: {model.num_trees()}")
    logger.info(f"Optimal threshold: {threshold:.3f}")
    logger.info(f"Target encoding: {target_encoder is not None} | Calibration: {calibrator is not None}")
    return model, float(threshold), target_encoder, calibrator


@task(name="evaluate-on-test", description="Evaluate final model on held-out test set")
def evaluate_on_test(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    target_encoder: Optional[dict] = None,
    calibrator=None,
) -> dict:
    """
    Compute AUROC, accuracy, precision, recall, F1 on test set.
    Applies target encoding and calibration if they were selected during HPO.
    """
    logger = get_run_logger()

    if target_encoder is not None:
        X_test = X_test.copy()
        for col, encoder in target_encoder.items():
            means = encoder["means"]
            global_mean = encoder["global_mean"]
            X_test[col] = X_test[col].astype(object).map(means).fillna(global_mean).astype(float)

    metrics = evaluate_model(model, X_test, y_test, threshold=threshold, calibrator=calibrator)

    cm = metrics["confusion_matrix"]
    results_table = Table(title="[bold]Test Set Results", show_header=True)
    results_table.add_column("Metric", style="dim")
    results_table.add_column("Value", justify="right", style="bold")

    results_table.add_row("AUROC",     f"{metrics['auroc']:.4f}")
    results_table.add_row("F1",        f"{metrics['f1_score']:.4f}")
    results_table.add_row("Precision", f"{metrics['precision']:.4f}")
    results_table.add_row("Recall",    f"{metrics['recall']:.4f}")
    results_table.add_row("Accuracy",  f"{metrics['accuracy']:.4f}")
    results_table.add_row("Threshold", f"{metrics['threshold']:.3f}")
    results_table.add_section()
    results_table.add_row("TN", f"{cm['tn']:,}")
    results_table.add_row("FP", f"{cm['fp']:,}")
    results_table.add_row("FN", f"{cm['fn']:,}")
    results_table.add_row("TP", f"{cm['tp']:,}")
    console.print(results_table)

    logger.info(
        f"Test results — AUROC: {metrics['auroc']:.4f}, "
        f"F1: {metrics['f1_score']:.4f}, "
        f"Recall: {metrics['recall']:.4f}"
    )
    return metrics


@task(name="shap-importance", description="TreeSHAP feature importance from trained LightGBM model")
def compute_shap_importance(
    model: lgb.Booster,
    X: pd.DataFrame,
    target_encoder: Optional[dict] = None,
    output_dir: Optional[str] = None,
    label: str = "",
) -> dict:
    """
    Compute mean absolute SHAP values using LightGBM native TreeSHAP (pred_contrib=True).
    Returns {"importance": {feature: float, ...}, "csv_path": ..., "png_path": ...}.
    Importance dict is sorted descending by mean |SHAP|.
    """
    logger = get_run_logger()

    X_enc = X.copy()
    if target_encoder is not None:
        for col, encoder in target_encoder.items():
            means = encoder["means"]
            global_mean = encoder["global_mean"]
            X_enc[col] = X_enc[col].astype(object).map(means).fillna(global_mean).astype(float)

    # (n_samples, n_features + 1); last column is the bias term
    shap_matrix = model.predict(X_enc, pred_contrib=True)
    mean_abs = np.abs(shap_matrix[:, :-1]).mean(axis=0)
    feature_names = model.feature_name()

    order = np.argsort(mean_abs)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = mean_abs[order]

    importance = {n: float(v) for n, v in zip(sorted_names, sorted_vals)}

    max_val = float(sorted_vals[0]) if len(sorted_vals) > 0 else 1.0
    bar_width = 28
    table = Table(title="[bold]SHAP Feature Importance (mean |SHAP|)", show_header=True)
    table.add_column("Rank", justify="right", style="dim", width=5)
    table.add_column("Feature", style="cyan")
    table.add_column("Mean |SHAP|", justify="right", style="bold", width=12)
    table.add_column("", no_wrap=True)
    for rank, (name, val) in enumerate(zip(sorted_names, sorted_vals), 1):
        bar = "█" * max(1, int(val / max_val * bar_width))
        table.add_row(str(rank), name, f"{val:.5f}", f"[green]{bar}[/green]")
    console.print(table)
    logger.info(f"SHAP top feature: {sorted_names[0]} ({sorted_vals[0]:.5f})")

    result: dict = {"importance": importance}

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""

        csv_path = output_path / f"shap_importance{suffix}_{timestamp}.csv"
        pd.DataFrame({"feature": sorted_names, "mean_abs_shap": sorted_vals}).to_csv(csv_path, index=False)

        fig = px.bar(
            x=list(sorted_vals[::-1]),
            y=list(sorted_names[::-1]),
            orientation="h",
            labels={"x": "Mean |SHAP|", "y": "Feature"},
            title=f"SHAP Feature Importance{' — ' + label if label else ''}",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(400, len(sorted_names) * 22),
        )
        png_path = output_path / f"shap_importance{suffix}_{timestamp}.png"
        fig.write_image(str(png_path))

        logger.info(f"Saved SHAP CSV → {csv_path}")
        logger.info(f"Saved SHAP PNG → {png_path}")
        console.print(
            f"[dim]shap-importance:[/dim] "
            f"[cyan]{csv_path.name}[/cyan]  [cyan]{png_path.name}[/cyan]"
        )
        result["csv_path"] = str(csv_path)
        result["png_path"] = str(png_path)

    return result


def _shap_markdown_section(shap_result: Optional[dict]) -> str:
    if not shap_result:
        return ""
    importance = shap_result.get("importance", {})
    if not importance:
        return ""
    top10 = list(importance.items())[:10]
    rows = "\n".join(f"| {i+1} | `{name}` | {val:.5f} |" for i, (name, val) in enumerate(top10))
    return f"""
### Top-10 Feature Importance (SHAP)
| Rank | Feature | Mean |SHAP| |
|------|---------|------|
{rows}
"""


@task(name="publish-results", description="Create Prefect artifact and save outputs to disk")
def publish_results(
    cv_score: float,
    test_metrics: dict,
    best_params: dict,
    data_info: dict,
    model: lgb.Booster,
    output_dir: str,
    target_encoder: Optional[dict] = None,
    calibrator=None,
    shap_result: Optional[dict] = None,
) -> dict:
    logger = get_run_logger()

    transfer_gap = cv_score - test_metrics["auroc"]
    cm = test_metrics["confusion_matrix"]

    params_table = "\n".join(
        f"| `{k}` | `{v}` |" for k, v in sorted(best_params.items())
    )
    markdown = f"""## LightGBM CV Pipeline Results

### Data
| | |
|---|---|
| Split mode | {data_info['split_mode']} |
| Train samples | {data_info['n_train']:,} |
| Test samples | {data_info['n_test']:,} |
| Features | {data_info['n_features']} |
| Target | `{data_info['target']}` |
| Class distribution (train) | {data_info['class_distribution_train']} |

### Performance
| Metric | Value |
|--------|-------|
| **CV AUROC** (Optuna best) | **{cv_score:.4f}** |
| **Test AUROC** | **{test_metrics['auroc']:.4f}** |
| Transfer gap | {transfer_gap:+.4f} |
| Accuracy | {test_metrics['accuracy']:.4f} |
| Precision | {test_metrics['precision']:.4f} |
| Recall | {test_metrics['recall']:.4f} |
| F1 Score | {test_metrics['f1_score']:.4f} |
| Threshold | {test_metrics['threshold']:.3f} |

### Confusion Matrix
| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | {cm['tn']:,} (TN) | {cm['fp']:,} (FP) |
| **Actual Yes** | {cm['fn']:,} (FN) | {cm['tp']:,} (TP) |

### HPO Decisions
| Decision | Selected |
|----------|---------|
| Target encoding | {"Yes" if target_encoder is not None else "No"} |
| Beta calibration | {"Yes" if calibrator is not None else "No"} |

### Best Hyperparameters
| Parameter | Value |
|-----------|-------|
{params_table}
{_shap_markdown_section(shap_result)}"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = output_path / f"lgbm_cv_{timestamp}.md"
    md_path.write_text(markdown)

    try:
        create_markdown_artifact(
            key="lgbm-cv-results",
            markdown=markdown,
            description="LightGBM CV pipeline summary",
        )
    except Exception:
        pass

    metrics_payload = {
        "timestamp": timestamp,
        "data_info": data_info,
        "cv_score": cv_score,
        "test_metrics": test_metrics,
        "best_params": best_params,
        "model_info": {
            "num_trees": model.num_trees(),
            "best_iteration": model.best_iteration,
        },
    }
    if shap_result:
        metrics_payload["shap"] = {
            "top_features": list(shap_result.get("importance", {}).items())[:10],
            "csv_path": shap_result.get("csv_path"),
            "png_path": shap_result.get("png_path"),
        }

    metrics_path = output_path / f"lgbm_cv_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    model_path = output_path / f"lgbm_cv_{timestamp}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    paths_table = Table(show_header=False, box=None, padding=(0, 2))
    paths_table.add_column(style="dim")
    paths_table.add_column(style="cyan")
    paths_table.add_row("report ", str(md_path))
    paths_table.add_row("metrics", str(metrics_path))
    paths_table.add_row("model  ", str(model_path))
    if shap_result:
        if shap_result.get("csv_path"):
            paths_table.add_row("shap csv", shap_result["csv_path"])
        if shap_result.get("png_path"):
            paths_table.add_row("shap png", shap_result["png_path"])
    console.print(Panel(paths_table, title="[bold]Outputs saved", expand=False))

    logger.info(f"Saved report  → {md_path}")
    logger.info(f"Saved metrics → {metrics_path}")
    logger.info(f"Saved model   → {model_path}")

    return {
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "report_path": str(md_path),
    }


# ---------------------------------------------------------------------------
# Standard flow
# ---------------------------------------------------------------------------

@flow(name="lgbm-cv-pipeline", log_prints=True)
def lgbm_cv_flow(
    dataset_path: str,
    target_col: str,
    test_dataset_path: Optional[str] = None,
    test_size: float = 0.2,
    n_folds: int = 5,
    n_trials: int = 50,
    seed: int = 42,
    timeout: Optional[int] = None,
    output_dir: str = "outputs/lgbm_runs",
) -> dict:
    console.print(Rule("[bold cyan]LightGBM CV Pipeline[/bold cyan]"))
    cfg_table = Table(show_header=False, box=None, padding=(0, 2))
    cfg_table.add_column(style="dim")
    cfg_table.add_column()
    cfg_table.add_row("dataset", dataset_path)
    if test_dataset_path:
        cfg_table.add_row("test", test_dataset_path)
    cfg_table.add_row("target",  f"[cyan]{target_col}[/cyan]")
    cfg_table.add_row("trials",  str(n_trials))
    cfg_table.add_row("folds",   str(n_folds))
    cfg_table.add_row("seed",    str(seed))
    if timeout:
        cfg_table.add_row("timeout", f"{timeout}s")
    console.print(cfg_table)
    console.print()

    X_train, X_test, y_train, y_test, data_info = load_data(
        dataset_path=dataset_path,
        target_col=target_col,
        test_dataset_path=test_dataset_path,
        test_size=test_size,
        seed=seed,
    )

    X_train_enc, X_test_enc = encode_features(X_train, X_test)

    best_params, cv_score, tuner = tune_hyperparameters(
        X_train=X_train_enc,
        y_train=y_train,
        n_folds=n_folds,
        n_trials=n_trials,
        seed=seed,
        timeout=timeout,
    )

    model, threshold, target_encoder, calibrator = train_final_model(
        X_train=X_train_enc,
        y_train=y_train,
        best_params=best_params,
        tuner=tuner,
        seed=seed,
    )

    test_metrics = evaluate_on_test(
        model=model,
        X_test=X_test_enc,
        y_test=y_test,
        threshold=threshold,
        target_encoder=target_encoder,
        calibrator=calibrator,
    )

    shap_result = compute_shap_importance(
        model=model,
        X=X_test_enc,
        target_encoder=target_encoder,
        output_dir=output_dir,
    )

    paths = publish_results(
        cv_score=cv_score,
        test_metrics=test_metrics,
        best_params=best_params,
        data_info=data_info,
        model=model,
        output_dir=output_dir,
        target_encoder=target_encoder,
        calibrator=calibrator,
        shap_result=shap_result,
    )

    if test_dataset_path:
        import shutil
        dseed_dir = Path(test_dataset_path).parent
        dest = dseed_dir / Path(paths["metrics_path"]).name
        if dest.resolve() != Path(paths["metrics_path"]).resolve():
            shutil.copy2(paths["metrics_path"], dest)
            console.print(f"[dim]Copied metrics JSON → {dest}[/dim]")

    gap = cv_score - test_metrics["auroc"]
    console.print()
    console.print(
        Panel(
            f"[bold green]CV AUROC[/bold green]   {cv_score:.4f}\n"
            f"[bold green]Test AUROC[/bold green] {test_metrics['auroc']:.4f}  "
            f"[dim](gap {gap:+.4f})[/dim]\n"
            f"[bold green]F1[/bold green]         {test_metrics['f1_score']:.4f}  "
            f"Precision {test_metrics['precision']:.4f}  Recall {test_metrics['recall']:.4f}",
            title="[bold]Pipeline Complete",
            expand=False,
        )
    )

    return {"cv_score": cv_score, "test_metrics": test_metrics, **paths}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LightGBM CV Prefect flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Path to training (or single) CSV")
    parser.add_argument("--target", required=True, help="Binary target column name")
    parser.add_argument("--test-dataset", default=None, help="Path to separate test CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction for single-file mode")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds (inner folds)")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna HPO trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timeout", type=int, default=None, help="HPO timeout in seconds")
    parser.add_argument("--output-dir", default="outputs/lgbm_runs", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    lgbm_cv_flow(
        dataset_path=args.dataset,
        target_col=args.target,
        test_dataset_path=args.test_dataset,
        test_size=args.test_size,
        n_folds=args.n_folds,
        n_trials=args.n_trials,
        seed=args.seed,
        timeout=args.timeout,
        output_dir=args.output_dir,
    )
