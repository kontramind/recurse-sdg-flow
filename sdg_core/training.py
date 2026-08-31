"""
Synthcity model construction, ported from sdpype/training.py.

Ported: create_synthcity_model (flattened from `cfg: DictConfig` to plain
model_type/parameters/seed args) and create_experiment_hash (same
flattening). Only the 5 model types this repo targets — arf, ctgan, ddpm,
rtvae, nflow — keep their explicit `match` branches; the other 8 case
blocks in the original (marginaldistributions, adsgan, aim, decaf, pategan,
privbayes, dpgan, bayesiannetwork) are dropped since they're unused by this
port, though the generic `case _` fallback is kept as-is — any other
synthcity plugin name still works via Plugins().get(model_type, **params).

create_sdv_model and create_synthpop_model are NOT ported: this repo is
synthcity-only (see the port plan for why sdv/synthpop are excluded).

Note: `data_shape`, a parameter on the original create_synthcity_model, is
dropped here — grep-confirmed it was never referenced in the function body.
"""

import hashlib
import json
from typing import Any, Dict, Optional

from synthcity.plugins import Plugins


def create_synthcity_model(model_type: str, parameters: Optional[Dict[str, Any]], seed: int):
    """
    Create a synthcity model/plugin with per-model-type parameter mapping.

    Args:
        model_type: One of "arf", "ctgan", "ddpm", "rtvae", "nflow" (or any
            other synthcity plugin name, via the generic fallback).
        parameters: Hyperparameter overrides (plain dict, or None/empty for
            all-defaults). Any key not recognized by the target model type
            is simply ignored (matches the original's model_params.get(...)
            pattern).
        seed: Random seed, used as each model's `random_state` default.

    Returns:
        An unfitted synthcity plugin instance.

    Raises:
        ValueError: If synthcity fails to construct the requested model.
    """
    all_params = dict(parameters) if parameters else {}

    # Filter out SDV-specific parameters that synthcity doesn't understand.
    # (Ported verbatim: this also means a caller-provided "verbose" is
    # always filtered out here, so e.g. arf's own verbose=... below always
    # falls back to its default — a pre-existing quirk in the original,
    # preserved rather than "fixed" during the port.)
    sdv_only_params = {'epochs', 'verbose', 'cuda', 'enforce_min_max_values', 'enforce_rounding', 'locales'}
    model_params = {k: v for k, v in all_params.items() if k not in sdv_only_params}

    try:
        match model_type:
            case "ctgan":
                if 'epochs' in all_params and 'n_iter' not in model_params:
                    model_params['n_iter'] = all_params['epochs']

                model = Plugins().get("ctgan",
                    # Training configuration
                    n_iter=model_params.get("n_iter", 2000),
                    batch_size=model_params.get("batch_size", 200),
                    random_state=model_params.get("random_state", seed),

                    # Generator architecture
                    generator_n_layers_hidden=model_params.get("generator_n_layers_hidden", 2),
                    generator_n_units_hidden=model_params.get("generator_n_units_hidden", 500),
                    generator_nonlin=model_params.get("generator_nonlin", "relu"),
                    generator_dropout=model_params.get("generator_dropout", 0.1),
                    generator_opt_betas=tuple(model_params.get("generator_opt_betas", [0.5, 0.999])),

                    # Discriminator architecture
                    discriminator_n_layers_hidden=model_params.get("discriminator_n_layers_hidden", 2),
                    discriminator_n_units_hidden=model_params.get("discriminator_n_units_hidden", 500),
                    discriminator_nonlin=model_params.get("discriminator_nonlin", "leaky_relu"),
                    discriminator_n_iter=model_params.get("discriminator_n_iter", 1),
                    discriminator_dropout=model_params.get("discriminator_dropout", 0.1),
                    discriminator_opt_betas=tuple(model_params.get("discriminator_opt_betas", [0.5, 0.999])),

                    # Learning rates and regularization
                    lr=model_params.get("lr", 1e-3),
                    weight_decay=model_params.get("weight_decay", 1e-3),

                    # Training stability
                    clipping_value=model_params.get("clipping_value", 1),
                    lambda_gradient_penalty=model_params.get("lambda_gradient_penalty", 10),

                    # Data encoding and handling
                    encoder_max_clusters=model_params.get("encoder_max_clusters", 10),
                    adjust_inference_sampling=model_params.get("adjust_inference_sampling", False),

                    # Early stopping and monitoring
                    patience=model_params.get("patience", 5),
                    n_iter_print=model_params.get("n_iter_print", 50),
                    n_iter_min=model_params.get("n_iter_min", 100),

                    # Core plugin settings
                    compress_dataset=model_params.get("compress_dataset", False),
                    sampling_patience=model_params.get("sampling_patience", 500),

                    # Advanced parameters (only if explicitly provided)
                    **{k: v for k, v in model_params.items() if k in [
                        'encoder', 'dataloader_sampler', 'patience_metric', 'workspace'
                    ] and v is not None}
                )
                return model

            case "nflow":
                if 'epochs' in all_params and 'n_iter' not in model_params:
                    model_params['n_iter'] = all_params['epochs']

                model = Plugins().get("nflow",
                    # Training configuration
                    n_iter=model_params.get("n_iter", 1000),
                    batch_size=model_params.get("batch_size", 200),
                    random_state=model_params.get("random_state", seed),

                    # Network architecture
                    n_layers_hidden=model_params.get("n_layers_hidden", 1),
                    n_units_hidden=model_params.get("n_units_hidden", 100),
                    num_transform_blocks=model_params.get("num_transform_blocks", 1),
                    dropout=model_params.get("dropout", 0.1),
                    batch_norm=model_params.get("batch_norm", False),

                    # Transform parameters
                    num_bins=model_params.get("num_bins", 8),
                    tail_bound=model_params.get("tail_bound", 3),

                    # Learning rate
                    lr=model_params.get("lr", 1e-3),

                    # Flow-specific settings
                    apply_unconditional_transform=model_params.get("apply_unconditional_transform", True),
                    base_distribution=model_params.get("base_distribution", "standard_normal"),
                    linear_transform_type=model_params.get("linear_transform_type", "permutation"),
                    base_transform_type=model_params.get("base_transform_type", "rq-autoregressive"),

                    # Data encoding
                    encoder_max_clusters=model_params.get("encoder_max_clusters", 10),
                    tabular=model_params.get("tabular", True),

                    # Early stopping and monitoring
                    n_iter_min=model_params.get("n_iter_min", 100),
                    n_iter_print=model_params.get("n_iter_print", 50),
                    patience=model_params.get("patience", 5),
                )
                return model

            case "arf":
                # Adversarial Random Forests - no epochs/iterations parameter
                model = Plugins().get("arf",
                    num_trees=model_params.get("num_trees", 30),
                    delta=model_params.get("delta", 0),
                    max_iters=model_params.get("max_iters", 10),
                    early_stop=model_params.get("early_stop", True),
                    verbose=model_params.get("verbose", True),
                    min_node_size=model_params.get("min_node_size", 5),

                    random_state=model_params.get("random_state", seed),
                    sampling_patience=model_params.get("sampling_patience", 500),
                    compress_dataset=model_params.get("compress_dataset", False),
                )
                return model

            case "ddpm":
                if 'epochs' in all_params and 'n_iter' not in model_params:
                    model_params['n_iter'] = all_params['epochs']

                model = Plugins().get("ddpm",
                    # Core training parameters
                    n_iter=model_params.get("n_iter", 1000),
                    lr=model_params.get("lr", 0.002),
                    weight_decay=model_params.get("weight_decay", 1e-4),
                    batch_size=model_params.get("batch_size", 1024),
                    random_state=model_params.get("random_state", seed),

                    # Task configuration
                    is_classification=model_params.get("is_classification", False),

                    # Diffusion process parameters
                    num_timesteps=model_params.get("num_timesteps", 1000),
                    gaussian_loss_type=model_params.get("gaussian_loss_type", "mse"),
                    scheduler=model_params.get("scheduler", "cosine"),

                    # Model architecture
                    model_type=model_params.get("model_type", "mlp"),
                    model_params=model_params.get("model_params", {
                        "n_layers_hidden": 3,
                        "n_units_hidden": 256,
                        "dropout": 0.0
                    }),
                    dim_embed=model_params.get("dim_embed", 128),

                    # Data encoding
                    continuous_encoder=model_params.get("continuous_encoder", "quantile"),
                    cont_encoder_params=model_params.get("cont_encoder_params", {}),

                    # Training monitoring and validation
                    log_interval=model_params.get("log_interval", 100),
                    validation_size=model_params.get("validation_size", 0),

                    # Core plugin settings
                    compress_dataset=model_params.get("compress_dataset", False),
                    sampling_patience=model_params.get("sampling_patience", 500),

                    # Advanced parameters (only if explicitly provided)
                    **{k: v for k, v in model_params.items() if k in [
                        'callbacks', 'validation_metric', 'workspace'
                    ] and v is not None}
                )
                return model

            case "rtvae":
                if 'epochs' in all_params and 'n_iter' not in model_params:
                    model_params['n_iter'] = all_params['epochs']

                model = Plugins().get("rtvae",
                    # Training configuration
                    n_iter=model_params.get("n_iter", 1000),
                    batch_size=model_params.get("batch_size", 200),
                    random_state=model_params.get("random_state", seed),
                    n_units_embedding=model_params.get("n_units_embedding", 500),

                    # Decoder architecture
                    decoder_n_layers_hidden=model_params.get("decoder_n_layers_hidden", 3),
                    decoder_n_units_hidden=model_params.get("decoder_n_units_hidden", 500),
                    decoder_nonlin=model_params.get("decoder_nonlin", "leaky_relu"),
                    decoder_dropout=model_params.get("decoder_dropout", 0),

                    # Encoder architecture
                    encoder_n_layers_hidden=model_params.get("encoder_n_layers_hidden", 3),
                    encoder_n_units_hidden=model_params.get("encoder_n_units_hidden", 500),
                    encoder_nonlin=model_params.get("encoder_nonlin", "leaky_relu"),
                    encoder_dropout=model_params.get("encoder_dropout", 0.1),

                    # Learning rates and regularization
                    lr=model_params.get("lr", 1e-3),
                    weight_decay=model_params.get("weight_decay", 1e-5),

                    # Robust divergence parameter (key feature of RTVAE)
                    robust_divergence_beta=model_params.get("robust_divergence_beta", 2),

                    # Data encoding
                    data_encoder_max_clusters=model_params.get("data_encoder_max_clusters", 10),

                    # Early stopping and monitoring
                    n_iter_print=model_params.get("n_iter_print", 50),
                    n_iter_min=model_params.get("n_iter_min", 100),
                    patience=model_params.get("patience", 5),
                )
                return model

            case _:
                # Fallback to generic plugin creation for any other synthcity model
                model = Plugins().get(model_type, **model_params)
                return model

    except Exception as e:
        raise ValueError(f"Failed to create Synthcity model '{model_type}': {e}")


def create_experiment_hash(sdg_params: Dict[str, Any], seed: int, training_file: str) -> str:
    """
    Create a short, deterministic hash identifying an experiment configuration.

    Args:
        sdg_params: e.g. {"library": ..., "model_type": ..., "parameters": ...}
        seed: Experiment/model seed
        training_file: Path to the training data file used

    Returns:
        8-character hex digest.
    """
    hash_dict = {
        "sdg": sdg_params,
        "seed": seed,
        "data_file": training_file,
    }

    hash_str = json.dumps(hash_dict, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:8]
