"""
Model serialization for the ported SDG pipeline, ported from
sdpype/serialization.py.

Ported: SerializationError/ModelNotFoundError/LibraryNotSupportedError,
save_model, load_model, create_model_metadata — synthcity-only (this repo's
locked scope), with save_model/load_model flattened to take an explicit
target path instead of reconstructing/globbing a Hydra/DVC-era
`.sdpype_config_hash`-keyed filename (that sentinel-file rendezvous has no
home in an argparse-only design — the flow now knows every path up front).

Dropped from the original: the SDV/synthpop branches (out of scope — see
the port plan), the DPGAN file-based special-casing (dpgan isn't one of
this repo's 5 target models), and get_model_info/list_saved_models/
validate_model/get_supported_libraries (confirmed unused by sdg_flow.py —
those were sdpype/cli/model.py-only).
"""

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

try:
    from synthcity.utils.serialization import save as synthcity_save, load as synthcity_load
    SYNTHCITY_AVAILABLE = True
except ImportError:
    SYNTHCITY_AVAILABLE = False
    synthcity_save = synthcity_load = None


SERIALIZE_FORMAT_VERSION = "2.1"


class SerializationError(Exception):
    """Custom exception for serialization errors"""
    pass


class ModelNotFoundError(SerializationError):
    """Raised when a model file is not found"""
    pass


class LibraryNotSupportedError(SerializationError):
    """Raised when trying to use an unsupported library"""
    pass


def save_model(model: Any, metadata: Dict[str, Any], library: str, model_path: Path | str) -> str:
    """
    Save a trained model with a unified interface (synthcity only for now —
    `library` is kept as an explicit arg rather than hardcoded so a future
    library addition is a contained change, not a rewrite).

    Args:
        model: Trained model object
        metadata: Experiment metadata (training time, config, etc.) — see
            create_model_metadata()
        library: Library name. Only "synthcity" is currently supported.
        model_path: Exact path to write the model pickle to.

    Returns:
        str: Path to the saved model file (same as model_path).

    Raises:
        LibraryNotSupportedError: If library is not "synthcity"
        SerializationError: If saving fails
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "format_version": SERIALIZE_FORMAT_VERSION,
        "library": library,
        "saved_at": datetime.now().isoformat(),
        **metadata,
    }

    try:
        if library == "synthcity":
            if not SYNTHCITY_AVAILABLE:
                raise LibraryNotSupportedError(
                    "Synthcity not available. Install with: pip install synthcity"
                )
            model_data["model_bytes"] = synthcity_save(model)
        else:
            raise LibraryNotSupportedError(f"Library '{library}' not supported")

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"📁 Model saved: {model_path} ({library} format)")
        return str(model_path)

    except LibraryNotSupportedError:
        raise
    except Exception as e:
        raise SerializationError(f"Failed to save {library} model: {e}") from e


def load_model(model_path: Path | str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a trained model with a unified interface (synthcity only for now).

    Args:
        model_path: Exact path to the model pickle.

    Returns:
        Tuple[model, metadata]: Loaded model object and its metadata dict.

    Raises:
        ModelNotFoundError: If model_path doesn't exist
        SerializationError: If loading fails
        LibraryNotSupportedError: If the pickled library is not "synthcity"
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise ModelNotFoundError(f"Model file not found: {model_path}")

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        if not isinstance(model_data, dict):
            raise SerializationError("Invalid model file format - expected dict with metadata")

        library = model_data.get("library", "synthcity")
        format_version = model_data.get("format_version", "1.0")

        if format_version != SERIALIZE_FORMAT_VERSION:
            warnings.warn(
                f"Model format version {format_version} differs from current "
                f"{SERIALIZE_FORMAT_VERSION}. Loading may fail."
            )

        if library == "synthcity":
            if not SYNTHCITY_AVAILABLE:
                raise LibraryNotSupportedError(
                    "Synthcity not available. Install with: pip install synthcity"
                )
            model = synthcity_load(model_data["model_bytes"])
        else:
            raise LibraryNotSupportedError(f"Library '{library}' not supported")

        return model, model_data

    except (ModelNotFoundError, LibraryNotSupportedError):
        raise
    except pickle.UnpicklingError as e:
        raise SerializationError(f"Failed to unpickle model file: {e}") from e
    except Exception as e:
        raise SerializationError(f"Failed to load model: {e}") from e


def create_model_metadata(
    model_type: str,
    library: str,
    seed: int,
    training_time: float,
    data: pd.DataFrame,
    experiment_id: str,
    experiment_hash: str,
    parameters: Dict[str, Any],
    run_params: Dict[str, Any],
    generation: int = 0,
    parent_model_id: Optional[str] = None,
    researcher: str = "anonymous",
) -> Dict[str, Any]:
    """
    Create standardized metadata for model serialization.

    Flattened from the original's `cfg: DictConfig` — `run_params` replaces
    `OmegaConf.to_container(cfg, resolve=True)` (pass e.g. vars(argparse
    Namespace) for full run provenance). `generation`/`parent_model_id`
    default to 0/None here (no recursion in this step yet — Step 5 will
    pass real values through). The lineage hash fields
    (root_training_hash/reference_hash/training_hash) don't have a real
    source at this step either; populated as "unknown" rather than dropped,
    to keep the metadata shape stable for Step 5 to fill in later.

    Args:
        model_type: e.g. "arf", "ctgan"
        library: e.g. "synthcity"
        seed: Model/experiment seed
        training_time: Training time in seconds
        data: Training data (used for shape/columns only)
        experiment_id: Unique experiment identifier
        experiment_hash: Configuration hash (see create_experiment_hash)
        parameters: The model hyperparameters actually used
        run_params: Full run provenance (e.g. the CLI args), stored as-is
        generation: Recursive generation counter (0 = non-recursive)
        parent_model_id: Parent model's experiment_id, if generation > 0
        researcher: Free-text attribution field

    Returns:
        Dict with standardized metadata structure.
    """
    return {
        "model_type": model_type,
        "experiment": {
            "id": experiment_id,
            "seed": seed,
            "hash": experiment_hash,
            "timestamp": datetime.now().isoformat(),
            "researcher": researcher,
        },
        "lineage": {
            "generation": generation,
            "parent_model_id": parent_model_id,
            "root_training_hash": "unknown",
            "reference_hash": "unknown",
            "training_hash": "unknown",
        },
        "params": run_params,
        "training_data_shape": list(data.shape),
        "training_time": training_time,
        "training_data_columns": list(data.columns),
        "parameters": dict(parameters) if parameters else {},
    }
