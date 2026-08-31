"""
RDT-based dataset encoding for the ported SDG pipeline.

Ported from sdpype/encoding.py — keeps the transformer registry,
load_encoding_config, validate_config_against_data, and the full
RDTDatasetEncoder class verbatim. The Hydra CLI entrypoint (main()) and its
.sdpype_config_hash sentinel-file helper are dropped: this module is a pure
library used by flows/sdg_flow.py's argparse-driven tasks, not a standalone
DVC/Hydra pipeline stage.

This module provides functionality to:
1. Load encoding configurations from YAML files
2. Instantiate RDT transformers from configuration specs
3. Fit transformers on training data
4. Transform/reverse-transform datasets (dual pipeline support)
5. Serialize fitted encoders for downstream use
"""

import inspect
import logging
import pickle
from pathlib import Path
from typing import Dict, Any

import yaml
import pandas as pd
from rdt import HyperTransformer
from rdt.transformers import (
    UniformEncoder,
    OrderedUniformEncoder,
    OrderedLabelEncoder,
    LabelEncoder,
    OneHotEncoder,
    FrequencyEncoder,
    UnixTimestampEncoder,
    FloatFormatter,
)
from rdt.transformers.boolean import BinaryEncoder

logger = logging.getLogger(__name__)


# =============================================================================
# TRANSFORMER REGISTRY
# =============================================================================

TRANSFORMER_REGISTRY = {
    'UniformEncoder': UniformEncoder,
    'OrderedUniformEncoder': OrderedUniformEncoder,
    'OrderedLabelEncoder': OrderedLabelEncoder,
    'LabelEncoder': LabelEncoder,
    'OneHotEncoder': OneHotEncoder,
    'FrequencyEncoder': FrequencyEncoder,
    'UnixTimestampEncoder': UnixTimestampEncoder,
    'FloatFormatter': FloatFormatter,
    'BinaryEncoder': BinaryEncoder,
}


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_encoding_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse encoding configuration from YAML file.

    Args:
        config_path: Path to YAML encoding configuration file

    Returns:
        Dictionary with structure:
        {
            'sdtypes': {col_name: sdtype, ...},
            'transformers': {col_name: transformer_instance, ...}
        }

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config structure is invalid
        KeyError: If transformer type is not in registry
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Encoding config not found: {config_path}")

    logger.info(f"Loading encoding config from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'sdtypes' not in config:
        raise ValueError("Config must contain 'sdtypes' section")
    if 'transformers' not in config:
        raise ValueError("Config must contain 'transformers' section")

    sdtypes = config['sdtypes']

    transformers = {}
    for col_name, transformer_spec in config['transformers'].items():
        if not isinstance(transformer_spec, dict):
            raise ValueError(
                f"Transformer spec for '{col_name}' must be a dict with 'type' and 'params'"
            )

        transformer_type = transformer_spec.get('type')
        if not transformer_type:
            raise ValueError(f"Transformer spec for '{col_name}' missing 'type' field")

        if transformer_type not in TRANSFORMER_REGISTRY:
            available = ', '.join(TRANSFORMER_REGISTRY.keys())
            raise KeyError(
                f"Unknown transformer type '{transformer_type}' for column '{col_name}'. "
                f"Available types: {available}"
            )

        transformer_class = TRANSFORMER_REGISTRY[transformer_type]

        params = transformer_spec.get('params', {})

        # Filter to only params the constructor accepts
        valid_keys = inspect.signature(transformer_class.__init__).parameters.keys() - {'self'}
        params = {k: v for k, v in params.items() if k in valid_keys}

        try:
            transformer = transformer_class(**params)
            transformers[col_name] = transformer
            logger.debug(f"  {col_name}: {transformer_type}({params})")
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {transformer_type} for '{col_name}' "
                f"with params {params}: {e}"
            )

    logger.info(f"Loaded {len(transformers)} transformer configurations")

    return {
        'sdtypes': sdtypes,
        'transformers': transformers,
        'config_path': str(config_path),
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config_against_data(
    config: Dict[str, Any],
    data: pd.DataFrame,
    require_all_columns: bool = True
) -> None:
    """
    Validate encoding configuration against actual dataset.

    Args:
        config: Config dictionary from load_encoding_config()
        data: DataFrame to validate against
        require_all_columns: If True, require all config columns exist in data

    Raises:
        ValueError: If validation fails
    """
    config_columns = set(config['sdtypes'].keys())
    data_columns = set(data.columns)

    missing_in_data = config_columns - data_columns
    if missing_in_data and require_all_columns:
        raise ValueError(
            f"Columns in encoding config not found in data: {missing_in_data}"
        )

    missing_in_config = data_columns - config_columns
    if missing_in_config:
        logger.warning(
            f"Columns in data not found in encoding config (will not be encoded): "
            f"{missing_in_config}"
        )

    transformer_columns = set(config['transformers'].keys())
    if transformer_columns != config_columns:
        sdtype_only = config_columns - transformer_columns
        transformer_only = transformer_columns - config_columns
        msg = []
        if sdtype_only:
            msg.append(f"sdtypes without transformers: {sdtype_only}")
        if transformer_only:
            msg.append(f"transformers without sdtypes: {transformer_only}")
        raise ValueError(
            f"Mismatch between sdtypes and transformers sections. {' | '.join(msg)}"
        )

    logger.info("✓ Config validation passed")


# =============================================================================
# RDT DATASET ENCODER
# =============================================================================

class RDTDatasetEncoder:
    """
    RDT-based dataset encoder for the SDG pipeline.

    Uses RDT's HyperTransformer under the hood for proper orchestration
    of multiple transformers.

    Supports:
    - Fitting transformers on training data
    - Transforming data to encoded (numeric) format
    - Reverse transforming encoded data back to original format (dual pipeline)
    - Serialization of fitted encoders for downstream use

    Usage:
        config = load_encoding_config('encoding_config.yaml')
        encoder = RDTDatasetEncoder(config)
        encoder.fit(training_df)
        encoded_train = encoder.transform(training_df)
        decoded_synthetic = encoder.reverse_transform(synthetic_encoded_df)
        encoder.save('fitted_encoders.pkl')
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize encoder with configuration.

        Args:
            config: Configuration dict from load_encoding_config()
        """
        self.config = config
        self.sdtypes = config['sdtypes']
        self.transformers = config['transformers']

        # Create HyperTransformer (will be configured during fit)
        self.ht = HyperTransformer()

        self._is_fitted = False
        logger.info(f"Initialized RDTDatasetEncoder with {len(self.sdtypes)} columns")

    def fit(self, training_data: pd.DataFrame) -> 'RDTDatasetEncoder':
        """
        Fit transformers on training data.

        Critical: Transformers are fitted ONLY on training data to prevent
        data leakage from reference/test sets.

        Args:
            training_data: Training DataFrame

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If validation fails
        """
        logger.info("Fitting encoders on training data...")

        validate_config_against_data(self.config, training_data)

        # Step 1: Detect initial config from data (required by RDT)
        self.ht.detect_initial_config(data=training_data)

        # Step 2: Update with custom sdtypes from config
        self.ht.update_sdtypes(column_name_to_sdtype=self.sdtypes)

        # Step 3: Update with custom transformers from config
        self.ht.update_transformers(column_name_to_transformer=self.transformers)

        # Step 4: Fit HyperTransformer on all data at once
        self.ht.fit(data=training_data)

        self._is_fitted = True
        logger.info(f"✓ Fitted {len(self.transformers)} transformers")

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers.

        Args:
            data: DataFrame to transform

        Returns:
            Transformed DataFrame (numeric format)

        Raises:
            RuntimeError: If encoder not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted. Call .fit() first.")

        logger.info(f"Transforming data ({len(data)} rows)...")

        transformed_data = self.ht.transform(data=data)

        logger.info(f"✓ Transformed to {transformed_data.shape[1]} columns")

        return transformed_data

    def reverse_transform(self, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse transform encoded data back to original format.

        This enables the dual pipeline approach where:
        - Metrics operate on encoded (numeric) data
        - Detection/privacy checks operate on native sdtype data

        Args:
            encoded_data: Encoded DataFrame from .transform()

        Returns:
            Decoded DataFrame with original sdtypes

        Raises:
            RuntimeError: If encoder not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted. Call .fit() first.")

        logger.info(f"Reverse transforming data ({len(encoded_data)} rows)...")

        decoded_data = self.ht.reverse_transform(data=encoded_data)

        logger.info(f"✓ Reverse transformed to {decoded_data.shape[1]} columns")

        return decoded_data

    @property
    def is_fitted(self) -> bool:
        """Check if encoder has been fitted."""
        return self._is_fitted

    def save(self, filepath: Path) -> None:
        """
        Save fitted encoders to disk.

        Saves:
        - Fitted HyperTransformer (with learned state)
        - Original config (for reference)
        - Metadata about encoding process

        Args:
            filepath: Path to save pickle file

        Raises:
            RuntimeError: If encoder not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted encoder. Call .fit() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'hyper_transformer': self.ht,
            'sdtypes': self.sdtypes,
            'config': self.config,
            'encoding_version': '2.0',  # Updated version for HyperTransformer
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"✓ Saved fitted encoders to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'RDTDatasetEncoder':
        """
        Load fitted encoders from disk.

        Args:
            filepath: Path to pickle file

        Returns:
            Fitted RDTDatasetEncoder instance

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Encoder file not found: {filepath}")

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        instance = cls(save_data['config'])

        instance.ht = save_data['hyper_transformer']
        instance._is_fitted = True

        logger.info(f"✓ Loaded fitted encoders from: {filepath}")

        return instance
