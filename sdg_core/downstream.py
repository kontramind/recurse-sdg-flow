"""
Downstream Task Training Module
Implements LGBM hyperparameter tuning with Bayesian Optimization for binary classification tasks.

Ported from sdpype/core/downstream.py — keeps LGBMBayesianTuner and evaluate_model
verbatim (no Hydra/DVC coupling in the original). The console-display helpers
(display_confusion_matrix, display_clinical_performance, display_transfer_gap_comparison)
and the older train_readmission_model/save_model_and_metrics convenience functions
were not ported — they aren't used by flows/lgbm_cv_flow.py.
"""

from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from betacal import BetaCalibration
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')


class LGBMBayesianTuner:
    """
    LGBM hyperparameter tuner using Bayesian Optimization

    Optimizes hyperparameters using Optuna with Tree-structured Parzen Estimator (TPE)
    and k-fold cross-validation with AUROC as the primary metric.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_folds: int = 5,
        n_trials: int = 100,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize the tuner

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels (binary)
        n_folds : int
            Number of cross-validation folds (default: 5)
        n_trials : int
            Number of Bayesian optimization trials (default: 100)
        random_state : int
            Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.study = None
        self.best_threshold = 0.5  # Default threshold
        self.n_jobs = n_jobs

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization

        Parameters:
        -----------
        trial : optuna.Trial
            A trial object from Optuna

        Returns:
        --------
        float : Mean AUROC across folds
        """

        # Define hyperparameter search space
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 4, 60),
            'max_depth': trial.suggest_int('max_depth', 1, 15),
            'learning_rate': trial.suggest_float('learning_rate', 2**(-8), 2**0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 60),
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }

        # Regularization parameters (helps with overfitting)
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 10.0)  # L1 regularization
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 10.0)  # L2 regularization

        # Feature sampling (column subsampling - helps generalization)
        params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.5, 1.0)

        # Leaf constraints — Pilgram Table S13: 1–60
        params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, 60)

        # Class imbalance handling (mutually exclusive options)
        imbalance_method = trial.suggest_categorical(
            'imbalance_method',
            ['none', 'scale_pos_weight', 'is_unbalance']
        )

        if imbalance_method == 'scale_pos_weight':
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            params['scale_pos_weight'] = scale_pos_weight
        elif imbalance_method == 'is_unbalance':
            params['is_unbalance'] = True

        # Early stopping rounds
        early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 7, 30)

        # Pilgram CV-based decisions: target encoding and beta calibration
        use_target_encoding = trial.suggest_categorical('use_target_encoding', [True, False])
        use_calibration = trial.suggest_categorical('use_calibration', [True, False])

        # Perform k-fold cross-validation
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # Target encoding: replace categoricals with per-category target mean
            if use_target_encoding:
                X_tr_enc, X_val_enc = X_tr.copy(), X_val.copy()
                cat_cols = X_tr.select_dtypes(include=['category', 'object']).columns
                for col in cat_cols:
                    means = y_tr.groupby(X_tr[col].astype(object)).mean()
                    fallback = means.mean()
                    X_tr_enc[col] = X_tr[col].astype(object).map(means).fillna(fallback).astype(float)
                    X_val_enc[col] = X_val[col].astype(object).map(means).fillna(fallback).astype(float)
            else:
                X_tr_enc, X_val_enc = X_tr, X_val

            # Beta calibration: hold out 20% of train fold to fit calibrator
            if use_calibration:
                X_tr_inner, X_cal, y_tr_inner, y_cal = train_test_split(
                    X_tr_enc, y_tr, test_size=0.2,
                    random_state=self.random_state, stratify=y_tr
                )
                train_data = lgb.Dataset(X_tr_inner, label=y_tr_inner)
            else:
                train_data = lgb.Dataset(X_tr_enc, label=y_tr)

            val_data = lgb.Dataset(X_val_enc, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )

            y_pred = model.predict(X_val_enc, num_iteration=model.best_iteration)

            if use_calibration:
                cal_preds = model.predict(X_cal, num_iteration=model.best_iteration).reshape(-1, 1)
                calibrator = BetaCalibration()
                calibrator.fit(cal_preds, y_cal)
                y_pred = calibrator.predict(y_pred.reshape(-1, 1))

            auroc = roc_auc_score(y_val, y_pred)
            cv_scores.append(auroc)

        return np.mean(cv_scores)

    def tune(self, timeout: Optional[int] = None, callbacks: Optional[list] = None) -> Dict[str, Any]:
        """
        Run Bayesian Optimization to find best hyperparameters

        Parameters:
        -----------
        timeout : int, optional
            Maximum time in seconds for optimization

        Returns:
        --------
        dict : Best hyperparameters found
        """

        # Create Optuna study with TPE sampler (Bayesian Optimization)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )

        # Optimize
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=timeout,
            callbacks=callbacks or [],
            show_progress_bar=False
        )

        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        return self.best_params

    def get_best_model_params(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get the best parameters formatted for LightGBM model training

        Returns:
        --------
        tuple : (lgbm_params, preprocessing_flags)
        """
        if self.best_params is None:
            raise ValueError("Must run tune() before getting best parameters")

        # Extract early stopping
        early_stopping_rounds = self.best_params.get('early_stopping_rounds', 10)

        # Format parameters for LightGBM
        lgbm_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': self.best_params['boosting_type'],
            'num_leaves': self.best_params['num_leaves'],
            'max_depth': self.best_params['max_depth'],
            'learning_rate': self.best_params['learning_rate'],
            'min_child_samples': self.best_params['min_child_samples'],
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': -1,
            # Regularization
            'reg_alpha': self.best_params.get('reg_alpha', 0.0),
            'reg_lambda': self.best_params.get('reg_lambda', 0.0),
            # Feature sampling
            'feature_fraction': self.best_params.get('feature_fraction', 1.0),
            # Leaf constraints
            'min_data_in_leaf': self.best_params.get('min_data_in_leaf', 20),
        }

        # Add class imbalance handling if selected
        imbalance_method = self.best_params.get('imbalance_method', 'none')
        if imbalance_method == 'scale_pos_weight':
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            lgbm_params['scale_pos_weight'] = scale_pos_weight
        elif imbalance_method == 'is_unbalance':
            lgbm_params['is_unbalance'] = True

        preprocessing_flags = {
            'early_stopping_rounds': early_stopping_rounds,
            'use_target_encoding': self.best_params.get('use_target_encoding', False),
            'use_calibration': self.best_params.get('use_calibration', False),
        }

        return lgbm_params, preprocessing_flags

    def train_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Tuple[lgb.Booster, Optional[Dict], Optional[Any]]:
        """
        Train final model with best parameters, applying target encoding and
        beta calibration if selected during HPO.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        X_val : pd.DataFrame, optional
            Validation features (used for early stopping)
        y_val : pd.Series, optional
            Validation labels

        Returns:
        --------
        tuple : (model, target_encoder, calibrator)
            target_encoder : dict {col -> means_series, '_global_mean' -> float} or None
            calibrator     : fitted BetaCalibration instance or None
        """
        if self.best_params is None:
            raise ValueError("Must run tune() before training final model")

        lgbm_params, preprocessing_flags = self.get_best_model_params()

        # Apply target encoding if selected
        target_encoder = None
        if preprocessing_flags['use_target_encoding']:
            target_encoder = {}
            cat_cols = X_train.select_dtypes(include=['category', 'object']).columns
            X_train = X_train.copy()
            if X_val is not None:
                X_val = X_val.copy()
            global_mean = float(y_train.mean())
            for col in cat_cols:
                means = y_train.groupby(X_train[col].astype(object)).mean()
                target_encoder[col] = {"means": means, "global_mean": global_mean}
                X_train[col] = X_train[col].astype(object).map(means).fillna(global_mean).astype(float)
                if X_val is not None:
                    X_val[col] = X_val[col].astype(object).map(means).fillna(global_mean).astype(float)

        # Fit beta calibrator on held-out 20% if selected
        calibrator = None
        if preprocessing_flags['use_calibration']:
            X_tr_inner, X_cal, y_tr_inner, y_cal = train_test_split(
                X_train, y_train, test_size=0.2,
                random_state=self.random_state, stratify=y_train
            )
            train_data = lgb.Dataset(X_tr_inner, label=y_tr_inner)
        else:
            train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)

        model = lgb.train(
            lgbm_params,
            train_data,
            num_boost_round=500,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=preprocessing_flags['early_stopping_rounds'],
                    verbose=False,
                ),
                lgb.log_evaluation(period=0)
            ]
        )

        if preprocessing_flags['use_calibration']:
            cal_preds = model.predict(X_cal, num_iteration=model.best_iteration).reshape(-1, 1)
            calibrator = BetaCalibration()
            calibrator.fit(cal_preds, y_cal)

        return model, target_encoder, calibrator

    def find_optimal_threshold(
        self,
        model: lgb.Booster,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = 'f1',
        calibrator: Optional[Any] = None,
    ) -> float:
        """
        Find optimal classification threshold on validation data.
        If a calibrator is provided, predictions are calibrated before thresholding.

        Parameters:
        -----------
        model : lgb.Booster
            Trained model
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation labels
        metric : str
            Metric to optimize ('f1', 'precision', 'recall')
        calibrator : BetaCalibration, optional
            Fitted calibrator to apply to raw predictions

        Returns:
        --------
        float : Optimal threshold
        """
        from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        if calibrator is not None:
            y_pred_proba = calibrator.predict(y_pred_proba.reshape(-1, 1))

        # Fine-grained sweep: 0.01 steps across full probability range
        thresholds = np.arange(0.01, 0.99, 0.01)
        best_score = -np.inf
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred, zero_division=0)
            elif metric == 'youden':
                # Youden's J = sensitivity + specificity - 1
                # More stable than F1 under class imbalance — has a sharper peak
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                score = sensitivity + specificity - 1.0
            else:
                raise ValueError(f"Unknown metric: {metric}. Choose from: f1, precision, recall, youden")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.best_threshold = best_threshold
        return best_threshold


def evaluate_model(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    calibrator: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on test set with multiple metrics.
    If a calibrator is provided, raw predictions are calibrated before evaluation.

    Parameters:
    -----------
    model : lgb.Booster
        Trained LightGBM model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    threshold : float
        Classification threshold for binary predictions
    calibrator : BetaCalibration, optional
        Fitted calibrator to apply to raw predictions

    Returns:
    --------
    dict : Dictionary of evaluation metrics including confusion matrix
    """
    from sklearn.metrics import confusion_matrix

    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    if calibrator is not None:
        y_pred_proba = calibrator.predict(y_pred_proba.reshape(-1, 1))

    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    metrics = {
        'auroc': float(roc_auc_score(y_test, y_pred_proba)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'threshold': threshold,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }

    return metrics
