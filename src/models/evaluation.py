"""Stratified CV comparison across all six models (PRD §4.5).

Fits each model's full pipeline (preprocessor [+ SMOTE] + classifier) inside
each CV fold — never on the full dataset first — and reports the imbalance-
aware metric suite for honest model comparison.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config.settings import Settings
from src.models.trainer import ModelZoo
from src.utils.logging_utils import get_logger
from src.utils.metrics import compute_full_metrics

logger = get_logger(__name__)


def cross_validate_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    settings: Settings,
) -> dict[str, dict]:
    """Run stratified CV for every registered model; return metrics per model."""
    zoo = ModelZoo(settings)
    skf = StratifiedKFold(
        n_splits=settings.models.cv_folds, shuffle=True, random_state=settings.seed
    )
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    use_smote = settings.imbalance.strategy == "smote"

    results: dict[str, dict] = {}

    for name in zoo.all_trainers():
        logger.info("Cross-validating %s ...", name)
        pipeline = zoo.build_pipeline(
            name, clone(preprocessor), scale_pos_weight=scale_pos_weight, use_smote=use_smote
        )
        y_prob = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
        metrics = compute_full_metrics(y.to_numpy(), y_prob, threshold=0.5)
        results[name] = metrics
        logger.info(
            "%s: PR-AUC=%.4f ROC-AUC=%.4f recall=%.4f precision=%.4f accuracy=%.4f",
            name,
            metrics["pr_auc"],
            metrics["roc_auc"],
            metrics["recall"],
            metrics["precision"],
            metrics["accuracy"],
        )

    return results


def rank_models(results: dict[str, dict], scoring: str = "pr_auc") -> list[str]:
    """Return model names sorted best-first by the given metric key."""
    return sorted(results.keys(), key=lambda n: results[n][scoring], reverse=True)
