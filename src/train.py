"""End-to-end training entrypoint (PRD §4.5 + §4.9).

Raw CSV -> engineered features -> leak-free preprocessing -> 6-model
stratified CV comparison -> best model refit on full train split -> value-based
threshold -> save versioned artifacts -> log everything to MLflow.

Run:
    python -m src.train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings
from src.data.loader import SessionDataLoader
from src.features.engineering import add_engineered_features
from src.models.evaluation import cross_validate_all_models, rank_models
from src.models.trainer import ModelZoo
from src.preprocessing.pipeline import build_preprocessor, get_feature_columns
from src.utils.io import save_artifact, save_json
from src.utils.logging_utils import get_logger
from src.utils.metrics import compute_full_metrics, find_value_based_threshold

logger = get_logger(__name__)


def run(config_path: str | None = None) -> dict:
    settings = get_settings(config_path)
    Path(settings.paths.resolve("model_dir")).mkdir(parents=True, exist_ok=True)
    Path(settings.paths.resolve("mlruns_dir")).mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{settings.paths.resolve('mlruns_dir')}")
    mlflow.set_experiment("purchase-intent-xai")

    # 1. Load + validate -----------------------------------------------------
    df = SessionDataLoader(settings=settings).load()

    # 2. Feature engineering --------------------------------------------------
    df = add_engineered_features(df)

    target = settings.data.target
    # config.yaml's skewed/plain numeric lists already include the engineered
    # columns added above, so get_feature_columns() is the single source of
    # truth for what the preprocessor (and therefore the model) sees.
    feature_cols = get_feature_columns(settings)
    X = df[feature_cols]
    y = df[target]

    # 3. Split: train / holdout test ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.data.test_size,
        stratify=y if settings.data.stratify else None,
        random_state=settings.seed,
    )
    logger.info(
        "Train: %d rows (%.1f%% positive) | Test: %d rows (%.1f%% positive)",
        len(X_train),
        100 * y_train.mean(),
        len(X_test),
        100 * y_test.mean(),
    )

    preprocessor = build_preprocessor(settings)

    with mlflow.start_run(run_name="six_model_comparison"):
        mlflow.log_params(
            {
                "seed": settings.seed,
                "imbalance_strategy": settings.imbalance.strategy,
                "use_page_values": settings.data.use_page_values,
                "cv_folds": settings.models.cv_folds,
            }
        )

        # 4. Stratified CV comparison across all 6 models on TRAIN split only --
        cv_results = cross_validate_all_models(X_train, y_train, preprocessor, settings)
        for name, metrics in cv_results.items():
            mlflow.log_metrics(
                {f"cv_{name}_{k}": v for k, v in metrics.items() if isinstance(v, int | float)}
            )

        ranking = rank_models(cv_results, scoring="pr_auc")
        best_name = ranking[0]
        logger.info("Model ranking by CV PR-AUC: %s", ranking)
        logger.info("Selected best model: %s", best_name)

        # 5. Refit best model on the full train split, evaluate on held-out test
        n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        use_smote = settings.imbalance.strategy == "smote"

        zoo = ModelZoo(settings)
        best_pipeline = zoo.build_pipeline(
            best_name, clone(preprocessor), scale_pos_weight=scale_pos_weight, use_smote=use_smote
        )
        best_pipeline.fit(X_train, y_train)

        y_prob_test = best_pipeline.predict_proba(X_test)[:, 1]

        # 6. Value-based decision threshold, chosen on the held-out test split -
        threshold_info = find_value_based_threshold(
            y_test.to_numpy(),
            y_prob_test,
            settings.threshold.value_per_conversion,
            settings.threshold.cost_per_intervention,
        )
        test_metrics = compute_full_metrics(
            y_test.to_numpy(), y_prob_test, threshold=threshold_info["threshold"]
        )
        test_metrics_default = compute_full_metrics(y_test.to_numpy(), y_prob_test, threshold=0.5)

        logger.info(
            "Chosen threshold=%.3f | test PR-AUC=%.4f recall=%.4f precision=%.4f",
            threshold_info["threshold"],
            test_metrics["pr_auc"],
            test_metrics["recall"],
            test_metrics["precision"],
        )

        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, int | float)}
        )
        mlflow.log_metric("chosen_threshold", threshold_info["threshold"])
        mlflow.sklearn.log_model(
            best_pipeline, artifact_path="model", registered_model_name="purchase_intent_best_model"
        )

        # 7. Persist versioned artifacts for API/dashboard reuse ---------------
        save_artifact(best_pipeline, settings.paths.resolve("model_artifact"))
        save_artifact(preprocessor, settings.paths.resolve("preprocessor_artifact"))
        save_artifact(feature_cols, settings.paths.resolve("feature_names_artifact"))

        report = {
            "best_model": best_name,
            "model_ranking_by_cv_pr_auc": ranking,
            "cv_results": cv_results,
            "test_metrics_default_threshold_0.5": test_metrics_default,
            "test_metrics_chosen_threshold": test_metrics,
            "chosen_threshold_info": threshold_info,
            "feature_columns": feature_cols,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        save_json(report, settings.paths.resolve("metrics_report"))

    logger.info(
        "Training complete. Best model: %s. Artifacts written to %s",
        best_name,
        settings.paths.resolve("model_dir"),
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the purchase-intent model.")
    parser.add_argument("--config", default=None, help="Path to config.yaml override")
    args = parser.parse_args()
    try:
        run(config_path=args.config)
    except Exception:  # noqa: BLE001
        logger.exception("Training failed.")
        sys.exit(1)
