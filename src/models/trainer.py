"""OOP model trainers with a common interface (PRD §4.5).

Each algorithm is wrapped in a ModelTrainer subclass exposing the same
`build()` -> sklearn-compatible estimator method, so `ModelZoo` can iterate
over all six uniformly and adding a seventh model later is a one-class change.

Class-imbalance handling (~15.7% positive rate) is via class_weight /
scale_pos_weight by default (see config.yaml: imbalance.strategy), applied
natively by each estimator — never by resampling the full dataset up front.
"""

from __future__ import annotations

import abc

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config.settings import Settings


class ModelTrainer(abc.ABC):
    """Common interface every model wrapper implements."""

    name: str

    def __init__(self, seed: int):
        self.seed = seed

    @abc.abstractmethod
    def build(self, scale_pos_weight: float) -> BaseEstimator:
        """Return an unfitted, ready-to-fit estimator."""


class LogisticRegressionTrainer(ModelTrainer):
    name = "logistic_regression"

    def build(self, scale_pos_weight: float) -> BaseEstimator:
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.seed,
        )


class DecisionTreeTrainer(ModelTrainer):
    name = "decision_tree"

    def build(self, scale_pos_weight: float) -> BaseEstimator:
        return DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=self.seed,
        )


class RandomForestTrainer(ModelTrainer):
    name = "random_forest"

    def build(self, scale_pos_weight: float) -> BaseEstimator:
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.seed,
        )


class XGBoostTrainer(ModelTrainer):
    name = "xgboost"

    def build(self, scale_pos_weight: float) -> BaseEstimator:
        return XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=self.seed,
            n_jobs=-1,
        )


class LightGBMTrainer(ModelTrainer):
    name = "lightgbm"

    def build(self, scale_pos_weight: float) -> BaseEstimator:
        return LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )


class CatBoostTrainer(ModelTrainer):
    name = "catboost"

    def build(self, scale_pos_weight: float) -> BaseEstimator:
        return CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            verbose=False,
        )


MODEL_REGISTRY: dict[str, type[ModelTrainer]] = {
    "logistic_regression": LogisticRegressionTrainer,
    "decision_tree": DecisionTreeTrainer,
    "random_forest": RandomForestTrainer,
    "xgboost": XGBoostTrainer,
    "lightgbm": LightGBMTrainer,
    "catboost": CatBoostTrainer,
}


class ModelZoo:
    """Iterates the full model registry through the common interface."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def all_trainers(self) -> dict[str, ModelTrainer]:
        return {name: cls(seed=self.settings.seed) for name, cls in MODEL_REGISTRY.items()}

    def build_pipeline(
        self, name: str, preprocessor, scale_pos_weight: float, use_smote: bool = False
    ):
        """Wrap preprocessor + (optional SMOTE) + estimator in one fittable Pipeline.

        SMOTE (if enabled) sits inside the pipeline via imblearn.Pipeline so it
        is refit per-CV-fold on the training rows only — it never sees
        validation/test rows, which is the leak that would otherwise inflate
        scores (PRD §4.2 watch-out).
        """
        trainer_cls = MODEL_REGISTRY[name]
        trainer = trainer_cls(seed=self.settings.seed)
        estimator = trainer.build(scale_pos_weight=scale_pos_weight)

        if use_smote:
            return ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=self.settings.seed)),
                    ("classifier", estimator),
                ]
            )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
