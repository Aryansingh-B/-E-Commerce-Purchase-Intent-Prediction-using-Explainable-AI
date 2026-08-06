"""Leak-free preprocessing pipeline (PRD §4.2).

A single reusable ColumnTransformer/Pipeline factory. The golden rule: this
must be fit INSIDE cross-validation folds / on the training split only, never
on the full dataset before splitting — otherwise scaler statistics and
one-hot categories leak information about the test rows.

- Categorical (incl. int-coded ID columns like OperatingSystems) -> OneHotEncoder
  with handle_unknown='ignore' so an unseen category at serving time degrades
  gracefully instead of crashing (PRD §4.7 watch-out on train/serve skew).
- Skewed numeric durations/counts -> log1p then RobustScaler (student hint,
  PRD §2.2): tames heavy right skew and the many exact zeros better than
  plain standardisation.
- Plain numeric (SpecialDay) -> RobustScaler only.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config.settings import Settings, get_settings


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """np.log1p wrapper as a proper (picklable, named) sklearn transformer."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn convention
        self.n_features_in_ = np.asarray(X).shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):  # noqa: N803
        X = np.asarray(X, dtype=float)
        X = np.clip(X, a_min=0, a_max=None)  # guard against any stray negatives
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def build_preprocessor(settings: Settings | None = None) -> ColumnTransformer:
    """Build the (unfitted) ColumnTransformer described in the module docstring."""
    settings = settings or get_settings()
    data_cfg = settings.data

    skewed_cols: list[str] = [
        c for c in data_cfg.skewed_numeric_columns if data_cfg.use_page_values or c != "PageValues"
    ]
    plain_cols: list[str] = list(data_cfg.plain_numeric_columns)
    categorical_cols: list[str] = list(data_cfg.all_categorical_columns)

    skewed_numeric_pipeline = Pipeline(
        steps=[
            ("log1p", Log1pTransformer()),
            ("scale", RobustScaler()),
        ]
    )

    plain_numeric_pipeline = Pipeline(
        steps=[
            ("scale", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if skewed_cols:
        transformers.append(("skewed_numeric", skewed_numeric_pipeline, skewed_cols))
    if plain_cols:
        transformers.append(("plain_numeric", plain_numeric_pipeline, plain_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_columns(settings: Settings | None = None) -> list[str]:
    """Ordered list of raw input columns the preprocessor expects."""
    settings = settings or get_settings()
    data_cfg = settings.data
    skewed_cols = [
        c for c in data_cfg.skewed_numeric_columns if data_cfg.use_page_values or c != "PageValues"
    ]
    return (
        skewed_cols + list(data_cfg.plain_numeric_columns) + list(data_cfg.all_categorical_columns)
    )
