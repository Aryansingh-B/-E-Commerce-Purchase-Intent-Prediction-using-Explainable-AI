"""Explainability wrappers (PRD §4.6).

Global + local SHAP for the chosen model, and an independent LIME lens for
single-session explanations. Both read the saved pipeline (preprocessor +
classifier) so the API, dashboard, and notebooks share one implementation and
never recompute an explainer per request.

Student hint honoured: use TreeExplainer (fast, exact) for tree models;
reserve KernelExplainer for non-tree models. The explainer is built once and
cached to disk via joblib by the caller (src/train_explainers.py), not
recomputed on every API call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline

TREE_MODEL_STEP_TYPES = (
    "CatBoostClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "RandomForestClassifier",
    "DecisionTreeClassifier",
)


class PurchaseIntentExplainer:
    """Wraps a fitted (preprocessor -> classifier) pipeline with SHAP + LIME.

    Works directly on the transformed (one-hot / scaled) feature space, since
    that's what the classifier actually sees. Transformed feature names come
    from the ColumnTransformer's get_feature_names_out().
    """

    def __init__(self, pipeline: Pipeline, background_df: pd.DataFrame):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]

        self.feature_names: list[str] = list(self.preprocessor.get_feature_names_out())

        Xt_background = self._transform(background_df)
        self._background = Xt_background

        model_type = type(self.classifier).__name__
        if model_type in TREE_MODEL_STEP_TYPES:
            self.shap_explainer = shap.TreeExplainer(self.classifier)
        else:
            # Non-tree model (e.g. LogisticRegression): use a small background
            # sample for a KernelExplainer / LinearExplainer-appropriate path.
            sample = shap.sample(Xt_background, min(100, len(Xt_background)))
            self.shap_explainer = shap.KernelExplainer(
                lambda data: self.classifier.predict_proba(data)[:, 1], sample
            )

        # LIME's discretizer holds unpicklable closures, so it is built lazily
        # (and rebuilt after unpickling) instead of stored directly — see
        # __getstate__/__setstate__ below. Rebuilding from the small cached
        # background sample is cheap (no data reload, no refit of the model).
        self._lime_explainer: LimeTabularExplainer | None = None

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        Xt = self.preprocessor.transform(df)
        return np.asarray(Xt)

    @property
    def lime_explainer(self) -> LimeTabularExplainer:
        if self._lime_explainer is None:
            self._lime_explainer = LimeTabularExplainer(
                training_data=self._background,
                feature_names=self.feature_names,
                class_names=["no_purchase", "purchase"],
                mode="classification",
                discretize_continuous=True,
            )
        return self._lime_explainer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lime_explainer"] = None  # never pickle LIME's internal lambdas
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # -- Global -----------------------------------------------------------
    def global_shap_values(self, X: pd.DataFrame, max_rows: int = 1000) -> dict:
        """Mean |SHAP value| per feature across up to max_rows sessions."""
        sample = X.sample(n=min(max_rows, len(X)), random_state=0) if len(X) > max_rows else X
        Xt = self._transform(sample)
        sv = self.shap_explainer.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv  # binary classifiers may return [class0, class1]
        mean_abs = np.abs(sv).mean(axis=0)
        ranked = sorted(
            zip(self.feature_names, mean_abs, strict=False), key=lambda t: t[1], reverse=True
        )
        return {
            "feature_importance": [{"feature": f, "mean_abs_shap": float(v)} for f, v in ranked]
        }

    # -- Local --------------------------------------------------------------
    def local_shap_explanation(self, session_row: pd.DataFrame, top_k: int = 5) -> dict:
        """Per-session SHAP contributions for a single-row DataFrame."""
        Xt = self._transform(session_row)
        sv = self.shap_explainer.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
        contributions = list(zip(self.feature_names, sv[0].tolist(), strict=False))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)
        base_value = self.shap_explainer.expected_value
        base_value = base_value[1] if isinstance(base_value, list | np.ndarray) else base_value
        return {
            "base_value": float(base_value),
            "top_contributors": [
                {"feature": f, "shap_value": float(v)} for f, v in contributions[:top_k]
            ],
        }

    def local_lime_explanation(self, session_row: pd.DataFrame, top_k: int = 5) -> dict:
        """Independent per-session explanation via LIME (PRD §4.6)."""
        Xt = self._transform(session_row)[0]

        def predict_fn(data: np.ndarray) -> np.ndarray:
            return self.classifier.predict_proba(data)

        exp = self.lime_explainer.explain_instance(Xt, predict_fn, num_features=top_k, labels=(1,))
        weights = exp.as_list(label=1)
        return {"top_contributors": [{"feature": f, "weight": float(w)} for f, w in weights]}
