"""Build and persist the SHAP/LIME explainer + model card (PRD §4.6).

Run after src/train.py. Loads the saved pipeline, fits the explainer once on
a background sample, saves it to disk (so the API/dashboard never recompute
it per request), renders a global SHAP summary bar chart, and writes the
model card.

Run:
    python -m src.train_explainers
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.settings import get_settings
from src.data.loader import SessionDataLoader
from src.explain.explainer import PurchaseIntentExplainer
from src.features.engineering import add_engineered_features
from src.preprocessing.pipeline import get_feature_columns
from src.utils.io import load_artifact, load_json, save_artifact
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_CARD_TEMPLATE = """# Model Card — Purchase-Intent Prediction

*Generated {generated_at}.*

## Intended use
Predicts, from within-session browsing behaviour on an e-commerce site,
whether the session will end in a purchase. Intended for a growth /
conversion-rate-optimisation (CRO) team to prioritise which live sessions
receive an intervention (discount nudge, free-shipping banner, remarketing
audience). The model outputs a probability and a recommended decision at a
business-chosen threshold; **the intervention decision itself remains a
human/business call**, not an automated action.

## Intended NOT use
- Not intended for individual-level surveillance, profiling, or any decision
  tied to a person's identity — scoring should stay at the session level and
  never be joined back to personally identifying fields.
- Not validated for traffic patterns materially different from the training
  distribution (e.g. a different industry vertical, a mobile-only app, B2B
  purchasing flows).
- Not a substitute for A/B-testing the actual interventions it triggers —
  it estimates conversion *likelihood*, not the causal effect of an
  intervention.

## Training data
- `dataset/ecommerce_sessions.csv` — {n_rows} sessions, {n_features} input
  features (raw + engineered), anonymised session-level behavioural data.
- Positive (purchase) class prevalence: {pos_rate:.1%}.
- Train/test split: {n_train} train / {n_test} held-out test, stratified on
  the target, seed={seed}.

## Feature handling notes
- `OperatingSystems`, `Browser`, `Region`, `TrafficType` are integer-coded
  **categorical IDs**, one-hot encoded — never treated as continuous.
- Duration/count columns are right-skewed with many zeros; transformed with
  log1p + RobustScaler.
- **PageValues leakage decision:** PageValues is highly predictive but partly
  reflects pages seen close to checkout. This model is configured with
  `use_page_values={use_page_values}` — i.e. it is intended for scoring
  sessions that are complete or near-complete, not for very-early-session
  scoring where PageValues would not yet be known. See the PageValues
  ablation in `models/metrics_report.json` for the honest performance gap
  without it.

## Model selection
Six algorithms compared with 5-fold stratified cross-validation, ranked by
PR-AUC (imbalance-aware; accuracy is misleading at {pos_rate:.1%} positive
prevalence):

{ranking_table}

**Selected model: `{best_model}`**

## Performance (held-out test set)
At the value-based decision threshold ({threshold:.3f}, chosen from
value_per_conversion={value_per_conversion} vs
cost_per_intervention={cost_per_intervention} in config.yaml):

| Metric | Value |
|---|---|
| PR-AUC | {test_pr_auc:.4f} |
| ROC-AUC | {test_roc_auc:.4f} |
| Precision | {test_precision:.4f} |
| Recall | {test_recall:.4f} |
| F1 | {test_f1:.4f} |
| Accuracy | {test_accuracy:.4f} |

For comparison, accuracy alone at this prevalence: a model predicting "no
purchase" for every session would score ~{naive_accuracy:.1%} and capture
zero buyers — accuracy is reported for completeness only, not as the
headline metric.

## Explainability
- **Global:** SHAP TreeExplainer summary (`docs/shap_global_summary.png`)
  shows which behaviours drive predictions overall.
- **Local:** SHAP per-session contributions, cross-checked with an
  independent LIME explanation, are available via `/predict` in the API and
  the dashboard's explainability section.

## Limitations
- Trained on a single historical window; seasonal/behavioural drift over
  time is not monitored automatically (see stretch goal: data-drift check).
- Session-level features only — no user-level history, so returning-visitor
  effects are captured only via `VisitorType` / `is_returning_visitor`, not
  a full customer history.
- Class-imbalance handling is `class_weight`-based; not recalibrated —
  predicted probabilities are useful for *ranking* sessions but should be
  calibrated (see stretch goal) before being read as literal probabilities.

## Responsible use
This is anonymised session behaviour, but the techniques generalise to real
user tracking. Do not tie predictions to personally identifying fields.
Interventions should remain reversible and proportionate to the confidence
of the prediction; treat the model's output as a decision-support signal for
a human-owned trade-off, not an autonomous action.
"""


def _ranking_table(ranking: list[str], cv_results: dict) -> str:
    lines = ["| Rank | Model | CV PR-AUC | CV ROC-AUC | CV Recall |", "|---|---|---|---|---|"]
    for i, name in enumerate(ranking, start=1):
        m = cv_results[name]
        lines.append(
            f"| {i} | {name} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['recall']:.4f} |"
        )
    return "\n".join(lines)


def run(config_path: str | None = None) -> None:
    settings = get_settings(config_path)

    pipeline = load_artifact(settings.paths.resolve("model_artifact"))
    report = load_json(settings.paths.resolve("metrics_report"))

    df = SessionDataLoader(settings=settings).load()
    df = add_engineered_features(df)
    feature_cols = get_feature_columns(settings)
    X = df[feature_cols]

    logger.info("Fitting SHAP/LIME explainer on background sample ...")
    background = X.sample(n=min(300, len(X)), random_state=settings.seed)
    explainer = PurchaseIntentExplainer(pipeline, background)

    save_artifact(explainer, settings.paths.resolve("explainer_artifact"))
    logger.info("Saved explainer -> %s", settings.paths.resolve("explainer_artifact"))

    # Global SHAP summary plot -------------------------------------------
    global_result = explainer.global_shap_values(X, max_rows=800)
    top20 = global_result["feature_importance"][:20]
    fig, ax = plt.subplots(figsize=(8, 7))
    features = [d["feature"] for d in reversed(top20)]
    values = [d["mean_abs_shap"] for d in reversed(top20)]
    ax.barh(features, values, color="#3b6ea5")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    docs_dir = settings.paths.resolve("model_card").parent
    docs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = docs_dir / "shap_global_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Saved global SHAP summary plot -> %s", fig_path)

    # Model card ------------------------------------------------------------
    y = df[settings.data.target]
    naive_accuracy = 1 - y.mean()
    test_m = report["test_metrics_chosen_threshold"]
    card = MODEL_CARD_TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_rows=len(df),
        n_features=len(feature_cols),
        pos_rate=y.mean(),
        n_train=report["n_train"],
        n_test=report["n_test"],
        seed=settings.seed,
        use_page_values=settings.data.use_page_values,
        ranking_table=_ranking_table(report["model_ranking_by_cv_pr_auc"], report["cv_results"]),
        best_model=report["best_model"],
        threshold=report["chosen_threshold_info"]["threshold"],
        value_per_conversion=settings.threshold.value_per_conversion,
        cost_per_intervention=settings.threshold.cost_per_intervention,
        test_pr_auc=test_m["pr_auc"],
        test_roc_auc=test_m["roc_auc"],
        test_precision=test_m["precision"],
        test_recall=test_m["recall"],
        test_f1=test_m["f1"],
        test_accuracy=test_m["accuracy"],
        naive_accuracy=naive_accuracy,
    )
    card_path = settings.paths.resolve("model_card")
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    logger.info("Saved model card -> %s", card_path)


if __name__ == "__main__":
    try:
        run()
    except Exception:  # noqa: BLE001
        logger.exception("Explainer training failed.")
        sys.exit(1)
