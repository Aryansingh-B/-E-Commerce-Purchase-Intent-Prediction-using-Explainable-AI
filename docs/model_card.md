# Model Card — Purchase-Intent Prediction

*Generated 2026-08-26 17:15 UTC.*

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
- `dataset/ecommerce_sessions.csv` — 12000 sessions, 23 input
  features (raw + engineered), anonymised session-level behavioural data.
- Positive (purchase) class prevalence: 15.7%.
- Train/test split: 9600 train / 2400 held-out test, stratified on
  the target, seed=42.

## Feature handling notes
- `OperatingSystems`, `Browser`, `Region`, `TrafficType` are integer-coded
  **categorical IDs**, one-hot encoded — never treated as continuous.
- Duration/count columns are right-skewed with many zeros; transformed with
  log1p + RobustScaler.
- **PageValues leakage decision:** PageValues is highly predictive but partly
  reflects pages seen close to checkout. This model is configured with
  `use_page_values=True` — i.e. it is intended for scoring
  sessions that are complete or near-complete, not for very-early-session
  scoring where PageValues would not yet be known. See the PageValues
  ablation in `models/metrics_report.json` for the honest performance gap
  without it.

## Model selection
Six algorithms compared with 5-fold stratified cross-validation, ranked by
PR-AUC (imbalance-aware; accuracy is misleading at 15.7% positive
prevalence):

| Rank | Model | CV PR-AUC | CV ROC-AUC | CV Recall |
|---|---|---|---|---|
| 1 | catboost | 0.8434 | 0.9671 | 0.8893 |
| 2 | xgboost | 0.8343 | 0.9651 | 0.8496 |
| 3 | lightgbm | 0.8316 | 0.9640 | 0.8105 |
| 4 | random_forest | 0.8243 | 0.9622 | 0.7946 |
| 5 | logistic_regression | 0.8198 | 0.9643 | 0.9264 |
| 6 | decision_tree | 0.7984 | 0.9484 | 0.9046 |

**Selected model: `catboost`**

## Performance (held-out test set)
At the value-based decision threshold (0.024, chosen from
value_per_conversion=50.0 vs
cost_per_intervention=2.0 in config.yaml):

| Metric | Value |
|---|---|
| PR-AUC | 0.8592 |
| ROC-AUC | 0.9715 |
| Precision | 0.4342 |
| Recall | 0.9973 |
| F1 | 0.6050 |
| Accuracy | 0.7954 |

For comparison, accuracy alone at this prevalence: a model predicting "no
purchase" for every session would score ~84.3% and capture
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
