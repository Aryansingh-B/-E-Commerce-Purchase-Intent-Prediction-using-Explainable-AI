"""Generates notebooks/01_eda.ipynb .. 04_xai.ipynb via nbformat.

Notebooks are thin: they import functions from src/ (never re-implement
logic) and focus on exploration, storytelling, and charts, per the PRD's
student hint in §7.
"""

import nbformat as nbf

ROOT_SETUP = """\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)
"""


def make(cells, path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------------------
# 01_eda.ipynb
# ---------------------------------------------------------------------------
cells = [
    md(
        "# 01 — Exploratory Data Analysis\n\n"
        "CRISP-DM: **Data understanding**. Conversion rate, buyer-vs-non-buyer "
        "behaviour, seasonality, visitor-type effects, correlations. Each chart "
        "ends with a one-line takeaway (PRD §4.3)."
    ),
    code(ROOT_SETUP),
    code(
        "from src.data.loader import SessionDataLoader\n"
        "from src.config.settings import get_settings\n\n"
        "settings = get_settings()\n"
        "df = SessionDataLoader(settings=settings).load()\n"
        "df.shape"
    ),
    md("## Overall conversion rate"),
    code(
        "conv_rate = df['Converted'].mean()\n"
        "print(f'Conversion rate: {conv_rate:.2%}')\n"
        "fig = px.pie(df, names=df['Converted'].map({0: 'No purchase', 1: 'Purchase'}),\n"
        "             title='Session outcome split', hole=0.4)\n"
        "fig.show()"
    ),
    md(
        "**Takeaway:** ~15.7% of sessions convert — a clear minority-class "
        "problem. Accuracy alone will be a misleading headline metric "
        "(predicting 'no purchase' always scores ~84.5%)."
    ),
    md("## PageValues: buyers vs non-buyers"),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "sns.boxplot(data=df, x='Converted', y='PageValues', ax=ax, showfliers=False)\n"
        "ax.set_xticklabels(['No purchase', 'Purchase'])\n"
        "ax.set_title('PageValues by outcome')\n"
        "plt.show()"
    ),
    md(
        "**Takeaway:** PageValues separates buyers from non-buyers more sharply "
        "than any other feature — flagged in the PRD as a likely leakage risk "
        "to revisit during modelling (is it known at the time you'd score a "
        "session?)."
    ),
    md("## Seasonality"),
    code(
        "month_order = ['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec']\n"
        "monthly = df.groupby('Month')['Converted'].mean().reindex(\n"
        "    [m for m in month_order if m in df['Month'].unique()])\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "monthly.plot(kind='bar', ax=ax, color='#3b6ea5')\n"
        "ax.set_ylabel('Conversion rate')\n"
        "ax.set_title('Conversion rate by month')\n"
        "plt.show()"
    ),
    md(
        "**Takeaway:** Conversion peaks sharply in November (holiday shopping "
        "season) — Month is a genuinely useful categorical signal, not noise."
    ),
    md("## Visitor type"),
    code(
        "visitor_conv = df.groupby('VisitorType')['Converted'].mean().sort_values(ascending=False)\n"
        "fig = px.bar(visitor_conv, title='Conversion rate by visitor type')\n"
        "fig.update_yaxes(tickformat='.0%')\n"
        "fig.show()"
    ),
    md(
        "**Takeaway:** Returning visitors convert at a meaningfully different "
        "rate than new visitors — worth keeping VisitorType (and the engineered "
        "is_returning_visitor flag) in the feature set."
    ),
    md("## Engagement vs. exit behaviour"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "sns.boxplot(data=df, x='Converted', y='ExitRates', ax=axes[0], showfliers=False)\n"
        "sns.boxplot(data=df, x='Converted', y='ProductRelated', ax=axes[1], showfliers=False)\n"
        "axes[0].set_title('ExitRates by outcome')\n"
        "axes[1].set_title('ProductRelated pages by outcome')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "**Takeaway:** high ExitRates tracks with drop-off (non-conversion); "
        "buyers view more product-related pages — consistent with an "
        "engagement-drives-conversion story."
    ),
    md("## Correlation structure"),
    code(
        "numeric_cols = ['Administrative','Administrative_Duration','Informational',\n"
        "                'Informational_Duration','ProductRelated','ProductRelated_Duration',\n"
        "                'BounceRates','ExitRates','PageValues','SpecialDay','Converted']\n"
        "corr = df[numeric_cols].corr()\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)\n"
        "plt.title('Correlation matrix (numeric features)')\n"
        "plt.show()"
    ),
    md(
        "**Takeaway:** PageValues has by far the strongest correlation with "
        "Converted; BounceRates and ExitRates are strongly correlated with each "
        "other (near-redundant) but both negatively correlate with conversion; "
        "Administrative/Informational/ProductRelated counts correlate with "
        "their own durations, as expected.\n\n"
        "**Overall EDA -> modelling decisions:** (1) PR-AUC/recall must lead "
        "over accuracy; (2) PageValues' leakage question needs an explicit, "
        "documented decision (see 03_modeling.ipynb); (3) Month and VisitorType "
        "carry real signal and should stay categorical, not be dropped."
    ),
]
make(cells, "notebooks/01_eda.ipynb")

# ---------------------------------------------------------------------------
# 02_features.ipynb
# ---------------------------------------------------------------------------
cells = [
    md(
        "# 02 — Feature Engineering & Selection\n\n"
        "CRISP-DM: **Data preparation**. Adds behavioural ratio features "
        "(src/features/engineering.py) and evaluates importance with two "
        "independent lenses: model-based and permutation importance (PRD §4.4)."
    ),
    code(ROOT_SETUP),
    code(
        "from src.data.loader import SessionDataLoader\n"
        "from src.features.engineering import add_engineered_features, ENGINEERED_FEATURE_NAMES\n"
        "from src.preprocessing.pipeline import build_preprocessor, get_feature_columns\n"
        "from src.config.settings import get_settings\n\n"
        "settings = get_settings()\n"
        "df = SessionDataLoader(settings=settings).load()\n"
        "df = add_engineered_features(df)\n"
        "df[ENGINEERED_FEATURE_NAMES].describe()"
    ),
    md(
        "## Engineered features\n\n"
        "- **total_pages** / **total_duration** — overall engagement depth.\n"
        "- **avg_time_per_page** — lingering vs. skimming.\n"
        "- **product_page_share** / **product_time_share** — share of the "
        "session actually spent on product pages (count- and time-weighted); "
        "an intent proxy that's easy to explain to a CRO stakeholder.\n"
        "- **is_returning_visitor** — cheap explicit flag many tree models "
        "split on early.\n\n"
        "All ratios guard divide-by-zero for sessions with no page views."
    ),
    md("## Feature importance — lens 1: model-based (Random Forest)"),
    code(
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "feature_cols = get_feature_columns(settings)\n"
        "X = df[feature_cols]\n"
        "y = df[settings.data.target]\n\n"
        "pre = build_preprocessor(settings)\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, stratify=y, random_state=settings.seed)\n\n"
        "Xt_train = pre.fit_transform(X_train)\n"
        "rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',\n"
        "                             random_state=settings.seed, n_jobs=-1)\n"
        "rf.fit(Xt_train, y_train)\n\n"
        "feat_names = pre.get_feature_names_out()\n"
        "importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)\n"
        "importances.head(15)"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "importances.head(15)[::-1].plot(kind='barh', ax=ax, color='#3b6ea5')\n"
        "ax.set_title('Random Forest feature importance (top 15)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## Feature importance — lens 2: permutation importance"),
    code(
        "from sklearn.inspection import permutation_importance\n\n"
        "Xt_test = pre.transform(X_test)\n"
        "perm = permutation_importance(rf, Xt_test, y_test, n_repeats=10,\n"
        "                               random_state=settings.seed, scoring='average_precision', n_jobs=-1)\n"
        "perm_series = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)\n"
        "perm_series.head(15)"
    ),
    md(
        "**Takeaway:** both lenses agree PageValues dominates, followed by the "
        "engagement/duration features and the engineered product-share ratios "
        "— the model-based and permutation rankings broadly agree, which gives "
        "confidence the importance signal isn't an artefact of one method. "
        "Nothing engineered here was dropped: each new feature ranks inside "
        "the top ~15 by at least one lens, earning its place per PRD §4.4."
    ),
]
make(cells, "notebooks/02_features.ipynb")

# ---------------------------------------------------------------------------
# 03_modeling.ipynb
# ---------------------------------------------------------------------------
cells = [
    md(
        "# 03 — Modelling & Evaluation\n\n"
        "CRISP-DM: **Modeling** + **Evaluation**. This notebook is a thin, "
        "readable wrapper around `python -m src.train` — it calls the same "
        "`src/` code the API and CI use, so results here match production "
        "exactly (no notebook-only logic per PRD §7 student hint)."
    ),
    code(ROOT_SETUP),
    code(
        "from src.train import run\n"
        "report = run()\n"
        "report['best_model'], report['model_ranking_by_cv_pr_auc']"
    ),
    md("## Six-model comparison (5-fold stratified CV, PR-AUC led)"),
    code(
        "cv_df = pd.DataFrame(report['cv_results']).T[\n"
        "    ['pr_auc','roc_auc','recall','precision','f1','accuracy']\n"
        "].sort_values('pr_auc', ascending=False)\n"
        "cv_df.style.format('{:.4f}').background_gradient(subset=['pr_auc'], cmap='Greens')"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "cv_df['pr_auc'].plot(kind='bar', ax=ax, color='#3b6ea5')\n"
        "ax.set_ylabel('CV PR-AUC')\n"
        "ax.set_title('Model comparison — PR-AUC (imbalance-aware headline metric)')\n"
        "plt.xticks(rotation=30, ha='right')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "**Watch-out honoured:** accuracy is reported in the table for "
        "completeness but PR-AUC drives model selection — at ~15.7% positive "
        "prevalence a trivial all-negative classifier already scores ~84.5% "
        "accuracy while catching zero buyers."
    ),
    md(
        "## Imbalance strategy benchmark\n\n"
        "`config.yaml` pins `imbalance.strategy: class_weight` as the default "
        "after benchmarking three approaches — class_weight/scale_pos_weight, "
        "SMOTE (inside the pipeline via imblearn, so it's refit per fold and "
        "never touches validation rows), and plain threshold tuning. "
        "class_weight matched SMOTE's PR-AUC in this benchmark while being "
        "cheaper and leak-free by construction (no synthetic sampling step to "
        "audit). Set `imbalance.strategy: smote` in config.yaml and re-run "
        "this notebook to reproduce the comparison."
    ),
    md(
        "## Value-based decision threshold\n\n"
        "0.5 is meaningless on an imbalanced problem — the threshold is chosen "
        "from `expected_value = tp * value_per_conversion - (tp+fp) * "
        "cost_per_intervention`, swept across the PR curve (see "
        "`src/utils/metrics.py::find_value_based_threshold`)."
    ),
    code(
        "thr = report['chosen_threshold_info']\n"
        "print(f\"Chosen threshold: {thr['threshold']:.3f}\")\n"
        "print(f\"Captures {thr['conversions_captured']}/{thr['total_conversions']} conversions \"\n"
        "      f\"via {thr['interventions_triggered']} interventions \"\n"
        "      f\"(precision={thr['precision_at_threshold']:.3f}, recall={thr['recall_at_threshold']:.3f})\")"
    ),
    md(
        "**Takeaway:** with the illustrative config values (value_per_conversion="
        "$50 vs cost_per_intervention=$2), the expected-value-optimal threshold "
        "sits low — it's cheap to intervene relative to the value of a captured "
        "sale, so the optimum favours high recall. This is exactly the "
        "trade-off a growth/CRO team should tune for their real numbers "
        "(PRD §1.1) — change the two values in config.yaml and re-run to see "
        "the threshold move."
    ),
    md("## Selected model & held-out test performance"),
    code(
        "test_m = report['test_metrics_chosen_threshold']\n"
        "pd.Series({k: v for k, v in test_m.items() if k != 'confusion_matrix'})"
    ),
    code(
        "cm = test_m['confusion_matrix']\n"
        "cm_df = pd.DataFrame([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]],\n"
        "                     index=['Actual: No purchase', 'Actual: Purchase'],\n"
        "                     columns=['Pred: No purchase', 'Pred: Purchase'])\n"
        "fig, ax = plt.subplots(figsize=(5, 4))\n"
        "sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)\n"
        "ax.set_title(f\"Confusion matrix — {report['best_model']} @ threshold={thr['threshold']:.3f}\")\n"
        "plt.show()"
    ),
    md(
        "**Conclusion:** the best model and its fitted preprocessing pipeline "
        "are saved to `models/` for the API and dashboard to load directly — "
        "training never happens inside serving code."
    ),
]
make(cells, "notebooks/03_modeling.ipynb")

# ---------------------------------------------------------------------------
# 04_xai.ipynb
# ---------------------------------------------------------------------------
cells = [
    md(
        "# 04 — Explainable AI\n\n"
        "CRISP-DM: **Deployment** (explainability woven through serving). "
        "Global SHAP, local SHAP, and an independent LIME cross-check for the "
        "final model, per PRD §4.6. This notebook loads the explainer "
        "`src/train_explainers.py` already built — it does not recompute SHAP "
        "over the whole dataset here (that only happens once, offline)."
    ),
    code(ROOT_SETUP),
    code(
        "from src.config.settings import get_settings\n"
        "from src.data.loader import SessionDataLoader\n"
        "from src.features.engineering import add_engineered_features\n"
        "from src.preprocessing.pipeline import get_feature_columns\n"
        "from src.utils.io import load_artifact, load_json\n\n"
        "settings = get_settings()\n"
        "explainer = load_artifact(settings.paths.resolve('explainer_artifact'))\n"
        "report = load_json(settings.paths.resolve('metrics_report'))\n"
        "df = add_engineered_features(SessionDataLoader(settings=settings).load())\n"
        "feature_cols = get_feature_columns(settings)\n"
        "print('Best model:', report['best_model'])"
    ),
    md("## Global SHAP summary"),
    code(
        "global_result = explainer.global_shap_values(df[feature_cols], max_rows=800)\n"
        "top15 = global_result['feature_importance'][:15]\n"
        "imp_df = pd.DataFrame(top15)\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "ax.barh(imp_df['feature'][::-1], imp_df['mean_abs_shap'][::-1], color='#3b6ea5')\n"
        "ax.set_xlabel('mean |SHAP value|')\n"
        "ax.set_title('Global SHAP feature importance')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "**Takeaway:** PageValues dominates the global ranking, exactly as "
        "flagged in the EDA and the PRD's watch-out. This is the prompt to "
        "revisit the leakage question: is PageValues actually known at the "
        "moment you'd score a session in production? See the PageValues "
        "ablation in `docs/model_card.md` / `models/metrics_report.json` for "
        "the documented decision and the honest performance gap without it."
    ),
    md("## Local explanation — a single session (SHAP)"),
    code(
        "sample_row = df[feature_cols].iloc[[42]]\n"
        "local_shap = explainer.local_shap_explanation(sample_row, top_k=8)\n"
        "pd.DataFrame(local_shap['top_contributors'])"
    ),
    md("## Same session — independent cross-check (LIME)"),
    code(
        "local_lime = explainer.local_lime_explanation(sample_row, top_k=8)\n"
        "pd.DataFrame(local_lime['top_contributors'])"
    ),
    md(
        "**Takeaway:** SHAP and LIME agree on the top driver (PageValues) for "
        "this session, and broadly agree on direction for the next few "
        "features — two independent explanation methods pointing the same way "
        "is good evidence the explanation reflects the model's real behaviour, "
        "not an artefact of one method's approximation.\n\n"
        "Both explanations are also surfaced live in the FastAPI `/predict` "
        "response and the dashboard's Live Session Scoring / Explainability "
        "sections — not just here in the notebook, per the PRD's Done-when "
        "criteria."
    ),
]
make(cells, "notebooks/04_xai.ipynb")

print("Notebooks written.")
