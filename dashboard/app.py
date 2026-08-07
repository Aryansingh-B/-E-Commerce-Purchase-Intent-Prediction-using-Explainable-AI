"""Streamlit analyst dashboard (PRD §4.8).

Six sections: overview, live session scoring, funnel/cohort analytics,
model comparison, explainability, performance metrics. Reads saved
artifacts directly (never retrains); caches the model/explainer as
resources and dataframes as data, per the PRD's student hint.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings  # noqa: E402
from src.data.loader import SessionDataLoader  # noqa: E402
from src.features.engineering import add_engineered_features  # noqa: E402
from src.preprocessing.pipeline import get_feature_columns  # noqa: E402
from src.utils.io import load_artifact, load_json  # noqa: E402

st.set_page_config(page_title="Purchase-Intent XAI Dashboard", layout="wide")


@st.cache_resource
def get_pipeline():
    settings = get_settings()
    return load_artifact(settings.paths.resolve("model_artifact"))


@st.cache_resource
def get_explainer():
    settings = get_settings()
    try:
        return load_artifact(settings.paths.resolve("explainer_artifact"))
    except FileNotFoundError:
        return None


@st.cache_data
def get_data() -> pd.DataFrame:
    settings = get_settings()
    df = SessionDataLoader(settings=settings).load()
    return add_engineered_features(df)


@st.cache_data
def get_metrics_report() -> dict:
    settings = get_settings()
    return load_json(settings.paths.resolve("metrics_report"))


settings = get_settings()
df = get_data()
feature_cols = get_feature_columns(settings)
pipeline = get_pipeline()
explainer = get_explainer()
report = get_metrics_report()
threshold = report["chosen_threshold_info"]["threshold"]

st.title("🛒 Purchase-Intent Prediction — Analyst Dashboard")

section = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Live Session Scoring",
        "Funnel & Cohort Analytics",
        "Model Comparison",
        "Explainability",
        "Performance Metrics",
    ],
)

# ---------------------------------------------------------------------------
if section == "Overview":
    st.header("Project overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", f"{len(df):,}")
    col2.metric("Conversion rate", f"{df['Converted'].mean():.1%}")
    col3.metric("Best model", report["best_model"])
    col4.metric("Decision threshold", f"{threshold:.3f}")

    st.markdown(
        """
        This system predicts, from within-session browsing behaviour, whether
        an online shopping session ends in a purchase. Only about **one
        session in six converts**, so the modelling challenge is imbalance-
        aware evaluation, not raw accuracy. Use the sections on the left to
        explore the data, score a live session, compare the six candidate
        models, and see why the chosen model made a given prediction.
        """
    )
    st.subheader("Sample of the raw data")
    st.dataframe(df.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
elif section == "Live Session Scoring":
    st.header("Score a session")
    st.caption("Fills a session form, calls the saved model + preprocessing pipeline directly.")

    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            administrative = st.number_input("Administrative pages", 0, 30, 2)
            administrative_duration = st.number_input(
                "Administrative duration (s)", 0.0, 5000.0, 40.0
            )
            informational = st.number_input("Informational pages", 0, 30, 0)
            informational_duration = st.number_input("Informational duration (s)", 0.0, 5000.0, 0.0)
            product_related = st.number_input("Product-related pages", 0, 800, 25)
            product_related_duration = st.number_input(
                "Product-related duration (s)", 0.0, 30000.0, 600.0
            )
        with c2:
            bounce_rates = st.slider("Bounce rate", 0.0, 1.0, 0.02)
            exit_rates = st.slider("Exit rate", 0.0, 1.0, 0.04)
            page_values = st.number_input("Page values", 0.0, 400.0, 10.0)
            special_day = st.slider("Special-day closeness", 0.0, 1.0, 0.0)
            month = st.selectbox(
                "Month",
                [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "June",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
                index=10,
            )
            weekend = st.checkbox("Weekend session")
        with c3:
            operating_systems = st.number_input("OperatingSystems (ID)", 1, 8, 2)
            browser = st.number_input("Browser (ID)", 1, 13, 2)
            region = st.number_input("Region (ID)", 1, 9, 1)
            traffic_type = st.number_input("TrafficType (ID)", 1, 20, 2)
            visitor_type = st.selectbox(
                "Visitor type", ["Returning_Visitor", "New_Visitor", "Other"]
            )

        submitted = st.form_submit_button("Score session")

    if submitted:
        row = pd.DataFrame(
            [
                {
                    "Administrative": administrative,
                    "Administrative_Duration": administrative_duration,
                    "Informational": informational,
                    "Informational_Duration": informational_duration,
                    "ProductRelated": product_related,
                    "ProductRelated_Duration": product_related_duration,
                    "BounceRates": bounce_rates,
                    "ExitRates": exit_rates,
                    "PageValues": page_values,
                    "SpecialDay": special_day,
                    "Month": month,
                    "OperatingSystems": operating_systems,
                    "Browser": browser,
                    "Region": region,
                    "TrafficType": traffic_type,
                    "VisitorType": visitor_type,
                    "Weekend": weekend,
                }
            ]
        )
        row = add_engineered_features(row)
        X = row[feature_cols]
        prob = float(pipeline.predict_proba(X)[0, 1])
        prediction = int(prob >= threshold)

        c1, c2 = st.columns(2)
        c1.metric("Conversion probability", f"{prob:.1%}")
        c2.metric("Decision", "🟢 Likely purchase" if prediction else "🔴 Unlikely to purchase")

        if explainer is not None:
            st.subheader("Why this prediction — SHAP")
            shap_res = explainer.local_shap_explanation(X, top_k=8)
            shap_df = pd.DataFrame(shap_res["top_contributors"])
            fig = px.bar(
                shap_df,
                x="shap_value",
                y="feature",
                orientation="h",
                color="shap_value",
                color_continuous_scale="RdBu",
                title="Top SHAP contributors",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cross-check — LIME")
            lime_res = explainer.local_lime_explanation(X, top_k=8)
            lime_df = pd.DataFrame(lime_res["top_contributors"])
            st.dataframe(lime_df, use_container_width=True)
        else:
            st.info("Explainer artifact not found. Run `python -m src.train_explainers`.")

# ---------------------------------------------------------------------------
elif section == "Funnel & Cohort Analytics":
    st.header("Funnel & cohort analytics")

    c1, c2 = st.columns(2)
    with c1:
        month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = (
            df.groupby("Month")["Converted"]
            .mean()
            .reindex([m for m in month_order if m in df["Month"].unique()])
            .reset_index()
        )
        fig = px.bar(monthly, x="Month", y="Converted", title="Conversion rate by month")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        visitor = df.groupby("VisitorType")["Converted"].mean().reset_index()
        fig = px.bar(
            visitor, x="VisitorType", y="Converted", title="Conversion rate by visitor type"
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(
            df,
            x="Converted",
            y="PageValues",
            points=False,
            title="PageValues: buyers vs non-buyers",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(
            df, x="Converted", y="ExitRates", points=False, title="ExitRates: buyers vs non-buyers"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Takeaways: conversion is highly seasonal (peaks around Nov), "
        "PageValues separates buyers from non-buyers most sharply, and high "
        "ExitRates tracks with drop-off."
    )

# ---------------------------------------------------------------------------
elif section == "Model Comparison":
    st.header("Model comparison (5-fold stratified CV)")
    cv = report["cv_results"]
    comp_df = pd.DataFrame(cv).T[["pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy"]]
    comp_df = comp_df.sort_values("pr_auc", ascending=False)
    st.dataframe(
        comp_df.style.format("{:.4f}").highlight_max(axis=0, color="#d4f4dd"),
        use_container_width=True,
    )

    fig = px.bar(
        comp_df.reset_index(),
        x="index",
        y="pr_auc",
        title="PR-AUC by model (headline metric — imbalance-aware)",
    )
    fig.update_layout(xaxis_title="model", yaxis_title="PR-AUC")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Selected model: **{report['best_model']}** (highest CV PR-AUC)")

# ---------------------------------------------------------------------------
elif section == "Explainability":
    st.header("Global explainability (SHAP)")
    shap_png = settings.paths.resolve("model_card").parent / "shap_global_summary.png"
    if shap_png.exists():
        st.image(
            str(shap_png), caption="Mean |SHAP value| — top drivers of purchase-intent predictions"
        )
    else:
        st.info("Run `python -m src.train_explainers` to generate the global SHAP summary.")

    model_card_path = settings.paths.resolve("model_card")
    if model_card_path.exists():
        with st.expander("📄 Full model card"):
            st.markdown(model_card_path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
elif section == "Performance Metrics":
    st.header("Held-out test performance")
    test_m = report["test_metrics_chosen_threshold"]
    default_m = report["test_metrics_default_threshold_0.5"]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"At chosen threshold ({threshold:.3f})")
        st.json({k: v for k, v in test_m.items() if k != "confusion_matrix"})
        st.write("Confusion matrix:", test_m["confusion_matrix"])
    with c2:
        st.subheader("At default threshold (0.5)")
        st.json({k: v for k, v in default_m.items() if k != "confusion_matrix"})
        st.write("Confusion matrix:", default_m["confusion_matrix"])

    st.caption(
        "Accuracy is reported for completeness only — with ~15.7% positive "
        "prevalence a model predicting 'no purchase' for everyone scores "
        "~84.5% accuracy while capturing zero buyers. PR-AUC and recall on "
        "the buying class are the metrics that matter here."
    )
