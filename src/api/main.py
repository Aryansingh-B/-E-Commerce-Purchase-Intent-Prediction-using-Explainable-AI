"""FastAPI real-time prediction service (PRD §4.7).

Loads the saved model pipeline + explainer ONCE at startup via FastAPI's
lifespan context (never per-request). Applies the exact same fitted
preprocessing the training pipeline saved, so train/serve skew is structural,
not something that can be re-implemented incorrectly by hand. Unknown
categories at serving time are handled gracefully because the fitted
OneHotEncoder was built with handle_unknown='ignore'.

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.schemas import Contributor, HealthResponse, PredictionResponse, SessionPayload
from src.config.settings import get_settings
from src.utils.io import load_artifact, load_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Loading model artifacts at startup ...")
    STATE["settings"] = settings
    STATE["pipeline"] = load_artifact(settings.paths.resolve("model_artifact"))
    STATE["feature_columns"] = load_artifact(settings.paths.resolve("feature_names_artifact"))
    try:
        STATE["explainer"] = load_artifact(settings.paths.resolve("explainer_artifact"))
    except FileNotFoundError:
        logger.warning(
            "Explainer artifact not found — /predict will skip explanations. "
            "Run `python -m src.train_explainers` to enable them."
        )
        STATE["explainer"] = None
    try:
        report = load_json(settings.paths.resolve("metrics_report"))
        STATE["threshold"] = report["chosen_threshold_info"]["threshold"]
        STATE["model_name"] = report["best_model"]
    except FileNotFoundError:
        STATE["threshold"] = 0.5
        STATE["model_name"] = type(STATE["pipeline"].named_steps["classifier"]).__name__
    logger.info(
        "Model '%s' loaded. Decision threshold=%.4f", STATE["model_name"], STATE["threshold"]
    )
    yield
    STATE.clear()


app = FastAPI(
    title="Purchase-Intent Prediction API",
    description="Scores an e-commerce session for conversion likelihood and explains the prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


def _confidence(prob: float, threshold: float) -> str:
    distance = abs(prob - threshold)
    if distance >= 0.3:
        return "high"
    if distance >= 0.1:
        return "medium"
    return "low"


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model_loaded = STATE.get("pipeline") is not None
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_name=str(STATE.get("model_name", "unknown")),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: SessionPayload) -> PredictionResponse:
    pipeline = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    feature_columns = STATE["feature_columns"]
    threshold = STATE["threshold"]

    row = pd.DataFrame([payload.model_dump()])
    # Re-derive engineered features so /predict matches training exactly.
    from src.features.engineering import add_engineered_features

    row = add_engineered_features(row)

    missing = [c for c in feature_columns if c not in row.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing derived columns: {missing}")

    X = row[feature_columns]

    try:
        prob = float(pipeline.predict_proba(X)[0, 1])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Could not score session: {exc}") from exc

    prediction = int(prob >= threshold)
    confidence = _confidence(prob, threshold)

    shap_contribs, lime_contribs = [], []
    explainer = STATE.get("explainer")
    if explainer is not None:
        try:
            shap_result = explainer.local_shap_explanation(X, top_k=5)
            shap_contribs = [
                Contributor(feature=c["feature"], value=c["shap_value"])
                for c in shap_result["top_contributors"]
            ]
            lime_result = explainer.local_lime_explanation(X, top_k=5)
            lime_contribs = [
                Contributor(feature=c["feature"], value=c["weight"])
                for c in lime_result["top_contributors"]
            ]
        except Exception:  # noqa: BLE001
            logger.exception("Explanation failed; returning prediction without it.")

    return PredictionResponse(
        prediction=prediction,
        conversion_probability=round(prob, 6),
        confidence=confidence,
        decision_threshold=threshold,
        top_contributors_shap=shap_contribs,
        top_contributors_lime=lime_contribs,
        model_name=str(STATE.get("model_name", "unknown")),
    )
