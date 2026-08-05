"""Pydantic schemas for the prediction API (PRD §4.7).

Built from the raw feature set so the API and the model never drift apart —
the same columns the training pipeline reads are what /predict validates.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionPayload(BaseModel):
    """One e-commerce session, as it would arrive from the tracking layer."""

    Administrative: int = Field(ge=0, description="Administrative pages visited")
    Administrative_Duration: float = Field(ge=0)
    Informational: int = Field(ge=0)
    Informational_Duration: float = Field(ge=0)
    ProductRelated: int = Field(ge=0)
    ProductRelated_Duration: float = Field(ge=0)
    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)
    PageValues: float = Field(ge=0)
    SpecialDay: float = Field(ge=0, le=1)
    Month: str = Field(description="Three-letter month, e.g. 'May'")
    OperatingSystems: int = Field(description="Categorical OS ID")
    Browser: int = Field(description="Categorical browser ID")
    Region: int = Field(description="Categorical region ID")
    TrafficType: int = Field(description="Categorical traffic-source ID")
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"]
    Weekend: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "Administrative": 2,
                "Administrative_Duration": 40.5,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 25,
                "ProductRelated_Duration": 620.3,
                "BounceRates": 0.01,
                "ExitRates": 0.03,
                "PageValues": 12.4,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        }
    }


class Contributor(BaseModel):
    feature: str
    value: float


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: Literal[0, 1]
    conversion_probability: float
    confidence: Literal["low", "medium", "high"]
    decision_threshold: float
    top_contributors_shap: list[Contributor]
    top_contributors_lime: list[Contributor]
    model_name: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: Literal["ok"]
    model_loaded: bool
    model_name: str
