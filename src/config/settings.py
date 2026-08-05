"""Typed settings loader.

Loads config.yaml once and exposes it as a validated, typed object so every
consumer (notebooks, training scripts, the API, the dashboard) reads the exact
same configuration instead of hard-coded paths scattered through the repo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# Repo root = two levels up from this file (src/config/settings.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class PathsConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    raw_data: str
    data_dictionary: str
    model_dir: str
    model_artifact: str
    preprocessor_artifact: str
    feature_names_artifact: str
    explainer_artifact: str
    metrics_report: str
    model_card: str
    mlruns_dir: str

    def resolve(self, field: str) -> Path:
        return REPO_ROOT / getattr(self, field)


class DataConfig(BaseModel):
    target: str
    test_size: float
    val_size: float
    stratify: bool
    categorical_id_columns: list[str]
    categorical_columns: list[str]
    skewed_numeric_columns: list[str]
    plain_numeric_columns: list[str]
    use_page_values: bool

    @property
    def all_categorical_columns(self) -> list[str]:
        return list(self.categorical_id_columns) + list(self.categorical_columns)

    @property
    def all_numeric_columns(self) -> list[str]:
        cols = list(self.skewed_numeric_columns) + list(self.plain_numeric_columns)
        if not self.use_page_values and "PageValues" in cols:
            cols = [c for c in cols if c != "PageValues"]
        return cols

    @property
    def feature_columns(self) -> list[str]:
        return self.all_numeric_columns + self.all_categorical_columns


class ImbalanceConfig(BaseModel):
    strategy: str = Field(pattern="^(class_weight|smote|none)$")


class ThresholdConfig(BaseModel):
    value_per_conversion: float
    cost_per_intervention: float


class ModelsConfig(BaseModel):
    cv_folds: int
    scoring: str


class ApiConfig(BaseModel):
    host: str
    port: int


class DashboardConfig(BaseModel):
    port: int


class Settings(BaseModel):
    seed: int
    paths: PathsConfig
    data: DataConfig
    imbalance: ImbalanceConfig
    threshold: ThresholdConfig
    models: ModelsConfig
    api: ApiConfig
    dashboard: DashboardConfig


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path | None = None) -> Settings:
    """Load and cache the project settings.

    Cached so repeated calls across modules in the same process are free and
    consistent; pass an explicit config_path (e.g. in tests) to bypass cache
    behaviour intentionally by calling get_settings.cache_clear() first.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings(**raw)
