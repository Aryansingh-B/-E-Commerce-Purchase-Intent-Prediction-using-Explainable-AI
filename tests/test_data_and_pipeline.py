"""Core data + preprocessing tests (PRD §5, §9)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config.settings import get_settings
from src.data.loader import DataValidationError, SessionDataLoader
from src.features.engineering import ENGINEERED_FEATURE_NAMES, add_engineered_features
from src.preprocessing.pipeline import build_preprocessor, get_feature_columns


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def raw_df(settings):
    return SessionDataLoader(settings=settings).load()


def test_loader_reads_expected_shape(raw_df):
    assert len(raw_df) > 0
    assert "Converted" in raw_df.columns


def test_loader_raises_on_missing_file(settings):
    with pytest.raises(FileNotFoundError):
        SessionDataLoader(settings=settings, path="does/not/exist.csv").load()


def test_loader_raises_on_missing_columns(settings, tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(bad_csv, index=False)
    with pytest.raises(DataValidationError):
        SessionDataLoader(settings=settings, path=bad_csv).load()


def test_engineered_features_present_and_bounded(raw_df):
    out = add_engineered_features(raw_df)
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in out.columns
    assert out["product_page_share"].between(0, 1).all()
    assert out["product_time_share"].between(0, 1).all()
    assert out["is_returning_visitor"].isin([0, 1]).all()


def test_engineered_features_handle_zero_pages(settings):
    zero_row = pd.DataFrame(
        [
            {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 0,
                "ProductRelated_Duration": 0.0,
                "BounceRates": 0.0,
                "ExitRates": 0.0,
                "PageValues": 0.0,
                "SpecialDay": 0.0,
                "Month": "Jan",
                "OperatingSystems": 1,
                "Browser": 1,
                "Region": 1,
                "TrafficType": 1,
                "VisitorType": "New_Visitor",
                "Weekend": False,
            }
        ]
    )
    out = add_engineered_features(zero_row)
    # Must not raise ZeroDivisionError / produce NaN or inf
    assert out["avg_time_per_page"].iloc[0] == 0.0
    assert out["product_page_share"].iloc[0] == 0.0
    assert not out.isna().any().any()


def test_preprocessor_fits_and_transforms_without_leakage_across_calls(settings, raw_df):
    df = add_engineered_features(raw_df)
    cols = get_feature_columns(settings)
    X = df[cols]

    pre_train = build_preprocessor(settings)
    Xt_train = pre_train.fit_transform(X.iloc[:8000])
    Xt_holdout = pre_train.transform(X.iloc[8000:])  # must reuse TRAIN-fitted stats

    assert Xt_train.shape[1] == Xt_holdout.shape[1]
    assert Xt_train.shape[0] == 8000


def test_preprocessor_treats_int_coded_ids_as_categorical(settings):
    """Region/Browser/OperatingSystems/TrafficType must be one-hot, not scaled."""
    feature_names = None
    pre = build_preprocessor(settings)
    cols = get_feature_columns(settings)
    dummy = pd.DataFrame(
        [
            {
                "Administrative": 1,
                "Administrative_Duration": 1.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 5,
                "ProductRelated_Duration": 50.0,
                "BounceRates": 0.01,
                "ExitRates": 0.02,
                "PageValues": 0.0,
                "total_pages": 6,
                "total_duration": 51.0,
                "avg_time_per_page": 8.5,
                "SpecialDay": 0.0,
                "product_page_share": 0.8,
                "product_time_share": 0.9,
                "is_returning_visitor": 1,
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 1,
                "Month": "May",
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        ]
    )[cols]
    pre.fit(dummy)
    feature_names = list(pre.get_feature_names_out())
    assert any("categorical__OperatingSystems" in f for f in feature_names)
    assert any("categorical__Region" in f for f in feature_names)
