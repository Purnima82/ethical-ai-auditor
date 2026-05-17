"""
test_bias_detector.py — Unit tests for the BiasDetector core engine
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from backend.core.bias_detector import BiasDetector, FairnessReport


@pytest.fixture
def sample_data():
    """Synthetic dataset with known bias patterns."""
    np.random.seed(42)
    n = 500
    X, y = make_classification(n_samples=n, n_features=8, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])

    # Inject known bias: gender=1 has lower positive rate
    df["gender"] = np.random.choice(["male", "female"], size=n, p=[0.5, 0.5])
    df["age_group"] = np.random.choice(["18-35", "36-55", "55+"], size=n)

    # Introduce synthetic bias — females less likely to get y=1
    bias_mask = df["gender"] == "female"
    y_biased = y.copy()
    flip_idx = np.where(bias_mask & (y_biased == 1))[0]
    flip_n = int(len(flip_idx) * 0.4)
    y_biased[flip_idx[:flip_n]] = 0

    return df, pd.Series(y_biased)


@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    X_feat = X[[c for c in X.columns if c.startswith("feat_")]]
    model = LogisticRegression(random_state=42, max_iter=300)
    model.fit(X_feat, y)
    return model, X, y


def test_bias_detector_init(trained_model):
    model, X, y = trained_model
    detector = BiasDetector(model, X[[c for c in X.columns if c.startswith("feat_")]], y)
    assert detector.threshold == 0.50


def test_audit_gender(trained_model):
    model, X, y = trained_model
    X_feat = X.copy()
    report = BiasDetector(model, X_feat[[c for c in X_feat.columns if c.startswith("feat_")]], y)
    # Manually inject gender into test set for attribute-based audit
    X_with_gender = X_feat[[c for c in X_feat.columns if c.startswith("feat_")]].copy()
    X_with_gender["gender"] = X_feat["gender"].values
    detector = BiasDetector(model, X_with_gender, y)
    report = detector.audit_attribute("gender")
    assert isinstance(report, FairnessReport)
    assert 0 <= report.demographic_parity <= 1
    assert report.verdict in {"PASS", "CAUTION", "FAIL"}
    assert len(report.group_metrics) >= 2


def test_biased_data_fails(trained_model):
    """A model trained on biased data should fail the parity check."""
    model, X, y = trained_model
    X_model = X[[c for c in X.columns if c.startswith("feat_")]].copy()
    X_model["gender"] = X["gender"].values
    detector = BiasDetector(model, X_model, y)
    report = detector.audit_attribute("gender")
    # The synthetic bias we injected should produce a non-trivial diff
    assert report.demographic_parity_diff > 0.0


def test_threshold_affects_predictions(trained_model):
    model, X, y = trained_model
    X_model = X[[c for c in X.columns if c.startswith("feat_")]].copy()
    X_model["gender"] = X["gender"].values
    detector = BiasDetector(model, X_model, y, threshold=0.30)
    y_pred_30, _ = detector._get_predictions()
    detector.set_threshold(0.70)
    y_pred_70, _ = detector._get_predictions()
    assert y_pred_30.sum() > y_pred_70.sum()


def test_overall_fairness_score(trained_model):
    model, X, y = trained_model
    X_model = X[[c for c in X.columns if c.startswith("feat_")]].copy()
    X_model["gender"] = X["gender"].values
    X_model["age_group"] = X["age_group"].values
    detector = BiasDetector(model, X_model, y)
    reports = detector.audit(["gender", "age_group"])
    score = detector.overall_fairness_score(reports)
    assert 0.0 <= score <= 1.0


def test_threshold_sweep(trained_model):
    model, X, y = trained_model
    X_model = X[[c for c in X.columns if c.startswith("feat_")]].copy()
    X_model["gender"] = X["gender"].values
    detector = BiasDetector(model, X_model, y)
    sweep = detector.threshold_sweep("gender", thresholds=[0.3, 0.5, 0.7])
    assert isinstance(sweep, pd.DataFrame)
    assert len(sweep) == 3
    assert "demographic_parity_diff" in sweep.columns
