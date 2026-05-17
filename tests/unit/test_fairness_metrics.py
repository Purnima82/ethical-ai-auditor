"""
test_fairness_metrics.py — Tests for FairnessMetrics calculator
"""
import numpy as np
import pandas as pd
import pytest

from backend.core.fairness_metrics import FairnessMetrics


@pytest.fixture
def fair_data():
    np.random.seed(0)
    n = 400
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true.copy()
    noise = np.random.rand(n) < 0.1
    y_pred[noise] = 1 - y_pred[noise]
    groups = np.random.choice(["A", "B"], n)
    return y_true, y_pred, pd.Series(groups)


@pytest.fixture
def biased_data():
    np.random.seed(42)
    n = 400
    y_true = np.random.randint(0, 2, n)
    groups = np.random.choice(["A", "B"], n)
    y_pred = y_true.copy()
    # Group B gets more false negatives (TPR drops for B)
    b_pos = np.where((groups == "B") & (y_true == 1))[0]
    y_pred[b_pos[:int(len(b_pos) * 0.5)]] = 0
    return y_true, y_pred, pd.Series(groups)


def test_demographic_parity_fair(fair_data):
    y_true, y_pred, groups = fair_data
    fm = FairnessMetrics(y_true, y_pred, groups)
    result = fm.demographic_parity()
    assert result.passes  # fair data should pass


def test_demographic_parity_biased(biased_data):
    y_true, y_pred, groups = biased_data
    fm = FairnessMetrics(y_true, y_pred, groups)
    result = fm.demographic_parity()
    assert result.diff >= 0.0


def test_equalized_odds_returns_two(fair_data):
    y_true, y_pred, groups = fair_data
    fm = FairnessMetrics(y_true, y_pred, groups)
    tpr, fpr = fm.equalized_odds()
    assert tpr.name == "Equalized Odds (TPR)"
    assert fpr.name == "Equalized Odds (FPR)"


def test_disparate_impact_80_rule(biased_data):
    y_true, y_pred, groups = biased_data
    fm = FairnessMetrics(y_true, y_pred, groups)
    result = fm.disparate_impact()
    assert 0.0 <= result.value <= 1.0
    assert result.threshold == 0.80


def test_summary_table(fair_data):
    y_true, y_pred, groups = fair_data
    fm = FairnessMetrics(y_true, y_pred, groups)
    df = fm.summary_table()
    assert isinstance(df, pd.DataFrame)
    assert "metric" in df.columns
    assert len(df) >= 5


def test_compute_all_keys(fair_data):
    y_true, y_pred, groups = fair_data
    fm = FairnessMetrics(y_true, y_pred, groups)
    results = fm.compute_all()
    expected = {
        "demographic_parity", "equalized_odds_tpr", "equalized_odds_fpr",
        "equal_opportunity", "disparate_impact", "calibration", "accuracy_parity"
    }
    assert set(results.keys()) == expected
