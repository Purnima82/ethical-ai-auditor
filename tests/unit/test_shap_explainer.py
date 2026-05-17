"""
test_shap_explainer.py — Unit tests for ShapExplainer
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from backend.core.shap_explainer import ShapExplainer, FeatureImportance, ShapReport


@pytest.fixture
def rf_and_data():
    rng = np.random.default_rng(42)
    n = 300
    feat_names = ["years_exp", "education", "skills", "zip_code", "prior_salary", "interview"]
    X = pd.DataFrame(rng.standard_normal((n, len(feat_names))), columns=feat_names)
    y = (X["years_exp"] + X["skills"] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=4)
    model.fit(X, y)
    X_train = X.iloc[:200]
    X_test  = X.iloc[200:]
    return model, X_train, X_test, y.iloc[200:]


class TestShapExplainer:
    def test_compute_shap_values_shape(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        sv = exp.compute_shap_values()
        assert sv.shape == X_te.shape

    def test_explain_returns_report(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        report = exp.explain()
        assert isinstance(report, ShapReport)
        assert len(report.feature_importances) > 0

    def test_proxy_detection_pattern_match(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        report = exp.explain()
        proxy_features = {fi.feature for fi in report.feature_importances if fi.is_proxy}
        # zip_code and prior_salary should be detected as proxies
        assert "zip_code" in proxy_features
        assert "prior_salary" in proxy_features

    def test_legit_features_not_flagged(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        report = exp.explain()
        legit = {fi.feature for fi in report.feature_importances if not fi.is_proxy}
        assert "years_exp" in legit
        assert "skills" in legit

    def test_feature_importance_rank_order(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        report = exp.explain()
        ranks = [fi.rank for fi in report.feature_importances]
        assert ranks == sorted(ranks)

    def test_abs_shap_descending(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        report = exp.explain()
        abs_vals = [fi.abs_shap_value for fi in report.feature_importances]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_direction_consistent(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        report = exp.explain()
        for fi in report.feature_importances:
            expected = "positive" if fi.shap_value >= 0 else "negative"
            assert fi.direction == expected

    def test_fidelity_between_0_and_1(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        report = exp.explain()
        assert 0.0 <= report.explanation_fidelity <= 1.0

    def test_waterfall_data_structure(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        wf = exp.waterfall_data(0)
        assert "base_value" in wf
        assert "final_value" in wf
        assert "contributions" in wf
        assert len(wf["contributions"]) > 0
        assert all("feature" in c and "shap_value" in c for c in wf["contributions"])

    def test_caching_consistency(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        sv1 = exp.compute_shap_values()
        sv2 = exp.compute_shap_values()
        assert np.array_equal(sv1, sv2)

    def test_top_n_respected(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        report = exp.explain(top_n=3)
        assert len(report.feature_importances) == 3


class TestDetectProxy:
    def test_known_proxy_patterns(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        assert "race" in exp._detect_proxy("zip_code") or "socioeconomic" in exp._detect_proxy("zip_code")
        assert "gender" in exp._detect_proxy("prior_salary")
        assert "race" in exp._detect_proxy("last_name")

    def test_non_proxy_returns_empty(self, rf_and_data):
        model, X_tr, X_te, _ = rf_and_data
        exp = ShapExplainer(model, X_tr, X_te)
        assert exp._detect_proxy("years_experience") == []
        assert exp._detect_proxy("gpa_score") == []
        assert exp._detect_proxy("interview_result") == []
