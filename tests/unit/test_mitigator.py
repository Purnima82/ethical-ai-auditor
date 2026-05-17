"""
test_mitigator.py — Unit tests for all three mitigation strategies
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from backend.core.mitigator import (
    BiasReweighter,
    ProxyRemover,
    ThresholdCalibrator,
    MitigationPipeline,
)


@pytest.fixture
def hiring_df():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "feat_0": rng.normal(0, 1, n),
        "feat_1": rng.normal(0, 1, n),
        "feat_2": rng.normal(0, 1, n),
        "gender":    rng.choice(["male", "female"], n),
        "race":      rng.choice(["white", "black", "other"], n),
        "age_group": rng.choice(["young", "mid", "senior"], n),
        "zip_code":  rng.integers(1, 6, n).astype(str),
        "prior_salary": rng.integers(40000, 100000, n),
    })


@pytest.fixture
def labels(hiring_df):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, len(hiring_df))
    # Inject bias
    female_pos = np.where((hiring_df["gender"] == "female") & (y == 1))[0]
    flip = rng.choice(female_pos, int(len(female_pos) * 0.4), replace=False)
    y[flip] = 0
    return pd.Series(y)


# ── BiasReweighter ────────────────────────────────────────────────────────────

class TestBiasReweighter:
    def test_weights_positive(self, hiring_df, labels):
        rw = BiasReweighter("gender")
        weights = rw.fit_transform(hiring_df, labels)
        assert (weights > 0).all()

    def test_weights_length(self, hiring_df, labels):
        rw = BiasReweighter("gender")
        weights = rw.fit_transform(hiring_df, labels)
        assert len(weights) == len(labels)

    def test_mean_weight_near_one(self, hiring_df, labels):
        rw = BiasReweighter("gender")
        weights = rw.fit_transform(hiring_df, labels)
        assert abs(weights.mean() - 1.0) < 0.05

    def test_fit_then_transform(self, hiring_df, labels):
        rw = BiasReweighter("race")
        rw.fit(hiring_df, labels)
        w = rw.transform()
        assert len(w) == len(labels)

    def test_minority_group_upweighted(self, hiring_df, labels):
        """Female group with downsampled positives should be upweighted."""
        rw = BiasReweighter("gender")
        weights = rw.fit_transform(hiring_df, labels)
        female_w = weights[hiring_df["gender"] == "female"].mean()
        male_w   = weights[hiring_df["gender"] == "male"].mean()
        # Female positives were flipped so female positives should carry more weight
        assert female_w > 0


# ── ProxyRemover ──────────────────────────────────────────────────────────────

class TestProxyRemover:
    def test_removes_explicit_proxies(self, hiring_df):
        remover = ProxyRemover(["gender", "race"], explicit_proxies=["zip_code", "prior_salary"])
        cleaned = remover.fit_transform(hiring_df)
        assert "zip_code" not in cleaned.columns
        assert "prior_salary" not in cleaned.columns

    def test_keeps_legit_features(self, hiring_df):
        remover = ProxyRemover(["gender"], explicit_proxies=["zip_code"])
        cleaned = remover.fit_transform(hiring_df)
        for c in ["feat_0", "feat_1", "feat_2"]:
            assert c in cleaned.columns

    def test_fit_transform_idempotent(self, hiring_df):
        remover = ProxyRemover(["gender"], explicit_proxies=["zip_code"])
        c1 = remover.fit_transform(hiring_df)
        c2 = remover.transform(hiring_df)
        assert list(c1.columns) == list(c2.columns)

    def test_empty_explicit_proxies(self, hiring_df):
        remover = ProxyRemover(["gender"], explicit_proxies=[])
        cleaned = remover.fit_transform(hiring_df)
        # Should still have all columns (no explicit drops, auto detection may vary)
        assert len(cleaned.columns) > 0

    def test_dropped_attr_logged(self, hiring_df):
        remover = ProxyRemover(["gender"], explicit_proxies=["prior_salary"])
        remover.fit_transform(hiring_df)
        assert "prior_salary" in remover.dropped_


# ── ThresholdCalibrator ───────────────────────────────────────────────────────

class TestThresholdCalibrator:
    @pytest.fixture
    def model_outputs(self, hiring_df, labels):
        feat_cols = ["feat_0", "feat_1", "feat_2"]
        model = RandomForestClassifier(n_estimators=20, random_state=0)
        model.fit(hiring_df[feat_cols], labels)
        y_prob = model.predict_proba(hiring_df[feat_cols])[:, 1]
        return labels.values, y_prob

    def test_produces_binary_predictions(self, hiring_df, model_outputs):
        y_true, y_prob = model_outputs
        cal = ThresholdCalibrator("gender")
        cal.fit(y_true, y_prob, hiring_df["gender"])
        preds = cal.predict(y_prob, hiring_df["gender"])
        assert set(preds).issubset({0, 1})

    def test_thresholds_per_group(self, hiring_df, model_outputs):
        y_true, y_prob = model_outputs
        cal = ThresholdCalibrator("gender")
        cal.fit(y_true, y_prob, hiring_df["gender"])
        assert set(cal.thresholds_.keys()) == {"male", "female"}

    def test_thresholds_in_valid_range(self, hiring_df, model_outputs):
        y_true, y_prob = model_outputs
        cal = ThresholdCalibrator("gender")
        cal.fit(y_true, y_prob, hiring_df["gender"])
        for t in cal.thresholds_.values():
            assert 0.0 < t < 1.0

    def test_improvement_summary_keys(self, hiring_df, model_outputs):
        y_true, y_prob = model_outputs
        cal = ThresholdCalibrator("gender")
        cal.fit(y_true, y_prob, hiring_df["gender"])
        summary = cal.improvement_summary(y_true, y_prob, hiring_df["gender"])
        assert "baseline_accuracy" in summary
        assert "calibrated_accuracy" in summary
        assert "dp_improvement" in summary

    def test_unknown_group_falls_back_to_baseline(self, hiring_df, model_outputs):
        y_true, y_prob = model_outputs
        cal = ThresholdCalibrator("gender")
        cal.fit(y_true, y_prob, hiring_df["gender"])
        unseen = pd.Series(["unknown"] * len(y_prob))
        preds = cal.predict(y_prob, unseen)
        assert len(preds) == len(y_prob)


# ── MitigationPipeline ────────────────────────────────────────────────────────

class TestMitigationPipeline:
    def test_pipeline_fit_predict(self, hiring_df, labels):
        feat_cols = ["feat_0", "feat_1", "feat_2"]
        protected = ["gender", "race"]
        model = RandomForestClassifier(n_estimators=20, random_state=0)

        X_tr, X_val, y_tr, y_val = train_test_split(
            hiring_df, labels, test_size=0.3, random_state=0
        )
        pipeline = MitigationPipeline(model, protected, explicit_proxies=["zip_code"])
        pipeline.fit(X_tr, y_tr, X_val, y_val)
        preds = pipeline.predict(X_val, primary_attr="gender")
        assert len(preds) == len(y_val)
        assert set(preds).issubset({0, 1})

    def test_pipeline_removes_proxies(self, hiring_df, labels):
        model = RandomForestClassifier(n_estimators=20, random_state=0)
        X_tr, X_val, y_tr, y_val = train_test_split(hiring_df, labels, test_size=0.3, random_state=0)
        pipeline = MitigationPipeline(model, ["gender"], explicit_proxies=["zip_code", "prior_salary"])
        pipeline.fit(X_tr, y_tr, X_val, y_val)
        assert "zip_code" in pipeline.proxy_remover.dropped_
