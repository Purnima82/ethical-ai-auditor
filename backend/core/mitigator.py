"""
mitigator.py
Bias mitigation strategies:
  1. Training data reweighting (pre-processing)
  2. Proxy variable removal (pre-processing)
  3. Per-group threshold calibration (post-processing)
  4. Fairness-constrained threshold search (post-processing)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score


class BiasReweighter:
    """
    Pre-processing mitigation: reweight training samples so each
    (class, protected_group) cell contributes equally to learning.

    Implements the method from Kamiran & Calders (2012).
    """

    def __init__(self, protected_attr: str):
        self.protected_attr = protected_attr
        self.weights_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BiasReweighter":
        n = len(y)
        combined = pd.DataFrame({
            "label": y.values,
            "group": X[self.protected_attr].values,
        })
        weights = np.ones(n)
        groups = combined["group"].unique()
        labels = combined["label"].unique()

        for g in groups:
            for lbl in labels:
                mask = (combined["group"] == g) & (combined["label"] == lbl)
                n_gl = mask.sum()
                n_g = (combined["group"] == g).sum()
                n_l = (combined["label"] == lbl).sum()
                if n_gl > 0 and n_g > 0 and n_l > 0:
                    expected = (n_g / n) * (n_l / n) * n
                    w = expected / n_gl
                    weights[mask.values] = w

        self.weights_ = pd.Series(weights, index=X.index)
        return self

    def transform(self) -> pd.Series:
        if self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        return self.weights_

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return self.fit(X, y).transform()


class ProxyRemover:
    """
    Pre-processing: remove or suppress proxy variables for protected attributes.
    Uses a list of known proxy features (from SHAP analysis) or auto-detects
    via correlation with protected attributes.
    """

    def __init__(
        self,
        protected_attrs: list[str],
        explicit_proxies: list[str] | None = None,
        correlation_threshold: float = 0.30,
    ):
        self.protected_attrs = protected_attrs
        self.explicit_proxies = explicit_proxies or []
        self.correlation_threshold = correlation_threshold
        self.detected_proxies_: list[str] = []
        self.dropped_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "ProxyRemover":
        auto_detected = []
        for col in X.columns:
            if col in self.protected_attrs:
                continue
            for attr in self.protected_attrs:
                if attr not in X.columns:
                    continue
                try:
                    corr = abs(
                        pd.get_dummies(X[col], drop_first=True)
                        .corrwith(pd.get_dummies(X[attr], drop_first=True).iloc[:, 0])
                        .abs()
                        .max()
                    )
                    if corr >= self.correlation_threshold:
                        auto_detected.append(col)
                        break
                except Exception:
                    pass

        self.detected_proxies_ = list(set(auto_detected + self.explicit_proxies))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in self.detected_proxies_ if c in X.columns]
        self.dropped_ = to_drop
        return X.drop(columns=to_drop)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class ThresholdCalibrator:
    """
    Post-processing: find per-group decision thresholds that equalise
    the False Positive Rate across demographic groups while preserving
    overall accuracy as much as possible.

    Based on Hardt, Price & Srebro (2016) equalized odds post-processing.
    """

    def __init__(
        self,
        protected_attr: str,
        fairness_metric: str = "fpr",   # "fpr" | "tpr" | "selection_rate"
        search_steps: int = 50,
    ):
        self.protected_attr = protected_attr
        self.fairness_metric = fairness_metric
        self.search_steps = search_steps
        self.thresholds_: dict[str, float] = {}
        self.baseline_threshold_: float = 0.50

    def _metric_at_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray, t: float
    ) -> float:
        y_pred = (y_prob >= t).astype(int)
        if self.fairness_metric == "fpr":
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            return fp / (fp + tn) if (fp + tn) > 0 else 0.0
        elif self.fairness_metric == "tpr":
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            return float(y_pred.mean())

    def fit(
        self,
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
        sensitive: pd.Series,
    ) -> "ThresholdCalibrator":
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        sensitive = pd.Series(sensitive).reset_index(drop=True)

        overall_metric = self._metric_at_threshold(y_true, y_prob, 0.50)
        thresholds = np.linspace(0.1, 0.9, self.search_steps)
        groups = sensitive.unique()

        for g in groups:
            mask = (sensitive == g).values
            y_true_g = y_true[mask]
            y_prob_g = y_prob[mask]
            best_t = 0.50
            best_dist = float("inf")
            for t in thresholds:
                m = self._metric_at_threshold(y_true_g, y_prob_g, t)
                dist = abs(m - overall_metric)
                if dist < best_dist:
                    best_dist = dist
                    best_t = t
            self.thresholds_[str(g)] = round(float(best_t), 3)

        self.baseline_threshold_ = 0.50
        return self

    def predict(
        self, y_prob: np.ndarray | pd.Series, sensitive: pd.Series
    ) -> np.ndarray:
        """Apply per-group thresholds to produce fair predictions."""
        y_prob = np.asarray(y_prob)
        sensitive = pd.Series(sensitive).reset_index(drop=True)
        y_pred = np.zeros(len(y_prob), dtype=int)
        for g, t in self.thresholds_.items():
            mask = (sensitive == g).values
            y_pred[mask] = (y_prob[mask] >= t).astype(int)
        unknown_mask = ~np.isin(sensitive.values, list(self.thresholds_.keys()))
        y_pred[unknown_mask] = (y_prob[unknown_mask] >= self.baseline_threshold_).astype(int)
        return y_pred

    def improvement_summary(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        sensitive: pd.Series,
    ) -> dict:
        """Report fairness improvement from threshold calibration."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        sensitive = pd.Series(sensitive).reset_index(drop=True)

        baseline = (y_prob >= 0.50).astype(int)
        calibrated = self.predict(y_prob, sensitive)

        baseline_acc = accuracy_score(y_true, baseline)
        calibrated_acc = accuracy_score(y_true, calibrated)

        def dp_diff(y_pred):
            rates = []
            for g in sensitive.unique():
                mask = sensitive == g
                rates.append(float(y_pred[mask].mean()))
            return max(rates) - min(rates) if rates else 0.0

        return {
            "baseline_accuracy": round(baseline_acc, 4),
            "calibrated_accuracy": round(calibrated_acc, 4),
            "accuracy_change": round(calibrated_acc - baseline_acc, 4),
            "baseline_dp_diff": round(dp_diff(baseline), 4),
            "calibrated_dp_diff": round(dp_diff(calibrated), 4),
            "dp_improvement": round(dp_diff(baseline) - dp_diff(calibrated), 4),
            "per_group_thresholds": self.thresholds_,
        }


class MitigationPipeline:
    """
    Full mitigation pipeline combining all three strategies:
    1. Proxy removal
    2. Data reweighting
    3. Model retraining
    4. Threshold calibration

    Returns a mitigated model + calibrator ready for fair predictions.
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        protected_attrs: list[str],
        explicit_proxies: list[str] | None = None,
    ):
        self.base_model = base_model
        self.protected_attrs = protected_attrs
        self.proxy_remover = ProxyRemover(protected_attrs, explicit_proxies)
        self.reweighters: dict[str, BiasReweighter] = {
            attr: BiasReweighter(attr) for attr in protected_attrs
        }
        self.calibrators: dict[str, ThresholdCalibrator] = {}
        self.mitigated_model_: BaseEstimator | None = None

    @staticmethod
    def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode any remaining categorical columns so sklearn can fit."""
        from sklearn.preprocessing import LabelEncoder
        out = df.copy()
        for col in out.select_dtypes(include=["object", "category"]).columns:
            out[col] = LabelEncoder().fit_transform(out[col].astype(str))
        return out

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> "MitigationPipeline":
        X_clean = self.proxy_remover.fit_transform(X_train)
        X_val_clean = self.proxy_remover.transform(X_val)

        combined_weights = np.ones(len(y_train))
        for attr in self.protected_attrs:
            if attr in X_clean.columns:
                w = self.reweighters[attr].fit_transform(X_clean, y_train)
                combined_weights *= w.values
        combined_weights /= combined_weights.mean()

        self.mitigated_model_ = clone(self.base_model)
        if hasattr(self.mitigated_model_, "fit"):
            # Drop protected attribute columns before fitting
            model_cols = [c for c in X_clean.columns if c not in self.protected_attrs]
            X_fit = self._encode_df(X_clean[model_cols])
            try:
                self.mitigated_model_.fit(X_fit, y_train, sample_weight=combined_weights)
            except TypeError:
                self.mitigated_model_.fit(X_fit, y_train)

        if hasattr(self.mitigated_model_, "predict_proba"):
            model_cols_val = [c for c in X_val_clean.columns if c not in self.protected_attrs]
            X_val_fit = self._encode_df(X_val_clean[model_cols_val])
            y_val_prob = self.mitigated_model_.predict_proba(X_val_fit)[:, 1]
            for attr in self.protected_attrs:
                if attr in X_val.columns:
                    cal = ThresholdCalibrator(attr)
                    cal.fit(y_val.values, y_val_prob, X_val[attr])
                    self.calibrators[attr] = cal

        self._model_cols = model_cols
        return self

    def predict(self, X: pd.DataFrame, primary_attr: str) -> np.ndarray:
        X_clean = self.proxy_remover.transform(X)
        model_cols = getattr(self, "_model_cols",
            [c for c in X_clean.columns if c not in self.protected_attrs])
        X_fit = self._encode_df(X_clean[model_cols])
        y_prob = self.mitigated_model_.predict_proba(X_fit)[:, 1]
        if primary_attr in self.calibrators and primary_attr in X.columns:
            return self.calibrators[primary_attr].predict(y_prob, X[primary_attr])
        return (y_prob >= 0.50).astype(int)
