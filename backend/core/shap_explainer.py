"""
shap_explainer.py
SHAP-based model explainability with proxy variable detection.
Identifies features that serve as proxies for protected attributes.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator

warnings.filterwarnings("ignore")


@dataclass
class FeatureImportance:
    feature: str
    shap_value: float
    abs_shap_value: float
    direction: str          # "positive" | "negative"
    is_proxy: bool
    proxy_for: list[str]    # which protected attributes this proxies
    rank: int


@dataclass
class ShapReport:
    feature_importances: list[FeatureImportance]
    proxy_variables: list[str]
    top_positive_features: list[str]
    top_negative_features: list[str]
    mean_abs_shap: float
    explanation_fidelity: float


# Known proxy patterns: feature name fragments → protected attribute
_PROXY_PATTERNS: dict[str, list[str]] = {
    "zip": ["race", "socioeconomic"],
    "postal": ["race", "socioeconomic"],
    "neighborhood": ["race", "socioeconomic"],
    "school": ["race", "socioeconomic"],
    "university": ["race", "socioeconomic"],
    "college": ["race", "socioeconomic"],
    "prior_salary": ["gender", "race"],
    "previous_salary": ["gender", "race"],
    "last_salary": ["gender", "race"],
    "salary_history": ["gender", "race"],
    "address": ["race", "socioeconomic"],
    "hometown": ["race", "socioeconomic"],
    "birthplace": ["race", "national_origin"],
    "language": ["national_origin", "race"],
    "name": ["gender", "race", "national_origin"],
    "first_name": ["gender", "race"],
    "last_name": ["race", "national_origin"],
    "military": ["gender", "disability"],
    "gaps": ["gender", "pregnancy"],
    "employment_gap": ["gender", "pregnancy"],
}


class ShapExplainer:
    """
    Computes SHAP values for model explanation and detects proxy variables.

    Supports tree-based models (XGBoost, LightGBM, RandomForest) via
    TreeExplainer, and falls back to KernelExplainer for other models.

    Usage:
        explainer = ShapExplainer(model, X_train, X_test)
        report = explainer.explain(protected_attrs=["gender", "race", "age"])
        print(report.proxy_variables)
    """

    def __init__(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        max_background_samples: int = 100,
    ):
        self.model = model
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.feature_names = list(X_train.columns)
        self._shap_values: np.ndarray | None = None
        self._explainer = None
        self._max_bg = max_background_samples

    def _build_explainer(self) -> None:
        """Auto-detect best explainer type for the model."""
        model_type = type(self.model).__name__.lower()
        tree_types = {"xgbclassifier", "lgbmclassifier", "randomforestclassifier",
                      "gradientboostingclassifier", "decisiontreeclassifier",
                      "extratreesclassifier", "xgbregressor"}
        if model_type in tree_types:
            self._explainer = shap.TreeExplainer(self.model)
        else:
            bg = shap.sample(self.X_train, min(self._max_bg, len(self.X_train)))
            self._explainer = shap.KernelExplainer(
                self.model.predict_proba, bg
            )

    def compute_shap_values(self) -> np.ndarray:
        """Compute and cache SHAP values for X_test."""
        if self._shap_values is not None:
            return self._shap_values

        if self._explainer is None:
            self._build_explainer()

        raw = self._explainer.shap_values(self.X_test)
        # Handle multiple SHAP output formats:
        # list [class0, class1]  — older SHAP + sklearn
        # 3D ndarray (n, feats, classes) — newer SHAP + sklearn RF
        # 2D ndarray (n, feats)          — single-output / XGBoost
        if isinstance(raw, list):
            self._shap_values = np.array(raw[1])
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            self._shap_values = raw[:, :, 1]
        else:
            self._shap_values = np.array(raw)

        return self._shap_values

    def _detect_proxy(self, feature: str) -> list[str]:
        """Return protected attributes this feature may proxy for."""
        feat_lower = feature.lower()
        proxied = []
        for pattern, attrs in _PROXY_PATTERNS.items():
            if pattern in feat_lower:
                proxied.extend(attrs)
        return list(set(proxied))

    def _detect_proxy_by_correlation(
        self, feature: str, protected_attrs: list[str], threshold: float = 0.30
    ) -> list[str]:
        """
        Detect proxy variables via Cramér's V correlation with protected attrs.
        Supplements pattern matching with statistical detection.
        """
        proxied = []
        for attr in protected_attrs:
            if attr not in self.X_test.columns or feature not in self.X_test.columns:
                continue
            try:
                from scipy.stats import chi2_contingency
                ct = pd.crosstab(self.X_test[feature], self.X_test[attr])
                chi2, _, _, _ = chi2_contingency(ct)
                n = ct.sum().sum()
                k = min(ct.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0.0
                if cramers_v >= threshold:
                    proxied.append(attr)
            except Exception:
                pass
        return proxied

    def explain(
        self,
        protected_attrs: list[str] | None = None,
        top_n: int = 15,
        use_correlation: bool = True,
    ) -> ShapReport:
        """
        Generate full SHAP explanation with proxy detection.

        Args:
            protected_attrs: List of protected attribute column names.
            top_n: Number of top features to include in report.
            use_correlation: Also detect proxies via statistical correlation.
        """
        shap_vals = self.compute_shap_values()
        mean_shap = np.abs(shap_vals).mean(axis=0)
        signed_mean = shap_vals.mean(axis=0)

        sorted_idx = np.argsort(np.abs(mean_shap))[::-1][:top_n]
        total_abs = float(np.abs(mean_shap).sum())

        feature_importances: list[FeatureImportance] = []
        for rank, idx in enumerate(sorted_idx):
            feat = self.feature_names[idx]
            sv = float(signed_mean[idx])
            abs_sv = float(mean_shap[idx])

            proxied_by_pattern = self._detect_proxy(feat)
            proxied_by_corr: list[str] = []
            if use_correlation and protected_attrs:
                proxied_by_corr = self._detect_proxy_by_correlation(
                    feat, protected_attrs
                )

            all_proxied = list(set(proxied_by_pattern + proxied_by_corr))
            is_proxy = len(all_proxied) > 0

            feature_importances.append(FeatureImportance(
                feature=feat,
                shap_value=round(sv, 5),
                abs_shap_value=round(abs_sv, 5),
                direction="positive" if sv >= 0 else "negative",
                is_proxy=is_proxy,
                proxy_for=all_proxied,
                rank=rank + 1,
            ))

        proxy_vars = [fi.feature for fi in feature_importances if fi.is_proxy]
        top_pos = [fi.feature for fi in feature_importances if fi.direction == "positive"][:5]
        top_neg = [fi.feature for fi in feature_importances if fi.direction == "negative"][:5]

        fidelity = min(1.0, float(mean_shap[sorted_idx].sum() / total_abs)) if total_abs > 0 else 0.0

        return ShapReport(
            feature_importances=feature_importances,
            proxy_variables=proxy_vars,
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            mean_abs_shap=round(float(mean_shap.mean()), 5),
            explanation_fidelity=round(fidelity, 4),
        )

    def group_shap_comparison(
        self, attribute: str, top_n: int = 10
    ) -> dict[str, list[FeatureImportance]]:
        """
        Compare SHAP feature importance between demographic groups.
        Reveals which features drive disparate outcomes per group.
        """
        shap_vals = self.compute_shap_values()
        if attribute not in self.X_test.columns:
            raise ValueError(f"Attribute '{attribute}' not in dataset.")

        result = {}
        groups = self.X_test[attribute].unique()

        for val in groups:
            mask = (self.X_test[attribute] == val).values
            group_shap = shap_vals[mask]
            mean_g = np.abs(group_shap).mean(axis=0)
            signed_g = group_shap.mean(axis=0)
            top_idx = np.argsort(mean_g)[::-1][:top_n]

            importances = []
            for rank, idx in enumerate(top_idx):
                feat = self.feature_names[idx]
                sv = float(signed_g[idx])
                abs_sv = float(mean_g[idx])
                importances.append(FeatureImportance(
                    feature=feat,
                    shap_value=round(sv, 5),
                    abs_shap_value=round(abs_sv, 5),
                    direction="positive" if sv >= 0 else "negative",
                    is_proxy=len(self._detect_proxy(feat)) > 0,
                    proxy_for=self._detect_proxy(feat),
                    rank=rank + 1,
                ))
            result[str(val)] = importances

        return result

    def waterfall_data(self, sample_idx: int) -> dict:
        """Return data for a single-prediction waterfall explanation."""
        shap_vals = self.compute_shap_values()
        row_shap = shap_vals[sample_idx]
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base = float(np.asarray(ev).flat[1] if len(np.asarray(ev)) > 1 else np.asarray(ev).flat[0])
        else:
            base = float(ev)
        contributions = sorted(
            zip(self.feature_names, row_shap.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return {
            "base_value": round(base, 4),
            "final_value": round(base + float(row_shap.sum()), 4),
            "contributions": [
                {"feature": f, "shap_value": round(v, 5)} for f, v in contributions[:15]
            ],
        }
