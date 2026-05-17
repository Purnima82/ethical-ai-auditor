"""
bias_detector.py
Core engine: detects bias across protected demographic attributes
using scikit-learn compatible models.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


@dataclass
class GroupMetrics:
    group: str
    attribute: str
    value: Any
    selection_rate: float
    true_positive_rate: float
    false_positive_rate: float
    true_negative_rate: float
    false_negative_rate: float
    sample_size: int
    positive_count: int


@dataclass
class FairnessReport:
    attribute: str
    demographic_parity: float
    demographic_parity_diff: float
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    equal_opportunity_diff: float
    disparate_impact_ratio: float
    group_metrics: list[GroupMetrics] = field(default_factory=list)
    passes_nist_threshold: bool = False
    verdict: str = "FAIL"

    def __post_init__(self):
        self.passes_nist_threshold = (
            abs(self.demographic_parity_diff) < 0.10
            and abs(self.equalized_odds_tpr_diff) < 0.10
            and abs(self.equalized_odds_fpr_diff) < 0.10
        )
        if self.passes_nist_threshold:
            self.verdict = "PASS"
        elif abs(self.demographic_parity_diff) < 0.20:
            self.verdict = "CAUTION"
        else:
            self.verdict = "FAIL"


class BiasDetector:
    """
    Detects algorithmic bias in ML model predictions across protected attributes.

    Metrics computed:
    - Demographic Parity (selection rate equality)
    - Equalized Odds (TPR + FPR equality across groups)
    - Equal Opportunity (TPR equality)
    - Disparate Impact Ratio (80% rule)

    Usage:
        detector = BiasDetector(model, X_test, y_test)
        reports = detector.audit(protected_attrs=["gender", "race", "age_group"])
        print(reports["gender"].verdict)
    """

    NIST_THRESHOLD = 0.10
    DISPARATE_IMPACT_THRESHOLD = 0.80

    def __init__(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.50,
        feature_cols: list | None = None,
    ):
        self.model = model
        self.X_test = X_test.copy().reset_index(drop=True)
        self.y_test = y_test.copy().reset_index(drop=True)
        self.threshold = threshold
        if feature_cols is not None:
            self._feature_cols = feature_cols
        elif hasattr(model, "feature_names_in_"):
            self._feature_cols = list(model.feature_names_in_)
        else:
            self._feature_cols = list(X_test.select_dtypes(include="number").columns)
        self._y_pred: np.ndarray | None = None
        self._y_prob: np.ndarray | None = None

    def _get_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        if self._y_pred is None:
            X_model = self.X_test[self._feature_cols]
            if hasattr(self.model, "predict_proba"):
                self._y_prob = self.model.predict_proba(X_model)[:, 1]
                self._y_pred = (self._y_prob >= self.threshold).astype(int)
            else:
                self._y_pred = self.model.predict(X_model)
                self._y_prob = self._y_pred.astype(float)
        return self._y_pred, self._y_prob

    def _group_metrics(
        self, attribute: str, group_value: Any, mask: np.ndarray
    ) -> GroupMetrics:
        y_pred, _ = self._get_predictions()
        y_true = self.y_test.values

        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        if len(y_true_g) == 0:
            raise ValueError(f"No samples for group {attribute}={group_value}")

        tn, fp, fn, tp = confusion_matrix(
            y_true_g, y_pred_g, labels=[0, 1]
        ).ravel() if len(np.unique(y_true_g)) > 1 else (
            np.sum(y_true_g == 0), 0, 0, np.sum(y_true_g == 1)
        )

        total = tp + tn + fp + fn
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        selection_rate = (tp + fp) / total if total > 0 else 0.0

        return GroupMetrics(
            group=f"{attribute}={group_value}",
            attribute=attribute,
            value=group_value,
            selection_rate=round(selection_rate, 4),
            true_positive_rate=round(tpr, 4),
            false_positive_rate=round(fpr, 4),
            true_negative_rate=round(tnr, 4),
            false_negative_rate=round(fnr, 4),
            sample_size=int(total),
            positive_count=int(tp + fp),
        )

    def audit_attribute(self, attribute: str) -> FairnessReport:
        """Compute all fairness metrics for one protected attribute."""
        if attribute not in self.X_test.columns:
            raise ValueError(f"Attribute '{attribute}' not found in dataset.")

        col = self.X_test[attribute]
        groups = col.unique()
        group_metrics = []

        for val in groups:
            mask = (col == val).values
            if mask.sum() < 10:
                continue
            gm = self._group_metrics(attribute, val, mask)
            group_metrics.append(gm)

        if len(group_metrics) < 2:
            raise ValueError(f"Need at least 2 groups for attribute '{attribute}'.")

        selection_rates = [gm.selection_rate for gm in group_metrics]
        tprs = [gm.true_positive_rate for gm in group_metrics]
        fprs = [gm.false_positive_rate for gm in group_metrics]

        dp_diff = max(selection_rates) - min(selection_rates)
        dp_score = 1.0 - dp_diff
        eo_tpr_diff = max(tprs) - min(tprs)
        eo_fpr_diff = max(fprs) - min(fprs)
        eop_diff = eo_tpr_diff

        min_sr = min(selection_rates)
        max_sr = max(selection_rates)
        di_ratio = min_sr / max_sr if max_sr > 0 else 0.0

        return FairnessReport(
            attribute=attribute,
            demographic_parity=round(dp_score, 4),
            demographic_parity_diff=round(dp_diff, 4),
            equalized_odds_tpr_diff=round(eo_tpr_diff, 4),
            equalized_odds_fpr_diff=round(eo_fpr_diff, 4),
            equal_opportunity_diff=round(eop_diff, 4),
            disparate_impact_ratio=round(di_ratio, 4),
            group_metrics=group_metrics,
        )

    def audit(self, protected_attrs: list[str]) -> dict[str, FairnessReport]:
        """Run full bias audit across all specified protected attributes."""
        results = {}
        for attr in protected_attrs:
            try:
                results[attr] = self.audit_attribute(attr)
            except ValueError as e:
                print(f"[BiasDetector] Skipping '{attr}': {e}")
        return results

    def overall_fairness_score(self, reports: dict[str, FairnessReport]) -> float:
        """Aggregate fairness score (0–1) across all attributes."""
        scores = [r.demographic_parity for r in reports.values()]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def set_threshold(self, threshold: float) -> None:
        """Update decision threshold and invalidate cached predictions."""
        self.threshold = threshold
        self._y_pred = None
        self._y_prob = None

    def threshold_sweep(
        self, attribute: str, thresholds: list[float] | None = None
    ) -> pd.DataFrame:
        """
        Sweep decision thresholds and return fairness metrics at each point.
        Useful for finding the fairness-accuracy Pareto frontier.
        """
        if thresholds is None:
            thresholds = [i / 100 for i in range(20, 81, 5)]

        rows = []
        original_threshold = self.threshold
        for t in thresholds:
            self.set_threshold(t)
            try:
                report = self.audit_attribute(attribute)
                rows.append({
                    "threshold": t,
                    "demographic_parity_diff": report.demographic_parity_diff,
                    "equalized_odds_tpr_diff": report.equalized_odds_tpr_diff,
                    "disparate_impact_ratio": report.disparate_impact_ratio,
                    "verdict": report.verdict,
                })
            except ValueError:
                pass

        self.set_threshold(original_threshold)
        return pd.DataFrame(rows)
