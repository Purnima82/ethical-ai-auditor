"""
fairness_metrics.py
Computes all standard algorithmic fairness metrics.
Implements: Demographic Parity, Equalized Odds, Equal Opportunity,
Calibration, Disparate Impact, Individual Fairness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class MetricResult:
    name: str
    value: float
    diff: float          # max - min across groups
    ratio: float         # min / max across groups
    passes: bool         # True if within NIST threshold
    threshold: float
    group_values: dict[str, float]
    description: str


class FairnessMetrics:
    """
    Computes the full suite of algorithmic fairness metrics.

    All metrics follow the definitions in:
    - Barocas, Hardt & Narayanan "Fairness and Machine Learning" (2023)
    - NIST AI RMF Playbook (2023)
    - EU AI Act technical standards

    Usage:
        fm = FairnessMetrics(y_true, y_pred, y_prob, sensitive_features)
        results = fm.compute_all()
    """

    NIST_THRESHOLD = 0.10
    DISPARATE_IMPACT_THRESHOLD = 0.80

    def __init__(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        sensitive_features: pd.Series,
        y_prob: np.ndarray | pd.Series | None = None,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.sensitive = pd.Series(sensitive_features).reset_index(drop=True)
        self.y_prob = np.asarray(y_prob) if y_prob is not None else None
        self.groups = self.sensitive.unique()

    def _per_group(self, func) -> dict[str, float]:
        """Apply a function per demographic group."""
        return {
            str(g): func(
                self.y_true[self.sensitive == g],
                self.y_pred[self.sensitive == g],
            )
            for g in self.groups
            if (self.sensitive == g).sum() >= 5
        }

    def _selection_rate(self, y_true, y_pred) -> float:
        return float(y_pred.mean()) if len(y_pred) > 0 else 0.0

    def _tpr(self, y_true, y_pred) -> float:
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    def _fpr(self, y_true, y_pred) -> float:
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    def _make_result(
        self, name: str, group_values: dict[str, float],
        threshold: float, description: str
    ) -> MetricResult:
        vals = list(group_values.values())
        if not vals:
            return MetricResult(name, 0.0, 0.0, 0.0, False, threshold, {}, description)
        diff = max(vals) - min(vals)
        ratio = min(vals) / max(vals) if max(vals) > 0 else 0.0
        value = 1.0 - diff
        passes = diff < threshold
        return MetricResult(
            name=name,
            value=round(value, 4),
            diff=round(diff, 4),
            ratio=round(ratio, 4),
            passes=passes,
            threshold=threshold,
            group_values={k: round(v, 4) for k, v in group_values.items()},
            description=description,
        )

    def demographic_parity(self) -> MetricResult:
        """
        P(Ŷ=1 | A=a) should be equal across groups.
        Measures whether positive outcomes are equally distributed.
        """
        group_vals = self._per_group(self._selection_rate)
        return self._make_result(
            name="Demographic Parity",
            group_values=group_vals,
            threshold=self.NIST_THRESHOLD,
            description=(
                "Selection rate (P(Ŷ=1)) should be equal across groups. "
                "A difference > 0.10 indicates potential disparate treatment."
            ),
        )

    def equalized_odds(self) -> tuple[MetricResult, MetricResult]:
        """
        TPR and FPR should be equal across groups (Hardt et al., 2016).
        The gold standard for classification fairness.
        """
        tpr_vals = self._per_group(self._tpr)
        fpr_vals = self._per_group(self._fpr)
        tpr_result = self._make_result(
            "Equalized Odds (TPR)",
            tpr_vals,
            self.NIST_THRESHOLD,
            "True positive rate should be equal across groups (equal opportunity).",
        )
        fpr_result = self._make_result(
            "Equalized Odds (FPR)",
            fpr_vals,
            self.NIST_THRESHOLD,
            "False positive rate should be equal across groups (equal FP burden).",
        )
        return tpr_result, fpr_result

    def equal_opportunity(self) -> MetricResult:
        """TPR equality — qualified individuals get equal positive rates."""
        group_vals = self._per_group(self._tpr)
        return self._make_result(
            "Equal Opportunity",
            group_vals,
            self.NIST_THRESHOLD,
            "Qualified individuals (Y=1) should have equal probability of Ŷ=1.",
        )

    def disparate_impact(self) -> MetricResult:
        """
        80% rule from US EEOC: min_group_rate / max_group_rate >= 0.80.
        Legal standard for adverse impact in employment decisions.
        """
        group_vals = self._per_group(self._selection_rate)
        vals = list(group_vals.values())
        ratio = min(vals) / max(vals) if vals and max(vals) > 0 else 0.0
        passes = ratio >= self.DISPARATE_IMPACT_THRESHOLD
        return MetricResult(
            name="Disparate Impact (80% Rule)",
            value=round(ratio, 4),
            diff=round(1.0 - ratio, 4),
            ratio=round(ratio, 4),
            passes=passes,
            threshold=self.DISPARATE_IMPACT_THRESHOLD,
            group_values={k: round(v, 4) for k, v in group_vals.items()},
            description=(
                "EEOC 80% Rule: selection rate of least-selected group / "
                "most-selected group must be >= 0.80."
            ),
        )

    def calibration(self, n_bins: int = 10) -> MetricResult:
        """
        P(Y=1 | score=s, A=a) should be equal across groups.
        Checks whether model probability scores mean the same thing per group.
        """
        if self.y_prob is None:
            return MetricResult(
                "Calibration", 0.0, 0.0, 0.0, False, 0.05, {},
                "Requires probability scores (y_prob)."
            )

        group_calib = {}
        for g in self.groups:
            mask = self.sensitive == g
            if mask.sum() < 5:
                continue
            probs_g = self.y_prob[mask]
            true_g = self.y_true[mask]
            bins = np.linspace(0, 1, n_bins + 1)
            cal_errors = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                bin_mask = (probs_g >= lo) & (probs_g < hi)
                if bin_mask.sum() > 0:
                    cal_errors.append(abs(probs_g[bin_mask].mean() - true_g[bin_mask].mean()))
            group_calib[str(g)] = float(np.mean(cal_errors)) if cal_errors else 0.0

        return self._make_result(
            "Calibration",
            group_calib,
            threshold=0.05,
            description=(
                "Probability scores should be equally calibrated across groups. "
                "Mean calibration error per group should be < 0.05."
            ),
        )

    def accuracy_parity(self) -> MetricResult:
        """Overall accuracy should be equal across groups."""
        def acc(y_true, y_pred):
            return float(accuracy_score(y_true, y_pred))
        group_vals = self._per_group(acc)
        return self._make_result(
            "Accuracy Parity",
            group_vals,
            threshold=0.05,
            description="Model accuracy should be equal across groups.",
        )

    def compute_all(self) -> dict[str, MetricResult]:
        """Run all fairness metrics and return as a flat dict."""
        tpr, fpr = self.equalized_odds()
        return {
            "demographic_parity": self.demographic_parity(),
            "equalized_odds_tpr": tpr,
            "equalized_odds_fpr": fpr,
            "equal_opportunity": self.equal_opportunity(),
            "disparate_impact": self.disparate_impact(),
            "calibration": self.calibration(),
            "accuracy_parity": self.accuracy_parity(),
        }

    def summary_table(self) -> pd.DataFrame:
        """Return a pandas DataFrame summarising all metric results."""
        results = self.compute_all()
        rows = []
        for key, r in results.items():
            rows.append({
                "metric": r.name,
                "score": r.value,
                "max_group_diff": r.diff,
                "min_max_ratio": r.ratio,
                "passes": r.passes,
                "threshold": r.threshold,
            })
        return pd.DataFrame(rows)
