"""
report_generator.py
Generates structured audit reports in JSON and Markdown formats.
Each report is fully reproducible from an AuditResult object.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GroupMetricRow:
    group: str
    selection_rate: float
    tpr: float
    fpr: float
    n_samples: int


@dataclass
class DimensionScore:
    name: str
    score: float          # 0–100
    interpretation: str
    passes_threshold: bool


@dataclass
class Finding:
    severity: str         # critical | warning | info
    title: str
    detail: str
    framework: str = ""   # EU AI Act | NIST | IEEE 7000
    remediation: str = ""


@dataclass
class AuditResult:
    """Complete structured result from a single audit run."""
    audit_id: str
    model_description: str
    domain: str
    depth: str
    protected_attributes: list[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Overall
    overall_score: int = 0
    verdict: str = "FAIL"          # PASS | CAUTION | FAIL
    verdict_reason: str = ""

    # Dimension scores
    dimension_scores: list[DimensionScore] = field(default_factory=list)

    # Per-attribute metrics
    fairness_before: dict[str, float] = field(default_factory=dict)
    fairness_after: dict[str, float] = field(default_factory=dict)

    # Group-level detail
    group_metrics: dict[str, list[GroupMetricRow]] = field(default_factory=dict)

    # SHAP
    top_features: list[dict[str, Any]] = field(default_factory=list)
    proxy_variables: list[str] = field(default_factory=list)

    # Findings & recommendations
    findings: list[Finding] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    regulatory_flags: list[str] = field(default_factory=list)

    # Improvement
    improvement_pct: float = 0.0
    mitigation_applied: list[str] = field(default_factory=list)

    @property
    def overall_improvement(self) -> float:
        if not self.fairness_before or not self.fairness_after:
            return self.improvement_pct
        before = sum(self.fairness_before.values()) / len(self.fairness_before)
        after  = sum(self.fairness_after.values())  / len(self.fairness_after)
        return round((after - before) / before * 100, 1) if before > 0 else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ReportGenerator:
    """
    Converts an AuditResult into human-readable Markdown or JSON reports.

    Reports include:
    - Executive summary
    - Fairness metrics table
    - SHAP feature importance section
    - Before/after mitigation comparison
    - Regulatory alignment (EU AI Act, NIST, IEEE 7000)
    - Findings ranked by severity
    - Actionable recommendations
    """

    SEVERITY_EMOJI = {"critical": "🔴", "warning": "🟡", "info": "🟢"}
    VERDICT_EMOJI  = {"PASS": "✅", "CAUTION": "⚠️", "FAIL": "❌"}

    def __init__(self, result: AuditResult):
        self.r = result

    def to_markdown(self) -> str:
        r = self.r
        lines: list[str] = []

        # Header
        lines += [
            f"# Ethical AI Audit Report",
            f"",
            f"**Audit ID:** `{r.audit_id}`  ",
            f"**Date:** {r.timestamp[:10]}  ",
            f"**Domain:** {r.domain.replace('-', ' ').title()}  ",
            f"**Depth:** {r.depth}  ",
            f"**Protected Attributes:** {', '.join(r.protected_attributes)}",
            f"",
            "---",
            "",
        ]

        # Executive summary
        verdict_icon = self.VERDICT_EMOJI.get(r.verdict, "")
        lines += [
            "## Executive Summary",
            "",
            f"| | |",
            f"|---|---|",
            f"| **Overall Score** | **{r.overall_score} / 100** |",
            f"| **Verdict** | {verdict_icon} **{r.verdict}** |",
            f"| **Summary** | {r.verdict_reason} |",
            f"| **Fairness Improvement** | +{r.overall_improvement:.0f}% after mitigation |",
            "",
        ]

        # Dimension scores
        if r.dimension_scores:
            lines += ["## Dimension Scores", "", "| Dimension | Score | Passes | Interpretation |", "|---|---|---|---|"]
            for ds in r.dimension_scores:
                tick = "✅" if ds.passes_threshold else "❌"
                lines.append(f"| {ds.name} | {ds.score:.0f}/100 | {tick} | {ds.interpretation} |")
            lines.append("")

        # Fairness metrics
        lines += ["## Fairness Metrics — Before vs After Mitigation", ""]
        if r.fairness_before:
            lines += ["| Attribute | Before | After | Improvement |", "|---|---|---|---|"]
            for attr in r.protected_attributes:
                b = r.fairness_before.get(attr, 0)
                a = r.fairness_after.get(attr, 0)
                imp = ((a - b) / b * 100) if b > 0 else 0
                lines.append(f"| {attr} | {b:.3f} | {a:.3f} | +{imp:.0f}% |")
            lines.append("")

        # SHAP section
        if r.top_features:
            lines += ["## SHAP Feature Importance", ""]
            if r.proxy_variables:
                lines.append(f"⚠️ **Proxy variables detected:** {', '.join(r.proxy_variables)}")
                lines.append("")
            lines += ["| Rank | Feature | SHAP Value | Direction | Proxy? |", "|---|---|---|---|---|"]
            for ft in r.top_features[:10]:
                proxy = f"⚠️ proxies `{', '.join(ft.get('proxy_for', []))}`" if ft.get("is_proxy") else "No"
                lines.append(f"| {ft.get('rank','-')} | `{ft.get('feature','')}` | {ft.get('shap_value',0):.4f} | {ft.get('direction','')} | {proxy} |")
            lines.append("")

        # Findings
        if r.findings:
            lines += ["## Findings", ""]
            for f_item in sorted(r.findings, key=lambda x: ["critical","warning","info"].index(x.severity)):
                icon = self.SEVERITY_EMOJI.get(f_item.severity, "")
                fw = f" _{f_item.framework}_" if f_item.framework else ""
                lines += [
                    f"### {icon} {f_item.title}{fw}",
                    f"{f_item.detail}",
                ]
                if f_item.remediation:
                    lines.append(f"**Remediation:** {f_item.remediation}")
                lines.append("")

        # Recommendations
        if r.recommendations:
            lines += ["## Recommendations", ""]
            for i, rec in enumerate(r.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        # Regulatory flags
        if r.regulatory_flags:
            lines += ["## Regulatory Flags", ""]
            for flag in r.regulatory_flags:
                lines.append(f"- ⚖️ {flag}")
            lines.append("")

        # Mitigation applied
        if r.mitigation_applied:
            lines += ["## Mitigation Techniques Applied", ""]
            for m in r.mitigation_applied:
                lines.append(f"- ✅ {m}")
            lines.append("")

        lines += [
            "---",
            f"*Generated by Ethical AI Auditor · {r.timestamp[:19]} UTC*",
            f"*Aligned with: EU AI Act (2024) · NIST AI RMF (2023) · IEEE 7000*",
        ]

        return "\n".join(lines)

    def to_json(self) -> str:
        return self.r.to_json()

    def executive_summary_text(self) -> str:
        r = self.r
        return (
            f"Audit of {r.domain} AI system ({r.audit_id[:8]}): "
            f"Overall score {r.overall_score}/100 — {r.verdict}. "
            f"{r.verdict_reason} "
            f"Fairness improved by +{r.overall_improvement:.0f}% across "
            f"{len(r.protected_attributes)} protected attributes after mitigation. "
            f"Proxy variables identified: {', '.join(r.proxy_variables) or 'none'}."
        )


def build_demo_result(audit_id: str, description: str, domain: str) -> AuditResult:
    """Build a realistic demo AuditResult for testing without a live model."""
    return AuditResult(
        audit_id=audit_id,
        model_description=description,
        domain=domain,
        depth="deep",
        protected_attributes=["gender", "race", "age_group"],
        overall_score=72,
        verdict="CAUTION",
        verdict_reason=(
            "Model exhibits measurable bias driven by proxy variables. "
            "Post-mitigation fairness meets NIST thresholds across all groups."
        ),
        dimension_scores=[
            DimensionScore("Bias Risk",     78, "Proxy variables removed; residual bias within NIST threshold.", True),
            DimensionScore("Transparency",  74, "SHAP values computed; model card documentation needed.", True),
            DimensionScore("Fairness",      81, "Demographic parity and equalized odds both pass after calibration.", True),
            DimensionScore("Accountability",77, "Audit trail maintained; human review process defined.", True),
            DimensionScore("Robustness",    80, "Tested across 3 demographic splits; stable under threshold sweep.", True),
            DimensionScore("Data Ethics",   76, "Training data reweighted; proxy columns excised.", True),
        ],
        fairness_before={"gender": 0.56, "race": 0.51, "age_group": 0.62},
        fairness_after= {"gender": 0.74, "race": 0.69, "age_group": 0.78},
        top_features=[
            {"rank":1,"feature":"years_experience","shap_value":0.38,"direction":"positive","is_proxy":False,"proxy_for":[]},
            {"rank":2,"feature":"education_level","shap_value":0.29,"direction":"positive","is_proxy":False,"proxy_for":[]},
            {"rank":3,"feature":"zip_code","shap_value":-0.22,"direction":"negative","is_proxy":True,"proxy_for":["race","socioeconomic"]},
            {"rank":4,"feature":"skills_match","shap_value":0.18,"direction":"positive","is_proxy":False,"proxy_for":[]},
            {"rank":5,"feature":"school_name","shap_value":-0.14,"direction":"negative","is_proxy":True,"proxy_for":["race","socioeconomic"]},
            {"rank":6,"feature":"prior_salary","shap_value":-0.11,"direction":"negative","is_proxy":True,"proxy_for":["gender","race"]},
        ],
        proxy_variables=["zip_code","school_name","prior_salary"],
        findings=[
            Finding("critical","Proxy Variable: zip_code","zip_code has Cramér's V=0.61 with race/ethnicity — strong proxy.","EU AI Act","Remove from feature set; verified absent in mitigated pipeline."),
            Finding("critical","Proxy Variable: prior_salary","prior_salary correlates with gender (Cramér's V=0.44) perpetuating historical pay gaps.","NIST AI RMF","Exclude from model; salary expectations from job posting are a safe alternative."),
            Finding("warning","Low Disparate Impact for Black applicants","Pre-mitigation DI ratio=0.60 violates EEOC 80% rule (threshold ≥0.80).","EU AI Act","Resolved post-mitigation: DI=0.85 after reweighting."),
            Finding("info","Model card incomplete","No documented intended use, limitations, or out-of-scope uses.","IEEE 7000","Complete model card per Mitchell et al. (2019) template."),
        ],
        recommendations=[
            "Deploy the mitigated pipeline (proxy removal + reweighting + calibration) as the production model.",
            "Implement quarterly bias audits re-running this pipeline on fresh production data.",
            "Complete the model card documenting intended use, limitations, and known failure modes.",
            "Register the system in the EU AI Act high-risk AI database before deployment in the EU.",
        ],
        regulatory_flags=[
            "EU AI Act Art. 10(3): Biometric data proxies (zip_code) must be explicitly justified or removed.",
            "NIST AI RMF MAP-1.5: Sociotechnical context not fully documented — complete stakeholder analysis.",
        ],
        mitigation_applied=[
            "Proxy variable removal: zip_code, school_name, prior_salary (ProxyRemover)",
            "Training data reweighting: Kamiran & Calders (2012) across gender × outcome cells",
            "Per-group threshold calibration: FPR equalised across gender groups",
        ],
        improvement_pct=20,
    )
