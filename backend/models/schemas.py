"""
schemas.py — Pydantic models for all API request/response contracts
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class AuditRequest(BaseModel):
    description: str = Field(..., min_length=20, description="AI system description")
    domain: str = Field(default="general", description="Domain: hiring, finance, healthcare, ...")
    depth: str = Field(default="quick", description="quick | deep | regulatory")
    protected_attrs: list[str] = Field(default=["gender", "race", "age_group"])


class GroupMetricSchema(BaseModel):
    group: str
    selection_rate: float
    true_positive_rate: float
    false_positive_rate: float
    sample_size: int


class FairnessMetricSchema(BaseModel):
    name: str
    value: float
    diff: float
    ratio: float
    passes: bool
    threshold: float
    group_values: dict[str, float]
    description: str


class FeatureImportanceSchema(BaseModel):
    feature: str
    shap_value: float
    abs_shap_value: float
    direction: str
    is_proxy: bool
    proxy_for: list[str]
    rank: int


class FindingSchema(BaseModel):
    severity: str      # critical | warning | info
    title: str
    detail: str
    framework: Optional[str] = None


class AuditResponse(BaseModel):
    audit_id: str
    overall_score: int
    verdict: str
    verdict_reason: str
    domain: str
    scores: dict[str, float]
    fairness_metrics: dict[str, FairnessMetricSchema]
    shap_features: list[FeatureImportanceSchema]
    findings: list[FindingSchema]
    recommendations: list[str]
    regulatory_flags: list[str]
    before_score: float
    after_score: float
    improvement_pct: float
    timestamp: str


class DatasetUploadRequest(BaseModel):
    dataset_name: str
    target_column: str
    protected_attrs: list[str]
    data: list[dict[str, Any]]


class ThresholdSweepRequest(BaseModel):
    attribute: str
    thresholds: Optional[list[float]] = None


class MitigationRequest(BaseModel):
    protected_attrs: list[str]
    explicit_proxies: Optional[list[str]] = None
    strategy: str = "full"   # "reweight" | "proxy_removal" | "threshold" | "full"
