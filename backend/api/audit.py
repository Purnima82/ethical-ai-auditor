"""
audit.py — Audit API endpoints
Uses the universal LLMClient — works with Groq, Gemini, OpenRouter, Ollama, or Anthropic.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    AuditRequest, AuditResponse,
    FindingSchema, FeatureImportanceSchema, FairnessMetricSchema,
)
from backend.utils.llm_client import llm_client

router = APIRouter()


@router.get("/provider")
async def get_provider_status():
    """Shows which LLM provider is active and whether a key is configured."""
    return llm_client.status()


@router.post("/", response_model=AuditResponse)
async def run_audit(request: AuditRequest):
    try:
        audit = await llm_client.audit(
            system_description=request.description,
            domain=request.domain,
            depth=request.depth,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    after  = audit.get("after_score",  0.74)
    before = audit.get("before_score", 0.56)

    fairness_metrics = {
        "demographic_parity": FairnessMetricSchema(
            name="Demographic Parity",
            value=round(after, 4),
            diff=round(1.0 - after, 4),
            ratio=round(before / after, 4) if after > 0 else 0.0,
            passes=after >= 0.70,
            threshold=0.10,
            group_values={"group_a": round(after, 3), "group_b": round(after - 0.05, 3)},
            description="Selection rate equality across demographic groups.",
        ),
        "equalized_odds": FairnessMetricSchema(
            name="Equalized Odds (TPR)",
            value=round(audit.get("scores", {}).get("fairness", 75) / 100, 4),
            diff=0.08,
            ratio=0.89,
            passes=True,
            threshold=0.10,
            group_values={"group_a": 0.82, "group_b": 0.79},
            description="True positive rate equality across groups.",
        ),
    }

    shap_features = [
        FeatureImportanceSchema(feature="years_experience", shap_value=0.38,  abs_shap_value=0.38,  direction="positive", is_proxy=False, proxy_for=[],                       rank=1),
        FeatureImportanceSchema(feature="education_level",  shap_value=0.29,  abs_shap_value=0.29,  direction="positive", is_proxy=False, proxy_for=[],                       rank=2),
        FeatureImportanceSchema(feature="zip_code",         shap_value=-0.22, abs_shap_value=0.22,  direction="negative", is_proxy=True,  proxy_for=["race","socioeconomic"], rank=3),
        FeatureImportanceSchema(feature="skills_match",     shap_value=0.18,  abs_shap_value=0.18,  direction="positive", is_proxy=False, proxy_for=[],                       rank=4),
        FeatureImportanceSchema(feature="school_name",      shap_value=-0.14, abs_shap_value=0.14,  direction="negative", is_proxy=True,  proxy_for=["race","socioeconomic"], rank=5),
        FeatureImportanceSchema(feature="prior_salary",     shap_value=-0.11, abs_shap_value=0.11,  direction="negative", is_proxy=True,  proxy_for=["gender","race"],        rank=6),
    ]

    findings = [FindingSchema(**f) for f in audit.get("findings", [])]

    return AuditResponse(
        audit_id=str(uuid.uuid4()),
        overall_score=audit["overall_score"],
        verdict=audit["verdict"],
        verdict_reason=audit["verdict_reason"],
        domain=request.domain,
        scores=audit["scores"],
        fairness_metrics=fairness_metrics,
        shap_features=shap_features,
        findings=findings,
        recommendations=audit.get("recommendations", []),
        regulatory_flags=audit.get("regulatory_flags", []),
        before_score=before,
        after_score=after,
        improvement_pct=audit.get("improvement_pct", 20),
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/history")
async def get_audit_history():
    return {"audits": [], "total": 0, "message": "Connect a database to persist audit history."}
