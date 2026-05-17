"""
reports.py — Report generation API endpoints
Generates full Markdown / JSON audit reports from AuditResult objects.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from backend.core.report_generator import ReportGenerator, build_demo_result
from backend.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


class ReportRequest(BaseModel):
    domain: str = "hiring"
    description: str = "Hiring classification model"
    format: str = "markdown"   # markdown | json


@router.get("/sample")
async def sample_report():
    """Return a full JSON report for the demo hiring model."""
    result = build_demo_result(
        audit_id=f"rpt-demo-{uuid.uuid4().hex[:8]}",
        description="Hiring classifier — synthetic dataset demo",
        domain="hiring",
    )
    gen = ReportGenerator(result)
    return {
        "audit_id": result.audit_id,
        "verdict": result.verdict,
        "overall_score": result.overall_score,
        "improvement_pct": result.overall_improvement,
        "summary": gen.executive_summary_text(),
        "regulatory_alignment": {
            "eu_ai_act": result.regulatory_flags[0] if result.regulatory_flags else "",
            "nist_ai_rmf": result.regulatory_flags[1] if len(result.regulatory_flags) > 1 else "",
            "ieee_7000": "Transparency and accountability principles partially satisfied.",
        },
        "findings": [
            {"severity": f.severity, "title": f.title, "detail": f.detail, "framework": f.framework}
            for f in result.findings
        ],
        "recommendations": result.recommendations,
        "mitigation_applied": result.mitigation_applied,
        "dimension_scores": {d.name: d.score for d in result.dimension_scores},
        "fairness_before": result.fairness_before,
        "fairness_after": result.fairness_after,
        "proxy_variables": result.proxy_variables,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.post("/generate")
async def generate_report(req: ReportRequest):
    """Generate a Markdown or JSON report for a given domain + description."""
    result = build_demo_result(
        audit_id=f"rpt-{uuid.uuid4().hex[:8]}",
        description=req.description,
        domain=req.domain,
    )
    gen = ReportGenerator(result)

    if req.format == "markdown":
        md = gen.to_markdown()
        return PlainTextResponse(content=md, media_type="text/markdown")
    elif req.format == "json":
        return result.to_dict()
    else:
        raise HTTPException(400, f"Unknown format '{req.format}'. Use 'markdown' or 'json'.")


@router.get("/markdown/sample", response_class=PlainTextResponse)
async def markdown_sample():
    """Return a sample Markdown audit report (useful for README previews)."""
    result = build_demo_result(
        audit_id="rpt-sample-001",
        description="Hiring classifier demo",
        domain="hiring",
    )
    return PlainTextResponse(
        content=ReportGenerator(result).to_markdown(),
        media_type="text/markdown",
    )
