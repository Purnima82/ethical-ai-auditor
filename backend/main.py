"""
main.py — FastAPI application entry point
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from backend.api.audit import router as audit_router
from backend.api.fairness import router as fairness_router
from backend.api.reports import router as reports_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Ethical AI Auditor API starting...")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Ethical AI Auditor API",
    description="Bias detection, fairness metrics, and SHAP explainability for ML models.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audit_router, prefix="/api/audit", tags=["Audit"])
app.include_router(fairness_router, prefix="/api/fairness", tags=["Fairness"])
app.include_router(reports_router, prefix="/api/reports", tags=["Reports"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ethical-ai-auditor"}
