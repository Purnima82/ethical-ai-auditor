"""
llm_client.py
Universal LLM client — switch between free providers with one .env line.

Supported providers (all FREE):
  - groq       → console.groq.com          (Llama 3, Mixtral — fastest)
  - gemini     → aistudio.google.com       (Gemini 1.5 Flash — best free quota)
  - openrouter → openrouter.ai             (50+ models, many free)
  - ollama     → ollama.ai                 (100% local, no key needed)
  - anthropic  → console.anthropic.com     (Claude — paid, best quality)

Set in .env:
  LLM_PROVIDER=groq
  LLM_API_KEY=your-key-here
  LLM_MODEL=llama-3.3-70b-versatile   (optional, has sensible defaults)
"""
from __future__ import annotations

import json
import os
import httpx
from typing import Optional


# ── Default models per provider ───────────────────────────────────────────────
PROVIDER_DEFAULTS = {
    "groq": {
        "model":   "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "style":   "openai",
    },
    "gemini": {
        "model":   "gemini-1.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "style":   "gemini",
    },
    "openrouter": {
        "model":   "meta-llama/llama-3.1-8b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "style":   "openai",
    },
    "ollama": {
        "model":   "llama3.2",
        "base_url": "http://localhost:11434/api/chat",
        "style":   "ollama",
    },
    "anthropic": {
        "model":   "claude-sonnet-4-20250514",
        "base_url": "https://api.anthropic.com/v1/messages",
        "style":   "anthropic",
    },
}

AUDIT_SYSTEM_PROMPT = """You are an expert AI Ethics Auditor. Analyze the described AI system and return ONLY valid JSON — no markdown, no explanation, no preamble.

Return exactly this structure:
{
  "overall_score": <integer 0-100>,
  "verdict": "<PASS|CAUTION|FAIL>",
  "verdict_reason": "<one sentence>",
  "scores": {
    "bias_risk": <0-100, higher = less bias>,
    "transparency": <0-100>,
    "fairness": <0-100>,
    "accountability": <0-100>,
    "robustness": <0-100>,
    "data_ethics": <0-100>
  },
  "findings": [
    {"severity": "critical", "title": "<short>", "detail": "<1-2 sentences>", "framework": "EU AI Act"},
    {"severity": "warning",  "title": "<short>", "detail": "<1-2 sentences>", "framework": "NIST"},
    {"severity": "warning",  "title": "<short>", "detail": "<1-2 sentences>", "framework": "IEEE 7000"},
    {"severity": "info",     "title": "<short>", "detail": "<1-2 sentences>", "framework": "NIST"}
  ],
  "recommendations": [
    "<specific action 1>",
    "<specific action 2>",
    "<specific action 3>",
    "<specific action 4>"
  ],
  "regulatory_flags": [
    "<EU AI Act concern>",
    "<NIST RMF flag>"
  ],
  "before_score": <float between 0.45-0.65>,
  "after_score": <float between 0.68-0.85>,
  "improvement_pct": <integer 15-30>
}

Rules: Be specific. Reference real metrics (demographic parity, equalized odds, SHAP). Cite real standards. Return ONLY the JSON object."""


class LLMClient:
    """
    Universal async LLM client.
    Auto-detects provider from LLM_PROVIDER env var.
    Falls back to a smart mock response if no API key is set
    (useful for demos and testing the UI without any account).
    """

    def __init__(self):
        self.provider  = os.getenv("LLM_PROVIDER", "groq").lower().strip()
        self.api_key   = os.getenv("LLM_API_KEY", "").strip()
        self.model     = os.getenv("LLM_MODEL", "").strip()
        self.config    = PROVIDER_DEFAULTS.get(self.provider, PROVIDER_DEFAULTS["groq"])
        if not self.model:
            self.model = self.config["model"]

    # ── Public method ─────────────────────────────────────────────────────────

    async def audit(self, system_description: str, domain: str, depth: str) -> dict:
        """
        Run an ethical audit of the described AI system.
        Returns a parsed dict matching the audit JSON schema.
        """
        user_message = (
            f"Audit this AI system:\n\n{system_description}\n\n"
            f"Domain: {domain}\n"
            f"Audit depth: {depth}\n"
            f"Protected attributes to assess: gender, race, age"
        )

        if not self.api_key and self.provider != "ollama":
            return self._mock_response(system_description, domain)

        style = self.config["style"]
        try:
            if style == "openai":
                raw = await self._call_openai_style(user_message)
            elif style == "gemini":
                raw = await self._call_gemini(user_message)
            elif style == "ollama":
                raw = await self._call_ollama(user_message)
            elif style == "anthropic":
                raw = await self._call_anthropic(user_message)
            else:
                return self._mock_response(system_description, domain)

            return self._parse(raw)

        except Exception as e:
            print(f"[LLMClient] {self.provider} error: {e} — using mock response")
            return self._mock_response(system_description, domain)

    # ── OpenAI-compatible (Groq + OpenRouter) ────────────────────────────────

    async def _call_openai_style(self, user_message: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/ethical-ai-auditor"
            headers["X-Title"]      = "Ethical AI Auditor"

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "temperature": 0.2,
            "max_tokens":  1500,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(self.config["base_url"], headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

    # ── Google Gemini ─────────────────────────────────────────────────────────

    async def _call_gemini(self, user_message: str) -> str:
        url = self.config["base_url"].format(model=self.model)
        url += f"?key={self.api_key}"
        body = {
            "contents": [{
                "parts": [{"text": AUDIT_SYSTEM_PROMPT + "\n\n" + user_message}]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1500,
            },
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    # ── Ollama (local) ────────────────────────────────────────────────────────

    async def _call_ollama(self, user_message: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(self.config["base_url"], json=body)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]

    # ── Anthropic Claude ──────────────────────────────────────────────────────

    async def _call_anthropic(self, user_message: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": 1500,
            "system": AUDIT_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(self.config["base_url"], headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            return "".join(
                b.get("text", "") for b in data.get("content", [])
                if b.get("type") == "text"
            )

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse(self, raw: str) -> dict:
        """Extract and parse JSON from LLM response."""
        text = raw.strip()
        # Strip markdown code fences if present
        for fence in ["```json", "```JSON", "```"]:
            if fence in text:
                text = text.split(fence, 1)[-1]
                text = text.rsplit("```", 1)[0]
                break
        # Find the first { ... } block
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        return json.loads(text)

    # ── Mock response (no API key / offline demo) ────────────────────────────

    def _mock_response(self, description: str, domain: str) -> dict:
        """
        Returns a realistic mock audit result.
        Used when no API key is configured so the UI still works fully.
        """
        desc_lower = description.lower()
        is_high_risk = any(w in desc_lower for w in [
            "criminal", "medical", "health", "loan", "credit", "hiring", "resume"
        ])
        score = 52 if is_high_risk else 68
        verdict = "FAIL" if is_high_risk else "CAUTION"

        return {
            "overall_score": score,
            "verdict": verdict,
            "verdict_reason": (
                "System shows significant demographic bias risk through proxy variable usage "
                "and training data reflecting historical inequities."
                if is_high_risk else
                "System has moderate fairness concerns that should be addressed before deployment."
            ),
            "scores": {
                "bias_risk":      48 if is_high_risk else 62,
                "transparency":   55,
                "fairness":       50 if is_high_risk else 65,
                "accountability": 60,
                "robustness":     65,
                "data_ethics":    45 if is_high_risk else 58,
            },
            "findings": [
                {
                    "severity": "critical",
                    "title": "Historical bias in training data",
                    "detail": "The model is trained on historical data that reflects past human discrimination. This perpetuates existing disparities rather than measuring actual potential.",
                    "framework": "EU AI Act",
                },
                {
                    "severity": "warning",
                    "title": "Proxy variable risk detected",
                    "detail": "Features correlated with protected attributes (zip code, prior salary, school name) may act as demographic proxies. SHAP analysis recommended.",
                    "framework": "NIST",
                },
                {
                    "severity": "warning",
                    "title": "Equalized odds not verified",
                    "detail": "No evidence that true positive rates are equal across demographic groups. Minority applicants may face higher false negative rates.",
                    "framework": "IEEE 7000",
                },
                {
                    "severity": "info",
                    "title": "Documentation gap",
                    "detail": "Model card and datasheet are missing. NIST AI RMF GOVERN function requires documented intended use, known limitations, and out-of-scope cases.",
                    "framework": "NIST",
                },
            ],
            "recommendations": [
                "Run SHAP analysis to identify proxy variables and remove zip code, prior salary, and school name from feature set",
                "Apply training data reweighting (Kamiran & Calders 2012) to balance class-group distributions before retraining",
                "Compute demographic parity and equalized odds across gender, race, and age — flag any group gap > 0.10",
                "Register system in EU AI database and complete conformity assessment before production deployment",
            ],
            "regulatory_flags": [
                "EU AI Act Annex III: This system may qualify as high-risk — requires conformity assessment",
                "NIST AI RMF: MEASURE function incomplete — fairness metrics not computed pre-deployment",
            ],
            "before_score": 0.54,
            "after_score":  0.74,
            "improvement_pct": 20,
            "_note": "Demo mode — set LLM_PROVIDER and LLM_API_KEY in .env for live AI analysis",
        }

    def status(self) -> dict:
        """Return current provider configuration (safe to expose in API)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "has_key": bool(self.api_key),
            "mode": "live" if (self.api_key or self.provider == "ollama") else "demo",
        }


# Singleton — imported by audit.py
llm_client = LLMClient()
