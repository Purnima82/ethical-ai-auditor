"""
test_generators.py — Tests for SampleDatasetGenerator and ReportGenerator
"""
import numpy as np
import pandas as pd
import pytest

from backend.data.sample_generator import SampleDatasetGenerator, BiasConfig
from backend.core.report_generator import ReportGenerator, build_demo_result


class TestSampleDatasetGenerator:
    def test_generates_correct_shape(self):
        gen = SampleDatasetGenerator(n_samples=200, seed=0)
        X, y = gen.generate()
        assert len(X) == 200
        assert len(y) == 200

    def test_protected_attrs_present(self):
        gen = SampleDatasetGenerator(n_samples=200, seed=0)
        X, y = gen.generate()
        for attr in ["gender", "race", "age_group"]:
            assert attr in X.columns

    def test_proxy_cols_present_when_enabled(self):
        gen = SampleDatasetGenerator(n_samples=200, seed=0, include_proxies=True)
        X, y = gen.generate()
        for col in gen.get_proxy_cols():
            assert col in X.columns

    def test_no_proxies_when_disabled(self):
        gen = SampleDatasetGenerator(n_samples=200, seed=0, include_proxies=False)
        X, y = gen.generate()
        for col in ["zip_code", "school_prestige", "prior_salary"]:
            assert col not in X.columns

    def test_bias_produces_unequal_rates(self):
        gen = SampleDatasetGenerator(n_samples=1000, seed=42)
        X, y = gen.generate()
        df = X.copy(); df["outcome"] = y.values
        male_rate   = df[df["gender"] == "male"]["outcome"].mean()
        female_rate = df[df["gender"] == "female"]["outcome"].mean()
        assert male_rate > female_rate, "Injected bias should produce lower female hire rate"

    def test_seed_reproducibility(self):
        gen1 = SampleDatasetGenerator(n_samples=100, seed=99)
        gen2 = SampleDatasetGenerator(n_samples=100, seed=99)
        X1, y1 = gen1.generate()
        X2, y2 = gen2.generate()
        assert (y1.values == y2.values).all()

    def test_bias_summary_all_attrs(self):
        gen = SampleDatasetGenerator(n_samples=300, seed=0)
        X, y = gen.generate()
        summary = gen.bias_summary(X, y)
        assert "attribute" in summary.columns
        assert "positive_rate" in summary.columns
        for attr in ["gender", "race", "age_group"]:
            assert attr in summary["attribute"].values

    def test_credit_domain(self):
        gen = SampleDatasetGenerator(domain="credit", n_samples=200, seed=0)
        X, y = gen.generate()
        assert "credit_score" in X.columns

    def test_custom_bias_config(self):
        low_bias = BiasConfig(gender_female_flip=0.0, race_black_flip=0.0)
        hi_bias  = BiasConfig(gender_female_flip=0.9, race_black_flip=0.9)
        gen_low = SampleDatasetGenerator(n_samples=800, seed=1, bias_config=low_bias)
        gen_hi  = SampleDatasetGenerator(n_samples=800, seed=1, bias_config=hi_bias)
        X_lo, y_lo = gen_low.generate()
        X_hi, y_hi = gen_hi.generate()
        df_lo = X_lo.copy(); df_lo["o"] = y_lo.values
        df_hi = X_hi.copy(); df_hi["o"] = y_hi.values
        female_lo = df_lo[df_lo["gender"] == "female"]["o"].mean()
        female_hi = df_hi[df_hi["gender"] == "female"]["o"].mean()
        assert female_lo > female_hi


class TestReportGenerator:
    @pytest.fixture
    def demo_result(self):
        return build_demo_result("test-001", "Test hiring model", "hiring")

    def test_markdown_contains_verdict(self, demo_result):
        gen = ReportGenerator(demo_result)
        md = gen.to_markdown()
        assert "CAUTION" in md or "PASS" in md or "FAIL" in md

    def test_markdown_has_findings_section(self, demo_result):
        gen = ReportGenerator(demo_result)
        md = gen.to_markdown()
        assert "## Findings" in md

    def test_markdown_has_recommendations(self, demo_result):
        gen = ReportGenerator(demo_result)
        md = gen.to_markdown()
        assert "## Recommendations" in md

    def test_json_is_valid(self, demo_result):
        import json
        gen = ReportGenerator(demo_result)
        parsed = json.loads(gen.to_json())
        assert "audit_id" in parsed
        assert "overall_score" in parsed

    def test_executive_summary_non_empty(self, demo_result):
        gen = ReportGenerator(demo_result)
        summary = gen.executive_summary_text()
        assert len(summary) > 50
        assert "CAUTION" in summary or demo_result.verdict in summary

    def test_improvement_calculation(self, demo_result):
        assert demo_result.overall_improvement > 0

    def test_markdown_has_regulatory_flags(self, demo_result):
        gen = ReportGenerator(demo_result)
        md = gen.to_markdown()
        assert "Regulatory Flags" in md or "EU AI Act" in md
