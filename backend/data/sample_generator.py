"""
sample_generator.py
Generates realistic synthetic hiring / credit / healthcare datasets
with known, controllable bias for demonstration and testing.

Usage:
    gen = SampleDatasetGenerator(domain="hiring", n_samples=2000, seed=42)
    df, y = gen.generate()
    print(gen.bias_summary(df, y))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


Domain = Literal["hiring", "credit", "healthcare", "criminal_justice"]


@dataclass
class BiasConfig:
    """Controls injected bias levels per group."""
    # Fraction of positive outcomes to flip to negative for each biased group
    gender_female_flip: float = 0.38
    gender_nonbinary_flip: float = 0.42
    race_black_flip: float = 0.45
    race_hispanic_flip: float = 0.35
    age_young_flip: float = 0.20    # 18-24
    age_old_flip: float = 0.30     # 55+


@dataclass
class DatasetSchema:
    """Column names for each domain."""
    domain: str
    feature_cols: list[str]
    proxy_cols: list[str]
    protected_attrs: list[str] = field(default_factory=lambda: ["gender", "race", "age_group"])
    target_col: str = "outcome"
    positive_label: str = "hired"


DOMAIN_SCHEMAS: dict[str, DatasetSchema] = {
    "hiring": DatasetSchema(
        domain="hiring",
        feature_cols=[
            "years_experience", "education_level", "skills_score",
            "interview_score", "cert_count", "gpa",
            "project_count", "reference_score",
        ],
        proxy_cols=["zip_code", "school_prestige", "prior_salary"],
        target_col="hired",
        positive_label="hired",
    ),
    "credit": DatasetSchema(
        domain="credit",
        feature_cols=[
            "credit_score", "debt_to_income", "employment_years",
            "num_accounts", "payment_history", "loan_amount",
            "collateral_value", "income_stability",
        ],
        proxy_cols=["zip_code", "neighborhood_income", "last_salary"],
        target_col="approved",
        positive_label="approved",
    ),
    "healthcare": DatasetSchema(
        domain="healthcare",
        feature_cols=[
            "bmi", "blood_pressure", "cholesterol", "glucose",
            "smoking_history", "family_history", "physical_activity",
            "prior_diagnoses",
        ],
        proxy_cols=["insurance_type", "zip_code", "hospital_type"],
        target_col="high_risk",
        positive_label="high_risk",
    ),
}


class SampleDatasetGenerator:
    """
    Generates synthetic ML datasets with injected demographic bias.

    The generated dataset includes:
    - Numeric feature columns (legitimately predictive)
    - Proxy columns (correlated with protected attributes)
    - Protected attribute columns (gender, race, age_group)
    - Binary target column with controllable per-group bias

    This mirrors real-world scenarios where historical data encodes
    systemic discrimination in its training labels.
    """

    def __init__(
        self,
        domain: Domain = "hiring",
        n_samples: int = 2000,
        seed: int = 42,
        bias_config: BiasConfig | None = None,
        include_proxies: bool = True,
    ):
        self.domain = domain
        self.n_samples = n_samples
        self.seed = seed
        self.bias_config = bias_config or BiasConfig()
        self.include_proxies = include_proxies
        self.schema = DOMAIN_SCHEMAS.get(domain, DOMAIN_SCHEMAS["hiring"])
        self._rng = np.random.default_rng(seed)

    def generate(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Generate the full dataset.

        Returns:
            (X DataFrame with features + protected attrs, y Series with labels)
        """
        n = self.n_samples
        schema = self.schema

        # Base features from sklearn (ensures realistic correlations)
        X_raw, y_base = make_classification(
            n_samples=n,
            n_features=len(schema.feature_cols),
            n_informative=5,
            n_redundant=2,
            random_state=self.seed,
        )
        df = pd.DataFrame(X_raw, columns=schema.feature_cols)

        # Normalise features to realistic ranges
        df = self._scale_features(df)

        # Protected attributes
        df["gender"] = self._rng.choice(
            ["male", "female", "non_binary"], n, p=[0.50, 0.45, 0.05]
        )
        df["race"] = self._rng.choice(
            ["white", "black", "hispanic", "asian", "other"],
            n, p=[0.50, 0.15, 0.15, 0.15, 0.05],
        )
        df["age_group"] = self._rng.choice(
            ["18-24", "25-34", "35-44", "45-54", "55+"], n
        )

        # Proxy variables (correlated with protected attrs)
        if self.include_proxies:
            df = self._add_proxies(df)

        # Inject bias into labels
        y_biased = self._inject_bias(y_base, df)
        df[schema.target_col] = y_biased

        X = df.drop(columns=[schema.target_col])
        y = df[schema.target_col]
        return X, y

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale raw sklearn features to domain-appropriate ranges."""
        schema = self.schema
        cols = schema.feature_cols
        if self.domain == "hiring":
            df[cols[0]] = (df[cols[0]] * 4 + 5).clip(0, 20).round(1)   # years_exp
            df[cols[1]] = self._rng.integers(1, 5, len(df))             # education 1-4
            df[cols[2]] = ((df[cols[2]] + 3) / 6 * 100).clip(0, 100)   # skills 0-100
            df[cols[3]] = ((df[cols[3]] + 3) / 6 * 10).clip(0, 10)     # interview 0-10
            df[cols[4]] = self._rng.integers(0, 6, len(df))             # certs
            df[cols[5]] = ((df[cols[5]] + 3) / 6 * 1.5 + 2.5).clip(2.0, 4.0).round(2)  # gpa
            df[cols[6]] = self._rng.integers(0, 10, len(df))            # projects
            df[cols[7]] = self._rng.integers(1, 6, len(df))             # refs
        else:
            # Generic 0-100 normalisation for other domains
            for c in cols:
                df[c] = ((df[c] - df[c].min()) / (df[c].max() - df[c].min()) * 100).round(2)
        return df

    def _add_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add proxy columns correlated with race and gender."""
        n = len(df)
        race_num = df["race"].map(
            {"white": 5, "asian": 4, "hispanic": 3, "black": 2, "other": 1}
        )
        gender_mult = df["gender"].map({"male": 1.0, "female": 0.82, "non_binary": 0.88})

        proxy_cols = self.schema.proxy_cols
        df[proxy_cols[0]] = (race_num + self._rng.integers(0, 3, n)).astype(str)   # zip/category
        df[proxy_cols[1]] = (race_num * 10 + self._rng.integers(0, 5, n)).astype(int)  # school prestige / neighborhood
        if self.domain == "hiring":
            df[proxy_cols[2]] = (
                gender_mult * 65000 + self._rng.normal(0, 8000, n)
            ).clip(30000, 150000).astype(int)
        else:
            df[proxy_cols[2]] = (
                gender_mult * 55000 + self._rng.normal(0, 7000, n)
            ).clip(20000, 120000).astype(int)
        return df

    def _inject_bias(self, y: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Flip positive labels for biased groups to simulate historical discrimination."""
        bc = self.bias_config
        y_biased = y.copy()

        flips: list[tuple[str, str, float]] = [
            ("gender", "female",     bc.gender_female_flip),
            ("gender", "non_binary", bc.gender_nonbinary_flip),
            ("race",   "black",      bc.race_black_flip),
            ("race",   "hispanic",   bc.race_hispanic_flip),
            ("age_group", "18-24",   bc.age_young_flip),
            ("age_group", "55+",     bc.age_old_flip),
        ]

        for col, val, frac in flips:
            if col not in df.columns:
                continue
            pos_in_group = np.where((df[col] == val) & (y_biased == 1))[0]
            n_flip = int(len(pos_in_group) * frac)
            if n_flip > 0:
                flip_idx = self._rng.choice(pos_in_group, n_flip, replace=False)
                y_biased[flip_idx] = 0

        return y_biased

    def bias_summary(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Return a DataFrame summarising hire rates per group."""
        rows = []
        df = X.copy()
        df["__outcome__"] = y.values
        for attr in self.schema.protected_attrs:
            if attr not in df.columns:
                continue
            grp = df.groupby(attr)["__outcome__"].agg(["mean", "count"])
            for val, row in grp.iterrows():
                rows.append({
                    "attribute": attr,
                    "group": val,
                    "positive_rate": round(float(row["mean"]), 4),
                    "n_samples": int(row["count"]),
                })
        return pd.DataFrame(rows).sort_values(["attribute", "positive_rate"], ascending=[True, False])

    def get_feature_cols(self) -> list[str]:
        return self.schema.feature_cols

    def get_proxy_cols(self) -> list[str]:
        return self.schema.proxy_cols if self.include_proxies else []

    def get_protected_attrs(self) -> list[str]:
        return self.schema.protected_attrs
