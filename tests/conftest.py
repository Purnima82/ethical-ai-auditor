"""
conftest.py — shared pytest fixtures
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def synthetic_hiring_dataset():
    """
    Synthetic hiring dataset with known bias.
    12 features, binary outcome (hired=1), 3 protected attributes.
    """
    np.random.seed(42)
    n = 800

    X_raw, y = make_classification(
        n_samples=n, n_features=8, n_informative=5,
        n_redundant=2, random_state=42
    )
    df = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(8)])

    # Protected attributes
    df["gender"]    = np.random.choice(["male", "female", "non_binary"], n, p=[0.5, 0.45, 0.05])
    df["race"]      = np.random.choice(["white", "black", "hispanic", "asian", "other"], n, p=[0.5, 0.15, 0.15, 0.15, 0.05])
    df["age_group"] = np.random.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n)

    # Proxy features (correlated with protected attrs)
    df["zip_code"]     = df["race"].map({"white": 1, "black": 2, "hispanic": 3, "asian": 4, "other": 5})
    df["prior_salary"] = df["gender"].map({"male": 1.0, "female": 0.82, "non_binary": 0.88}) * np.random.normal(60000, 10000, n)

    # Inject gender bias
    y_biased = y.copy()
    female_pos = np.where((df["gender"] == "female") & (y_biased == 1))[0]
    flip_n = int(len(female_pos) * 0.35)
    y_biased[female_pos[:flip_n]] = 0

    return df, pd.Series(y_biased)


@pytest.fixture(scope="session")
def trained_rf(synthetic_hiring_dataset):
    """Pre-trained RandomForest on feature columns only."""
    df, y = synthetic_hiring_dataset
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feat_cols]
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6)
    model.fit(X, y)
    return model, df, y, feat_cols
