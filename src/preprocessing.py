"""Data loading, cleaning and the preprocessing pipeline.

Key design decision (avoids data leakage): imputation, scaling and encoding are
wrapped in a scikit-learn ColumnTransformer that is fitted ONLY on the training
data inside a Pipeline. The test set is only ever transformed, never fitted.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (CATEGORICAL_FEATURES, DATA_PATH, DEPARTMENTS, FEATURE_COLUMNS,
                   ID_COLUMN, NUMERIC_FEATURES, RANDOM_STATE, TARGET)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the CSV with clear errors if the file is missing or unreadable."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python src/generate_data.py` first."
        )
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file {path} is empty.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent data: duplicates, wrong types, impossible values, bad labels."""
    df = df.copy()
    df = df.drop_duplicates()
    if ID_COLUMN in df.columns:
        df = df.drop_duplicates(subset=ID_COLUMN)

    # Standardise text columns (plain object dtype with NaN keeps sklearn happy)
    def tidy_text(series: pd.Series) -> pd.Series:
        return series.astype(object).where(series.notna(), np.nan).map(
            lambda v: v.strip().title() if isinstance(v, str) else np.nan)

    df["Department"] = tidy_text(df["Department"]).replace({"Hr": "HR"})
    df.loc[~df["Department"].isin(DEPARTMENTS), "Department"] = np.nan  # invalid -> missing
    df["OverTime"] = tidy_text(df["OverTime"])
    df.loc[~df["OverTime"].isin(["Yes", "No"]), "OverTime"] = np.nan
    df[TARGET] = tidy_text(df[TARGET])

    # Numeric columns: coerce bad text to NaN, then blank out impossible values
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[(df["Age"] < 18) | (df["Age"] > 70), "Age"] = np.nan
    df.loc[df["MonthlyIncome"] <= 0, "MonthlyIncome"] = np.nan
    df.loc[df["YearsAtCompany"] < 0, "YearsAtCompany"] = np.nan
    df.loc[df["DistanceFromHome"] < 0, "DistanceFromHome"] = np.nan
    for col in ["JobSatisfaction", "WorkLifeBalance", "PerformanceRating"]:
        df.loc[~df[col].between(1, 5), col] = np.nan

    # Rows without a valid target cannot be used for supervised learning
    return df[df[TARGET].isin(["Yes", "No"])].reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Impute -> scale numeric; impute -> one-hot encode categorical."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # median is robust to outliers
        ("scaler", StandardScaler()),                     # needed for Logistic Regression
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def get_train_test(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified split so train and test keep the same attrition ratio."""
    X = df[FEATURE_COLUMNS]
    y = (df[TARGET] == "Yes").astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
