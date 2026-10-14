"""Shared constants, paths and input-validation helpers."""
from pathlib import Path

# ---------- Paths (always relative to the project root, so scripts run from anywhere) ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "employee_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
METRICS_PATH = ROOT_DIR / "models" / "model_comparison.csv"
VISUALS_DIR = ROOT_DIR / "visuals"

# ---------- Dataset schema ----------
TARGET = "Attrition"
ID_COLUMN = "EmployeeID"
DEPARTMENTS = ["Sales", "Engineering", "HR", "Marketing", "Finance"]
NUMERIC_FEATURES = [
    "Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
    "WorkLifeBalance", "PerformanceRating", "DistanceFromHome",
]
CATEGORICAL_FEATURES = ["Department", "OverTime"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42


class InputValidationError(ValueError):
    """Raised when user-supplied prediction input is invalid."""


def validate_employee_input(data: dict) -> dict:
    """Validate and clean a single employee record.

    Returns a cleaned dict, or raises InputValidationError with a clear message.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in data or data[c] in (None, "")]
    if missing:
        raise InputValidationError(f"Missing required fields: {', '.join(missing)}")

    cleaned = {}

    # Numeric fields with allowed ranges
    ranges = {
        "Age": (18, 70),
        "MonthlyIncome": (1, 1_000_000),
        "YearsAtCompany": (0, 50),
        "JobSatisfaction": (1, 5),
        "WorkLifeBalance": (1, 5),
        "PerformanceRating": (1, 5),
        "DistanceFromHome": (0, 500),
    }
    for col, (low, high) in ranges.items():
        try:
            value = float(data[col])
        except (TypeError, ValueError):
            raise InputValidationError(f"{col} must be a number (got '{data[col]}').")
        if value != value:  # NaN check
            raise InputValidationError(f"{col} must be a valid number.")
        if not low <= value <= high:
            raise InputValidationError(f"{col} must be between {low} and {high} (got {value:g}).")
        cleaned[col] = value

    # Tenure cannot exceed working life
    if cleaned["YearsAtCompany"] > cleaned["Age"] - 15:
        raise InputValidationError("YearsAtCompany is not possible for the given Age.")

    # Department (case-insensitive)
    dept_lookup = {d.lower(): d for d in DEPARTMENTS}
    dept = str(data["Department"]).strip().lower()
    if dept not in dept_lookup:
        raise InputValidationError(
            f"Invalid Department '{data['Department']}'. Choose from: {', '.join(DEPARTMENTS)}."
        )
    cleaned["Department"] = dept_lookup[dept]

    # OverTime
    overtime = str(data["OverTime"]).strip().lower()
    if overtime not in ("yes", "no", "y", "n"):
        raise InputValidationError("OverTime must be Yes or No.")
    cleaned["OverTime"] = "Yes" if overtime in ("yes", "y") else "No"

    return cleaned
