"""Generate a realistic synthetic HR dataset (imbalanced, with some missing values)."""
import numpy as np
import pandas as pd

from utils import DATA_PATH, DEPARTMENTS, RANDOM_STATE


def generate_dataset(n_rows: int = 1500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 60, n_rows)
    department = rng.choice(DEPARTMENTS, n_rows, p=[0.28, 0.30, 0.10, 0.17, 0.15])
    years = np.minimum(rng.gamma(2.0, 2.5, n_rows).round(1), age - 20)

    base_income = {"Sales": 45000, "Engineering": 70000, "HR": 42000, "Marketing": 50000, "Finance": 60000}
    income = np.array([base_income[d] for d in department]) + years * 2500 + rng.normal(0, 8000, n_rows)
    income = np.clip(income, 15000, None).round(-2)

    job_sat = rng.integers(1, 6, n_rows)
    work_life = rng.integers(1, 6, n_rows)
    performance = rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.03, 0.07, 0.45, 0.30, 0.15])
    overtime = rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70])
    distance = rng.integers(1, 51, n_rows)

    # Hidden attrition logic: low satisfaction, overtime, low income, long commute, short tenure -> leave
    logit = (
        -1.6
        + 1.0 * (overtime == "Yes")
        - 0.45 * (job_sat - 3)
        - 0.30 * (work_life - 3)
        - 0.00004 * (income - 55000)
        + 0.02 * (distance - 25)
        - 0.12 * years
        - 0.015 * (age - 35)
        + 0.3 * (department == "Sales")
    )
    prob = 1 / (1 + np.exp(-logit))
    attrition = np.where(rng.random(n_rows) < prob, "Yes", "No")

    df = pd.DataFrame({
        "EmployeeID": np.arange(1001, 1001 + n_rows),
        "Age": age,
        "Department": department,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "JobSatisfaction": job_sat,
        "WorkLifeBalance": work_life,
        "PerformanceRating": performance,
        "OverTime": overtime,
        "DistanceFromHome": distance,
        "Attrition": attrition,
    })

    # Inject ~2% missing values so the preprocessing step has real work to do
    for col in ["MonthlyIncome", "JobSatisfaction", "DistanceFromHome", "Department"]:
        mask = rng.random(n_rows) < 0.02
        df[col] = df[col].astype(object)
        df.loc[mask, col] = np.nan
    return df


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    data.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH}  shape={data.shape}")
    print(data["Attrition"].value_counts(normalize=True).round(3))
