"""Exploratory data analysis + visualisations (saved into /visuals)."""
import matplotlib
matplotlib.use("Agg")  # no display needed; works on any machine
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing import clean_data, load_data
from utils import NUMERIC_FEATURES, TARGET, VISUALS_DIR

sns.set_theme(style="whitegrid")


def _save(fig, name):
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  saved visuals/{name}")


def attrition_rate_by(df: pd.DataFrame, column: str) -> pd.Series:
    return (df.groupby(column)[TARGET].apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False).round(2))


def run_eda(df: pd.DataFrame) -> None:
    df = df.copy()
    df["AttritionFlag"] = (df[TARGET] == "Yes").astype(int)

    print("\n=== OVERALL ATTRITION RATE ===")
    print(f"{df['AttritionFlag'].mean() * 100:.2f}% of employees left "
          f"({df['AttritionFlag'].sum()} of {len(df)}) -> imbalanced dataset")

    for col in ["Department", "OverTime", "JobSatisfaction"]:
        print(f"\n=== ATTRITION RATE (%) BY {col.upper()} ===")
        print(attrition_rate_by(df, col))

    corr = df[NUMERIC_FEATURES + ["AttritionFlag"]].corr()
    target_corr = corr["AttritionFlag"].drop("AttritionFlag").sort_values(key=abs, ascending=False)
    print("\n=== CORRELATION OF NUMERIC FEATURES WITH ATTRITION ===")
    print(target_corr.round(3))
    print("\n=== TOP FACTORS (by absolute correlation) ===")
    for feat, val in target_corr.head(3).items():
        direction = "more" if val > 0 else "less"
        print(f"  {feat}: higher values -> {direction} attrition (r={val:.3f})")

    print("\nSaving plots...")
    # 1. Bar chart: attrition rate by department
    rate = attrition_rate_by(df, "Department")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette="viridis", legend=False, ax=ax)
    ax.set(title="Attrition Rate by Department", xlabel="Department", ylabel="Attrition Rate (%)")
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
    _save(fig, "attrition_by_department.png")

    # 2. Box plot: MonthlyIncome vs Attrition
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=df, x=TARGET, y="MonthlyIncome", hue=TARGET, palette="Set2", legend=False, ax=ax)
    ax.set(title="Monthly Income vs Attrition", ylabel="Monthly Income (INR)")
    _save(fig, "income_vs_attrition_boxplot.png")

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features")
    _save(fig, "correlation_heatmap.png")

    # 4. Extra: attrition by overtime and by job satisfaction
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ot = attrition_rate_by(df, "OverTime")
    sns.barplot(x=ot.index, y=ot.values, hue=ot.index, palette="rocket", legend=False, ax=axes[0])
    axes[0].set(title="Attrition Rate by OverTime", ylabel="Attrition Rate (%)")
    js = attrition_rate_by(df, "JobSatisfaction").sort_index()
    sns.barplot(x=js.index.astype(int), y=js.values, hue=js.index.astype(int), palette="mako",
                legend=False, ax=axes[1])
    axes[1].set(title="Attrition Rate by Job Satisfaction", xlabel="JobSatisfaction", ylabel="Attrition Rate (%)")
    _save(fig, "attrition_by_overtime_and_satisfaction.png")


if __name__ == "__main__":
    run_eda(clean_data(load_data()))
