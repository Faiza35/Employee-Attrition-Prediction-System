"""Prediction module + console interface.

Run:  python src/predict.py
"""
import joblib
import pandas as pd

from utils import (DEPARTMENTS, FEATURE_COLUMNS, MODEL_PATH, InputValidationError,
                   validate_employee_input)


def load_model(path=MODEL_PATH):
    """Load the saved pipeline with clear errors."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found at {path}. Train it first: python src/train_model.py"
        )
    except Exception as exc:  # corrupted file / version mismatch
        raise RuntimeError(f"Could not load model from {path}: {exc}")


def predict_attrition(model, employee: dict) -> dict:
    """Validate input, then return the predicted label and probability."""
    cleaned = validate_employee_input(employee)
    row = pd.DataFrame([cleaned])[FEATURE_COLUMNS]
    probability = float(model.predict_proba(row)[0, 1])
    label = "Yes" if probability >= 0.5 else "No"
    return {"prediction": label, "probability": probability}


def _ask(prompt: str, cast=str):
    """Keep asking until the value can be converted (basic type check)."""
    while True:
        raw = input(prompt).strip()
        try:
            return cast(raw) if cast is not str else raw
        except ValueError:
            print("  Invalid input, please enter a valid number.")


def run_console() -> None:
    try:
        model = load_model()
    except (FileNotFoundError, RuntimeError) as err:
        print(f"ERROR: {err}")
        return

    print("\n=== Employee Attrition Prediction ===  (Ctrl+C to quit)")
    while True:
        try:
            employee = {
                "Age": _ask("Age: ", float),
                "Department": _ask(f"Department ({'/'.join(DEPARTMENTS)}): "),
                "MonthlyIncome": _ask("Monthly Income (INR): ", float),
                "YearsAtCompany": _ask("Years at Company: ", float),
                "JobSatisfaction": _ask("Job Satisfaction (1-5): ", float),
                "WorkLifeBalance": _ask("Work-Life Balance (1-5): ", float),
                "PerformanceRating": _ask("Performance Rating (1-5): ", float),
                "OverTime": _ask("OverTime (Yes/No): "),
                "DistanceFromHome": _ask("Distance from Home (km): ", float),
            }
            result = predict_attrition(model, employee)
            print(f"\n>> Predicted Attrition: {result['prediction']}")
            print(f">> Probability of leaving: {result['probability'] * 100:.2f}%\n")
        except InputValidationError as err:
            print(f"\nInput error: {err}\nPlease try again.\n")
            continue
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if input("Predict another? (y/n): ").strip().lower() != "y":
            print("Goodbye!")
            break


if __name__ == "__main__":
    run_console()
