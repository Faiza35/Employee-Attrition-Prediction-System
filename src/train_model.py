"""Train, tune, compare and save the attrition model.

Class imbalance is handled with class_weight="balanced" (no extra libraries needed).
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, clean_data, get_train_test, load_data
from utils import METRICS_PATH, MODEL_PATH, RANDOM_STATE, VISUALS_DIR


def get_candidate_models() -> dict:
    """Each entry: (classifier, hyperparameter grid). Grid keys use the 'clf__' prefix."""
    return {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [5, 10, None],
             "clf__min_samples_leaf": [1, 5]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]},
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }


def get_feature_importance(pipeline: Pipeline) -> pd.Series:
    """Works for tree models (feature_importances_) and Logistic Regression (|coef|)."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]  # strip 'num__' / 'cat__' prefixes
    clf = pipeline.named_steps["clf"]
    values = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    return pd.Series(values, index=names).sort_values(ascending=False)


def save_model_plots(best_name, best_model, X_test, y_test) -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Feature importance chart (required plot)
    imp = get_feature_importance(best_model).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp.index, imp.values, color=sns.color_palette("viridis", len(imp)))
    ax.set_title(f"Top Feature Importances - {best_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=["Stay", "Leave"],
                                          cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(f"ROC Curve - {best_name}")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    print("Saved feature_importance.png, confusion_matrix.png, roc_curve.png in visuals/")


def train_and_compare():
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = get_train_test(df)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)} | "
          f"Attrition rate (train): {y_train.mean() * 100:.1f}%")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = [], {}

    for name, (clf, grid) in get_candidate_models().items():
        print(f"\nTuning {name} ...")
        # Preprocessing lives INSIDE the pipeline -> re-fitted on each CV training fold (no leakage)
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        metrics.update({"Model": name, "CV ROC-AUC": search.best_score_,
                        "Best Params": str(search.best_params_)})
        results.append(metrics)
        fitted[name] = search.best_estimator_
        print(f"  best params: {search.best_params_}")
        print(f"  CV ROC-AUC: {search.best_score_:.4f} | Test ROC-AUC: {metrics['ROC-AUC']:.4f}")

    table = pd.DataFrame(results).set_index("Model")
    table = table[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "CV ROC-AUC", "Best Params"]]

    # Select on cross-validation score (NOT on the test set) to keep the test set unbiased
    best_name = table["CV ROC-AUC"].idxmax()
    best_model = fitted[best_name]

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(METRICS_PATH)
    joblib.dump(best_model, MODEL_PATH)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (test set)")
    print("=" * 70)
    print(table.drop(columns="Best Params").round(4).to_string())
    print(f"\nBEST MODEL: {best_name}")
    print(f"Justification: highest cross-validated ROC-AUC ({table.loc[best_name, 'CV ROC-AUC']:.4f}); "
          f"on the unseen test set it reaches ROC-AUC {table.loc[best_name, 'ROC-AUC']:.4f}, "
          f"recall {table.loc[best_name, 'Recall']:.4f} and F1 {table.loc[best_name, 'F1-score']:.4f}. "
          "ROC-AUC is preferred because attrition data is imbalanced and accuracy alone is misleading.")
    print(f"Model saved to {MODEL_PATH}")

    save_model_plots(best_name, best_model, X_test, y_test)
    return table, best_name


if __name__ == "__main__":
    train_and_compare()
