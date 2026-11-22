"""Classify Iris species from the measurements supplied with this project.

Run from any directory with: python "Task 1/iris_flower_classification.py"
The script writes its evaluation tables, figure, and fitted model to Task 1/outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Iris.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_STATE = 42
FEATURE_COLUMNS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET_COLUMN = "Species"


def build_candidates() -> dict[str, Pipeline | RandomForestClassifier]:
    """Return comparable classifiers; scaling is fitted only on training folds."""
    return {
        "Logistic regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
            ]
        ),
        "K-nearest neighbors": Pipeline(
            [("scale", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if data.empty:
        raise ValueError("No complete Iris records remain after removing missing values.")

    features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_rows = features.notna().all(axis=1)
    features = features.loc[valid_rows]
    target = data.loc[valid_rows, TARGET_COLUMN].astype(str)
    if target.nunique() < 2:
        raise ValueError("At least two species classes are required for classification.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    comparison_rows: list[dict[str, float | str]] = []
    candidates = build_candidates()
    for name, estimator in candidates.items():
        scores = cross_val_score(estimator, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
        comparison_rows.append(
            {
                "model": name,
                "cv_macro_f1_mean": float(scores.mean()),
                "cv_macro_f1_std": float(scores.std(ddof=1)),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "cv_macro_f1_mean", ascending=False, ignore_index=True
    )
    best_name = str(comparison.loc[0, "model"])
    best_model = candidates[best_name]
    best_model.fit(x_train, y_train)
    predictions = best_model.predict(x_test)
    labels = sorted(target.unique())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / "classification_report.csv")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=labels, cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(f"Iris test-set confusion matrix\nSelected model: {best_name}")
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    metrics = {
        "dataset": DATA_PATH.name,
        "rows_used": int(len(features)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "random_state": RANDOM_STATE,
        "selection_metric": "mean 5-fold training macro F1",
        "selected_model": best_name,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classes": labels,
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")
    joblib.dump(best_model, OUTPUT_DIR / "iris_classifier.joblib")

    print(f"Selected model: {best_name}")
    print(f"Holdout accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Holdout macro F1: {metrics['test_macro_f1']:.3f}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
