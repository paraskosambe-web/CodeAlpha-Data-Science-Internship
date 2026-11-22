"""Analyze advertising spend and predict sales from the supplied dataset.

Run from any directory with: python "Task 4/sales_prediction.py"
Evaluation artifacts and the selected model are written to Task 4/outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Advertising.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
RANDOM_STATE = 42


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    missing_columns = sorted(set(FEATURES + [TARGET]).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")
    # The source's Unnamed: 0 column is only a row index and is not a predictor.
    data = frame[FEATURES + [TARGET]].copy()
    for column in FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    source_rows = len(data)
    data = data.dropna().copy()
    return data[FEATURES], data[TARGET], source_rows - len(data)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    features, target, rows_removed = clean_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE
    )

    candidates = {
        "Mean baseline": Pipeline([("regressor", DummyRegressor(strategy="mean"))]),
        "Linear regression": Pipeline([("regressor", LinearRegression())]),
        "Random forest": Pipeline(
            [
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=400, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1
                    ),
                )
            ]
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison_rows: list[dict[str, float | str]] = []
    for name, estimator in candidates.items():
        scores = cross_val_score(
            estimator,
            x_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        rmse_scores = -scores
        comparison_rows.append(
            {
                "model": name,
                "cv_rmse_mean": float(rmse_scores.mean()),
                "cv_rmse_std": float(rmse_scores.std(ddof=1)),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "cv_rmse_mean", ignore_index=True
    )
    best_name = str(comparison.loc[0, "model"])
    best_model = candidates[best_name].fit(x_train, y_train)
    predictions = best_model.predict(x_test)

    # Coefficients from a separate, interpretable linear model quantify the
    # fitted association with each channel while holding the other channels fixed.
    insight_model = LinearRegression().fit(x_train, y_train)
    coefficients = pd.DataFrame(
        {"channel": FEATURES, "sales_change_per_spend_unit": insight_model.coef_}
    ).sort_values("sales_change_per_spend_unit", ascending=False, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "linear_model_channel_coefficients.csv", index=False)
    pd.DataFrame({"actual_sales": y_test.to_numpy(), "predicted_sales": predictions}).to_csv(
        OUTPUT_DIR / "test_set_predictions.csv", index=False
    )

    metrics = {
        "dataset": DATA_PATH.name,
        "rows_used": int(len(features)),
        "rows_removed_for_missing_or_invalid_values": int(rows_removed),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": FEATURES,
        "target": TARGET,
        "selection_metric": "mean 5-fold training RMSE",
        "selected_model": best_name,
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "test_r2": float(r2_score(y_test, predictions)),
        "random_state": RANDOM_STATE,
        "interpretation_note": (
            "Coefficients describe associations in this observational dataset, "
            "not causal effects."
        ),
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(y_test, predictions, alpha=0.75, edgecolor="white", linewidth=0.4)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    ax.set(
        title=f"Advertising sales prediction: {best_name}",
        xlabel="Actual sales",
        ylabel="Predicted sales",
    )
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    coefficients.sort_values("sales_change_per_spend_unit").plot(
        x="channel",
        y="sales_change_per_spend_unit",
        kind="barh",
        ax=ax,
        legend=False,
        color="#4472C4",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title="Linear-model association by advertising channel",
        xlabel="Change in predicted sales per spend unit",
        ylabel="Channel",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "channel_coefficients.png", dpi=180)
    plt.close(fig)

    residuals = y_test.to_numpy() - predictions
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(predictions, residuals, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set(
        title="Prediction residuals",
        xlabel="Predicted sales",
        ylabel="Actual minus predicted",
    )
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIR / "residuals.png", dpi=180)
    plt.close(fig)

    best_channel = str(coefficients.iloc[0]["channel"])
    best_coefficient = float(coefficients.iloc[0]["sales_change_per_spend_unit"])
    weakest_channel = str(coefficients.iloc[-1]["channel"])
    weakest_coefficient = float(coefficients.iloc[-1]["sales_change_per_spend_unit"])
    findings = [
        "ADVERTISING SALES - KEY FINDINGS",
        (
            f"Selected model: {best_name}; holdout MAE {metrics['test_mae']:.3f}, "
            f"RMSE {metrics['test_rmse']:.3f}, R² {metrics['test_r2']:.3f}."
        ),
        (
            f"In a multiple linear regression fitted on the training split, "
            f"{best_channel} had the largest positive coefficient "
            f"({best_coefficient:.4f} sales units per spend unit)."
        ),
        (
            f"{weakest_channel} had the smallest coefficient "
            f"({weakest_coefficient:.4f}) after accounting for the other channels."
        ),
        (
            "Use channel coefficients as directional associations in this dataset, "
            "not as guaranteed returns or causal effects. Validate campaigns with "
            "controlled experiments and current spend data before reallocating budget."
        ),
        (
            "The supplied data has no time column, so this is an ad-spend-to-sales "
            "regression rather than a chronological forecast."
        ),
    ]
    (OUTPUT_DIR / "key_findings.txt").write_text("\n".join(findings) + "\n", encoding="utf-8")

    print(f"Selected model: {best_name}")
    print(
        f"Test MAE: {metrics['test_mae']:.3f} | RMSE: {metrics['test_rmse']:.3f} "
        f"| R2: {metrics['test_r2']:.3f}"
    )
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
