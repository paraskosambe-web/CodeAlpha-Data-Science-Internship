"""Train and evaluate models for predicting used-car selling prices.

Run from any directory with: python "Task 3/car_price_prediction.py"
Evaluation artifacts and the selected model are written to Task 3/outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "car data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET = "Selling_Price"
NUMERIC_FEATURES = ["VehicleAgeYears", "Present_Price", "Driven_kms", "Owner"]
CATEGORICAL_FEATURES = ["Fuel_Type", "Selling_type", "Transmission"]
RANDOM_STATE = 42


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    required = {
        "Year",
        TARGET,
        "Present_Price",
        "Driven_kms",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
    }
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = frame.copy()
    # Car_Name is a high-cardinality model label in this small dataset, so the
    # predictor uses general vehicle attributes instead of memorizing names.
    for column in ["Year", TARGET, "Present_Price", "Driven_kms", "Owner"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("string").str.strip()
        data[column] = data[column].replace("", pd.NA)
    data = data.dropna(subset=list(required)).copy()
    if data.empty:
        raise ValueError("No complete car records remain after cleaning.")

    reference_year = int(data["Year"].max())
    data["VehicleAgeYears"] = (reference_year - data["Year"]).clip(lower=0)
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data[TARGET]
    return features, target, reference_year


def make_pipeline(estimator) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")
    features, target, reference_year = prepare_data(pd.read_csv(DATA_PATH))
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE
    )

    candidates = {
        "Mean baseline": make_pipeline(DummyRegressor(strategy="mean")),
        "Ridge regression": make_pipeline(Ridge(alpha=1.0)),
        "Random forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "Gradient boosting": make_pipeline(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.04,
                max_depth=2,
                random_state=RANDOM_STATE,
            )
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(
        {"actual_selling_price": y_test.to_numpy(), "predicted_selling_price": predictions}
    ).to_csv(
        OUTPUT_DIR / "test_set_predictions.csv",
        index=False,
    )

    metrics = {
        "dataset": DATA_PATH.name,
        "rows_used": int(len(features)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "target": TARGET,
        "currency_units": "same units as Selling_Price in the supplied dataset",
        "vehicle_age_reference_year": reference_year,
        "selection_metric": "mean 5-fold training RMSE",
        "selected_model": best_name,
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "test_r2": float(r2_score(y_test, predictions)),
        "random_state": RANDOM_STATE,
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
        title=f"Used-car price prediction: {best_name}",
        xlabel="Actual selling price",
        ylabel="Predicted selling price",
    )
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    residuals = y_test.to_numpy() - predictions
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(predictions, residuals, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set(
        title="Prediction residuals",
        xlabel="Predicted selling price",
        ylabel="Actual minus predicted",
    )
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIR / "residuals.png", dpi=180)
    plt.close(fig)

    joblib.dump(best_model, OUTPUT_DIR / "car_price_model.joblib")
    print(f"Selected model: {best_name}")
    print(
        f"Test MAE: {metrics['test_mae']:.3f} | RMSE: {metrics['test_rmse']:.3f} "
        f"| R2: {metrics['test_r2']:.3f}"
    )
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
