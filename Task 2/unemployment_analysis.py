"""Clean and analyze the supplied Indian unemployment datasets.

Run from any directory with: python "Task 2/unemployment_analysis.py"
Tables, figures, and a concise findings summary are saved under Task 2/outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
AREA_DATA_PATH = PROJECT_DIR / "Unemployment in India.csv"
STATE_DATA_PATH = PROJECT_DIR / "Unemployment_Rate_upto_11_2020.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RATE = "Estimated Unemployment Rate (%)"
EMPLOYED = "Estimated Employed"
AREA_REQUIRED = ["Region", "Date", RATE, EMPLOYED, "Area"]
STATE_REQUIRED = ["Region", "Date", RATE, EMPLOYED]


def load_clean_dataset(
    path: Path, required_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize column/value whitespace, dates, and numeric fields consistently."""
    if not path.is_file():
        raise FileNotFoundError(f"Expected dataset at {path}")

    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    missing_columns = sorted(set(required_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {missing_columns}")

    for column in frame.columns:
        if not pd.api.types.is_string_dtype(frame[column].dtype):
            continue
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

    raw_rows = len(frame)
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    for column in (RATE, EMPLOYED, "Estimated Labour Participation Rate (%)"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=[column for column in required_columns if column in frame]).copy()
    frame = frame.loc[frame[RATE].between(0, 100)].copy()
    frame["Month"] = frame["Date"].dt.to_period("M").dt.to_timestamp()
    stats = {
        "raw_rows": int(raw_rows),
        "duplicate_rows_removed": duplicate_rows,
        "clean_rows": int(len(frame)),
        "rows_removed_or_invalid": int(raw_rows - len(frame)),
    }
    return frame, stats


def add_weighted_rate(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate labor-force weights from employed count and unemployment rate."""
    result = frame.copy()
    denominator = 1.0 - result[RATE] / 100.0
    result["Estimated Labour Force"] = result[EMPLOYED].div(denominator.where(denominator > 0))
    return result


def weighted_monthly_rate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    weighted = add_weighted_rate(frame)
    weighted = weighted.dropna(subset=["Estimated Labour Force"])
    weighted["Estimated Unemployed"] = (
        weighted["Estimated Labour Force"] * weighted[RATE] / 100.0
    )
    grouped = weighted.groupby(group_columns, as_index=False).agg(
        estimated_unemployed=("Estimated Unemployed", "sum"),
        estimated_labour_force=("Estimated Labour Force", "sum"),
    )
    grouped["weighted_unemployment_rate_pct"] = (
        grouped["estimated_unemployed"] / grouped["estimated_labour_force"] * 100
    )
    return grouped


def main() -> None:
    area_data, area_quality = load_clean_dataset(AREA_DATA_PATH, AREA_REQUIRED)
    state_data, state_quality = load_clean_dataset(STATE_DATA_PATH, STATE_REQUIRED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    area_monthly = (
        area_data.groupby(["Month", "Area"], as_index=False)[RATE]
        .mean()
        .rename(columns={RATE: "mean_unemployment_rate_pct"})
    )
    area_weighted = weighted_monthly_rate(area_data, ["Month", "Area"])
    area_monthly = area_monthly.merge(area_weighted, on=["Month", "Area"], how="left")
    area_monthly.to_csv(OUTPUT_DIR / "monthly_rural_urban_trends.csv", index=False)

    state_monthly = (
        state_data.groupby("Month", as_index=False)[RATE]
        .agg(
            mean_unemployment_rate_pct="mean",
            median_unemployment_rate_pct="median",
            reported_state_count="count",
        )
    )
    state_weighted = weighted_monthly_rate(state_data, ["Month"])
    state_monthly = state_monthly.merge(state_weighted, on="Month", how="left")
    state_monthly.to_csv(OUTPUT_DIR / "monthly_state_trends.csv", index=False)

    state_monthly_by_region = (
        state_data.groupby(["Month", "Region"], as_index=False)[RATE]
        .mean()
        .rename(columns={RATE: "mean_unemployment_rate_pct"})
    )
    state_monthly_by_region.to_csv(OUTPUT_DIR / "monthly_state_rates.csv", index=False)

    # January-February is the 2020 pre-lockdown comparison period. March is
    # reported separately because the national lockdown began part-way through it.
    state_data["Period"] = pd.cut(
        state_data["Month"],
        bins=[
            pd.Timestamp("2019-12-01"),
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2020-04-01"),
            pd.Timestamp("2020-06-01"),
            pd.Timestamp("2020-11-01"),
        ],
        labels=[
            "Jan-Feb 2020 baseline",
            "March 2020 transition",
            "Apr-May 2020 shock",
            "Jun-Oct 2020",
        ],
        right=False,
    )
    period_summary = (
        state_data.dropna(subset=["Period"])
        .groupby("Period", observed=False)[RATE]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_unemployment_rate_pct",
                "median": "median_unemployment_rate_pct",
                "count": "state_month_records",
            }
        )
    )
    period_summary.to_csv(OUTPUT_DIR / "covid_period_comparison.csv", index=False)

    baseline = state_data.loc[state_data["Period"] == "Jan-Feb 2020 baseline", RATE]
    shock = state_data.loc[state_data["Period"] == "Apr-May 2020 shock", RATE]
    baseline_mean = float(baseline.mean()) if not baseline.empty else float("nan")
    shock_mean = float(shock.mean()) if not shock.empty else float("nan")
    difference = shock_mean - baseline_mean
    relative_change = difference / baseline_mean * 100 if baseline_mean else float("nan")

    comparison_periods = ["Jan-Feb 2020 baseline", "Apr-May 2020 shock"]
    regional_period_stats = (
        state_data[state_data["Period"].isin(comparison_periods)]
        .groupby(["Region", "Period"], observed=True)[RATE]
        .agg(["mean", "count"])
        .unstack("Period")
    )
    required_stats = [
        (stat, period)
        for stat in ("mean", "count")
        for period in comparison_periods
    ]
    missing_stats = [
        column
        for column in required_stats
        if column not in regional_period_stats.columns
    ]
    if missing_stats:
        raise ValueError("The state dataset does not contain both COVID comparison periods.")
    state_period = pd.DataFrame(index=regional_period_stats.index)
    state_period["jan_feb_mean_pct"] = regional_period_stats[("mean", comparison_periods[0])]
    state_period["jan_feb_records"] = (
        regional_period_stats[("count", comparison_periods[0])].fillna(0).astype(int)
    )
    state_period["apr_may_mean_pct"] = regional_period_stats[("mean", comparison_periods[1])]
    state_period["apr_may_records"] = (
        regional_period_stats[("count", comparison_periods[1])].fillna(0).astype(int)
    )
    state_period["change_percentage_points"] = (
        state_period["apr_may_mean_pct"] - state_period["jan_feb_mean_pct"]
    )
    matched_states = state_period.dropna(
        subset=["jan_feb_mean_pct", "apr_may_mean_pct"]
    ).copy()
    matched_states = matched_states.sort_values(
        "change_percentage_points", ascending=False
    )
    state_period = state_period.sort_values("change_percentage_points", ascending=False)
    state_period.to_csv(OUTPUT_DIR / "state_covid_period_changes.csv")

    quality = pd.DataFrame(
        [
            {"dataset": AREA_DATA_PATH.name, **area_quality},
            {"dataset": STATE_DATA_PATH.name, **state_quality},
        ]
    )
    quality.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for area, series in area_monthly.groupby("Area"):
        ax.plot(
            series["Month"],
            series["mean_unemployment_rate_pct"],
            marker="o",
            linewidth=2,
            label=area,
        )
    ax.set(
        title="Average unemployment rate by area",
        xlabel="Month",
        ylabel="Unemployment rate (%)",
    )
    ax.grid(alpha=0.25)
    ax.legend(title="Area")
    fig.savefig(OUTPUT_DIR / "rural_urban_unemployment_trend.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(
        state_monthly["Month"],
        state_monthly["mean_unemployment_rate_pct"],
        marker="o",
        label="Simple state mean",
    )
    ax.plot(
        state_monthly["Month"],
        state_monthly["weighted_unemployment_rate_pct"],
        marker="s",
        label="Estimated labor-force weighted",
    )
    ax.axvspan(
        pd.Timestamp("2020-03-25"),
        pd.Timestamp("2020-05-31"),
        color="tomato",
        alpha=0.14,
        label="Initial lockdown period",
    )
    ax.set(
        title="Monthly unemployment rate across states, 2020",
        xlabel="Month",
        ylabel="Unemployment rate (%)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "state_unemployment_2020.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    largest_changes = matched_states["change_percentage_points"].sort_values().tail(12)
    largest_changes.plot(kind="barh", ax=ax, color="#4472C4")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title="Largest state-level increases: Jan-Feb vs Apr-May 2020",
        xlabel="Change in mean unemployment rate (percentage points)",
        ylabel="State / region",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "state_covid_shock_comparison.png", dpi=180)
    plt.close(fig)

    peak_row = state_monthly.loc[state_monthly["mean_unemployment_rate_pct"].idxmax()]
    state_end_month = state_data["Month"].max()
    top_state = matched_states.index[0] if not matched_states.empty else "Not available"
    top_state_change = (
        matched_states.iloc[0]["change_percentage_points"]
        if not matched_states.empty
        else float("nan")
    )
    matched_mean_change = float(matched_states["change_percentage_points"].mean())
    matched_median_change = float(matched_states["change_percentage_points"].median())
    area_peaks = area_monthly.loc[
        area_monthly.groupby("Area")["mean_unemployment_rate_pct"].idxmax()
    ]
    area_peak_text = "; ".join(
        f"{row['Area'].lower()} peak "
        f"{row['mean_unemployment_rate_pct']:.2f}% in {row['Month']:%B %Y}"
        for _, row in area_peaks.sort_values("Area").iterrows()
    )
    summary = [
        "UNEMPLOYMENT ANALYSIS - KEY FINDINGS",
        (
            f"Cleaned area-level records: {len(area_data):,}; "
            f"cleaned state-month records: {len(state_data):,}."
        ),
        (
            f"The area-level file spans {area_data['Month'].min():%B %Y} to "
            f"{area_data['Month'].max():%B %Y}; the state file ends in "
            f"{state_end_month:%B %Y} despite its filename referring to November 2020."
        ),
        f"Jan-Feb 2020 state mean: {baseline_mean:.2f}%.",
        f"Apr-May 2020 state mean: {shock_mean:.2f}%.",
        (
            f"Early-pandemic change: {difference:+.2f} percentage points "
            f"({relative_change:+.1f}% relative to Jan-Feb)."
        ),
        (
            f"Among {len(matched_states)} states with records in both comparison periods, "
            f"the mean state-level change was {matched_mean_change:+.2f} percentage "
            f"points and the median was {matched_median_change:+.2f} points."
        ),
        (
            f"Highest simple-mean state rate occurred in {peak_row['Month']:%B %Y}: "
            f"{peak_row['mean_unemployment_rate_pct']:.2f}%."
        ),
        (
            f"Largest state-level Jan-Feb to Apr-May increase: {top_state} "
            f"({top_state_change:+.2f} percentage points); check the state change "
            "table for each state's observation count."
        ),
        f"Rural/urban series: {area_peak_text}.",
        (
            "The area-level file supports rural/urban monthly comparisons; the second "
            "file supports state-level 2020 comparisons."
        ),
        (
            "Unweighted state means give each reported state record equal influence. "
            "The weighted series estimates labor-force weights from the supplied "
            "employed counts and rate estimates."
        ),
        (
            "State coverage varies by month, so period means may have different "
            "state composition. The matched-state comparison reduces that issue; "
            "interpret extreme state changes alongside their record counts."
        ),
        (
            "The short observation window cannot establish recurring annual "
            "seasonality. The before/after comparison is descriptive and does not "
            "by itself prove that COVID caused the observed changes."
        ),
    ]
    (OUTPUT_DIR / "key_findings.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Area-level records: {len(area_data):,}; state-month records: {len(state_data):,}")
    print(
        f"Jan-Feb mean: {baseline_mean:.2f}% | Apr-May mean: {shock_mean:.2f}% "
        f"| Change: {difference:+.2f} pp"
    )
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
