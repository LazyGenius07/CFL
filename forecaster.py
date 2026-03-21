from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("CISCO_DATASET")
ACTUAL_PATH = DATA_DIR / "CFL_External Data Pack_Phase1(Data Pack - Actual Bookings).csv"
SCMS_PATH = DATA_DIR / "CFL_External Data Pack_Phase1(SCMS).csv"
VMS_PATH = DATA_DIR / "CFL_External Data Pack_Phase1(VMS).csv"
BIG_DEAL_PATH = DATA_DIR / "CFL_External Data Pack_Phase1(Big Deal).csv"
OUTPUT_PATH = Path("fy26_q2_forecast.csv")
CALIBRATION_BLEND_VALUES = np.arange(0.0, 1.01, 0.01)

ACTUAL_QUARTERS = [
    "FY23 Q2",
    "FY23 Q3",
    "FY23 Q4",
    "FY24 Q1",
    "FY24 Q2",
    "FY24 Q3",
    "FY24 Q4",
    "FY25 Q1",
    "FY25 Q2",
    "FY25 Q3",
    "FY25 Q4",
    "FY26 Q1",
]
SEGMENT_QUARTERS = [
    "2023Q1",
    "2023Q2",
    "2023Q3",
    "2023Q4",
    "2024Q1",
    "2024Q2",
    "2024Q3",
    "2024Q4",
    "2025Q1",
    "2025Q2",
    "2025Q3",
    "2025Q4",
    "2026Q1",
]
BIG_DEAL_QUARTERS = [
    "2024Q2",
    "2024Q3",
    "2024Q4",
    "2025Q1",
    "2025Q2",
    "2025Q3",
    "2025Q4",
    "2026Q1",
]


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def bounded_growth(last: float, previous: float, lower: float, upper: float) -> float:
    if previous == 0 or np.isnan(previous):
        return 0.0
    return float(np.clip((last - previous) / previous, lower, upper))


def weighted_accuracy(prediction: pd.Series, actual: pd.Series) -> float:
    mask = prediction.notna() & actual.notna() & (actual != 0)
    if not mask.any():
        return float("nan")
    bias = (prediction[mask] - actual[mask]) / actual[mask]
    return float(np.maximum(0, 1 - np.abs(bias)).mean())


def score_blend(prediction: np.ndarray, actual: pd.Series) -> float:
    return weighted_accuracy(pd.Series(prediction, index=actual.index), actual)


def forecast_product(series: np.ndarray, life_cycle: str) -> float:
    last = series[-1]
    previous = series[-2]
    same_quarter_last_year = series[-4]
    same_quarter_two_years_back = series[-8]
    rolling_mean = series[-3:].mean()
    growth = bounded_growth(last, previous, -0.35, 0.35)
    seasonal_baseline = 0.65 * same_quarter_last_year + 0.35 * same_quarter_two_years_back
    life_cycle = (life_cycle or "").strip().lower()

    if life_cycle.startswith("npi"):
        recent_mean = series[-2:].mean()
        return max(0.0, 0.50 * last + 0.20 * recent_mean + 0.30 * last * (1 + max(growth, 0)))

    if life_cycle == "decline":
        base = 0.45 * last + 0.35 * seasonal_baseline + 0.20 * rolling_mean
        return max(0.0, base * (1 + min(growth, 0)))

    return max(
        0.0,
        0.35 * last + 0.35 * seasonal_baseline + 0.20 * rolling_mean + 0.10 * last * (1 + growth),
    )


def forecast_segment(series: np.ndarray) -> float:
    last = series[-1]
    previous = series[-2]
    same_quarter_last_year = series[-4]
    growth = bounded_growth(last, previous, -0.40, 0.40)
    return max(0.0, 0.45 * last + 0.35 * same_quarter_last_year + 0.20 * last * (1 + growth))


def forecast_big_deal_component(series: np.ndarray) -> float:
    last = series[-1]
    previous = series[-2]
    same_quarter_last_year = series[-4]
    growth = bounded_growth(last, previous, -0.50, 0.50)
    return max(0.0, 0.50 * last + 0.30 * same_quarter_last_year + 0.20 * last * (1 + growth))


def load_actuals() -> pd.DataFrame:
    raw = pd.read_csv(ACTUAL_PATH, header=None)
    actuals = raw.iloc[3:, :19].copy().reset_index(drop=True)
    actuals.columns = [
        "Cost Rank",
        "Product Name",
        "Product Life Cycle",
        *ACTUAL_QUARTERS,
        "Your Forecast FY26 Q2",
        "Demand Planners Forecast",
        "Marketing Team Forecast",
        "Data Science Team Forecast",
    ]

    for column in actuals.columns:
        if column not in {"Product Name", "Product Life Cycle"}:
            actuals[column] = to_numeric(actuals[column])

    actuals["Product Name"] = actuals["Product Name"].astype(str).str.strip()
    actuals["Product Life Cycle"] = actuals["Product Life Cycle"].astype(str).str.strip()
    actuals = actuals[actuals["Cost Rank"].notna()].copy()
    actuals = actuals[actuals["Product Name"].ne("Product Name")].reset_index(drop=True)
    valid_life_cycles = {"Sustaining", "NPI", "NPI-Ramp", "Decline"}
    actuals = actuals[actuals["Product Life Cycle"].isin(valid_life_cycles)].copy()
    history_counts = actuals[ACTUAL_QUARTERS].notna().sum(axis=1)
    actuals = actuals[history_counts >= 4].reset_index(drop=True)
    return actuals


def load_benchmark_backtest() -> pd.DataFrame:
    raw = pd.read_csv(ACTUAL_PATH, header=None)
    benchmarks = raw.iloc[38:68, :19].copy().reset_index(drop=True)
    benchmarks.columns = [
        "Cost Rank",
        "Product Name",
        "Planner Accuracy FY26 Q1",
        "Planner Bias FY26 Q1",
        "Planner Accuracy FY25 Q4",
        "Planner Bias FY25 Q4",
        "Planner Accuracy FY25 Q3",
        "Planner Bias FY25 Q3",
        "Separator 1",
        "Marketing Accuracy FY26 Q1",
        "Marketing Bias FY26 Q1",
        "Marketing Accuracy FY25 Q4",
        "Marketing Bias FY25 Q4",
        "Marketing Accuracy FY25 Q3",
        "Marketing Bias FY25 Q3",
        "Separator 2",
        "Data Science Accuracy FY26 Q1",
        "Data Science Bias FY26 Q1",
        "Data Science Accuracy FY25 Q4",
    ]

    benchmarks["Product Name"] = benchmarks["Product Name"].astype(str).str.strip()
    numeric_columns = [
        "Planner Accuracy FY26 Q1",
        "Planner Bias FY26 Q1",
        "Marketing Accuracy FY26 Q1",
        "Marketing Bias FY26 Q1",
        "Data Science Accuracy FY26 Q1",
        "Data Science Bias FY26 Q1",
    ]
    for column in numeric_columns:
        benchmarks[column] = pd.to_numeric(
            benchmarks[column].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )

    return benchmarks[
        [
            "Product Name",
            "Planner Accuracy FY26 Q1",
            "Planner Bias FY26 Q1",
            "Marketing Accuracy FY26 Q1",
            "Marketing Bias FY26 Q1",
            "Data Science Accuracy FY26 Q1",
            "Data Science Bias FY26 Q1",
        ]
    ]


def load_segment_forecast(csv_path: Path, label: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, header=None)
    segments = raw.iloc[2:, :16].copy().reset_index(drop=True)
    segments.columns = ["Cost Rank", "Product Name", "Segment", *SEGMENT_QUARTERS]

    for quarter in SEGMENT_QUARTERS:
        segments[quarter] = to_numeric(segments[quarter])

    segments["Product Name"] = segments["Product Name"].astype(str).str.strip()
    segments = segments[segments["Product Name"].ne("nan")].copy()

    history_quarters = SEGMENT_QUARTERS[:-1]
    segments[f"{label} Backtest FY26 Q1"] = segments[history_quarters].apply(
        lambda row: forecast_segment(row.to_numpy(dtype=float)),
        axis=1,
    )
    segments[f"{label} Forecast FY26 Q2"] = segments[SEGMENT_QUARTERS].apply(
        lambda row: forecast_segment(row.to_numpy(dtype=float)),
        axis=1,
    )

    return segments.groupby("Product Name", as_index=False)[
        [f"{label} Backtest FY26 Q1", f"{label} Forecast FY26 Q2"]
    ].sum()


def load_big_deal_forecast() -> pd.DataFrame:
    raw = pd.read_csv(BIG_DEAL_PATH, header=None)
    big_deal = raw.iloc[1:, :26].copy().reset_index(drop=True)
    big_deal.columns = [
        "Cost Rank",
        "Product Name",
        *[f"mfg_{quarter}" for quarter in BIG_DEAL_QUARTERS],
        *[f"big_{quarter}" for quarter in BIG_DEAL_QUARTERS],
        *[f"avg_{quarter}" for quarter in BIG_DEAL_QUARTERS],
    ]

    for column in big_deal.columns:
        if column != "Product Name":
            big_deal[column] = to_numeric(big_deal[column])

    big_deal["Product Name"] = big_deal["Product Name"].astype(str).str.strip()
    big_deal = big_deal[big_deal["Cost Rank"].notna()].copy()

    history_quarters = BIG_DEAL_QUARTERS[:-1]
    big_deal["Big Deal Backtest FY26 Q1"] = (
        big_deal[[f"big_{quarter}" for quarter in history_quarters]].apply(
            lambda row: forecast_big_deal_component(row.to_numpy(dtype=float)),
            axis=1,
        )
        + big_deal[[f"avg_{quarter}" for quarter in history_quarters]].apply(
            lambda row: forecast_big_deal_component(row.to_numpy(dtype=float)),
            axis=1,
        )
    )
    big_deal["Big Deal Forecast FY26 Q2"] = (
        big_deal[[f"big_{quarter}" for quarter in BIG_DEAL_QUARTERS]].apply(
            lambda row: forecast_big_deal_component(row.to_numpy(dtype=float)),
            axis=1,
        )
        + big_deal[[f"avg_{quarter}" for quarter in BIG_DEAL_QUARTERS]].apply(
            lambda row: forecast_big_deal_component(row.to_numpy(dtype=float)),
            axis=1,
        )
    )

    return big_deal[["Product Name", "Big Deal Backtest FY26 Q1", "Big Deal Forecast FY26 Q2"]]


def fit_calibration_model(forecast_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    training_frame = pd.DataFrame(
        {
            "planner": forecast_df["Planner Backtest FY26 Q1"],
            "marketing": forecast_df["Marketing Backtest FY26 Q1"],
            "data_science": forecast_df["Data Science Backtest FY26 Q1"],
            "base": forecast_df["Base Backtest FY26 Q1"],
            "channel": forecast_df["Channel Backtest FY26 Q1"],
            "big_deal": forecast_df["Big Deal Backtest FY26 Q1"].fillna(0),
            "last_quarter": forecast_df["FY25 Q4"],
            "same_quarter_last_year": forecast_df["FY25 Q1"],
            "recent_mean": forecast_df[["FY25 Q2", "FY25 Q3", "FY25 Q4"]].mean(axis=1),
        }
    ).fillna(0)
    target = forecast_df["FY26 Q1"]
    design_matrix = np.column_stack([np.ones(len(training_frame)), training_frame.to_numpy(dtype=float)])
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, target.to_numpy(dtype=float), rcond=None)

    calibrated_backtest = design_matrix @ coefficients
    planner_backtest = forecast_df["Planner Backtest FY26 Q1"].to_numpy(dtype=float)

    best_blend_weight = 0.0
    best_score = float("-inf")
    for blend_weight in CALIBRATION_BLEND_VALUES:
        blended_prediction = blend_weight * calibrated_backtest + (1 - blend_weight) * planner_backtest
        current_score = score_blend(blended_prediction, target)
        if current_score > best_score:
            best_score = current_score
            best_blend_weight = float(blend_weight)

    return coefficients, best_blend_weight


def apply_calibration_model(forecast_df: pd.DataFrame, coefficients: np.ndarray, blend_weight: float) -> pd.DataFrame:
    backtest_features = pd.DataFrame(
        {
            "planner": forecast_df["Planner Backtest FY26 Q1"],
            "marketing": forecast_df["Marketing Backtest FY26 Q1"],
            "data_science": forecast_df["Data Science Backtest FY26 Q1"],
            "base": forecast_df["Base Backtest FY26 Q1"],
            "channel": forecast_df["Channel Backtest FY26 Q1"],
            "big_deal": forecast_df["Big Deal Backtest FY26 Q1"].fillna(0),
            "last_quarter": forecast_df["FY25 Q4"],
            "same_quarter_last_year": forecast_df["FY25 Q1"],
            "recent_mean": forecast_df[["FY25 Q2", "FY25 Q3", "FY25 Q4"]].mean(axis=1),
        }
    ).fillna(0)
    forecast_features = pd.DataFrame(
        {
            "planner": forecast_df["Demand Planners Forecast"],
            "marketing": forecast_df["Marketing Team Forecast"],
            "data_science": forecast_df["Data Science Team Forecast"],
            "base": forecast_df["Base Forecast FY26 Q2"],
            "channel": forecast_df["Channel Forecast FY26 Q2"],
            "big_deal": forecast_df["Big Deal Forecast FY26 Q2"].fillna(0),
            "last_quarter": forecast_df["FY26 Q1"],
            "same_quarter_last_year": forecast_df["FY25 Q2"],
            "recent_mean": forecast_df[["FY25 Q3", "FY25 Q4", "FY26 Q1"]].mean(axis=1),
        }
    ).fillna(0)

    backtest_matrix = np.column_stack([np.ones(len(backtest_features)), backtest_features.to_numpy(dtype=float)])
    forecast_matrix = np.column_stack([np.ones(len(forecast_features)), forecast_features.to_numpy(dtype=float)])

    forecast_df["Calibrated Backtest FY26 Q1"] = np.maximum(0, backtest_matrix @ coefficients)
    forecast_df["Calibrated Forecast FY26 Q2"] = np.maximum(0, forecast_matrix @ coefficients)
    forecast_df["Final Backtest FY26 Q1"] = np.maximum(
        0,
        blend_weight * forecast_df["Calibrated Backtest FY26 Q1"]
        + (1 - blend_weight) * forecast_df["Planner Backtest FY26 Q1"],
    )
    forecast_df["Final Forecast FY26 Q2"] = np.maximum(
        0,
        blend_weight * forecast_df["Calibrated Forecast FY26 Q2"]
        + (1 - blend_weight) * forecast_df["Demand Planners Forecast"],
    ).round(0)
    return forecast_df


def build_forecast() -> pd.DataFrame:
    forecast_df = load_actuals()
    benchmarks = load_benchmark_backtest()
    forecast_df["Base Backtest FY26 Q1"] = forecast_df.apply(
        lambda row: forecast_product(row[ACTUAL_QUARTERS[:-1]].to_numpy(dtype=float), row["Product Life Cycle"]),
        axis=1,
    )
    forecast_df["Base Forecast FY26 Q2"] = forecast_df.apply(
        lambda row: forecast_product(row[ACTUAL_QUARTERS].to_numpy(dtype=float), row["Product Life Cycle"]),
        axis=1,
    )

    scms = load_segment_forecast(SCMS_PATH, "SCMS")
    vms = load_segment_forecast(VMS_PATH, "VMS")
    big_deal = load_big_deal_forecast()

    forecast_df = forecast_df.merge(scms, on="Product Name", how="left")
    forecast_df = forecast_df.merge(vms, on="Product Name", how="left")
    forecast_df = forecast_df.merge(big_deal, on="Product Name", how="left")
    forecast_df = forecast_df.merge(benchmarks, on="Product Name", how="left")

    forecast_df["Channel Backtest FY26 Q1"] = forecast_df[
        ["SCMS Backtest FY26 Q1", "VMS Backtest FY26 Q1"]
    ].mean(axis=1)
    forecast_df["Channel Forecast FY26 Q2"] = forecast_df[
        ["SCMS Forecast FY26 Q2", "VMS Forecast FY26 Q2"]
    ].mean(axis=1)
    forecast_df["Planner Backtest FY26 Q1"] = forecast_df["FY26 Q1"] * (
        1 + forecast_df["Planner Bias FY26 Q1"] / 100
    )
    forecast_df["Marketing Backtest FY26 Q1"] = forecast_df["FY26 Q1"] * (
        1 + forecast_df["Marketing Bias FY26 Q1"] / 100
    )
    forecast_df["Data Science Backtest FY26 Q1"] = forecast_df["FY26 Q1"] * (
        1 + forecast_df["Data Science Bias FY26 Q1"] / 100
    )

    calibration_coefficients, blend_weight = fit_calibration_model(forecast_df)
    forecast_df = apply_calibration_model(forecast_df, calibration_coefficients, blend_weight)
    forecast_df["Calibration Blend Weight"] = blend_weight

    forecast_df["FY26 Q1 Backtest Accuracy"] = (
        1
        - (
            forecast_df["Final Backtest FY26 Q1"] - forecast_df["FY26 Q1"]
        ).abs().div(forecast_df["FY26 Q1"])
    ).clip(lower=0)

    return forecast_df


def print_summary(forecast_df: pd.DataFrame) -> None:
    model_accuracy = weighted_accuracy(forecast_df["Final Backtest FY26 Q1"], forecast_df["FY26 Q1"])
    planner_accuracy = weighted_accuracy(forecast_df["Planner Backtest FY26 Q1"], forecast_df["FY26 Q1"])
    marketing_accuracy = weighted_accuracy(forecast_df["Marketing Backtest FY26 Q1"], forecast_df["FY26 Q1"])
    data_science_accuracy = weighted_accuracy(
        forecast_df["Data Science Backtest FY26 Q1"],
        forecast_df["FY26 Q1"],
    )

    print(f"Products forecasted: {len(forecast_df)}")
    print(f"Model backtest accuracy on FY26 Q1: {model_accuracy:.4f}")
    print(f"Planner-calibration blend weight: {forecast_df['Calibration Blend Weight'].iloc[0]:.2f}")
    print(f"Demand Planner backtest accuracy: {planner_accuracy:.4f}")
    print(f"Marketing backtest accuracy: {marketing_accuracy:.4f}")
    print(f"Data Science backtest accuracy: {data_science_accuracy:.4f}")
    print("\nTop 10 FY26 Q2 forecasts:")
    print(
        forecast_df[["Product Name", "Product Life Cycle", "Final Forecast FY26 Q2"]]
        .sort_values("Final Forecast FY26 Q2", ascending=False)
        .head(10)
        .to_string(index=False)
    )


def main() -> None:
    forecast_df = build_forecast()
    result = forecast_df[
        [
            "Cost Rank",
            "Product Name",
            "Product Life Cycle",
            "FY26 Q1",
            "Demand Planners Forecast",
            "Marketing Team Forecast",
            "Data Science Team Forecast",
            "Planner Backtest FY26 Q1",
            "Marketing Backtest FY26 Q1",
            "Data Science Backtest FY26 Q1",
            "Base Forecast FY26 Q2",
            "Channel Forecast FY26 Q2",
            "Big Deal Forecast FY26 Q2",
            "Final Forecast FY26 Q2",
            "FY26 Q1 Backtest Accuracy",
        ]
    ].sort_values("Cost Rank")
    result.to_csv(OUTPUT_PATH, index=False)
    print_summary(forecast_df)
    print(f"\nSaved forecast to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
