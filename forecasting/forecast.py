"""
Medical Supply Intelligence - Demand Forecasting & Shortage Prediction Module
=============================================================================
This module provides end-to-end machine learning forecasting for daily medicine demand,
calculates days until stockout based on current inventory, and flags shortage risks
by comparing against supplier lead times across the hospital network.

Features:
- Chronological time-series feature engineering (lag_1, lag_7, rolling_7, calendar, operational signals)
- Strict chronological train-test split (no data leakage / no random shuffling)
- High-efficiency Gradient Boosted Tree regression (HistGradientBoostingRegressor)
- 7-day multi-step recursive demand forecasting per hospital-medicine pair
- Shortage risk classification (HIGH / MEDIUM / LOW) vs supplier lead times
- Modular architecture designed for easy integration with redistribution and dashboard modules.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def load_and_preprocess_data(csv_path: str | Path = "data/medical_supply_20_medicines.csv") -> pd.DataFrame:
    """
    Load dataset, ensure datetime formatting, and sort chronologically.

    Parameters
    ----------
    csv_path : str or Path
        Path to the medical supply CSV file.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with parsed datetime.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["hospital_id", "medicine_id", "date"]).reset_index(drop=True)
    return df


def create_time_series_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int], Dict[str, int]]:
    """
    Generate time-series and calendar features without lookahead leakage.

    Features generated:
    - lag_1: Prior day's daily demand for the specific (hospital, medicine) pair
    - lag_7: Daily demand 7 days prior for the specific pair
    - rolling_7: 7-day rolling mean of past daily demand (shifted by 1 day to prevent leakage)
    - day_of_week: Day of week integer (0=Monday, 6=Sunday)
    - month: Month integer (1 to 12)
    - outbreak_signal: Active epidemic / outbreak indicator
    - patient_load: Hospital patient load volume
    - emergency_demand: Emergency department demand volume
    - hospital_code: Categorical code for hospital_id
    - medicine_code: Categorical code for medicine_id

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], Dict[str, int], Dict[str, int]]
        Cleaned feature DataFrame, feature column names, hospital mapping, medicine mapping.
    """
    df_feat = df.copy()

    # Time series lags and rolling averages computed per (hospital_id, medicine_id) series
    group = df_feat.groupby(["hospital_id", "medicine_id"])
    df_feat["lag_1"] = group["daily_demand"].shift(1)
    df_feat["lag_7"] = group["daily_demand"].shift(7)
    df_feat["rolling_7"] = group["daily_demand"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
    )

    # Calendar features
    df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
    df_feat["month"] = df_feat["date"].dt.month

    # Categorical encoders
    hospital_cats = sorted(df_feat["hospital_id"].unique())
    medicine_cats = sorted(df_feat["medicine_id"].unique())
    hosp_to_code = {h: i for i, h in enumerate(hospital_cats)}
    med_to_code = {m: i for i, m in enumerate(medicine_cats)}

    df_feat["hospital_code"] = df_feat["hospital_id"].map(hosp_to_code)
    df_feat["medicine_code"] = df_feat["medicine_id"].map(med_to_code)

    feature_cols = [
        "lag_1",
        "lag_7",
        "rolling_7",
        "day_of_week",
        "month",
        "outbreak_signal",
        "patient_load",
        "emergency_demand",
        "hospital_code",
        "medicine_code",
    ]

    # Drop initial warm-up period where lag_7 is NaN
    df_clean = df_feat.dropna(subset=["lag_1", "lag_7", "rolling_7"]).copy()
    return df_clean, feature_cols, hosp_to_code, med_to_code


def train_and_evaluate(
    df_clean: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "daily_demand",
    test_days: int = 30,
) -> Tuple[HistGradientBoostingRegressor, Dict[str, float]]:
    """
    Perform a strict chronological train/test split, train HistGradientBoostingRegressor,
    and report evaluation metrics (MAE and RMSE).

    Parameters
    ----------
    df_clean : pd.DataFrame
        Dataset containing engineered features.
    feature_cols : List[str]
        List of input feature names.
    target_col : str
        Target column name (default 'daily_demand').
    test_days : int
        Number of trailing days reserved for hold-out test evaluation.

    Returns
    -------
    Tuple[HistGradientBoostingRegressor, Dict[str, float]]
        Trained model instance and dictionary containing MAE and RMSE metrics.
    """
    split_date = df_clean["date"].max() - pd.Timedelta(days=test_days)
    train_df = df_clean[df_clean["date"] <= split_date]
    test_df = df_clean[df_clean["date"] > split_date]

    # Identify categorical feature indices for HistGradientBoosting
    cat_indices = [
        feature_cols.index("hospital_code"),
        feature_cols.index("medicine_code"),
    ]

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL TRAIN-TEST SPLIT EVALUATION")
    print("=" * 60)
    print(f"Training Period:   {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df):,} rows)")
    print(f"Test Hold-out:     {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df):,} rows)")

    # Train evaluation model
    eval_model = HistGradientBoostingRegressor(
        categorical_features=cat_indices,
        random_state=42,
        max_iter=100,
        learning_rate=0.1,
    )
    eval_model.fit(train_df[feature_cols], train_df[target_col])

    # Evaluate on hold-out set
    y_test = test_df[target_col]
    y_pred = eval_model.predict(test_df[feature_cols])

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))

    print(f"Hold-out Test MAE:  {mae:.2f} units")
    print(f"Hold-out Test RMSE: {rmse:.2f} units")
    print("=" * 60 + "\n")

    # Fit final operational model on all available historical data
    print("Training final operational forecasting model on complete dataset...")
    final_model = HistGradientBoostingRegressor(
        categorical_features=cat_indices,
        random_state=42,
        max_iter=100,
        learning_rate=0.1,
    )
    final_model.fit(df_clean[feature_cols], df_clean[target_col])
    print("Final model trained successfully.")

    metrics = {"mae": mae, "rmse": rmse}
    return final_model, metrics


def categorize_shortage_risk(days_until_stockout: float, lead_time_days: int) -> str:
    """
    Determine shortage risk level by comparing days of stock remaining against supplier lead time.

    - HIGH:   days_until_stockout <= supplier_lead_time_days
              (Stock runs out before a new supplier order can arrive)
    - MEDIUM: days_until_stockout <= supplier_lead_time_days * 1.5
              (Tight buffer; vulnerable to transit delay or demand surge)
    - LOW:    days_until_stockout > supplier_lead_time_days * 1.5
              (Adequate safety buffer)
    """
    if days_until_stockout <= lead_time_days:
        return "HIGH"
    elif days_until_stockout <= lead_time_days * 1.5:
        return "MEDIUM"
    else:
        return "LOW"


def generate_7_day_forecast(
    model: HistGradientBoostingRegressor,
    df_raw: pd.DataFrame,
    feature_cols: List[str],
    hosp_to_code: Dict[str, int],
    med_to_code: Dict[str, int],
    forecast_days: int = 7,
) -> pd.DataFrame:
    """
    Generate recursive multi-step 7-day demand forecasts and assess shortage risk
    for each hospital-medicine combination.

    Parameters
    ----------
    model : HistGradientBoostingRegressor
        Fitted forecasting model.
    df_raw : pd.DataFrame
        Complete historical dataset.
    feature_cols : List[str]
        List of feature column names used by the model.
    hosp_to_code : Dict[str, int]
        Hospital ID to categorical integer code mapping.
    med_to_code : Dict[str, int]
        Medicine ID to categorical integer code mapping.
    forecast_days : int
        Number of forward days to forecast (default 7).

    Returns
    -------
    pd.DataFrame
        DataFrame containing forecast metrics and shortage risk classifications.
    """
    latest_date = df_raw["date"].max()
    latest_snapshot = (
        df_raw[df_raw["date"] == latest_date]
        .drop_duplicates(subset=["hospital_id", "medicine_id"])
        .sort_values(["hospital_id", "medicine_id"])
        .reset_index(drop=True)
    )

    # Cache recent historical daily demands (at least 14 days) per pair
    recent_demands_dict = {}
    for (h_id, m_id), grp in df_raw.groupby(["hospital_id", "medicine_id"]):
        recent_demands_dict[(h_id, m_id)] = grp.sort_values("date")["daily_demand"].tolist()[-14:]

    # Storage for multi-step predictions: pair_key -> list of daily predictions
    predictions_by_pair: Dict[Tuple[str, str], List[float]] = {
        (row["hospital_id"], row["medicine_id"]): [] for _, row in latest_snapshot.iterrows()
    }

    # Recursive multi-day forecast across all pairs
    for step in range(1, forecast_days + 1):
        target_date = latest_date + pd.Timedelta(days=step)
        dow = target_date.dayofweek
        mon = target_date.month

        step_feature_rows = []
        pair_keys = []

        for _, row in latest_snapshot.iterrows():
            h_id = row["hospital_id"]
            m_id = row["medicine_id"]
            hist = recent_demands_dict[(h_id, m_id)]

            lag_1 = hist[-1]
            lag_7 = hist[-7]
            rolling_7 = float(np.mean(hist[-7:]))

            step_feature_rows.append({
                "lag_1": lag_1,
                "lag_7": lag_7,
                "rolling_7": rolling_7,
                "day_of_week": dow,
                "month": mon,
                "outbreak_signal": row["outbreak_signal"],
                "patient_load": row["patient_load"],
                "emergency_demand": row["emergency_demand"],
                "hospital_code": hosp_to_code[h_id],
                "medicine_code": med_to_code[m_id],
            })
            pair_keys.append((h_id, m_id))

        step_df = pd.DataFrame(step_feature_rows)[feature_cols]
        step_preds = np.maximum(0.0, model.predict(step_df))

        # Append step prediction to rolling histories
        for (h_id, m_id), pred_val in zip(pair_keys, step_preds):
            recent_demands_dict[(h_id, m_id)].append(pred_val)
            predictions_by_pair[(h_id, m_id)].append(pred_val)

    # Compile final results table
    results = []
    for _, row in latest_snapshot.iterrows():
        h_id = row["hospital_id"]
        m_id = row["medicine_id"]
        m_name = row["medicine_name"]
        curr_stock = int(row["current_stock"])
        lead_time = int(row["supplier_lead_time_days"])

        preds = predictions_by_pair[(h_id, m_id)]
        pred_7_day = float(np.sum(preds))
        pred_daily = float(np.mean(preds))

        if pred_daily > 0:
            days_stockout = curr_stock / pred_daily
        else:
            days_stockout = 999.0 if curr_stock > 0 else 0.0

        risk = categorize_shortage_risk(days_stockout, lead_time)

        results.append({
            "hospital_id": h_id,
            "medicine_id": m_id,
            "medicine_name": m_name,
            "current_stock": curr_stock,
            "predicted_daily_demand": round(pred_daily, 2),
            "predicted_7_day_demand": round(pred_7_day, 2),
            "days_until_stockout": round(days_stockout, 2),
            "supplier_lead_time_days": lead_time,
            "shortage_risk": risk,
        })

    return pd.DataFrame(results)


def save_forecast_results(
    results_df: pd.DataFrame,
    output_path: str | Path = "data/forecast_results.csv",
) -> Path:
    """
    Save forecast and risk results to CSV.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output DataFrame.
    output_path : str or Path
        Destination CSV path.

    Returns
    -------
    Path
        Resolved destination path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"Forecast results saved successfully to: {out_path.resolve()} ({len(results_df)} rows)")
    return out_path


def run_forecasting_pipeline(
    data_path: str | Path = "data/medical_supply_20_medicines.csv",
    output_path: str | Path = "data/forecast_results.csv",
    test_days: int = 30,
    forecast_days: int = 7,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Execute complete end-to-end forecasting pipeline.

    Parameters
    ----------
    data_path : str or Path
        Path to the source CSV dataset.
    output_path : str or Path
        Path to save the forecast results CSV.
    test_days : int
        Days to hold out for evaluation.
    forecast_days : int
        Days to forecast ahead (default 7).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        Forecast results DataFrame and performance metrics.
    """
    print("1. Loading raw dataset...")
    df_raw = load_and_preprocess_data(data_path)

    print("2. Engineering time-series features (lags, rolling averages, calendar, operational signals)...")
    df_clean, feature_cols, hosp_to_code, med_to_code = create_time_series_features(df_raw)

    print("3. Training regression model with chronological validation...")
    model, metrics = train_and_evaluate(df_clean, feature_cols, test_days=test_days)

    print(f"4. Generating {forecast_days}-day demand forecast and shortage risk analysis...")
    results_df = generate_7_day_forecast(
        model=model,
        df_raw=df_raw,
        feature_cols=feature_cols,
        hosp_to_code=hosp_to_code,
        med_to_code=med_to_code,
        forecast_days=forecast_days,
    )

    print("5. Saving forecast results...")
    save_forecast_results(results_df, output_path)

    print("\n" + "=" * 60)
    print("FORECAST PIPELINE SUMMARY")
    print("=" * 60)
    print("Shortage Risk Distribution:")
    print(results_df["shortage_risk"].value_counts().to_string())
    print("\nSample High-Risk Predictions:")
    high_risks = results_df[results_df["shortage_risk"] == "HIGH"].head(5)
    print(high_risks[["hospital_id", "medicine_name", "current_stock", "predicted_daily_demand", "days_until_stockout", "supplier_lead_time_days"]].to_string(index=False))
    print("=" * 60 + "\n")

    return results_df, metrics


if __name__ == "__main__":
    run_forecasting_pipeline()
