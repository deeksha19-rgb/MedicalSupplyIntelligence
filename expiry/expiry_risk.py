"""
Expiry & Wastage Risk Module
Medical Supply Intelligence Platform

Part 1: Detect medicines at risk of expiring before clinical consumption,
compute projected wastage (excess stock), and classify expiry risk (HIGH, MEDIUM, LOW).
Supports seamless integration with demand forecasting outputs (data/forecast_results.csv)
when available, with graceful fallback to historical consumption rates.
"""

import os
import argparse
import logging
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Essential medicine criticality ratings (for risk weighting)
CRITICAL_MEDICINE_IDS = {
    "M07",  # Insulin
    "M19",  # Adrenaline
    "M05",  # Ceftriaxone
    "M10",  # IV Fluids
    "M16",  # Dexamethasone
    "M18",  # Furosemide
    "M20",  # Salbutamol Inhaler
}


def load_inventory_and_forecast_data(
    data_path: str = "data/medical_supply_20_medicines.csv",
    forecast_path: Optional[str] = "data/forecast_results.csv",
    evaluation_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load hospital inventory records and merge demand forecasts if available.

    Args:
        data_path: Path to the main inventory CSV dataset.
        forecast_path: Path to forecast results CSV (if available).
        evaluation_date: Specific snapshot date (e.g. '2025-12-31').
                         If None, defaults to the latest date in the dataset.
                         If 'all', preserves all historical records.

    Returns:
        pd.DataFrame containing inventory records with an optional 'predicted_demand' column.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Inventory dataset not found at: {data_path}")

    logger.info(f"Loading inventory dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Ensure required columns exist
    required_cols = {"hospital_id", "medicine_id", "daily_demand", "current_stock", "expiry_date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Inventory dataset missing required columns: {missing}")

    # Standardize types
    df["date"] = pd.to_datetime(df["date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["current_stock"] = pd.to_numeric(df["current_stock"], errors="coerce").fillna(0).astype(int)
    df["daily_demand"] = pd.to_numeric(df["daily_demand"], errors="coerce").fillna(0).astype(int)

    # Determine snapshot / evaluation date
    if evaluation_date is None or evaluation_date.lower() == "latest":
        latest_date = df["date"].max()
        logger.info(f"Filtering to latest inventory snapshot date: {latest_date.strftime('%Y-%m-%d')}")
        df = df[df["date"] == latest_date].copy()
    elif evaluation_date.lower() != "all":
        target_date = pd.to_datetime(evaluation_date)
        logger.info(f"Filtering to specified evaluation date: {target_date.strftime('%Y-%m-%d')}")
        df = df[df["date"] == target_date].copy()
        if df.empty:
            raise ValueError(f"No records found for specified evaluation date: {evaluation_date}")
    else:
        logger.info("Preserving all historical dates for evaluation.")
        df = df.copy()

    # Load and merge forecast results if available
    df["predicted_demand"] = np.nan
    if forecast_path and os.path.exists(forecast_path):
        logger.info(f"Forecast results detected at: {forecast_path}. Integrating forecasts...")
        try:
            forecast_df = pd.read_csv(forecast_path)
            # Find candidate predicted demand column
            demand_col = None
            for candidate in [
                "predicted_demand",
                "forecasted_demand",
                "predicted_daily_demand",
                "daily_demand_forecast",
                "forecast_demand",
                "forecast",
                "y_pred",
            ]:
                if candidate in forecast_df.columns:
                    demand_col = candidate
                    break

            if demand_col:
                # Merge keys: hospital_id, medicine_id, and optionally date
                merge_keys = ["hospital_id", "medicine_id"]
                has_date = "date" in forecast_df.columns and "date" in df.columns
                if has_date:
                    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
                    merge_keys.append("date")

                # Remove predicted_demand column from df before merging if it already exists
                df_base = df.drop(columns=["predicted_demand"])
                fc_subset = forecast_df[merge_keys + [demand_col]].rename(columns={demand_col: "predicted_demand"})
                df = pd.merge(
                    df_base,
                    fc_subset,
                    on=merge_keys,
                    how="left",
                )
                logger.info(f"Successfully integrated '{demand_col}' from forecast file.")
            else:
                logger.warning(
                    f"Forecast file found at {forecast_path}, but no known predicted demand column detected. "
                    f"Columns present: {forecast_df.columns.tolist()}"
                )
        except Exception as e:
            logger.warning(f"Failed to parse forecast file ({forecast_path}): {e}. Using actual daily demand.")
    else:
        logger.info(
            f"Forecast file not found at '{forecast_path}'. Proceeding with actual daily demand as demand baseline."
        )

    return df


def calculate_expiry_risk(
    df: pd.DataFrame,
    evaluation_date: Optional[Any] = None,
    quarantine_buffer_days: int = 0,
) -> pd.DataFrame:
    """
    Calculate batch shelf-life metrics, estimated usage before expiry, excess stock,
    and classify expiry risk into HIGH, MEDIUM, LOW categories.

    Formulas & Metrics:
    - days_to_expiry: Remaining shelf-life in days relative to evaluation date.
    - effective_daily_demand: Uses predicted_demand if available, otherwise daily_demand.
    - days_of_supply: current_stock / effective_daily_demand.
    - estimated_usage_before_expiry: Realistically consumable units before expiry date:
      min(current_stock, round(effective_daily_demand * max(0, days_to_expiry - quarantine_buffer_days))).
    - excess_stock: Units guaranteed to expire unused at the current hospital:
      max(0, current_stock - estimated_usage_before_expiry).
    - expiry_risk: Categorical rating (HIGH, MEDIUM, LOW) based on excess volume and shelf-life urgency.

    Args:
        df: Input DataFrame containing hospital inventory.
        evaluation_date: Date to calculate days_to_expiry against. If None, uses record's 'date'.
        quarantine_buffer_days: Minimum shelf-life buffer days before expiration where medication
                                can no longer be administered to patients (default: 0).

    Returns:
        pd.DataFrame enriched with expiry analysis columns.
    """
    res = df.copy()

    # Determine reference date for each row
    if evaluation_date is not None and str(evaluation_date).lower() != "all":
        ref_date = pd.to_datetime(evaluation_date)
        res["eval_date"] = ref_date
    else:
        res["eval_date"] = pd.to_datetime(res["date"])

    res["expiry_date_dt"] = pd.to_datetime(res["expiry_date"])
    res["days_to_expiry"] = (res["expiry_date_dt"] - res["eval_date"]).dt.days

    # Determine effective daily demand rate
    if "predicted_demand" in res.columns:
        # Use predicted demand if valid (> 0 and not null), otherwise fallback to daily_demand
        has_pred = res["predicted_demand"].notnull() & (res["predicted_demand"] > 0)
        res["effective_daily_demand"] = np.where(has_pred, res["predicted_demand"], res["daily_demand"])
    else:
        res["predicted_demand"] = np.nan
        res["effective_daily_demand"] = res["daily_demand"]

    # Protect against non-positive demands
    safe_demand = res["effective_daily_demand"].clip(lower=0.1)

    # Days of supply currently on hand
    res["days_of_supply"] = np.where(
        res["current_stock"] > 0,
        np.round(res["current_stock"] / safe_demand, 2),
        0.0,
    )

    # Usable days before expiry after optional quarantine buffer
    usable_days = (res["days_to_expiry"] - quarantine_buffer_days).clip(lower=0)

    # Maximum projected consumption the hospital can realistically achieve before expiry
    projected_consumption = (safe_demand * usable_days).round().astype(int)

    # Estimated usage before expiry cannot exceed current stock on hand
    res["estimated_usage_before_expiry"] = np.where(
        res["days_to_expiry"] <= 0,
        0,
        np.minimum(res["current_stock"], projected_consumption),
    ).astype(int)

    # Excess stock is stock that CANNOT be consumed before the expiration date
    res["excess_stock"] = np.where(
        res["days_to_expiry"] <= 0,
        res["current_stock"],
        np.maximum(0, res["current_stock"] - res["estimated_usage_before_expiry"]),
    ).astype(int)

    # Shelf-life consumption ratio (fraction of remaining shelf-life stocked)
    safe_expiry = res["days_to_expiry"].clip(lower=1)
    res["shelf_life_ratio"] = np.where(
        res["current_stock"] > 0,
        np.round(res["days_of_supply"] / safe_expiry, 3),
        0.0,
    )

    # Continuous expiry risk score (0.0 to 1.0)
    # Combines proportion of stock that will expire with shelf-life urgency
    waste_fraction = np.where(
        res["current_stock"] > 0,
        res["excess_stock"] / res["current_stock"],
        0.0,
    )
    # Urgency multiplier: decays as days_to_expiry grows (half-life at 60 days)
    urgency = np.where(
        res["days_to_expiry"] <= 0,
        1.0,
        np.exp(-res["days_to_expiry"] / 60.0),
    )
    # Base risk incorporates excess waste, urgency, and shelf life ratio
    risk_score = np.where(
        res["current_stock"] == 0,
        0.0,
        np.clip(0.6 * waste_fraction + 0.3 * urgency + 0.1 * np.clip(res["shelf_life_ratio"], 0, 1), 0.0, 1.0),
    )
    res["expiry_risk_score"] = np.round(risk_score, 3)

    # Categorize expiry risk into HIGH, MEDIUM, LOW
    def assign_category_and_reason(row: pd.Series) -> Tuple[str, str]:
        stock = row["current_stock"]
        days = row["days_to_expiry"]
        excess = row["excess_stock"]
        slr = row["shelf_life_ratio"]
        dos = row["days_of_supply"]
        score = row["expiry_risk_score"]

        # 1. Zero stock
        if stock == 0:
            return "LOW", "No current stock on hand; zero wastage risk."

        # 2. Already expired
        if days <= 0:
            return "HIGH", f"ALREADY EXPIRED ({abs(days)}d past expiry date). All {stock} units unusable."

        # 3. Direct excess stock detected (unusable before expiry at current demand rate)
        if excess > 0:
            if days <= 90 or excess >= 50 or (excess / stock >= 0.25):
                return (
                    "HIGH",
                    f"Excess stock of {excess} units ({round(excess/stock*100, 1)}% of inventory) "
                    f"projected to expire unused within {days} days.",
                )
            return (
                "MEDIUM",
                f"Moderate excess of {excess} units expiring in {days} days; potential for redistribution.",
            )

        # 4. Critical shelf-life windows (even if zero excess under steady demand)
        # In hospital inventory, stock expiring in <= 45 days is at high risk of unexpected demand dip
        if days <= 45:
            return (
                "HIGH",
                f"Imminent shelf-life expiration in {days} days (stock={stock}, supply={dos}d). "
                f"High sensitivity to consumption slowdown.",
            )
        elif days <= 60 and (slr >= 0.4 or dos >= 15):
            return (
                "HIGH",
                f"Narrow shelf-life of {days} days with substantial inventory ({dos}d supply, "
                f"{round(slr*100, 1)}% shelf-life coverage).",
            )
        elif days <= 60:
            return (
                "MEDIUM",
                f"Approaching expiry window ({days} days remaining). Usage must be closely monitored.",
            )
        elif days <= 90 and (slr >= 0.3 or dos >= 20):
            return (
                "MEDIUM",
                f"Elevated inventory ({dos}d supply) with expiration within {days} days.",
            )
        elif score >= 0.35:
            return (
                "MEDIUM",
                f"Moderate expiry risk score ({score}) with {days} days remaining shelf life.",
            )

        # 5. Low risk
        return (
            "LOW",
            f"Sufficient shelf-life ({days} days) relative to stock level ({dos}d supply). Low wastage risk.",
        )

    classified = [assign_category_and_reason(row) for _, row in res.iterrows()]
    res["expiry_risk"] = [c[0] for c in classified]
    res["risk_reason"] = [c[1] for c in classified]

    # Clean up auxiliary columns
    output_columns = [
        "hospital_id",
        "medicine_id",
        "medicine_name",
        "date",
        "current_stock",
        "daily_demand",
        "predicted_demand",
        "expiry_date",
        "days_to_expiry",
        "days_of_supply",
        "estimated_usage_before_expiry",
        "excess_stock",
        "expiry_risk",
        "expiry_risk_score",
        "risk_reason",
    ]
    # Filter to existing columns in output_columns
    final_cols = [c for c in output_columns if c in res.columns]
    return res[final_cols]


def get_expiry_summary(expiry_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate aggregate expiry risk summary metrics for reporting and dashboard cards.

    Args:
        expiry_df: Result DataFrame from calculate_expiry_risk.

    Returns:
        Dictionary containing key operational metrics.
    """
    total_items = len(expiry_df)
    items_with_stock = int((expiry_df["current_stock"] > 0).sum())
    total_stock = int(expiry_df["current_stock"].sum())
    total_excess_stock = int(expiry_df["excess_stock"].sum())

    risk_counts = expiry_df["expiry_risk"].value_counts().to_dict()
    high_count = risk_counts.get("HIGH", 0)
    med_count = risk_counts.get("MEDIUM", 0)
    low_count = risk_counts.get("LOW", 0)

    # Top items by excess stock or risk
    top_excess = (
        expiry_df[expiry_df["excess_stock"] > 0]
        .sort_values(by="excess_stock", ascending=False)
        .head(5)[["hospital_id", "medicine_name", "current_stock", "excess_stock", "days_to_expiry", "expiry_risk"]]
        .to_dict(orient="records")
    )

    top_urgent_expiries = (
        expiry_df[expiry_df["current_stock"] > 0]
        .sort_values(by="days_to_expiry", ascending=True)
        .head(5)[["hospital_id", "medicine_name", "current_stock", "days_to_expiry", "expiry_risk"]]
        .to_dict(orient="records")
    )

    return {
        "total_items_evaluated": total_items,
        "items_with_stock": items_with_stock,
        "total_stock_units": total_stock,
        "total_excess_units_at_risk": total_excess_stock,
        "high_risk_count": high_count,
        "medium_risk_count": med_count,
        "low_risk_count": low_count,
        "top_excess_items": top_excess,
        "top_urgent_expiries": top_urgent_expiries,
    }


def run_expiry_pipeline(
    data_path: str = "data/medical_supply_20_medicines.csv",
    forecast_path: Optional[str] = "data/forecast_results.csv",
    output_path: str = "data/expiry_results.csv",
    evaluation_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Execute full end-to-end expiry risk analysis pipeline and save results to CSV.

    Args:
        data_path: Input dataset path.
        forecast_path: Forecast dataset path.
        output_path: Target CSV path for expiry results.
        evaluation_date: Snapshot date or None for latest date.

    Returns:
        Result DataFrame.
    """
    logger.info("Starting Expiry Risk Analysis Pipeline...")

    df = load_inventory_and_forecast_data(
        data_path=data_path,
        forecast_path=forecast_path,
        evaluation_date=evaluation_date,
    )

    results = calculate_expiry_risk(df, evaluation_date=evaluation_date)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(results)} expiry risk evaluations to: {output_path}")

    # Log summary
    summary = get_expiry_summary(results)
    logger.info("=== Expiry Risk Summary ===")
    logger.info(f"Total Evaluated: {summary['total_items_evaluated']} records")
    logger.info(f"Items with Stock: {summary['items_with_stock']}")
    logger.info(f"Risk Breakdown: HIGH={summary['high_risk_count']}, MEDIUM={summary['medium_risk_count']}, LOW={summary['low_risk_count']}")
    logger.info(f"Total Excess Stock At Risk: {summary['total_excess_units_at_risk']} units")

    return results


def main():
    parser = argparse.ArgumentParser(description="Medical Supply Expiry & Wastage Risk Module")
    parser.add_argument("--data", default="data/medical_supply_20_medicines.csv", help="Path to inventory dataset")
    parser.add_argument("--forecast", default="data/forecast_results.csv", help="Path to forecast results CSV")
    parser.add_argument("--output", default="data/expiry_results.csv", help="Path to save expiry risk results")
    parser.add_argument("--date", default=None, help="Evaluation date (YYYY-MM-DD or 'latest' or 'all')")
    args = parser.parse_args()

    run_expiry_pipeline(
        data_path=args.data,
        forecast_path=args.forecast,
        output_path=args.output,
        evaluation_date=args.date,
    )


if __name__ == "__main__":
    main()
