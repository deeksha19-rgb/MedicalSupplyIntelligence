"""
Hospital-to-Hospital Redistribution Module
Medical Supply Intelligence Platform

Part 2: Multi-criteria optimization engine to match surplus hospitals with deficit
hospitals for the SAME medicine, ensuring logical volume conservation and zero impossible transfers.
Part 3: Clinical Priority Scoring system factoring shortage risk, emergency demand,
patient load, and days until stockout to classify urgency into CRITICAL, HIGH, MEDIUM, LOW.

Future Forecast Integration:
When data/forecast_results.csv becomes available, automatically incorporates:
- predicted_daily_demand
- days_until_stockout
- shortage_risk
"""

import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Ensure parent directory is in python path for relative package imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from expiry.expiry_risk import (
        calculate_expiry_risk,
        CRITICAL_MEDICINE_IDS,
    )
except ImportError:
    CRITICAL_MEDICINE_IDS = {"M07", "M19", "M05", "M10", "M16", "M18", "M20"}
    calculate_expiry_risk = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard Regional Hospital Geographic Coordinates (km relative to central dispatch hub)
# Represents realistic metropolitan & regional health authority network layout
DEFAULT_HOSPITAL_COORDINATES: Dict[str, Tuple[float, float]] = {
    "H01": (10.0, 15.0),  # Central General Hospital
    "H02": (15.0, 25.0),  # North District Medical Center
    "H03": (25.0, 10.0),  # East Valley Hospital
    "H04": (5.0, 30.0),   # Northwest Community Clinic
    "H05": (20.0, 20.0),  # Central Children & Specialized
    "H06": (30.0, 25.0),  # Northeast Regional Hospital
    "H07": (12.0, 5.0),   # South Metro Emergency Care
    "H08": (35.0, 15.0),  # East Suburban Hospital
    "H09": (5.0, 10.0),   # Southwest Primary Care Center
    "H10": (22.0, 35.0),  # North Ridge Trauma Center
}


def calculate_distance_km(
    hosp_a: str,
    hosp_b: str,
    coords: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """
    Compute Euclidean driving distance (km) between two hospitals using existing coordinates.

    Args:
        hosp_a: Source hospital ID (e.g. 'H01')
        hosp_b: Destination hospital ID (e.g. 'H02')
        coords: Optional custom coordinate mapping; defaults to DEFAULT_HOSPITAL_COORDINATES

    Returns:
        Distance in kilometers rounded to 1 decimal place.
    """
    if hosp_a == hosp_b:
        return 0.0
    coords_map = coords or DEFAULT_HOSPITAL_COORDINATES
    p1 = coords_map.get(hosp_a, (0.0, 0.0))
    p2 = coords_map.get(hosp_b, (20.0, 20.0))
    dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return round(dist, 1)


def load_redistribution_data(
    data_path: str = "data/medical_supply_20_medicines.csv",
    forecast_path: Optional[str] = "data/forecast_results.csv",
    evaluation_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load hospital inventory records and optionally merge forecast predictions.

    When data/forecast_results.csv is available, seamlessly integrates:
    - predicted_daily_demand (overriding historical daily_demand)
    - days_until_stockout (from forecast engine)
    - shortage_risk (from forecast engine)

    Args:
        data_path: Path to the main inventory CSV dataset.
        forecast_path: Path to forecast results CSV (if available).
        evaluation_date: Specific snapshot date (e.g. '2025-12-31').
                         If None, defaults to the latest date in the dataset.

    Returns:
        pd.DataFrame ready for redistribution optimization.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Inventory dataset not found at: {data_path}")

    logger.info(f"Loading inventory dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Standardize data types
    df["date"] = pd.to_datetime(df["date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["current_stock"] = pd.to_numeric(df["current_stock"], errors="coerce").fillna(0).astype(int)
    df["daily_demand"] = pd.to_numeric(df["daily_demand"], errors="coerce").fillna(0).astype(int)
    df["supplier_lead_time_days"] = pd.to_numeric(df["supplier_lead_time_days"], errors="coerce").fillna(7).astype(int)
    df["patient_load"] = pd.to_numeric(df["patient_load"], errors="coerce").fillna(200).astype(int)
    df["emergency_demand"] = pd.to_numeric(df["emergency_demand"], errors="coerce").fillna(0).astype(int)

    # Filter to snapshot date
    if evaluation_date is None or evaluation_date.lower() == "latest":
        latest_date = df["date"].max()
        logger.info(f"Evaluating as of latest inventory date: {latest_date.strftime('%Y-%m-%d')}")
        df = df[df["date"] == latest_date].copy()
    elif evaluation_date.lower() != "all":
        target_date = pd.to_datetime(evaluation_date)
        logger.info(f"Evaluating as of specified date: {target_date.strftime('%Y-%m-%d')}")
        df = df[df["date"] == target_date].copy()
        if df.empty:
            raise ValueError(f"No records found for evaluation date: {evaluation_date}")
    else:
        df = df.copy()

    # Baseline effective demand
    df["effective_demand"] = df["daily_demand"]

    # Check for forecast results
    df["predicted_daily_demand"] = np.nan
    df["forecast_days_until_stockout"] = np.nan
    df["forecast_shortage_risk"] = np.nan

    if forecast_path and os.path.exists(forecast_path):
        logger.info(f"Forecast results detected at: {forecast_path}. Integrating forecast features...")
        try:
            fc_df = pd.read_csv(forecast_path)

            # Identify predicted daily demand column
            demand_col = None
            for col in [
                "predicted_daily_demand",
                "predicted_demand",
                "forecasted_demand",
                "daily_demand_forecast",
                "forecast_demand",
                "forecast",
            ]:
                if col in fc_df.columns:
                    demand_col = col
                    break

            # Identify days until stockout column
            dos_col = None
            for col in [
                "days_until_stockout",
                "days_to_stockout",
                "predicted_days_until_stockout",
                "stockout_days",
            ]:
                if col in fc_df.columns:
                    dos_col = col
                    break

            # Identify shortage risk column
            risk_col = None
            for col in [
                "shortage_risk",
                "predicted_shortage_risk",
                "shortage_probability",
                "shortage_prob",
                "risk_level",
            ]:
                if col in fc_df.columns:
                    risk_col = col
                    break

            # Merge keys: hospital_id, medicine_id, and optionally date
            merge_keys = ["hospital_id", "medicine_id"]
            if "date" in fc_df.columns and "date" in df.columns:
                fc_df["date"] = pd.to_datetime(fc_df["date"])
                merge_keys.append("date")

            cols_to_pull = list(merge_keys)
            rename_map = {}
            if demand_col:
                cols_to_pull.append(demand_col)
                rename_map[demand_col] = "predicted_daily_demand"
            if dos_col:
                cols_to_pull.append(dos_col)
                rename_map[dos_col] = "forecast_days_until_stockout"
            if risk_col:
                cols_to_pull.append(risk_col)
                rename_map[risk_col] = "forecast_shortage_risk"

            fc_sub = fc_df[cols_to_pull].drop_duplicates(subset=merge_keys).rename(columns=rename_map)

            # Drop pre-initialized placeholder columns before merge to prevent _x/_y collisions
            df_base = df.drop(
                columns=["predicted_daily_demand", "forecast_days_until_stockout", "forecast_shortage_risk"],
                errors="ignore",
            )
            df = pd.merge(df_base, fc_sub, on=merge_keys, how="left")

            # Ensure columns exist even if not in forecast file
            for c in ["predicted_daily_demand", "forecast_days_until_stockout", "forecast_shortage_risk"]:
                if c not in df.columns:
                    df[c] = np.nan

            # Apply predicted demand if available
            has_pred = df["predicted_daily_demand"].notnull() & (df["predicted_daily_demand"] > 0)
            df["effective_demand"] = np.where(has_pred, df["predicted_daily_demand"], df["daily_demand"])

            logger.info(
                f"Forecast features successfully integrated: demand={demand_col}, "
                f"stockout_days={dos_col}, shortage_risk={risk_col}"
            )
        except Exception as e:
            logger.warning(f"Error parsing forecast file ({forecast_path}): {e}. Using baseline dataset features.")
    else:
        logger.info(
            f"Forecast file not found at '{forecast_path}'. Using current inventory features from {data_path}."
        )

    return df


def calculate_priority_score(row: pd.Series) -> Tuple[float, str, float]:
    """
    Calculate clinical shortage priority score (0-100) and priority category (Part 3).

    Considers:
    - current_stock
    - daily_demand (or predicted_daily_demand if available)
    - patient_load
    - emergency_demand
    - supplier_lead_time_days
    - days_until_stockout (calculated or from forecast)
    - shortage_risk (from forecast if available, or evaluated from stockout trajectory)

    Returns:
        Tuple of (priority_score, priority_category, days_until_stockout)
        priority_category is one of: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    """
    stock = float(row.get("current_stock", 0))
    demand = float(row.get("effective_demand", row.get("daily_demand", 1)))
    demand = max(demand, 0.1)
    lead_time = float(row.get("supplier_lead_time_days", 7))
    emerg_demand = float(row.get("emergency_demand", 0))
    patient_load = float(row.get("patient_load", 200))
    med_id = str(row.get("medicine_id", ""))
    outbreak = float(row.get("outbreak_signal", 0))

    # Determine days until stockout: prefer forecast value if valid, else calculate
    fc_dos = row.get("forecast_days_until_stockout", np.nan)
    if pd.notnull(fc_dos) and not np.isnan(fc_dos):
        dos = round(float(fc_dos), 1)
    else:
        dos = round(stock / demand, 1)

    # Component 1: Shortage Risk Score (0 to 50 pts)
    fc_risk = row.get("forecast_shortage_risk", np.nan)
    if pd.notnull(fc_risk):
        # Handle forecast shortage risk
        if isinstance(fc_risk, (int, float)) and not np.isnan(fc_risk):
            if fc_risk <= 1.0:
                shortage_score = float(fc_risk) * 50.0
            else:
                shortage_score = float(np.clip(fc_risk, 0.0, 50.0))
        elif isinstance(fc_risk, str):
            r_str = fc_risk.strip().upper()
            if r_str == "CRITICAL":
                shortage_score = 50.0
            elif r_str == "HIGH":
                shortage_score = 42.0
            elif r_str == "MEDIUM":
                shortage_score = 25.0
            else:
                shortage_score = 5.0
        else:
            shortage_score = 25.0
    else:
        # Evaluate shortage risk directly from current stock, demand, and supplier lead time
        if stock == 0:
            shortage_score = 50.0
        elif dos < lead_time:
            shortage_score = 42.0 * (1.0 - (dos / lead_time))
        elif dos < lead_time * 1.5:
            shortage_score = 18.0 * (1.0 - (dos / (lead_time * 1.5)))
        else:
            shortage_score = 0.0

    # Component 2: Emergency Demand & Outbreak surge (0 to 25 pts)
    emerg_score = min(20.0, emerg_demand * 2.0) + (5.0 if outbreak > 0 else 0.0)

    # Component 3: Patient Load Impact (0 to 15 pts, scaled from 100 to 400 patients)
    load_score = min(15.0, max(0.0, (patient_load - 100.0) / 300.0 * 15.0))

    # Component 4: Medicine Clinical Criticality (0 to 10 pts)
    crit_score = 10.0 if med_id in CRITICAL_MEDICINE_IDS else 0.0

    # Total Priority Score (0 to 100)
    total_score = round(shortage_score + emerg_score + load_score + crit_score, 1)
    total_score = float(np.clip(total_score, 0.0, 100.0))

    # Priority Category: CRITICAL, HIGH, MEDIUM, LOW
    if stock == 0 and (emerg_demand > 0 or outbreak > 0 or med_id in CRITICAL_MEDICINE_IDS or patient_load >= 250):
        category = "CRITICAL"
    elif total_score >= 65.0:
        category = "CRITICAL"
    elif total_score >= 45.0 or (stock == 0):
        category = "HIGH"
    elif total_score >= 25.0 or (dos < lead_time):
        category = "MEDIUM"
    else:
        category = "LOW"

    return total_score, category, dos


def calculate_surplus_and_deficit(
    df: pd.DataFrame,
    expiry_df: Optional[pd.DataFrame] = None,
    safety_buffer_days: int = 2,
) -> pd.DataFrame:
    """
    Identify hospitals with genuine surplus stock and hospitals with genuine shortage need.

    Rules:
    - Source Safe Retention: Source hospital retains sufficient stock to cover its own
      demand during supplier lead time + safety buffer days:
      safe_retention = effective_demand * (supplier_lead_time_days + safety_buffer_days)
    - Available Surplus: Stock above safe retention floor, plus any batch-level excess stock.
      available_surplus = max(excess_stock, current_stock - safe_retention)
    - Destination Genuine Need: Deficit required to survive until next supplier delivery:
      destination_need = max(0, safe_retention - current_stock)

    Returns:
        pd.DataFrame with available_surplus, destination_need, priority, and priority_score.
    """
    working_df = df.copy()

    # Merge expiry metrics if available
    if expiry_df is not None and not expiry_df.empty:
        expiry_cols = ["hospital_id", "medicine_id", "days_to_expiry", "excess_stock", "expiry_risk"]
        present_cols = [c for c in expiry_cols if c in expiry_df.columns]
        merge_keys = ["hospital_id", "medicine_id"]
        drop_cols = [c for c in present_cols if c in working_df.columns and c not in merge_keys]
        working_df = pd.merge(
            working_df.drop(columns=drop_cols),
            expiry_df[present_cols],
            on=merge_keys,
            how="left",
        )

    # Ensure days to expiry is calculated
    if "days_to_expiry" not in working_df.columns:
        date_col = pd.to_datetime(working_df["date"])
        exp_col = pd.to_datetime(working_df["expiry_date"])
        working_df["days_to_expiry"] = (exp_col - date_col).dt.days

    # Ensure effective demand
    if "effective_demand" not in working_df.columns:
        if "predicted_daily_demand" in working_df.columns:
            has_pred = working_df["predicted_daily_demand"].notnull() & (working_df["predicted_daily_demand"] > 0)
            working_df["effective_demand"] = np.where(has_pred, working_df["predicted_daily_demand"], working_df["daily_demand"])
        else:
            working_df["effective_demand"] = working_df["daily_demand"]

    # Calculate Priority Score, Category, and Days Until Stockout
    p_results = [calculate_priority_score(row) for _, row in working_df.iterrows()]
    working_df["priority_score"] = [r[0] for r in p_results]
    working_df["priority"] = [r[1] for r in p_results]
    working_df["days_until_stockout"] = [r[2] for r in p_results]

    # Calculate Safe Retention Floor & Available Surplus
    lead_time = working_df["supplier_lead_time_days"].fillna(7).astype(float)
    demand = working_df["effective_demand"].fillna(1).astype(float)
    coverage_days = lead_time + float(safety_buffer_days)

    working_df["safe_retention_stock"] = (demand * coverage_days).round().astype(int)

    # Operational surplus: stock above safe retention floor
    op_surplus = (working_df["current_stock"] - working_df["safe_retention_stock"]).clip(lower=0)

    # Expiry surplus: stock at risk of expiry before consumption
    if "excess_stock" in working_df.columns:
        exp_surplus = working_df["excess_stock"].fillna(0).astype(int)
    else:
        proj_use = (demand * working_df["days_to_expiry"].clip(lower=0)).round().astype(int)
        exp_surplus = (working_df["current_stock"] - proj_use).clip(lower=0)

    working_df["available_surplus"] = np.maximum(op_surplus, exp_surplus).astype(int)

    # Destination Genuine Need:
    working_df["target_coverage_stock"] = (demand * coverage_days).round().astype(int)
    working_df["destination_need"] = (working_df["target_coverage_stock"] - working_df["current_stock"]).clip(lower=0).astype(int)

    return working_df


def generate_redistribution_recommendations(
    inventory_df: pd.DataFrame,
    expiry_df: Optional[pd.DataFrame] = None,
    safety_buffer_days: int = 2,
    max_transit_distance_km: float = 120.0,
    coords: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Generate practical hospital-to-hospital redistribution recommendations.

    Strict Transfer Rules:
    - Source hospital MUST have genuine surplus (available_surplus > 0).
    - Destination hospital MUST have genuine need (destination_need > 0).
    - Both hospitals MUST have the SAME medicine (grouped by medicine_id).
    - Transfer quantity NEVER exceeds available surplus.
    - Transfer quantity NEVER exceeds destination need.
    - Transfer quantity NEVER exceeds destination's capacity to consume before expiry.
    - No self-transfers (from_hospital != to_hospital).
    - Stock closer to expiry is prioritized for transfer.

    Returns:
        pd.DataFrame containing columns:
        from_hospital, to_hospital, medicine_id, medicine_name,
        recommended_quantity, distance, priority, reason
    """
    logger.info("Evaluating hospital network for redistribution matches...")
    df_eval = calculate_surplus_and_deficit(
        inventory_df,
        expiry_df=expiry_df,
        safety_buffer_days=safety_buffer_days,
    )

    priority_rank_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations: List[Dict[str, Any]] = []

    # Process each medicine independently (SAME medicine requirement)
    for med_id, med_group in df_eval.groupby("medicine_id"):
        med_name = med_group["medicine_name"].iloc[0]

        # 1. Hospitals with genuine surplus
        sources = med_group[med_group["available_surplus"] > 0].copy()

        # 2. Hospitals with genuine shortage/critical need
        destinations = med_group[med_group["destination_need"] > 0].copy()

        if sources.empty or destinations.empty:
            continue

        # Sort destinations: highest shortage priority first
        destinations["pri_rank"] = destinations["priority"].map(priority_rank_map)
        destinations = destinations.sort_values(
            by=["pri_rank", "priority_score", "days_until_stockout"],
            ascending=[True, False, True],
        )

        # In-memory tracking of remaining surplus and need to conserve volume
        src_surplus = sources.set_index("hospital_id")["available_surplus"].to_dict()
        dest_need = destinations.set_index("hospital_id")["destination_need"].to_dict()

        for _, dest_row in destinations.iterrows():
            to_h = dest_row["hospital_id"]
            need = dest_need.get(to_h, 0)
            if need <= 0:
                continue

            # Eligible sources: surplus > 0 and not the same hospital
            valid_src_ids = [h for h, sur in src_surplus.items() if sur > 0 and h != to_h]
            if not valid_src_ids:
                break

            sub_sources = sources[sources["hospital_id"].isin(valid_src_ids)].copy()

            # Calculate distance using existing hospital coordinates
            sub_sources["distance"] = sub_sources["hospital_id"].apply(
                lambda h: calculate_distance_km(h, to_h, coords=coords)
            )

            # Filter within maximum transit distance
            sub_sources = sub_sources[sub_sources["distance"] <= max_transit_distance_km]
            if sub_sources.empty:
                continue

            # Prioritize:
            # 1. days_to_expiry ascending (near-expiry stock moved first to prevent waste)
            # 2. distance ascending (shorter distance)
            # 3. available_surplus descending
            sub_sources = sub_sources.sort_values(
                by=["days_to_expiry", "distance", "available_surplus"],
                ascending=[True, True, False],
            )

            for _, src_row in sub_sources.iterrows():
                from_h = src_row["hospital_id"]
                surplus_avail = src_surplus.get(from_h, 0)
                if surplus_avail <= 0 or need <= 0:
                    continue

                days_exp = int(src_row["days_to_expiry"])
                dist_km = float(src_row["distance"])

                # Skip batches that expire too quickly (<= 2 days)
                if days_exp <= 2:
                    continue

                dest_daily_demand = max(float(dest_row["effective_demand"]), 0.1)
                max_dest_consume = int(dest_daily_demand * days_exp)

                # Transfer quantity bounded by available surplus, need, and shelf life
                transfer_qty = min(surplus_avail, need, max_dest_consume)
                transfer_qty = int(transfer_qty)

                if transfer_qty <= 0:
                    continue

                # Conserve quantities
                src_surplus[from_h] -= transfer_qty
                dest_need[to_h] -= transfer_qty
                need -= transfer_qty

                # Clinical reason text
                dest_dos = dest_row["days_until_stockout"]
                dest_lead = dest_row["supplier_lead_time_days"]
                dest_pri = dest_row["priority"]
                dest_emerg = dest_row.get("emergency_demand", 0)
                cov_days = round(transfer_qty / dest_daily_demand, 1)

                emerg_text = f" (+{dest_emerg} emergency surge)" if dest_emerg > 0 else ""
                reason = (
                    f"{to_h} faces {dest_pri} shortage risk ({dest_dos}d supply remaining{emerg_text}, "
                    f"lead time {dest_lead}d). {from_h} holds surplus of {surplus_avail} units (expires in {days_exp}d). "
                    f"Transfer of {transfer_qty} units covers {cov_days}d demand until supplier delivery."
                )

                recommendations.append({
                    "from_hospital": from_h,
                    "to_hospital": to_h,
                    "medicine_id": med_id,
                    "medicine_name": med_name,
                    "recommended_quantity": transfer_qty,
                    "distance": dist_km,
                    "priority": dest_pri,
                    "reason": reason,
                })

                if need <= 0:
                    break

    recs_df = pd.DataFrame(recommendations)
    output_cols = [
        "from_hospital",
        "to_hospital",
        "medicine_id",
        "medicine_name",
        "recommended_quantity",
        "distance",
        "priority",
        "reason",
    ]

    if recs_df.empty:
        logger.warning("No valid redistribution transfers found.")
        return pd.DataFrame(columns=output_cols)

    # Sort final recommendations by priority rank (CRITICAL first)
    recs_df["pri_sort"] = recs_df["priority"].map(priority_rank_map)
    recs_df = recs_df.sort_values(
        by=["pri_sort", "recommended_quantity"],
        ascending=[True, False],
    ).drop(columns=["pri_sort"])

    return recs_df[output_cols]


def get_redistribution_summary(recs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary metrics of redistribution recommendations.
    """
    if recs_df.empty:
        return {
            "total_transfers": 0,
            "total_units_transferred": 0,
            "critical_transfers": 0,
            "high_transfers": 0,
            "medium_transfers": 0,
            "low_transfers": 0,
            "participating_sources": 0,
            "relieved_destinations": 0,
            "medicines_balanced": 0,
        }

    pri_counts = recs_df["priority"].value_counts().to_dict()
    return {
        "total_transfers": len(recs_df),
        "total_units_transferred": int(recs_df["recommended_quantity"].sum()),
        "critical_transfers": pri_counts.get("CRITICAL", 0),
        "high_transfers": pri_counts.get("HIGH", 0),
        "medium_transfers": pri_counts.get("MEDIUM", 0),
        "low_transfers": pri_counts.get("LOW", 0),
        "participating_sources": int(recs_df["from_hospital"].nunique()),
        "relieved_destinations": int(recs_df["to_hospital"].nunique()),
        "medicines_balanced": int(recs_df["medicine_id"].nunique()),
    }


def run_redistribution_pipeline(
    data_path: str = "data/medical_supply_20_medicines.csv",
    expiry_path: str = "data/expiry_results.csv",
    forecast_path: Optional[str] = "data/forecast_results.csv",
    output_path: str = "data/redistribution_results.csv",
    evaluation_date: Optional[str] = None,
    safety_buffer_days: int = 2,
) -> pd.DataFrame:
    """
    Execute full redistribution pipeline and save results to CSV.

    Args:
        data_path: Inventory dataset path.
        expiry_path: Expiry results path.
        forecast_path: Forecast results path (used when available).
        output_path: Output CSV path for recommendations.
        evaluation_date: Snapshot date or 'latest'.
        safety_buffer_days: Days of safety stock to retain at source.

    Returns:
        pd.DataFrame containing generated redistribution recommendations.
    """
    logger.info("Starting Redistribution & Priority Pipeline...")

    # Load inventory dataset and forecast if present
    inventory_df = load_redistribution_data(
        data_path=data_path,
        forecast_path=forecast_path,
        evaluation_date=evaluation_date,
    )

    # Load or compute expiry results
    expiry_df = None
    if os.path.exists(expiry_path):
        logger.info(f"Loading pre-calculated expiry metrics from: {expiry_path}")
        expiry_df = pd.read_csv(expiry_path)
    elif calculate_expiry_risk is not None:
        logger.info("Computing expiry metrics inline...")
        expiry_df = calculate_expiry_risk(inventory_df, evaluation_date=evaluation_date)

    # Generate recommendations
    recs_df = generate_redistribution_recommendations(
        inventory_df=inventory_df,
        expiry_df=expiry_df,
        safety_buffer_days=safety_buffer_days,
    )

    # Save to output CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    recs_df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(recs_df)} redistribution recommendations to: {output_path}")

    # Log summary
    summary = get_redistribution_summary(recs_df)
    logger.info("=== Redistribution Recommendation Summary ===")
    logger.info(f"Total Recommended Transfers: {summary['total_transfers']}")
    logger.info(f"Total Units Redistributed: {summary['total_units_transferred']} units")
    logger.info(
        f"Priority Breakdown: CRITICAL={summary['critical_transfers']}, "
        f"HIGH={summary['high_transfers']}, "
        f"MEDIUM={summary['medium_transfers']}, "
        f"LOW={summary['low_transfers']}"
    )
    logger.info(
        f"Network Reach: {summary['participating_sources']} source hospitals -> "
        f"{summary['relieved_destinations']} recipient hospitals across {summary['medicines_balanced']} medicines"
    )

    return recs_df


def main():
    parser = argparse.ArgumentParser(description="Hospital Medical Supply Redistribution Module")
    parser.add_argument("--data", default="data/medical_supply_20_medicines.csv", help="Path to inventory dataset")
    parser.add_argument("--expiry", default="data/expiry_results.csv", help="Path to expiry results CSV")
    parser.add_argument("--forecast", default="data/forecast_results.csv", help="Path to forecast results CSV")
    parser.add_argument("--output", default="data/redistribution_results.csv", help="Path to save redistribution results")
    parser.add_argument("--date", default=None, help="Evaluation date (YYYY-MM-DD or 'latest')")
    parser.add_argument("--buffer", type=int, default=2, help="Safety buffer days to retain at source")
    args = parser.parse_args()

    run_redistribution_pipeline(
        data_path=args.data,
        expiry_path=args.expiry,
        forecast_path=args.forecast,
        output_path=args.output,
        evaluation_date=args.date,
        safety_buffer_days=args.buffer,
    )


if __name__ == "__main__":
    main()
