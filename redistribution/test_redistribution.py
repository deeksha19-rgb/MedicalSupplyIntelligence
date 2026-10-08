"""
Comprehensive Test Suite for Medical Supply Redistribution Module
Validates all requirements, constraint checks, and future forecast integration.
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from redistribution.redistribution import (
    load_redistribution_data,
    calculate_priority_score,
    calculate_surplus_and_deficit,
    calculate_distance_km,
    generate_redistribution_recommendations,
    run_redistribution_pipeline,
    DEFAULT_HOSPITAL_COORDINATES,
)


def run_tests():
    print("==================================================")
    print("STARTING REDISTRIBUTION TEST SUITE")
    print("==================================================")

    data_path = "data/medical_supply_20_medicines.csv"
    output_path = "data/redistribution_results.csv"

    # --- TEST 1: Load Base Dataset & Date Snapshot ---
    print("\n--- TEST 1: Base Dataset & Snapshot Loading ---")
    df = load_redistribution_data(data_path=data_path)
    assert len(df) == 200, f"Expected 200 items on latest snapshot, got {len(df)}"
    assert "date" in df.columns
    assert "hospital_id" in df.columns
    assert "medicine_id" in df.columns
    assert "current_stock" in df.columns
    assert "daily_demand" in df.columns
    assert "patient_load" in df.columns
    assert "emergency_demand" in df.columns
    assert "supplier_lead_time_days" in df.columns
    print("PASSED: 200 snapshot records loaded with all required feature columns.")

    # --- TEST 2: Hospital Coordinates & Distance Calculation ---
    print("\n--- TEST 2: Existing Hospital Coordinates & Distance Calculation ---")
    for h in [f"H{i:02d}" for i in range(1, 11)]:
        assert h in DEFAULT_HOSPITAL_COORDINATES, f"Missing coordinate for {h}"

    # Verify Euclidean distance math
    d_h1_h2 = calculate_distance_km("H01", "H02")
    # H01=(10,15), H02=(15,25) -> hypot(5, 10) = 11.18 -> 11.2
    assert d_h1_h2 == 11.2, f"Expected 11.2 km, got {d_h1_h2}"

    d_self = calculate_distance_km("H01", "H01")
    assert d_self == 0.0, f"Self distance must be 0.0, got {d_self}"
    print("PASSED: Coordinate definitions and distance calculations verified.")

    # --- TEST 3: Priority Scoring ---
    print("\n--- TEST 3: Priority Scoring Evaluation ---")
    # Zero stock item with high patient load
    row_crit = pd.Series({
        "current_stock": 0,
        "daily_demand": 100,
        "supplier_lead_time_days": 8,
        "patient_load": 320,
        "emergency_demand": 0,
        "medicine_id": "M01",
        "outbreak_signal": 0,
    })
    score_crit, cat_crit, dos_crit = calculate_priority_score(row_crit)
    assert cat_crit == "CRITICAL", f"Expected CRITICAL, got {cat_crit}"
    assert dos_crit == 0.0, f"Expected 0.0 days of stock, got {dos_crit}"

    # Well-stocked item
    row_low = pd.Series({
        "current_stock": 2000,
        "daily_demand": 50,
        "supplier_lead_time_days": 6,
        "patient_load": 150,
        "emergency_demand": 0,
        "medicine_id": "M01",
        "outbreak_signal": 0,
    })
    score_low, cat_low, dos_low = calculate_priority_score(row_low)
    assert cat_low == "LOW", f"Expected LOW, got {cat_low}"
    assert dos_low == 40.0, f"Expected 40.0 days of stock, got {dos_low}"
    print("PASSED: Priority scoring accurately stratifies risk into CRITICAL and LOW.")

    # --- TEST 4: Surplus and Deficit Identification ---
    print("\n--- TEST 4: Surplus and Deficit Identification ---")
    eval_df = calculate_surplus_and_deficit(df, safety_buffer_days=2)
    surplus_hospitals = eval_df[eval_df["available_surplus"] > 0]
    deficit_hospitals = eval_df[eval_df["destination_need"] > 0]

    assert len(surplus_hospitals) > 0, "No surplus hospitals detected!"
    assert len(deficit_hospitals) > 0, "No deficit hospitals detected!"
    # Verify mutual exclusion per hospital-medicine pair:
    # A pair with surplus > 0 should have destination_need == 0
    overlap = eval_df[(eval_df["available_surplus"] > 0) & (eval_df["destination_need"] > 0)]
    assert len(overlap) == 0, f"Found {len(overlap)} pairs with both surplus and deficit!"
    print(f"PASSED: Found {len(surplus_hospitals)} surplus pairs and {len(deficit_hospitals)} deficit pairs (zero overlap).")

    # --- TEST 5: Redistribution Recommendations Generation & Schema ---
    print("\n--- TEST 5: Recommendation Generation & Output Schema ---")
    recs_df = generate_redistribution_recommendations(df, safety_buffer_days=2)
    expected_cols = [
        "from_hospital",
        "to_hospital",
        "medicine_id",
        "medicine_name",
        "recommended_quantity",
        "distance",
        "priority",
        "reason",
    ]
    assert list(recs_df.columns) == expected_cols, f"Columns mismatch: {recs_df.columns.tolist()}"
    assert len(recs_df) > 0, "No recommendations generated!"

    # Verify no self transfers
    assert (recs_df["from_hospital"] == recs_df["to_hospital"]).sum() == 0, "Self transfers found!"

    # Verify quantities > 0
    assert (recs_df["recommended_quantity"] <= 0).sum() == 0, "Non-positive quantities found!"

    # Verify distance calculation in output matches coordinates
    for _, r in recs_df.head(10).iterrows():
        expected_d = calculate_distance_km(r["from_hospital"], r["to_hospital"])
        assert r["distance"] == expected_d, f"Distance mismatch for {r['from_hospital']}->{r['to_hospital']}"

    # Verify source surplus conservation:
    # Total transferred from source for medicine <= available surplus
    src_surplus_map = eval_df.set_index(["hospital_id", "medicine_id"])["available_surplus"].to_dict()
    transferred_by_src = recs_df.groupby(["from_hospital", "medicine_id"])["recommended_quantity"].sum().to_dict()
    for (src, med), trans_qty in transferred_by_src.items():
        avail = src_surplus_map.get((src, med), 0)
        assert trans_qty <= avail, f"Source {src} for {med} transferred {trans_qty} > available surplus {avail}!"

    # Verify destination need constraint:
    # Total transferred to destination for medicine <= destination need
    dest_need_map = eval_df.set_index(["hospital_id", "medicine_id"])["destination_need"].to_dict()
    transferred_to_dest = recs_df.groupby(["to_hospital", "medicine_id"])["recommended_quantity"].sum().to_dict()
    for (dst, med), trans_qty in transferred_to_dest.items():
        need = dest_need_map.get((dst, med), 0)
        assert trans_qty <= need, f"Destination {dst} for {med} received {trans_qty} > need {need}!"

    print("PASSED: All 8 output columns match exactly. Surplus and deficit constraints fully satisfied.")

    # --- TEST 6: Future Forecast Integration ---
    print("\n--- TEST 6: Future Forecast Integration (forecast_results.csv) ---")
    mock_forecast_path = "data/mock_forecast_results.csv"
    mock_fc = pd.DataFrame([
        {
            "hospital_id": "H02",
            "medicine_id": "M01",
            "predicted_daily_demand": 250, # High forecasted demand
            "days_until_stockout": 0.0,
            "shortage_risk": "CRITICAL",
        },
        {
            "hospital_id": "H05",
            "medicine_id": "M01",
            "predicted_daily_demand": 80,  # Lower forecasted demand -> higher surplus!
            "days_until_stockout": 13.0,
            "shortage_risk": "LOW",
        },
    ])
    mock_fc.to_csv(mock_forecast_path, index=False)

    df_fc = load_redistribution_data(data_path=data_path, forecast_path=mock_forecast_path)
    h2_m1 = df_fc[(df_fc["hospital_id"] == "H02") & (df_fc["medicine_id"] == "M01")].iloc[0]
    assert h2_m1["effective_demand"] == 250, f"Expected 250 predicted demand, got {h2_m1['effective_demand']}"
    assert h2_m1["forecast_shortage_risk"] == "CRITICAL"
    assert h2_m1["forecast_days_until_stockout"] == 0.0

    h5_m1 = df_fc[(df_fc["hospital_id"] == "H05") & (df_fc["medicine_id"] == "M01")].iloc[0]
    assert h5_m1["effective_demand"] == 80, f"Expected 80 predicted demand, got {h5_m1['effective_demand']}"

    # Generate recommendations with forecast
    recs_fc = generate_redistribution_recommendations(df_fc, safety_buffer_days=2)
    assert len(recs_fc) > 0, "Forecast recommendations failed!"
    print("PASSED: Forecast columns (predicted_daily_demand, days_until_stockout, shortage_risk) seamlessly incorporated.")

    # Clean up mock file
    if os.path.exists(mock_forecast_path):
        os.remove(mock_forecast_path)

    # --- TEST 7: End-to-End Pipeline & Output File Verification ---
    print("\n--- TEST 7: End-to-End Pipeline Execution ---")
    final_recs = run_redistribution_pipeline(
        data_path=data_path,
        output_path=output_path,
    )
    assert os.path.exists(output_path), f"Output file not found: {output_path}"
    saved_df = pd.read_csv(output_path)
    assert list(saved_df.columns) == expected_cols
    assert len(saved_df) == len(final_recs)
    print(f"PASSED: Results saved to {output_path} ({len(saved_df)} transfer orders).")

    print("\n==================================================")
    print("ALL 7 TEST SUITES PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
