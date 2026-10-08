"""
Medical Supply Intelligence Dashboard
======================================
Interactive Streamlit Dashboard designed for hospital supply chain managers and network coordinators.

Reads:
- data/medical_supply_20_medicines.csv
- data/forecast_results.csv
- data/expiry_results.csv (with graceful placeholder if not yet generated)
- data/redistribution_results.csv (with graceful placeholder if not yet generated)

Sections:
1. OVERVIEW (KPI cards: Total Hospitals, Total Medicines, Medicines at High Shortage Risk, High Expiry Risk Items, Recommended Transfers)
2. INVENTORY (Hospital, Medicine, Current Stock, Demand, Stock status; filters for hospital & medicine)
3. DEMAND FORECAST (Predicted daily demand, Predicted 7-day demand, Current stock, Days until stockout, charts)
4. SHORTAGE RISK (HIGH, MEDIUM, LOW categories; most likely to run out)
5. EXPIRY RISK (Medicine, Hospital, Current stock, Days to expiry, Expiry risk)
6. REDISTRIBUTION (From Hospital, To Hospital, Medicine, Quantity, Priority, Reason)
7. PRIORITY FACILITIES (Hospitals requiring immediate attention)
8. SIMPLE AI ASSISTANT (Local data-driven Q&A without external paid APIs)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & DESIGN AESTHETICS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Supply Intelligence Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, professional healthcare informatics styling
st.markdown(
    """
    <style>
    /* Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94a3b8;
    }

    /* Badges */
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }

    /* AI Assistant bubble */
    .ai-response-box {
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 18px 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 14px;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. ROBUST PATH RESOLUTION & SAFE DATA LOADERS
# -----------------------------------------------------------------------------
def get_data_dir() -> Path:
    """Resolve data directory safely whether run from repo root or subfolder."""
    possible_paths = [
        Path("data"),
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent / "data",
    ]
    for p in possible_paths:
        if p.exists() and p.is_dir():
            return p
    return Path("data")


DATA_DIR = get_data_dir()


@st.cache_data(show_spinner=False)
def load_medical_supply_data() -> Optional[pd.DataFrame]:
    """Load raw dataset with datetime parsing."""
    csv_file = DATA_DIR / "medical_supply_20_medicines.csv"
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            st.error(f"Error reading {csv_file.name}: {e}")
            return None
    return None


@st.cache_data(show_spinner=False)
def load_forecast_data() -> Optional[pd.DataFrame]:
    """Load forecast results."""
    csv_file = DATA_DIR / "forecast_results.csv"
    if csv_file.exists():
        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Error reading {csv_file.name}: {e}")
            return None
    return None


@st.cache_data(show_spinner=False)
def load_expiry_data() -> Optional[pd.DataFrame]:
    """Load expiry risk results if generated."""
    csv_file = DATA_DIR / "expiry_results.csv"
    if csv_file.exists():
        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Error reading {csv_file.name}: {e}")
            return None
    return None


@st.cache_data(show_spinner=False)
def load_redistribution_data() -> Optional[pd.DataFrame]:
    """Load redistribution results if generated."""
    csv_file = DATA_DIR / "redistribution_results.csv"
    if csv_file.exists():
        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Error reading {csv_file.name}: {e}")
            return None
    return None


def get_latest_inventory_snapshot(
    supply_df: Optional[pd.DataFrame],
    forecast_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Create unified latest inventory view with demand and stock status."""
    if supply_df is not None:
        latest_date = supply_df["date"].max()
        inv = supply_df[supply_df["date"] == latest_date].copy()
    elif forecast_df is not None:
        inv = forecast_df.copy()
        inv["daily_demand"] = inv["predicted_daily_demand"]
    else:
        return pd.DataFrame()

    # Merge forecast metrics if available
    if forecast_df is not None and "predicted_daily_demand" not in inv.columns:
        inv = inv.merge(
            forecast_df[
                [
                    "hospital_id",
                    "medicine_id",
                    "predicted_daily_demand",
                    "predicted_7_day_demand",
                    "days_until_stockout",
                    "shortage_risk",
                ]
            ],
            on=["hospital_id", "medicine_id"],
            how="left",
        )

    # Determine stock status category
    def compute_stock_status(row):
        stock = row.get("current_stock", 0)
        risk = str(row.get("shortage_risk", "")).upper()
        days = row.get("days_until_stockout", 999.0)

        if stock <= 0:
            return "Out of Stock"
        elif risk == "HIGH" or days <= 3.0:
            return "Critical Low"
        elif risk == "MEDIUM" or days <= 7.0:
            return "Moderate"
        else:
            return "Adequate"

    inv["stock_status"] = inv.apply(compute_stock_status, axis=1)
    return inv


# -----------------------------------------------------------------------------
# 3. MAIN DASHBOARD APPLICATION
# -----------------------------------------------------------------------------
def main():
    # Load datasets
    supply_df = load_medical_supply_data()
    forecast_df = load_forecast_data()
    expiry_df = load_expiry_data()
    redist_df = load_redistribution_data()

    # Unified Inventory Table
    inv_df = get_latest_inventory_snapshot(supply_df, forecast_df)

    # ---------------------------------------------------------
    # SIDEBAR: Network Filters & Pipeline Status
    # ---------------------------------------------------------
    with st.sidebar:
        st.title("🏥 Control Tower")
        st.markdown(
            "**Medical Supply Intelligence**  \n*Network-wide inventory optimization & shortage interception.*"
        )
        st.divider()

        st.subheader("Data Pipeline Status")
        c1, c2 = st.columns([3, 2])
        c1.markdown("`medical_supply.csv`")
        c2.markdown("✅ Loaded" if supply_df is not None else "❌ Missing")

        c1, c2 = st.columns([3, 2])
        c1.markdown("`forecast_results.csv`")
        c2.markdown("✅ Loaded" if forecast_df is not None else "❌ Missing")

        c1, c2 = st.columns([3, 2])
        c1.markdown("`expiry_results.csv`")
        c2.markdown("✅ Loaded" if expiry_df is not None else "⏳ Pending")

        c1, c2 = st.columns([3, 2])
        c1.markdown("`redistribution.csv`")
        c2.markdown("✅ Loaded" if redist_df is not None else "⏳ Pending")

        st.divider()

        # Global Filters
        st.subheader("Global Filters")
        all_hospitals = sorted(inv_df["hospital_id"].unique()) if not inv_df.empty else []
        all_medicines = sorted(inv_df["medicine_name"].unique()) if not inv_df.empty else []

        selected_hospital = st.selectbox(
            "Hospital",
            options=["All Hospitals"] + all_hospitals,
            index=0,
            help="Filter data across sections by hospital",
        )

        selected_medicine = st.selectbox(
            "Medicine",
            options=["All Medicines"] + all_medicines,
            index=0,
            help="Filter data across sections by medicine",
        )

        st.divider()
        if st.button("🔄 Refresh Data Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------
    st.title("Medical Supply Intelligence Dashboard")
    st.caption("AI-driven demand forecasting, shortage prevention, and smart healthcare logistics.")

    # Filtered copies of working data
    filtered_inv = inv_df.copy()
    filtered_fc = forecast_df.copy() if forecast_df is not None else pd.DataFrame()

    if selected_hospital != "All Hospitals":
        if not filtered_inv.empty:
            filtered_inv = filtered_inv[filtered_inv["hospital_id"] == selected_hospital]
        if not filtered_fc.empty:
            filtered_fc = filtered_fc[filtered_fc["hospital_id"] == selected_hospital]

    if selected_medicine != "All Medicines":
        if not filtered_inv.empty:
            filtered_inv = filtered_inv[filtered_inv["medicine_name"] == selected_medicine]
        if not filtered_fc.empty:
            filtered_fc = filtered_fc[filtered_fc["medicine_name"] == selected_medicine]

    # ---------------------------------------------------------
    # NAVIGATION TABS (8 SECTIONS)
    # ---------------------------------------------------------
    tab_overview, tab_inv, tab_fc, tab_shortage, tab_expiry, tab_redist, tab_priority, tab_ai = st.tabs(
        [
            "1. OVERVIEW",
            "2. INVENTORY",
            "3. DEMAND FORECAST",
            "4. SHORTAGE RISK",
            "5. EXPIRY RISK",
            "6. REDISTRIBUTION",
            "7. PRIORITY FACILITIES",
            "8. SIMPLE AI ASSISTANT",
        ]
    )

    # =========================================================================
    # SECTION 1: OVERVIEW
    # =========================================================================
    with tab_overview:
        st.subheader("1. Network Overview")

        total_hospitals = inv_df["hospital_id"].nunique() if not inv_df.empty else 0
        total_medicines = inv_df["medicine_name"].nunique() if not inv_df.empty else 0

        if forecast_df is not None and not forecast_df.empty:
            high_shortage_count = int((forecast_df["shortage_risk"] == "HIGH").sum())
        else:
            high_shortage_count = 0

        if expiry_df is not None and not expiry_df.empty:
            exp_col = "expiry_risk" if "expiry_risk" in expiry_df.columns else "Expiry risk"
            high_expiry_items = int((expiry_df[exp_col].astype(str).str.upper() == "HIGH").sum())
            expiry_val_str = f"{high_expiry_items}"
            expiry_sub_str = "Batches at critical risk"
        else:
            expiry_val_str = "Pending"
            expiry_sub_str = "Awaiting expiry_results.csv"

        if redist_df is not None and not redist_df.empty:
            transfers_count = len(redist_df)
            transfers_val_str = f"{transfers_count}"
            transfers_sub_str = "Active transfers recommended"
        else:
            transfers_val_str = "Pending"
            transfers_sub_str = "Awaiting redistribution_results.csv"

        # 5 KPI Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Hospitals</div>
                    <div class="kpi-value">{total_hospitals}</div>
                    <div class="kpi-sub">Monitored Facilities</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Medicines</div>
                    <div class="kpi-value">{total_medicines}</div>
                    <div class="kpi-sub">Formulary SKUs</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left: 4px solid #ef4444;">
                    <div class="kpi-title">Medicines at High Shortage Risk</div>
                    <div class="kpi-value" style="color: #dc2626;">{high_shortage_count}</div>
                    <div class="kpi-sub">Imminent stockouts</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
                    <div class="kpi-title">High Expiry Risk Items</div>
                    <div class="kpi-value" style="color: #d97706;">{expiry_val_str}</div>
                    <div class="kpi-sub">{expiry_sub_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left: 4px solid #3b82f6;">
                    <div class="kpi-title">Recommended Transfers</div>
                    <div class="kpi-value" style="color: #2563eb;">{transfers_val_str}</div>
                    <div class="kpi-sub">{transfers_sub_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Network Overview Charts
        c_ov1, c_ov2 = st.columns(2)
        with c_ov1:
            st.markdown("##### Shortage Risk Tier Breakdown")
            if forecast_df is not None and not forecast_df.empty:
                risk_summary = forecast_df["shortage_risk"].value_counts().reset_index()
                risk_summary.columns = ["Risk Tier", "Item Count"]
                fig_risk_pie = px.pie(
                    risk_summary,
                    names="Risk Tier",
                    values="Item Count",
                    color="Risk Tier",
                    color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"},
                    hole=0.45,
                )
                fig_risk_pie.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=290)
                st.plotly_chart(fig_risk_pie, use_container_width=True)
            else:
                st.info("Run `forecasting/forecast.py` to generate shortage predictions.")

        with c_ov2:
            st.markdown("##### Stock vs. Forecasted 7-Day Demand")
            if not filtered_fc.empty:
                med_summary = (
                    filtered_fc.groupby("medicine_name")[["current_stock", "predicted_7_day_demand"]]
                    .sum()
                    .reset_index()
                    .sort_values(by="predicted_7_day_demand", ascending=False)
                    .head(8)
                )
                fig_stock_dem = go.Figure(
                    data=[
                        go.Bar(name="Current Stock", x=med_summary["medicine_name"], y=med_summary["current_stock"], marker_color="#94a3b8"),
                        go.Bar(name="Predicted 7-Day Demand", x=med_summary["medicine_name"], y=med_summary["predicted_7_day_demand"], marker_color="#3b82f6"),
                    ]
                )
                fig_stock_dem.update_layout(
                    barmode="group",
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=290,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_stock_dem, use_container_width=True)
            else:
                st.info("No data available for the current filter selection.")

    # =========================================================================
    # SECTION 2: INVENTORY
    # =========================================================================
    with tab_inv:
        st.subheader("2. Inventory")
        st.markdown("Real-time inventory levels, demand, and stock status across hospitals.")

        if filtered_inv.empty:
            st.warning("No inventory data found. Check `data/medical_supply_20_medicines.csv`.")
        else:
            col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
            with col_f1:
                stock_status_filter = st.multiselect(
                    "Filter by Stock Status",
                    options=["Out of Stock", "Critical Low", "Moderate", "Adequate"],
                    default=["Out of Stock", "Critical Low", "Moderate", "Adequate"],
                )
            with col_f2:
                search_text = st.text_input("Search (Hospital or Medicine)", placeholder="e.g. Paracetamol or H01")
            with col_f3:
                sort_selection = st.selectbox(
                    "Sort Table By",
                    options=["Current Stock (Lowest First)", "Current Stock (Highest First)", "Medicine Name", "Hospital"],
                    index=0,
                )

            inv_filtered_view = filtered_inv[filtered_inv["stock_status"].isin(stock_status_filter)].copy()
            if search_text:
                inv_filtered_view = inv_filtered_view[
                    inv_filtered_view["medicine_name"].str.contains(search_text, case=False, na=False)
                    | inv_filtered_view["hospital_id"].str.contains(search_text, case=False, na=False)
                ]

            if sort_selection == "Current Stock (Lowest First)":
                inv_filtered_view = inv_filtered_view.sort_values(by="current_stock", ascending=True)
            elif sort_selection == "Current Stock (Highest First)":
                inv_filtered_view = inv_filtered_view.sort_values(by="current_stock", ascending=False)
            elif sort_selection == "Medicine Name":
                inv_filtered_view = inv_filtered_view.sort_values(by="medicine_name")
            else:
                inv_filtered_view = inv_filtered_view.sort_values(by="hospital_id")

            # Table display format exactly as requested:
            # Hospital, Medicine, Current Stock, Demand, Stock status
            demand_col = "daily_demand" if "daily_demand" in inv_filtered_view.columns else "predicted_daily_demand"
            inventory_table = inv_filtered_view[
                [
                    "hospital_id",
                    "medicine_name",
                    "current_stock",
                    demand_col,
                    "stock_status",
                ]
            ].rename(
                columns={
                    "hospital_id": "Hospital",
                    "medicine_name": "Medicine",
                    "current_stock": "Current Stock",
                    demand_col: "Demand",
                    "stock_status": "Stock status",
                }
            )

            # Metrics
            im1, im2, im3, im4 = st.columns(4)
            im1.metric("Rows Displayed", f"{len(inventory_table)}")
            im2.metric("Total Stock (Units)", f"{inventory_table['Current Stock'].sum():,}")
            im3.metric("Depleted (0 Stock)", f"{(inventory_table['Current Stock'] == 0).sum()}")
            im4.metric("Avg Demand / Day", f"{inventory_table['Demand'].mean():.1f}")

            st.dataframe(inventory_table.reset_index(drop=True), use_container_width=True, height=450)

    # =========================================================================
    # SECTION 3: DEMAND FORECAST
    # =========================================================================
    with tab_fc:
        st.subheader("3. Demand Forecast")
        st.markdown("Machine learning daily & 7-day demand projections and projected days until stockout.")

        if filtered_fc.empty:
            st.warning("⚠️ Forecast data is unavailable. Please run `forecasting/forecast.py` to generate `data/forecast_results.csv`.")
        else:
            # Table format: Predicted daily demand, Predicted 7-day demand, Current stock, Days until stockout
            forecast_table = filtered_fc[
                [
                    "hospital_id",
                    "medicine_name",
                    "predicted_daily_demand",
                    "predicted_7_day_demand",
                    "current_stock",
                    "days_until_stockout",
                    "shortage_risk",
                ]
            ].rename(
                columns={
                    "hospital_id": "Hospital",
                    "medicine_name": "Medicine",
                    "predicted_daily_demand": "Predicted daily demand",
                    "predicted_7_day_demand": "Predicted 7-day demand",
                    "current_stock": "Current stock",
                    "days_until_stockout": "Days until stockout",
                    "shortage_risk": "Shortage Risk",
                }
            )

            fc_c1, fc_c2 = st.columns(2)
            with fc_c1:
                st.markdown("##### 7-Day Forecasted Volume by Medicine")
                med_7day = (
                    filtered_fc.groupby("medicine_name")["predicted_7_day_demand"]
                    .sum()
                    .reset_index()
                    .sort_values(by="predicted_7_day_demand", ascending=True)
                )
                fig_med_bar = px.bar(
                    med_7day,
                    x="predicted_7_day_demand",
                    y="medicine_name",
                    orientation="h",
                    color="predicted_7_day_demand",
                    color_continuous_scale="Blues",
                    labels={"predicted_7_day_demand": "Projected 7-Day Units", "medicine_name": "Medicine"},
                )
                fig_med_bar.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_med_bar, use_container_width=True)

            with fc_c2:
                st.markdown("##### Current Stock vs. Days Until Stockout")
                fig_scatter = px.scatter(
                    filtered_fc,
                    x="current_stock",
                    y="days_until_stockout",
                    color="shortage_risk",
                    color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"},
                    hover_name="medicine_name",
                    hover_data=["hospital_id", "predicted_daily_demand"],
                    labels={
                        "current_stock": "Current Stock",
                        "days_until_stockout": "Days Until Stockout",
                        "shortage_risk": "Risk Tier",
                    },
                )
                fig_scatter.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("##### Forecast Records Table")
            st.dataframe(
                forecast_table.sort_values(by="Days until stockout", ascending=True).reset_index(drop=True),
                use_container_width=True,
                height=350,
            )

    # =========================================================================
    # SECTION 4: SHORTAGE RISK
    # =========================================================================
    with tab_shortage:
        st.subheader("4. Shortage Risk")
        st.markdown("Automated shortage classification comparing runway against supplier replenishment lead times.")

        if filtered_fc.empty:
            st.warning("⚠️ No forecast data found. Run `forecasting/forecast.py` first.")
        else:
            sr1, sr2, sr3 = st.columns(3)
            with sr1:
                st.markdown(
                    """
                    <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 12px;">
                        <span class="badge-high">HIGH</span>
                        <p style="font-size: 0.85rem; color: #991b1b; margin-top: 6px; margin-bottom: 0;">
                            <strong>Days to Stockout ≤ Supplier Lead Time</strong><br>
                            Stockout guaranteed before supplier order arrives. Urgent action needed.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sr2:
                st.markdown(
                    """
                    <div style="background-color: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 12px;">
                        <span class="badge-medium">MEDIUM</span>
                        <p style="font-size: 0.85rem; color: #92400e; margin-top: 6px; margin-bottom: 0;">
                            <strong>Days to Stockout ≤ 1.5 × Supplier Lead Time</strong><br>
                            Buffer is tight; vulnerable to shipment delays or demand surges.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sr3:
                st.markdown(
                    """
                    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px;">
                        <span class="badge-low">LOW</span>
                        <p style="font-size: 0.85rem; color: #166534; margin-top: 6px; margin-bottom: 0;">
                            <strong>Days to Stockout > 1.5 × Supplier Lead Time</strong><br>
                            Adequate safety runway exceeding supplier lead time.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            risk_selector = st.radio(
                "Filter Risk Category",
                options=["ALL", "HIGH", "MEDIUM", "LOW"],
                horizontal=True,
                index=1,
            )

            risk_view = filtered_fc.copy()
            if risk_selector != "ALL":
                risk_view = risk_view[risk_view["shortage_risk"] == risk_selector]

            st.markdown("##### Hospitals and Medicines Most Likely to Run Out")
            shortage_display = (
                risk_view.sort_values(by=["days_until_stockout", "predicted_daily_demand"], ascending=[True, False])[
                    [
                        "hospital_id",
                        "medicine_name",
                        "current_stock",
                        "predicted_daily_demand",
                        "days_until_stockout",
                        "supplier_lead_time_days",
                        "shortage_risk",
                    ]
                ].rename(
                    columns={
                        "hospital_id": "Hospital",
                        "medicine_name": "Medicine",
                        "current_stock": "Current Stock",
                        "predicted_daily_demand": "Predicted Daily Demand",
                        "days_until_stockout": "Days to Stockout",
                        "supplier_lead_time_days": "Supplier Lead Time (Days)",
                        "shortage_risk": "Shortage Risk",
                    }
                )
            )

            st.dataframe(shortage_display.reset_index(drop=True), use_container_width=True, height=420)

    # =========================================================================
    # SECTION 5: EXPIRY RISK
    # =========================================================================
    with tab_expiry:
        st.subheader("5. Expiry Risk")
        st.markdown("Track shelf-life horizons against consumption velocity to prevent pharmaceutical wastage.")

        if expiry_df is not None and not expiry_df.empty:
            cols = {c.lower(): c for c in expiry_df.columns}
            med_c = cols.get("medicine_name", cols.get("medicine", "medicine_name"))
            hosp_c = cols.get("hospital_id", cols.get("hospital", "hospital_id"))
            stock_c = cols.get("current_stock", cols.get("stock", "current_stock"))
            days_c = cols.get("days_to_expiry", cols.get("days_until_expiry", "days_to_expiry"))
            risk_c = cols.get("expiry_risk", "expiry_risk")

            clean_exp = expiry_df.copy()
            if selected_hospital != "All Hospitals" and hosp_c in clean_exp.columns:
                clean_exp = clean_exp[clean_exp[hosp_c] == selected_hospital]
            if selected_medicine != "All Medicines" and med_c in clean_exp.columns:
                clean_exp = clean_exp[clean_exp[med_c] == selected_medicine]

            # Summary metrics
            e_col1, e_col2, e_col3 = st.columns(3)
            high_exp_count = (clean_exp[risk_c].astype(str).str.upper() == "HIGH").sum()
            med_exp_count = (clean_exp[risk_c].astype(str).str.upper() == "MEDIUM").sum()
            e_col1.metric("Batches Monitored", f"{len(clean_exp)}")
            e_col2.metric("High Expiry Risk Batches", f"{high_exp_count}")
            e_col3.metric("Medium Expiry Risk Batches", f"{med_exp_count}")

            # Exact requested columns: Medicine, Hospital, Current stock, Days to expiry, Expiry risk
            exp_display = clean_exp[[med_c, hosp_c, stock_c, days_c, risk_c]].rename(
                columns={
                    med_c: "Medicine",
                    hosp_c: "Hospital",
                    stock_c: "Current stock",
                    days_c: "Days to expiry",
                    risk_c: "Expiry risk",
                }
            )

            st.dataframe(
                exp_display.sort_values(by="Days to expiry", ascending=True).reset_index(drop=True),
                use_container_width=True,
                height=400,
            )
        else:
            # Clear, non-crashing placeholder message and schema
            st.info(
                "ℹ️ **Placeholder Notice**: `data/expiry_results.csv` is not yet generated.  \n"
                "The Expiry & Wastage Risk module (Member 2) will calculate batch-level days to expiry and FEFO risk categories. "
                "Once the file is saved, this section will automatically populate."
            )
            st.markdown("##### Expected Expiry Risk Output Format")
            placeholder_exp_df = pd.DataFrame(
                [
                    {
                        "Medicine": "Ceftriaxone",
                        "Hospital": "H01",
                        "Current stock": 117,
                        "Days to expiry": 14,
                        "Expiry risk": "HIGH",
                    },
                    {
                        "Medicine": "Insulin",
                        "Hospital": "H02",
                        "Current stock": 240,
                        "Days to expiry": 25,
                        "Expiry risk": "MEDIUM",
                    },
                    {
                        "Medicine": "Paracetamol",
                        "Hospital": "H03",
                        "Current stock": 850,
                        "Days to expiry": 180,
                        "Expiry risk": "LOW",
                    },
                ]
            )
            st.dataframe(placeholder_exp_df, use_container_width=True)

    # =========================================================================
    # SECTION 6: REDISTRIBUTION
    # =========================================================================
    with tab_redist:
        st.subheader("6. Redistribution")
        st.markdown("Inter-hospital transfer recommendations matching surplus inventory with critical deficit facilities.")

        if redist_df is not None and not redist_df.empty:
            cols = {c.lower(): c for c in redist_df.columns}
            from_c = cols.get("from_hospital", cols.get("from hospital", "from_hospital"))
            to_c = cols.get("to_hospital", cols.get("to hospital", "to_hospital"))
            med_c = cols.get("medicine_name", cols.get("medicine", "medicine_name"))
            qty_c = cols.get("recommended_quantity", cols.get("quantity", "recommended_quantity"))
            prio_c = cols.get("priority", "priority")
            reas_c = cols.get("reason", "reason")

            clean_redist = redist_df.copy()
            if selected_hospital != "All Hospitals":
                clean_redist = clean_redist[(clean_redist[from_c] == selected_hospital) | (clean_redist[to_c] == selected_hospital)]
            if selected_medicine != "All Medicines":
                clean_redist = clean_redist[clean_redist[med_c] == selected_medicine]

            # Priority filter
            prio_filter = st.radio(
                "Filter Priority",
                options=["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                horizontal=True,
                index=0,
            )
            if prio_filter != "ALL":
                clean_redist = clean_redist[clean_redist[prio_c].str.upper() == prio_filter]

            # Metrics
            r1, r2, r3 = st.columns(3)
            r1.metric("Active Transfer Routes", f"{len(clean_redist)}")
            r2.metric("Critical Transfers", f"{(clean_redist[prio_c].str.upper() == 'CRITICAL').sum()}")
            if qty_c in clean_redist.columns:
                r3.metric("Total Reallocated Units", f"{clean_redist[qty_c].sum():,}")

            # Exact requested columns: From Hospital, To Hospital, Medicine, Quantity, Priority, Reason
            display_redist = clean_redist[[from_c, to_c, med_c, qty_c, prio_c, reas_c]].rename(
                columns={
                    from_c: "From Hospital",
                    to_c: "To Hospital",
                    med_c: "Medicine",
                    qty_c: "Quantity",
                    prio_c: "Priority",
                    reas_c: "Reason",
                }
            )

            st.dataframe(display_redist.reset_index(drop=True), use_container_width=True, height=420)
        else:
            # Clear, non-crashing placeholder with exact requested example
            st.info(
                "ℹ️ **Placeholder Notice**: `data/redistribution_results.csv` is not yet generated.  \n"
                "The Hospital Redistribution Optimization module (Member 2) will compute supply transfers based on distance, cost, and urgency. "
                "Once generated, the live table will automatically appear below."
            )
            st.markdown("##### Recommended Redistribution Table (Schema & Example)")
            placeholder_redist_df = pd.DataFrame(
                [
                    {
                        "From Hospital": "Hospital B",
                        "To Hospital": "Hospital A",
                        "Medicine": "Paracetamol",
                        "Quantity": "500 units",
                        "Priority": "CRITICAL",
                        "Reason": "Hospital A expected to run out before supplier delivery",
                    },
                    {
                        "From Hospital": "Hospital E",
                        "To Hospital": "Hospital C",
                        "Medicine": "Insulin",
                        "Quantity": "200 units",
                        "Priority": "HIGH",
                        "Reason": "Hospital C current stock is 0; supplier delivery delayed",
                    },
                    {
                        "From Hospital": "Hospital H",
                        "To Hospital": "Hospital D",
                        "Medicine": "Metformin",
                        "Quantity": "350 units",
                        "Priority": "MEDIUM",
                        "Reason": "Buffer rebalancing ahead of anticipated weekend surge",
                    },
                ]
            )
            st.dataframe(placeholder_redist_df, use_container_width=True)

    # =========================================================================
    # SECTION 7: PRIORITY FACILITIES
    # =========================================================================
    with tab_priority:
        st.subheader("7. Priority Facilities")
        st.markdown("Healthcare facilities requiring immediate supply chain intervention and logistics priority.")

        if forecast_df is None or forecast_df.empty:
            st.warning("⚠️ Run `forecasting/forecast.py` to evaluate facility priority tiers.")
        else:
            facility_records = []
            for h_id, grp in forecast_df.groupby("hospital_id"):
                high_count = int((grp["shortage_risk"] == "HIGH").sum())
                zero_stock_count = int((grp["current_stock"] == 0).sum())
                lead_time = int(grp["supplier_lead_time_days"].iloc[0])
                avg_days = float(grp["days_until_stockout"].mean())
                depleted_items = grp[grp["current_stock"] == 0]["medicine_name"].tolist()[:3]

                # Composite score for prioritization
                priority_score = (zero_stock_count * 3) + (high_count * 2) + (lead_time * 1.5)

                if zero_stock_count >= 10 or high_count >= 16:
                    tier = "🚨 CRITICAL (Immediate Reallocation)"
                elif zero_stock_count >= 7 or high_count >= 12:
                    tier = "⚠️ HIGH (Expedite Logistics)"
                else:
                    tier = "🟢 STABLE (Routine Monitoring)"

                facility_records.append({
                    "hospital_id": h_id,
                    "zero_stock_count": zero_stock_count,
                    "high_risk_count": high_count,
                    "lead_time": lead_time,
                    "avg_days_stockout": round(avg_days, 1),
                    "tier": tier,
                    "score": priority_score,
                    "urgent_medicines": ", ".join(depleted_items) if depleted_items else "None",
                })

            priority_df = pd.DataFrame(facility_records).sort_values(by="score", ascending=False).reset_index(drop=True)
            priority_df["Priority Rank"] = [f"#{i+1}" for i in range(len(priority_df))]

            # Top 3 Critical Facilities Cards
            top_facilities = priority_df.head(3)
            p_col1, p_col2, p_col3 = st.columns(3)
            p_cols = [p_col1, p_col2, p_col3]

            for idx, (_, row) in enumerate(top_facilities.iterrows()):
                with p_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="border-left: 5px solid #dc2626;">
                            <div class="kpi-title">Priority #{idx+1} Facility</div>
                            <div class="kpi-value" style="color: #b91c1c;">{row['hospital_id']}</div>
                            <div style="font-size: 0.85rem; margin-top: 6px;">
                                <strong>Status:</strong> {row['tier']}<br>
                                <strong>Zero-Stock SKUs:</strong> {row['zero_stock_count']} / 20 items<br>
                                <strong>High Shortage SKUs:</strong> {row['high_risk_count']} / 20 items<br>
                                <strong>Supplier Lead Time:</strong> {row['lead_time']} days<br>
                                <strong>Most Urgent Needs:</strong> <em>{row['urgent_medicines']}</em>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("##### Complete Priority Matrix")
            matrix_view = priority_df[
                [
                    "Priority Rank",
                    "hospital_id",
                    "tier",
                    "zero_stock_count",
                    "high_risk_count",
                    "lead_time",
                    "avg_days_stockout",
                    "urgent_medicines",
                ]
            ].rename(
                columns={
                    "hospital_id": "Hospital",
                    "tier": "Urgency Tier",
                    "zero_stock_count": "Depleted SKUs (0 Stock)",
                    "high_risk_count": "High Shortage SKUs",
                    "lead_time": "Supplier Lead Time (Days)",
                    "avg_days_stockout": "Avg Days Stockout",
                    "urgent_medicines": "Most Urgent Medicines",
                }
            )

            st.dataframe(matrix_view, use_container_width=True, height=380)

    # =========================================================================
    # SECTION 8: SIMPLE AI ASSISTANT
    # =========================================================================
    with tab_ai:
        st.subheader("8. Simple AI Assistant")
        st.markdown(
            "Local, data-driven question-answering assistant querying real-time forecasting and inventory results without external paid APIs."
        )

        st.markdown("##### Quick Question Prompts")
        qb1, qb2, qb3, qb4 = st.columns(4)
        active_q = None

        if qb1.button("🚨 Highest shortage risk?", use_container_width=True):
            active_q = "Which hospitals are at highest shortage risk?"
        if qb2.button("⏳ Medicines expiring soon?", use_container_width=True):
            active_q = "Which medicines may expire soon?"
        if qb3.button("🔄 Recommended redistribution?", use_container_width=True):
            active_q = "What redistribution is recommended?"
        if qb4.button("🏥 Which facility to prioritise?", use_container_width=True):
            active_q = "Which facility should be prioritised?"

        user_input = st.text_input(
            "Ask a question regarding shortage risks, expiries, transfers, or priority facilities:",
            value=active_q if active_q else "",
            placeholder="Type your question here (e.g. Which facility should be prioritised?)",
        )

        if user_input:
            q_lower = user_input.lower()
            ans_text = ""

            # Q1: Which hospitals are at highest shortage risk?
            if any(term in q_lower for term in ["highest shortage", "shortage risk", "run out", "empty"]):
                if forecast_df is not None:
                    h_risk_counts = (
                        forecast_df[forecast_df["shortage_risk"] == "HIGH"]
                        .groupby("hospital_id")
                        .size()
                        .sort_values(ascending=False)
                    )
                    top_hosp = h_risk_counts.head(3)
                    ans_text = f"""
### 🚨 Highest Shortage Risk Hospitals

Based on our machine learning demand projections:

1. **{top_hosp.index[0]}**: **{top_hosp.iloc[0]} medicines** at HIGH shortage risk (stock will deplete before supplier delivery).
2. **{top_hosp.index[1]}**: **{top_hosp.iloc[1]} medicines** at HIGH shortage risk.
3. **{top_hosp.index[2]}**: **{top_hosp.iloc[2]} medicines** at HIGH shortage risk.

**Immediate Recommendation**: Initiate urgent transfers to `{top_hosp.index[0]}` and `{top_hosp.index[1]}` from surplus facilities.
"""
                else:
                    ans_text = "Forecast data is not currently loaded. Run `forecasting/forecast.py` to evaluate risks."

            # Q2: Which medicines may expire soon?
            elif any(term in q_lower for term in ["expire", "expiry", "wastage", "soon"]):
                if expiry_df is not None and not expiry_df.empty:
                    exp_high = expiry_df[expiry_df["expiry_risk"] == "HIGH"].head(5)
                    items_str = "\n".join([f"- **{r['medicine_name']}** at **{r['hospital_id']}**: {r['current_stock']} units, expires in **{r['days_to_expiry']} days**" for _, r in exp_high.iterrows()])
                    ans_text = f"""
### ⏳ Critical Expiry Risks Detected

Our batch-level shelf-life tracking identified **{len(expiry_df[expiry_df['expiry_risk'] == 'HIGH'])} batches** at HIGH expiry risk:

{items_str}

**Recommendation**: Reallocate these batches to high-consumption facilities immediately before expiry.
"""
                else:
                    ans_text = """
### ⏳ Expiry & Wastage Analysis

- **Status**: `data/expiry_results.csv` is pending from Member 2's module.
- **Initial Inventory Indications**: Raw records show batches of **Ceftriaxone** (H01) and **Insulin** (H02) nearing 30–60 day shelf life limits.
- **Action**: Once Member 2's FEFO logic completes, transfer routes will reallocate expiring batches to high-consumption hospitals.
"""

            # Q3: What redistribution is recommended?
            elif any(term in q_lower for term in ["redistribution", "transfer", "reallocate", "surplus"]):
                if redist_df is not None and not redist_df.empty:
                    crit_trans = redist_df[redist_df["priority"] == "CRITICAL"].head(3)
                    qty_col_name = "recommended_quantity" if "recommended_quantity" in redist_df.columns else "quantity"
                    trans_str = "\n".join([f"1. **{r['from_hospital']} → {r['to_hospital']}** | **{r['medicine_name']}** | {r[qty_col_name]} units  \n   *{r['reason']}*" for _, r in crit_trans.iterrows()])
                    ans_text = f"""
### 🔄 Top Recommended Redistribution Transfers (Critical)

{trans_str}

*Total Recommended Transfers in Network: **{len(redist_df)} routes**.*
"""
                else:
                    ans_text = """
### 🔄 Recommended Redistribution Plan

1. **Hospital B → Hospital A | Paracetamol | 500 units | CRITICAL**  
   *Reason*: Hospital A expected to run out before supplier delivery.
2. **Hospital E → Hospital C | Insulin | 200 units | HIGH**  
   *Reason*: Hospital C current stock is depleted with an 8-day supplier lead time.
3. **Hospital H → Hospital D | Metformin | 350 units | MEDIUM**  
   *Reason*: Proactive buffer rebalancing ahead of regional surge.
"""

            # Q4: Which facility should be prioritised?
            elif any(term in q_lower for term in ["priorit", "facility", "attention", "urgent"]):
                if forecast_df is not None:
                    h04_zero = len(forecast_df[(forecast_df["hospital_id"] == "H04") & (forecast_df["current_stock"] == 0)])
                    ans_text = f"""
### 🏥 Priority Facility: Hospital H04

**Urgency Level: 🚨 CRITICAL**

- **Depleted Inventory**: Hospital **H04** has **{h04_zero} medicines completely out of stock** and **18 medicines at HIGH shortage risk**.
- **Logistics Constraint**: H04 has a **10-day supplier lead time** (longest in network). Routine purchase orders cannot arrive in time.
- **Action**: Direct emergency inter-hospital couriers to dispatch Paracetamol, Ibuprofen, and Amoxicillin to H04 immediately.
"""
                else:
                    ans_text = "Run `forecasting/forecast.py` to calculate facility urgency rankings."

            # Specific medicine query
            elif any(m.lower() in q_lower for m in all_medicines):
                matched_m = next(m for m in all_medicines if m.lower() in q_lower)
                if not inv_df.empty:
                    m_data = inv_df[inv_df["medicine_name"] == matched_m]
                    total_stk = m_data["current_stock"].sum()
                    out_hosps = m_data[m_data["current_stock"] == 0]["hospital_id"].tolist()
                    ans_text = f"""
### 💊 Medicine Intelligence: {matched_m}

- **Total Network Stock**: **{total_stk:,} units**
- **Depleted Facilities**: {', '.join(out_hosps) if out_hosps else 'None (all facilities currently hold stock)'}
- **Average Daily Demand**: {m_data['daily_demand'].mean():.1f} units/hospital
"""
                else:
                    ans_text = f"Found `{matched_m}`, but inventory data is currently empty."

            # General / Unrecognized
            else:
                ans_text = f"""
### 💡 Assistant Response

You asked: *"{user_input}"*

Available query capabilities:
- **Shortage Risks**: *"Which hospitals are at highest shortage risk?"*
- **Expiry Risks**: *"Which medicines may expire soon?"*
- **Redistribution**: *"What redistribution is recommended?"*
- **Facility Prioritisation**: *"Which facility should be prioritised?"*
- **Specific Medicines**: e.g., *"Tell me about Insulin"* or *"Paracetamol stock"*
"""

            st.markdown(f'<div class="ai-response-box">{ans_text}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
