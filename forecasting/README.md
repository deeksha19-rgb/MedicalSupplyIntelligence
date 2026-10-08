# Forecasting Module

This module handles **demand forecasting** and **shortage risk prediction** across the hospital network.

## Planned Responsibilities

1. **Demand Forecasting**:
   - Forecast supply consumption at hospital and item levels over defined horizons (e.g., 7-day, 14-day, 30-day).
   - Capture temporal patterns, seasonality, and consumption spikes.
2. **Shortage Risk Prediction**:
   - Compare projected demand against current inventory and replenishment lead times.
   - Calculate run-out dates and flag critical shortage probabilities before supplies deplete.

## Inputs & Outputs

- **Inputs**: Historical consumption logs, current stock levels, replenishment lead times.
- **Outputs**: Forecasted demand series, projected days-of-supply remaining, and binary/probability shortage risk alerts.
