# Dashboard Module

This module houses the interactive **Medical Supply Intelligence Dashboard** (e.g., Streamlit application) designed for hospital supply chain managers and network coordinators.

## Planned Views & Features

1. **Executive Overview**:
   - Total network inventory, active critical alerts, near-term shortage count, and wastage risk summary.
2. **Inventory & Demand Forecasts**:
   - Interactive charts of historical vs. predicted demand across hospitals and item categories.
   - Days-of-supply remaining per item.
3. **Shortage & Expiry Risk Tracker**:
   - Ranked list of high-risk items and facilities with real-time risk indicators and countdowns to stockout/expiry.
4. **Redistribution Action Center**:
   - Interactive recommendation cards displaying suggested hospital-to-hospital transfers with accept/reject or export options.
   - Network transfer map / distance matrix visualization.

## Running the Dashboard

Once the application code is added, run:
```bash
streamlit run app.py
```
*(or the designated dashboard entry point)*
