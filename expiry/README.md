# Expiry & Wastage Risk Module

This module detects **expiry and wastage risks** and prioritizes **critical medical supplies** before they expire.

## Planned Responsibilities

1. **Expiry & Wastage Risk Detection**:
   - Track batch-level expiry timelines against projected consumption velocity (FEFO — First-Expired, First-Out).
   - Flag batches at risk of expiring before being consumed at the current facility.
   - Calculate potential financial and clinical loss from projected wastage.
2. **Critical Supply Prioritisation**:
   - Stratify supplies by clinical criticality (e.g., life-saving medicines vs. routine consumables).
   - Feed high-risk expiry alerts into the redistribution engine to reallocate expiring stock to high-demand hospitals.

## Inputs & Outputs

- **Inputs**: Batch-level inventory data (quantities, expiry dates), forecasted consumption rates, item criticality ratings.
- **Outputs**: Expiry risk index, estimated units at risk of expiration, recommended actions (expedite usage / flag for redistribution).
