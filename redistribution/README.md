# Redistribution Module

This module generates **hospital-to-hospital redistribution recommendations** to balance surplus inventory and eliminate urgent shortages.

## Planned Responsibilities

1. **Surplus & Deficit Detection**:
   - Identify hospitals with excess inventory (low near-term consumption relative to stock).
   - Match surplus facilities against hospitals facing impending stockouts or high-risk shortages.
2. **Transfer Optimization & Routing**:
   - Evaluate multi-criteria trade-offs: geographic proximity, transit time, transfer cost, and urgency.
   - Prioritize critical life-saving items (e.g., blood supplies, critical pharmaceuticals, ICU disposables).
3. **Actionable Transfer Plans**:
   - Generate transfer orders containing source hospital, destination hospital, recommended transfer quantity, and priority rank.

## Inputs & Outputs

- **Inputs**: Current inventories, forecasted shortage risk, batch expiry dates, inter-hospital distance/transit matrix.
- **Outputs**: Prioritized redistribution recommendations with quantity, estimated cost, and risk reduction score.
