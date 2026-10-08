# Data Directory

This directory will store all raw, simulated, and processed datasets for the **Medical Supply Intelligence** platform.

## Planned Data Structure

- `raw/`: Original hospital records, supply inventories, and demand logs.
- `processed/`: Cleaned, merged, and feature-engineered datasets ready for model training and dashboard consumption.

## Key Datasets & Expected Schemas

1. **Hospital Inventory & Stock Records**:
   - `hospital_id`, `item_id`, `item_name`, `category`, `current_stock`, `reorder_threshold`, `unit_cost`, `is_critical`
2. **Historical Consumption & Demand Logs**:
   - `date`, `hospital_id`, `item_id`, `units_consumed`, `department`
3. **Batch Expiry Records**:
   - `batch_id`, `hospital_id`, `item_id`, `expiry_date`, `batch_quantity`
4. **Hospital Network & Logistics**:
   - `hospital_a_id`, `hospital_b_id`, `distance_km`, `estimated_transit_hours`, `transfer_cost_per_unit`

> **Note**: Do not generate or modify datasets until team alignment on final schema is complete.
