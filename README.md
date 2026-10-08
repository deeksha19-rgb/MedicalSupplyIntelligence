# Medical Supply Intelligence

A collaborative intelligence system designed to optimize healthcare inventory, predict critical supply shortages, minimize expiry wastage, and enable smart hospital-to-hospital stock redistribution.

---

## Project Overview

In healthcare networks, inefficient supply allocation often leads to simultaneous supply shortages at one hospital and costly wastage from expired inventory at another. **Medical Supply Intelligence** addresses this imbalance with an end-to-end data-driven pipeline:

1. **Demand Forecasting & Shortage Prediction**: Proactively predicting item consumption and alerting on impending stockouts before supplies deplete.
2. **Expiry & Wastage Risk Detection**: Tracking batch-level shelf lives against consumption rates to intercept expiring supplies.
3. **Hospital-to-Hospital Redistribution**: Matching surplus facilities with deficit facilities to recommend cost- and time-effective stock transfers.
4. **Critical Supply Prioritisation**: Prioritizing life-saving and critical medical supplies across forecasting and transfer pipelines.
5. **Interactive Dashboard**: A unified control tower displaying live inventory levels, forecasted demand, risk alerts, and transfer recommendations.

---

## Project Structure

```text
medical-supply-intelligence/
│
├── data/
│   └── README.md
│
├── forecasting/
│   └── README.md
│
├── redistribution/
│   └── README.md
│
├── expiry/
│   └── README.md
│
├── dashboard/
│   └── README.md
│
├── models/
│   └── README.md
│
├── requirements.txt
└── README.md
```

---

## Team Division (3-Member Hackathon Team)

| Member | Focus Area | Primary Directories | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Data & Demand Forecasting** | `data/`, `forecasting/`, `models/` | Data schemas/simulation, demand forecasting models, and shortage risk threshold logic. |
| **Member 2** | **Expiry Risk & Redistribution Logic** | `expiry/`, `redistribution/` | Expiry risk scoring (FEFO), critical supply weighting, and inter-hospital transfer optimization engine. |
| **Member 3** | **Dashboard & System Integration** | `dashboard/`, Root integration | Interactive Streamlit/UI dashboard, metric cards, transfer recommendation interface, and end-to-end demo flow. |

---

## Setup & Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/MedicalSupplyIntelligence.git
   cd MedicalSupplyIntelligence
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```