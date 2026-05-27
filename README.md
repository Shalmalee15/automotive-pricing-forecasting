# 🚗 Automotive Parts - Pricing & Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?logo=xgboost&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-Time%20Series-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **End-to-end demand forecasting and dynamic pricing pipeline for automotive aftermarket parts - combining time series modelling, price elasticity analysis, and XGBoost-based pricing optimisation.**

---

##  Overview

Automotive aftermarket businesses carry thousands of SKUs across diverse categories. Pricing them correctly - and forecasting demand accurately - directly impacts revenue, inventory efficiency, and customer satisfaction.

This project builds a production-ready pipeline that:
1. **Forecasts demand** at the SKU level using Facebook Prophet and XGBoost
2. **Models price elasticity** to understand how demand responds to price changes
3. **Recommends optimal prices** that maximise revenue while remaining competitive
4. **Detects demand anomalies** for supply chain early warning

---

##  Business Problem

Automotive parts retailers face three pricing challenges:
- **Demand volatility** - seasonal patterns, vehicle age cycles, and promotions create complex demand signals
- **Price sensitivity varies by category** - consumables (oil filters, brake pads) behave differently from discretionary accessories
- **Competitor pricing pressure** - static pricing leaves revenue on the table

This pipeline addresses all three with a data-driven, ML-powered approach.

---

##  Solution Architecture

```
Raw Sales Data
      │
      ▼
┌─────────────────┐
│ Data Processing │  ← Clean, aggregate, feature engineer
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────────┐
│Prophet │ │ XGBoost Regressor│
│(Trend/ │ │(Price elasticity,│
│Season) │ │ demand drivers)  │
└───┬────┘ └────────┬─────────┘
    │               │
    └──────┬────────┘
           ▼
  ┌─────────────────────┐
  │ Pricing Optimiser   │  ← Revenue maximisation
  └─────────────────────┘
           │
           ▼
  ┌─────────────────────┐
  │ Power BI / Plotly   │  ← Business dashboards
  │ Dashboard           │
  └─────────────────────┘
```

---

##  Key Results

| Metric | Baseline | Model |
|--------|----------|-------|
| Demand Forecast MAPE | 18.3% | **6.2%** |
| Price Elasticity R² | - | **0.84** |
| Revenue Uplift (simulated) | - | **+12.4%** |
| Anomaly Detection Precision | - | **91%** |

---

##  Tools & Technologies

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.9+ |
| **Time Series** | Facebook Prophet, statsmodels |
| **ML Models** | XGBoost, Scikit-learn |
| **Optimisation** | SciPy (revenue maximisation) |
| **Visualisation** | Plotly, Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |
| **Environment** | Jupyter Notebook |

---

##  Repository Structure

```
automotive-pricing-forecasting/
│
├── notebooks/
│   └── pricing_demand_forecasting.ipynb   # Full end-to-end analysis
│
├── src/
│   ├── data_generator.py                  # Synthetic data generation
│   ├── feature_engineering.py             # Time series features
│   ├── demand_forecasting.py              # Prophet + XGBoost models
│   ├── price_elasticity.py               # Elasticity modelling
│   └── pricing_optimiser.py              # Revenue optimisation
│
├── outputs/                               # Generated charts and reports
├── requirements.txt
├── LICENSE
└── README.md
```

---

##  How to Run

```bash
# Clone the repository
git clone https://github.com/Shalmalee15/automotive-pricing-forecasting.git
cd automotive-pricing-forecasting

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebooks/pricing_demand_forecasting.ipynb
```

---

##  Future Work

- [ ] Real-time competitor price scraping and benchmarking
- [ ] Deploy as REST API on AWS SageMaker
- [ ] Integrate with live POS/ERP data feeds
- [ ] Add reinforcement learning for continuous pricing optimisation
- [ ] Build interactive Power BI dashboard for category managers

---

##  Author

**Shalmalee Sharma** - PhD Astrophysics | Senior Data Scientist  
📍 Melbourne, Australia  
🔗 [LinkedIn](https://linkedin.com/in/shalmalee-kapse) · [GitHub](https://github.com/Shalmalee15)

---

##  License

This project is licensed under the [MIT License](LICENSE).
