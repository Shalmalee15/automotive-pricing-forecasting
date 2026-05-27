"""
data_generator.py
-----------------
Generates realistic synthetic automotive parts sales data
for demand forecasting and pricing analysis.

Author: Shalmalee Sharma
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Product categories with realistic pricing and demand profiles
CATEGORIES = {
    "Brake Pads":       {"base_price": 85,  "base_demand": 120, "elasticity": -1.8, "seasonality": "winter"},
    "Oil Filters":      {"base_price": 18,  "base_demand": 350, "elasticity": -1.2, "seasonality": "flat"},
    "Air Filters":      {"base_price": 25,  "base_demand": 200, "elasticity": -1.4, "seasonality": "spring"},
    "Spark Plugs":      {"base_price": 45,  "base_demand": 180, "elasticity": -1.6, "seasonality": "flat"},
    "Wiper Blades":     {"base_price": 30,  "base_demand": 160, "elasticity": -2.1, "seasonality": "winter"},
    "Shock Absorbers":  {"base_price": 220, "base_demand": 40,  "elasticity": -0.9, "seasonality": "flat"},
    "Tyres":            {"base_price": 180, "base_demand": 80,  "elasticity": -1.1, "seasonality": "winter"},
    "Car Batteries":    {"base_price": 150, "base_demand": 65,  "elasticity": -0.8, "seasonality": "winter"},
}

SKUS = {cat: [f"{cat[:3].upper()}-{str(i).zfill(4)}" for i in range(1, 6)]
        for cat in CATEGORIES}


def generate_seasonality(date: pd.Timestamp, season_type: str) -> float:
    """Generate seasonal multiplier based on month and season type."""
    month = date.month
    if season_type == "winter":
        # Peak in June-August (Australian winter)
        return 1.0 + 0.4 * np.sin((month - 6) * np.pi / 6 + np.pi / 2) * -1 + 0.2
    elif season_type == "spring":
        # Peak in September-November
        return 1.0 + 0.3 * np.sin((month - 9) * np.pi / 6 + np.pi / 2)
    else:
        return 1.0


def generate_price_series(base_price: float, n_days: int, volatility: float = 0.05) -> np.ndarray:
    """Generate realistic price series with occasional promotions."""
    prices = np.full(n_days, base_price, dtype=float)
    # Add promotional periods (10% of days have discounts)
    promo_days = np.random.choice(n_days, size=int(n_days * 0.1), replace=False)
    for day in promo_days:
        promo_length = np.random.randint(3, 14)
        end = min(day + promo_length, n_days)
        discount = np.random.uniform(0.10, 0.25)
        prices[day:end] *= (1 - discount)
    # Add small daily noise
    prices *= (1 + np.random.normal(0, volatility, n_days))
    return prices.round(2)


def generate_demand(prices: np.ndarray, base_demand: float, elasticity: float,
                    base_price: float, dates: pd.DatetimeIndex, season_type: str) -> np.ndarray:
    """
    Generate demand using price elasticity model:
    demand = base_demand * (price / base_price) ^ elasticity
    with seasonal adjustments and noise.
    """
    seasonal_multipliers = np.array([generate_seasonality(d, season_type) for d in dates])
    # Price elasticity effect
    price_effect = (prices / base_price) ** elasticity
    # Weekend boost
    weekend_boost = np.where(dates.dayofweek >= 5, 1.15, 1.0)
    # Trend (slight growth over time)
    trend = 1 + np.linspace(0, 0.1, len(dates))
    # Combine all effects
    demand = base_demand * price_effect * seasonal_multipliers * weekend_boost * trend
    # Add noise
    demand *= (1 + np.random.normal(0, 0.08, len(dates)))
    # Add occasional demand spikes (supply disruptions, promotions)
    spike_days = np.random.choice(len(dates), size=int(len(dates) * 0.02), replace=False)
    demand[spike_days] *= np.random.uniform(1.5, 2.5, len(spike_days))
    return np.maximum(demand, 0).round(0).astype(int)


def generate_dataset(start_date: str = "2021-01-01",
                     end_date: str = "2024-12-31",
                     output_path: str = "../data/automotive_sales.csv") -> pd.DataFrame:
    """
    Generate full synthetic automotive parts sales dataset.

    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        output_path: Path to save CSV

    Returns:
        DataFrame with daily sales data per SKU
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    records = []
    logger.info(f"Generating {n_days} days of data for {sum(len(v) for v in SKUS.values())} SKUs...")

    for category, params in CATEGORIES.items():
        for sku in SKUS[category]:
            prices = generate_price_series(params["base_price"], n_days)
            demands = generate_demand(
                prices, params["base_demand"], params["elasticity"],
                params["base_price"], dates, params["seasonality"]
            )
            for i, date in enumerate(dates):
                records.append({
                    "date":          date,
                    "sku":           sku,
                    "category":      category,
                    "price":         prices[i],
                    "base_price":    params["base_price"],
                    "demand":        demands[i],
                    "revenue":       round(prices[i] * demands[i], 2),
                    "elasticity":    params["elasticity"],
                    "is_promoted":   int(prices[i] < params["base_price"] * 0.95),
                    "day_of_week":   date.dayofweek,
                    "month":         date.month,
                    "quarter":       date.quarter,
                    "year":          date.year,
                })

    df = pd.DataFrame(records)
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Shape: {df.shape} | Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    logger.info(f"Total revenue: ${df['revenue'].sum():,.0f}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.groupby("category")[["price", "demand", "revenue"]].mean().round(2))
