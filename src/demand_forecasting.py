"""
demand_forecasting.py
---------------------
Demand forecasting using XGBoost with time series features.
Includes model training, evaluation, and visualisation.

Author: Shalmalee Sharma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer time series features from date column."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"]   = df["date"].dt.dayofweek
    df["day_of_month"]  = df["date"].dt.day
    df["month"]         = df["date"].dt.month
    df["quarter"]       = df["date"].dt.quarter
    df["year"]          = df["date"].dt.year
    df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"]  = df["date"].dt.is_month_end.astype(int)
    df["is_quarter_end"]= df["date"].dt.is_quarter_end.astype(int)

    # Lag features (previous demand signals)
    df = df.sort_values("date")
    df["demand_lag_7"]  = df["demand"].shift(7)
    df["demand_lag_14"] = df["demand"].shift(14)
    df["demand_lag_28"] = df["demand"].shift(28)
    df["demand_roll_7"] = df["demand"].shift(1).rolling(7).mean()
    df["demand_roll_28"]= df["demand"].shift(1).rolling(28).mean()

    return df.dropna()


FEATURES = [
    "price", "is_promoted", "day_of_week", "day_of_month", "month",
    "quarter", "year", "week_of_year", "is_weekend", "is_month_end",
    "demand_lag_7", "demand_lag_14", "demand_lag_28",
    "demand_roll_7", "demand_roll_28"
]


def train_xgboost_model(df: pd.DataFrame, sku: str) -> tuple:
    """
    Train XGBoost demand forecasting model for a single SKU.

    Returns:
        (model, test_df with predictions, metrics dict)
    """
    sku_df = df[df["sku"] == sku].copy()
    sku_df = create_time_features(sku_df)

    # Time-based train/test split (last 90 days = test)
    split_date = sku_df["date"].max() - pd.Timedelta(days=90)
    train = sku_df[sku_df["date"] <= split_date]
    test  = sku_df[sku_df["date"] >  split_date]

    X_train, y_train = train[FEATURES], train["demand"]
    X_test,  y_test  = test[FEATURES],  test["demand"]

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="rmse", early_stopping_rounds=20,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    test = test.copy()
    test["predicted_demand"] = model.predict(X_test).round(0)

    metrics = {
        "sku":   sku,
        "mape":  round(mean_absolute_percentage_error(y_test, test["predicted_demand"]) * 100, 2),
        "rmse":  round(np.sqrt(mean_squared_error(y_test, test["predicted_demand"])), 1),
        "n_train": len(train),
        "n_test":  len(test),
    }
    logger.info(f"SKU {sku} | MAPE: {metrics['mape']}% | RMSE: {metrics['rmse']}")
    return model, test, metrics


def plot_forecast(test_df: pd.DataFrame, sku: str, save_path: str = None):
    """Plot actual vs predicted demand for a SKU."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"Demand Forecast — {sku}", fontsize=13, fontweight="bold")

    # Actual vs Predicted
    axes[0].plot(test_df["date"], test_df["demand"],
                 label="Actual", color="#3498DB", linewidth=2)
    axes[0].plot(test_df["date"], test_df["predicted_demand"],
                 label="Predicted", color="#E74C3C", linewidth=2, linestyle="--")
    axes[0].fill_between(test_df["date"],
                         test_df["predicted_demand"] * 0.9,
                         test_df["predicted_demand"] * 1.1,
                         alpha=0.2, color="#E74C3C", label="±10% band")
    axes[0].set_title("Actual vs Predicted Demand (Test Period)")
    axes[0].set_ylabel("Units")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = test_df["demand"] - test_df["predicted_demand"]
    axes[1].bar(test_df["date"], residuals,
                color=np.where(residuals >= 0, "#2ECC71", "#E74C3C"), alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Forecast Residuals")
    axes[1].set_ylabel("Actual − Predicted")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model: XGBRegressor, save_path: str = None):
    """Plot XGBoost feature importance."""
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colours = ["#3498DB" if i >= len(importance) - 5 else "#BDC3C7"
               for i in range(len(importance))]
    importance.plot(kind="barh", ax=ax, color=colours)
    ax.set_title("XGBoost Feature Importance — Demand Forecasting",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
