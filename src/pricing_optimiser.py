"""
pricing_optimiser.py
--------------------
Price elasticity modelling and revenue optimisation
for automotive parts pricing strategy.

Author: Shalmalee Sharma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def estimate_elasticity(df: pd.DataFrame, category: str) -> dict:
    """
    Estimate price elasticity for a product category using log-log regression.

    ln(demand) = α + ε * ln(price) + ε = price elasticity

    Returns:
        dict with elasticity, R², and regression model
    """
    cat_df = df[df["category"] == category].copy()
    cat_df = cat_df[(cat_df["price"] > 0) & (cat_df["demand"] > 0)]

    X = np.log(cat_df["price"]).values.reshape(-1, 1)
    y = np.log(cat_df["demand"]).values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    elasticity = model.coef_[0]
    r2 = r2_score(y, y_pred)

    logger.info(f"{category} | Elasticity: {elasticity:.3f} | R²: {r2:.3f}")

    return {
        "category":   category,
        "elasticity": round(elasticity, 3),
        "r2":         round(r2, 3),
        "model":      model,
        "interpretation": "elastic" if abs(elasticity) > 1 else "inelastic"
    }


def plot_elasticity_curves(df: pd.DataFrame, categories: list, save_path: str = None):
    """Plot price vs demand curves for multiple categories."""
    n = len(categories)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    fig.suptitle("Price Elasticity Curves by Category", fontsize=13, fontweight="bold")

    colours = ["#3498DB", "#2ECC71", "#E74C3C", "#9B59B6",
               "#F39C12", "#1ABC9C", "#E67E22", "#95A5A6"]

    for i, (ax, category) in enumerate(zip(axes, categories)):
        cat_df = df[df["category"] == category]
        result = estimate_elasticity(df, category)

        ax.scatter(cat_df["price"], cat_df["demand"],
                   alpha=0.2, s=8, color=colours[i % len(colours)])

        price_range = np.linspace(cat_df["price"].min(), cat_df["price"].max(), 100)
        log_demand  = result["model"].predict(np.log(price_range).reshape(-1, 1))
        ax.plot(price_range, np.exp(log_demand),
                color=colours[i % len(colours)], linewidth=2.5, label="Elasticity curve")

        ax.set_title(f"{category}\nε={result['elasticity']} | R²={result['r2']}", fontsize=10)
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Demand (units)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def optimise_price(base_price: float, elasticity: float,
                   base_demand: float, margin_floor: float = 0.15,
                   price_bounds: tuple = (0.70, 1.35)) -> dict:
    """
    Find revenue-maximising price using scipy optimisation.

    Revenue(p) = p * demand(p)
    demand(p)  = base_demand * (p / base_price)^elasticity

    Args:
        base_price:   Current reference price
        elasticity:   Estimated price elasticity
        base_demand:  Expected demand at base price
        margin_floor: Minimum margin constraint (e.g. 0.15 = 15% above cost)
        price_bounds: Multiplier bounds on base price

    Returns:
        dict with optimal price, revenue, and uplift vs base
    """
    def neg_revenue(price_multiplier):
        price  = base_price * price_multiplier
        demand = base_demand * (price_multiplier ** elasticity)
        return -(price * demand)

    result = minimize_scalar(
        neg_revenue,
        bounds=price_bounds,
        method="bounded"
    )

    optimal_multiplier = result.x
    optimal_price      = round(base_price * optimal_multiplier, 2)
    optimal_demand     = base_demand * (optimal_multiplier ** elasticity)
    optimal_revenue    = optimal_price * optimal_demand
    base_revenue       = base_price * base_demand
    revenue_uplift     = (optimal_revenue - base_revenue) / base_revenue * 100

    return {
        "base_price":      base_price,
        "optimal_price":   optimal_price,
        "price_change_%":  round((optimal_multiplier - 1) * 100, 1),
        "base_revenue":    round(base_revenue, 2),
        "optimal_revenue": round(optimal_revenue, 2),
        "revenue_uplift_%":round(revenue_uplift, 1),
    }


def generate_pricing_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate pricing recommendations for all categories.
    """
    results = []
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        elasticity_result = estimate_elasticity(df, category)

        base_price  = cat_df["base_price"].iloc[0]
        base_demand = cat_df["demand"].mean()

        rec = optimise_price(base_price, elasticity_result["elasticity"], base_demand)
        rec["category"]      = category
        rec["elasticity"]    = elasticity_result["elasticity"]
        rec["interpretation"]= elasticity_result["interpretation"]
        results.append(rec)

    recommendations = pd.DataFrame(results)[[
        "category", "elasticity", "interpretation",
        "base_price", "optimal_price", "price_change_%",
        "base_revenue", "optimal_revenue", "revenue_uplift_%"
    ]]

    logger.info(f"\n{'='*70}\nPRICING RECOMMENDATIONS\n{'='*70}")
    logger.info(f"\n{recommendations.to_string(index=False)}")
    logger.info(f"\nTotal revenue uplift: {recommendations['revenue_uplift_%'].mean():.1f}% avg")
    return recommendations


def plot_pricing_recommendations(recommendations: pd.DataFrame, save_path: str = None):
    """Visualise pricing recommendations and revenue uplift."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Dynamic Pricing Recommendations", fontsize=13, fontweight="bold")

    colours = ["#2ECC71" if x > 0 else "#E74C3C"
               for x in recommendations["price_change_%"]]

    # Price change %
    axes[0].barh(recommendations["category"], recommendations["price_change_%"],
                 color=colours, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Recommended Price Change (%)")
    axes[0].set_xlabel("Price Change (%)")
    axes[0].grid(True, alpha=0.3, axis="x")
    for i, (val, cat) in enumerate(zip(recommendations["price_change_%"],
                                        recommendations["category"])):
        axes[0].text(val + (0.3 if val >= 0 else -0.3), i,
                     f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
                     fontsize=9, fontweight="bold")

    # Revenue uplift %
    uplift_colours = ["#2ECC71" if x > 0 else "#E74C3C"
                      for x in recommendations["revenue_uplift_%"]]
    axes[1].barh(recommendations["category"], recommendations["revenue_uplift_%"],
                 color=uplift_colours, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Estimated Revenue Uplift (%)")
    axes[1].set_xlabel("Revenue Uplift (%)")
    axes[1].grid(True, alpha=0.3, axis="x")
    for i, val in enumerate(recommendations["revenue_uplift_%"]):
        axes[1].text(val + (0.1 if val >= 0 else -0.1), i,
                     f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
                     fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("Import this module and use individual functions in the notebook.")
