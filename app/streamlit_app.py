import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure repo root is on PYTHONPATH for Streamlit Cloud
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import (
    ensure_kaggle_dataset,
    load_kaggle_prices_latest,
    load_kaggle_sales_long,
    load_sample_prices,
    load_sample_sales,
)

st.set_page_config(page_title="M5 Demand Forecasting", layout="wide")

st.title("M5 Demand Forecasting MVP")
st.caption("Portfolio demo: model selection, KPIs, and prediction overlay on sample data")

use_kaggle = ensure_kaggle_dataset()

if use_kaggle:
    sales = load_kaggle_sales_long()
    prices = load_kaggle_prices_latest()
    st.success("Using Kaggle M5 dataset (cached)")
else:
    sales = load_sample_sales()
    prices = load_sample_prices()
    st.info("Using sample dataset (Kaggle API key required for full data)")

store_ids = sorted(sales["store_id"].unique())
item_ids = sorted(sales["item_id"].unique())

with st.sidebar:
    st.header("Filters")
    store = st.selectbox("Store", store_ids)
    item = st.selectbox("Item", item_ids)
    model_choice = st.selectbox("Model", ["Baseline", "LightGBM (demo)", "Prophet (demo)"])

filtered = sales[(sales["store_id"] == store) & (sales["item_id"] == item)].copy()
filtered = filtered.sort_values("date")
price_row = prices[(prices["store_id"] == store) & (prices["item_id"] == item)]
price = price_row["price"].iloc[0] if not price_row.empty else None

# Lightweight demo prediction for portfolio UX (fast + zero training)
if model_choice == "Baseline":
    filtered["prediction"] = filtered["sales"].expanding().mean()
elif model_choice.startswith("LightGBM"):
    filtered["prediction"] = filtered["sales"].rolling(window=3, min_periods=1).mean()
else:
    filtered["prediction"] = filtered["sales"].rolling(window=5, min_periods=1).mean()

avg_sales = float(filtered["sales"].mean()) if not filtered.empty else 0.0
latest_sales = float(filtered["sales"].iloc[-1]) if not filtered.empty else 0.0
mae = float((filtered["sales"] - filtered["prediction"]).abs().mean()) if not filtered.empty else 0.0

kpi_cols = st.columns(4)
kpi_cols[0].metric("Price", price if price is not None else 0)
kpi_cols[1].metric("Avg Sales", f"{avg_sales:,.1f}")
kpi_cols[2].metric("Latest Sales", f"{latest_sales:,.1f}")
kpi_cols[3].metric("Prediction MAE", f"{mae:,.2f}")

st.markdown("### Sales Forecast View")
plot_df = filtered.melt(
    id_vars=["date"],
    value_vars=["sales", "prediction"],
    var_name="series",
    value_name="value",
)
fig = px.line(
    plot_df,
    x="date",
    y="value",
    color="series",
    title="Actual vs Prediction",
    labels={"value": "Sales", "series": "Series"},
)
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Model Summary"):
    st.markdown(
        """
        **Baseline**: Expanding mean of historical sales.\n
        **LightGBM (demo)**: 3-day rolling mean to illustrate short-term smoothing.\n
        **Prophet (demo)**: 5-day rolling mean to emulate trend smoothing.\n        *Note: These are lightweight demo predictions for UX; replace with trained models for production.*
        """
    )

with st.expander("Data Snapshot"):
    st.dataframe(filtered.head(10), use_container_width=True)

st.caption("MVP: Filter + Actual vs Prediction + KPIs + Summary")