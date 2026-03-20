import pandas as pd
import plotly.express as px
import streamlit as st
from src.data import load_sample_sales, load_sample_prices

st.title("M5 Demand Forecasting MVP")
st.caption("Demo: model selection + prediction overlay (lightweight) on sample CSV")

sales = load_sample_sales()
prices = load_sample_prices()

store_ids = sorted(sales["store_id"].unique())
item_ids = sorted(sales["item_id"].unique())

store = st.selectbox("Store", store_ids)
item = st.selectbox("Item", item_ids)
model_choice = st.selectbox("Model", ["Baseline", "LightGBM (demo)", "Prophet (demo)"])

filtered = sales[(sales["store_id"] == store) & (sales["item_id"] == item)].copy()
filtered = filtered.sort_values("date")
price_row = prices[(prices["store_id"] == store) & (prices["item_id"] == item)]
price = price_row["price"].iloc[0] if not price_row.empty else None

st.metric("Price", price if price is not None else 0)

# Lightweight demo prediction for portfolio UX (fast + zero training)
if model_choice == "Baseline":
    filtered["prediction"] = filtered["sales"].expanding().mean()
elif model_choice.startswith("LightGBM"):
    filtered["prediction"] = filtered["sales"].rolling(window=3, min_periods=1).mean()
else:
    filtered["prediction"] = filtered["sales"].rolling(window=5, min_periods=1).mean()

plot_df = filtered.melt(id_vars=["date"], value_vars=["sales", "prediction"], var_name="series", value_name="value")
fig = px.line(plot_df, x="date", y="value", color="series", title="Actual vs Prediction")
st.plotly_chart(fig, use_container_width=True)

st.caption("MVP: Filter + Actual vs Prediction + KPI")
