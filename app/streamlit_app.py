import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from lightgbm import LGBMRegressor

# Ensure repo root is on PYTHONPATH for Streamlit Cloud
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import (
    ensure_kaggle_dataset,
    get_kaggle_debug_status,
    load_kaggle_prices_latest,
    load_kaggle_sales_for_item,
    load_kaggle_store_item_index,
    load_sample_prices,
    load_sample_sales,
)

st.set_page_config(page_title="M5 Demand Forecasting", layout="wide")

st.title("M5 Demand Forecasting MVP")
st.caption("Portfolio demo: model selection, KPIs, and prediction overlay on sample data")

use_kaggle = ensure_kaggle_dataset()

if use_kaggle:
    store_ids, item_ids = load_kaggle_store_item_index()
    prices = load_kaggle_prices_latest()
    st.success("Using Kaggle M5 dataset (cached)")
else:
    sales = load_sample_sales()
    prices = load_sample_prices()
    store_ids = sorted(sales["store_id"].unique())
    item_ids = sorted(sales["item_id"].unique())
    st.info("Using sample dataset (Kaggle API key required for full data)")

kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")
kaggle_status = get_kaggle_debug_status()


def _mask_secret(value: str | None) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 4:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def _format_store(store_id: str) -> str:
    if not store_id:
        return "(unknown store)"
    parts = store_id.split("_")
    if len(parts) == 2:
        state, store_num = parts
        return f"{store_id} (State {state}, Store {store_num})"
    return store_id


def _format_item(item_id: str) -> str:
    if not item_id:
        return "(unknown item)"
    parts = item_id.split("_")
    if len(parts) >= 3:
        category = parts[0]
        dept = parts[1]
        item_num = "_".join(parts[2:])
        return f"{item_id} (Category {category}, Dept {dept}, Item {item_num})"
    return item_id


def _build_lag_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"sales": series})
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)
    df["rolling_7"] = df["sales"].rolling(7).mean()
    df["rolling_28"] = df["sales"].rolling(28).mean()
    return df


def _train_lightgbm_model(series: pd.Series) -> tuple[LGBMRegressor, pd.DataFrame]:
    features = _build_lag_features(series).dropna()
    X = features.drop(columns=["sales"])
    y = features["sales"]
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X, y)
    return model, features


def _forecast_lightgbm(model: LGBMRegressor, history: pd.Series, horizon: int) -> pd.Series:
    values = history.tolist()
    preds = []
    for _ in range(horizon):
        window = pd.Series(values)
        features = _build_lag_features(window).iloc[-1]
        X_next = features.drop(labels=["sales"]).to_frame().T
        next_pred = float(model.predict(X_next)[0])
        preds.append(next_pred)
        values.append(next_pred)
    return pd.Series(preds)


with st.sidebar:
    st.header("Filters")
    store = st.selectbox("Store", store_ids, format_func=_format_store)
    item = st.selectbox("Item", item_ids, format_func=_format_item)
    forecast_days = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=28)
    model_choice = st.selectbox("Model", ["Baseline", "LightGBM (trained)"])
    st.caption("Store/Item labels are decoded for readability; the IDs are the canonical Kaggle keys.")
    with st.expander("Kaggle Debug"):
        st.write("KAGGLE_USERNAME set:", bool(kaggle_username))
        st.write("KAGGLE_USERNAME (masked):", _mask_secret(kaggle_username))
        st.write("KAGGLE_KEY set:", bool(kaggle_key))
        st.write("KAGGLE_KEY (masked):", _mask_secret(kaggle_key))
        st.write("Kaggle dataset ready:", use_kaggle)
        st.write("Required files present:", kaggle_status["required_files_present"])
        st.write("Last error:", kaggle_status["last_error"])

if use_kaggle:
    filtered = load_kaggle_sales_for_item(store, item, last_n_days=730)
else:
    filtered = sales[(sales["store_id"] == store) & (sales["item_id"] == item)].copy()

filtered = filtered.sort_values("date")
price_row = prices[(prices["store_id"] == store) & (prices["item_id"] == item)]
price = price_row["price"].iloc[0] if not price_row.empty else None

filtered["prediction"] = np.nan

if not filtered.empty:
    sales_series = filtered["sales"].astype(float).reset_index(drop=True)

    if model_choice == "Baseline":
        filtered["prediction"] = sales_series.expanding().mean()
        baseline_forecast = float(filtered["prediction"].iloc[-1])
        forecast_values = pd.Series([baseline_forecast] * forecast_days)
    else:
        model, features = _train_lightgbm_model(sales_series)
        in_sample_preds = model.predict(features.drop(columns=["sales"]))
        filtered.loc[features.index, "prediction"] = in_sample_preds
        forecast_values = _forecast_lightgbm(model, sales_series, forecast_days)

    last_date = pd.to_datetime(filtered["date"].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    forecast_df = pd.DataFrame(
        {
            "date": future_dates,
            "sales": np.nan,
            "prediction": forecast_values.values,
        }
    )
    plot_data = pd.concat(
        [
            filtered[["date", "sales", "prediction"]],
            forecast_df,
        ],
        ignore_index=True,
    )
else:
    plot_data = filtered[["date", "sales", "prediction"]]

avg_sales = float(filtered["sales"].mean()) if not filtered.empty else 0.0
latest_sales = float(filtered["sales"].iloc[-1]) if not filtered.empty else 0.0
mae = (
    float((filtered["sales"] - filtered["prediction"]).abs().mean())
    if not filtered.empty
    else 0.0
)

kpi_cols = st.columns(4)
kpi_cols[0].metric("Price", price if price is not None else 0)
kpi_cols[1].metric("Avg Sales", f"{avg_sales:,.1f}")
kpi_cols[2].metric("Latest Sales", f"{latest_sales:,.1f}")
kpi_cols[3].metric("Prediction MAE", f"{mae:,.2f}")

st.markdown("### Sales Forecast View")
plot_df = plot_data.melt(
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
        **Baseline**: Expanding mean of historical sales + flat future forecast.\n        **LightGBM (trained)**: Lightweight lag/rolling features trained on selected series and used for recursive forecasting.\n        *Note: This is a fast demo model; replace with production training pipeline for higher accuracy.*
        """
    )

with st.expander("Data Snapshot"):
    st.dataframe(filtered.head(10), use_container_width=True)

st.caption("MVP: Filter + Actual vs Prediction + KPIs + Summary")
