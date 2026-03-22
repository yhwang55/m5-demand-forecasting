import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

BUILD_VERSION = "2026-03-22-1719"

st.set_page_config(page_title="M5 Demand Forecasting", layout="wide")

st.markdown(
    '''
    <style>
        :root {
            --primary: #1f77b4;
            --accent: #ff7f0e;
            --success: #2ca02c;
            --bg-card: #f7f9fc;
            --text-muted: #6b7280;
        }
        .kpi-card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 16px 18px;
            border: 1px solid rgba(31, 119, 180, 0.15);
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        }
        .kpi-card .kpi-title {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 6px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .kpi-card .kpi-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
        }
        .kpi-card .kpi-icon {
            font-size: 1.6rem;
            margin-right: 10px;
        }
        .kpi-row {
            display: flex;
            gap: 14px;
        }
        .kpi-highlight {
            border-left: 4px solid var(--primary);
        }
        .kpi-accent {
            border-left: 4px solid var(--accent);
        }
        .kpi-success {
            border-left: 4px solid var(--success);
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin: 10px 0 6px 0;
        }
    </style>
    ''',
    unsafe_allow_html=True,
)

st.title("M5 Demand Forecasting MVP")
st.caption("Portfolio demo: model selection, KPIs, and prediction overlay on sample data")
st.caption(f"Build: {BUILD_VERSION}")

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
    return f"{value[:2]}***{value[-2:] }"


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
        st.write("App build:", BUILD_VERSION)
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
        in_sample_preds = model.predict(features.drop(columns=["sales"])).astype(float)
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

    plot_history = filtered[["date", "sales"]].copy()
    plot_history["sales"] = plot_history["sales"].astype(float)
    plot_history["prediction"] = np.nan

    plot_data = pd.concat(
        [
            plot_history,
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

kpi_cols[0].markdown(
    f"""
    <div class="kpi-card kpi-highlight">
        <div class="kpi-title">Price</div>
        <div class="kpi-value">{price if price is not None else 0}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols[1].markdown(
    f"""
    <div class="kpi-card kpi-accent">
        <div class="kpi-title">Avg Sales</div>
        <div class="kpi-value">{avg_sales:,.1f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols[2].markdown(
    f"""
    <div class="kpi-card kpi-highlight">
        <div class="kpi-title">Latest Sales</div>
        <div class="kpi-value">{latest_sales:,.1f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols[3].markdown(
    f"""
    <div class="kpi-card kpi-success">
        <div class="kpi-title">Prediction MAE</div>
        <div class="kpi-value">{mae:,.2f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Sales Forecast View")

history_series = plot_data.dropna(subset=["sales"]).copy()
forecast_series = plot_data.dropna(subset=["prediction"]).copy()

history_series["sales"] = pd.to_numeric(history_series["sales"], errors="coerce")
forecast_series["prediction"] = pd.to_numeric(forecast_series["prediction"], errors="coerce")

last_actual_date = None
last_actual_value = None
if not history_series.empty:
    last_actual_date = history_series["date"].iloc[-1]
    last_actual_value = float(history_series["sales"].iloc[-1])

if history_series.empty:
    st.warning("No non-null sales values to plot for the selected store/item.")

fig = go.Figure()
if not history_series.empty:
    fig.add_trace(
        go.Scatter(
            x=history_series["date"],
            y=history_series["sales"],
            mode="lines+markers",
            name="sales",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=5),
        )
    )

forecast_plot = forecast_series.dropna(subset=["prediction"]).copy()
if last_actual_value is not None and not forecast_plot.empty:
    first_pred = float(forecast_plot["prediction"].iloc[0])
    offset = last_actual_value - first_pred
    forecast_plot["prediction"] = forecast_plot["prediction"] + offset
    anchor_row = pd.DataFrame(
        {"date": [last_actual_date], "prediction": [last_actual_value]}
    )
    forecast_plot = pd.concat([anchor_row, forecast_plot], ignore_index=True)

if not forecast_plot.empty:
    fig.add_trace(
        go.Scatter(
            x=forecast_plot["date"],
            y=forecast_plot["prediction"],
            mode="lines+markers",
            name="prediction",
            line=dict(color="#ff7f0e", width=3),
            marker=dict(size=5),
        )
    )

fig.update_layout(
    title="Actual vs Prediction",
    template="plotly_white",
    hovermode="x unified",
    legend_title_text="",
    xaxis_title="",
    yaxis_title="Sales",
    margin=dict(l=10, r=10, t=40, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.08)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.08)", zeroline=False)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Model Summary"):
    st.markdown(
        """
        **Baseline**: Expanding mean of historical sales + flat future forecast.
        **LightGBM (trained)**: Lightweight lag/rolling features trained on selected series and used for recursive forecasting.
        *Note: This is a fast demo model; replace with production training pipeline for higher accuracy.*
        """
    )

with st.expander("Data Snapshot"):
    snapshot = filtered.head(10).copy()
    snapshot = snapshot.astype(str)
    st.markdown(snapshot.to_html(index=False), unsafe_allow_html=True)

st.caption("MVP: Filter + Actual vs Prediction + KPIs + Summary")