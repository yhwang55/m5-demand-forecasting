import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lightgbm import LGBMRegressor

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

BUILD_VERSION = "2026-03-23-2145"

STATE_LABELS = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}
CATEGORY_LABELS = {"FOODS": "🍎 Foods", "HOBBIES": "🎮 Hobbies", "HOUSEHOLD": "🏠 Household"}
DEPT_DESC = {
    "FOODS_1": "Produce & Deli", "FOODS_2": "Dry Goods", "FOODS_3": "Beverages & Snacks",
    "HOBBIES_1": "Toys & Games", "HOBBIES_2": "Crafts & Outdoors",
    "HOUSEHOLD_1": "Cleaning & Paper", "HOUSEHOLD_2": "Kitchen & Home",
}

def _parse_store_id(store_id: str) -> dict:
    parts = store_id.split("_")
    if len(parts) == 2:
        return {"state": parts[0], "store_num": parts[1]}
    return {"state": store_id, "store_num": "?"}

def _parse_item_id(item_id: str) -> dict:
    parts = item_id.split("_")
    if len(parts) >= 3:
        cat = parts[0]; dept_num = parts[1]
        return {"category": cat, "dept_num": dept_num,
                "dept_key": f"{cat}_{dept_num}", "item_num": "_".join(parts[2:])}
    return {"category": item_id, "dept_num": "?", "dept_key": item_id, "item_num": "?"}

def _fmt_dept(dk: str) -> str:
    num = dk.split("_")[1] if "_" in dk else dk
    desc = DEPT_DESC.get(dk, "")
    return f"Dept {num}  —  {desc}" if desc else f"Dept {num}"

def _fmt_item_num(item_id: str) -> str:
    return f"Item #{_parse_item_id(item_id)['item_num']}"

def _mask_secret(value) -> str:
    if not value: return "(not set)"
    if len(value) <= 4: return "***"
    return f"{value[:2]}***{value[-2:]}"

def _build_lag_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"sales": series})
    df["lag_1"] = df["sales"].shift(1); df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14); df["rolling_7"] = df["sales"].rolling(7).mean()
    df["rolling_28"] = df["sales"].rolling(28).mean()
    return df

def _train_lightgbm_model(series: pd.Series):
    features = _build_lag_features(series).dropna()
    X = features.drop(columns=["sales"]); y = features["sales"]
    model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=-1,
                          subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(X, y)
    return model, features

def _forecast_lightgbm(model, history: pd.Series, horizon: int) -> pd.Series:
    values = history.tolist(); preds = []
    for _ in range(horizon):
        window = pd.Series(values)
        feat = _build_lag_features(window).iloc[-1]
        X_next = feat.drop(labels=["sales"]).to_frame().T
        p = float(model.predict(X_next)[0])
        preds.append(p); values.append(p)
    return pd.Series(preds)

st.set_page_config(page_title="M5 Demand Forecasting", layout="wide")

st.markdown("""<style>
:root { --primary:#1f77b4; --accent:#ff7f0e; --success:#2ca02c; --bg-card:#f7f9fc; --text-muted:#6b7280; }
.kpi-card { background:var(--bg-card); border-radius:12px; padding:16px 18px;
    border:1px solid rgba(31,119,180,.15); box-shadow:0 6px 18px rgba(15,23,42,.08); }
.kpi-card .kpi-title { font-size:.85rem; color:var(--text-muted); margin-bottom:6px;
    font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.kpi-card .kpi-value { font-size:1.6rem; font-weight:700; color:#0f172a; }
.kpi-highlight { border-left:4px solid var(--primary); }
.kpi-accent    { border-left:4px solid var(--accent); }
.kpi-success   { border-left:4px solid var(--success); }
.info-card { background:linear-gradient(135deg,#f0f7ff 0%,#fafafa 100%);
    border:1px solid rgba(31,119,180,.2); border-radius:12px;
    padding:14px 20px; margin-bottom:18px;
    display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
.info-badge { display:inline-flex; align-items:center; gap:6px; background:white;
    border:1px solid #dbeafe; border-radius:20px; padding:4px 14px;
    font-size:.85rem; font-weight:600; color:#1e40af; }
.info-section { display:flex; flex-direction:column; gap:2px; }
.info-label { font-size:.72rem; color:var(--text-muted); text-transform:uppercase;
    letter-spacing:.05em; font-weight:600; }
.info-value { font-size:.95rem; font-weight:700; color:#0f172a; }
.info-divider { color:#cbd5e1; font-size:1.4rem; padding:0 4px; }
</style>""", unsafe_allow_html=True)

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
    item_ids  = sorted(sales["item_id"].unique())
    st.info("Using sample dataset (Kaggle API key required for full data)")

kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key      = os.getenv("KAGGLE_KEY")
kaggle_status   = get_kaggle_debug_status()

# Pre-parse all IDs into dicts — avoids re-parsing inside lambdas
store_parsed_map = {s: _parse_store_id(s) for s in store_ids}
item_parsed_map  = {i: _parse_item_id(i)  for i in item_ids}

with st.sidebar:
    st.header("Filters")

    # Store — 2 steps
    st.markdown("**Store**")
    states = sorted({v["state"] for v in store_parsed_map.values()})
    sel_state = st.selectbox(
        "① State", states,
        format_func=lambda s: f"{s}  —  {STATE_LABELS.get(s, s)}"
    )
    stores_in_state = sorted([s for s, p in store_parsed_map.items() if p["state"] == sel_state])
    store = st.selectbox(
        "② Store", stores_in_state,
        format_func=lambda s: f"Store #{store_parsed_map[s]['store_num']}  ({s})"
    )

    st.divider()

    # Item — 3 steps
    st.markdown("**Item**")
    st.caption("Item IDs follow the format `CATEGORY_DEPT_NUMBER` in the M5 dataset.")
    categories = sorted({v["category"] for v in item_parsed_map.values()})
    sel_category = st.selectbox(
        "① Category", categories,
        format_func=lambda c: CATEGORY_LABELS.get(c, c)
    )
    depts_in_cat = sorted({v["dept_key"] for v in item_parsed_map.values() if v["category"] == sel_category})
    sel_dept = st.selectbox("② Department", depts_in_cat, format_func=_fmt_dept)
    items_in_dept = sorted([i for i, p in item_parsed_map.items() if p["dept_key"] == sel_dept])
    item = st.selectbox("③ Item Number", items_in_dept, format_func=_fmt_item_num)

    st.divider()

    forecast_days = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=28)
    model_choice  = st.selectbox("Model", ["Baseline", "LightGBM (trained)"])

    with st.expander("Kaggle Debug"):
        st.write("App build:", BUILD_VERSION)
        st.write("KAGGLE_USERNAME set:", bool(kaggle_username))
        st.write("KAGGLE_USERNAME (masked):", _mask_secret(kaggle_username))
        st.write("KAGGLE_KEY set:", bool(kaggle_key))
        st.write("KAGGLE_KEY (masked):", _mask_secret(kaggle_key))
        st.write("Kaggle dataset ready:", use_kaggle)
        st.write("Required files present:", kaggle_status["required_files_present"])
        st.write("Last error:", kaggle_status["last_error"])

# Info card
sp = store_parsed_map[store]
ip = item_parsed_map[item]

st.markdown(f"""
<div class="info-card">
    <div class="info-section">
        <span class="info-label">Store</span>
        <span class="info-value">{store}</span>
    </div>
    <div class="info-badge">📍 {sp["state"]} — {STATE_LABELS.get(sp["state"], sp["state"])}</div>
    <div class="info-section">
        <span class="info-label">Store #</span>
        <span class="info-value">{sp["store_num"]}</span>
    </div>
    <span class="info-divider">|</span>
    <div class="info-section">
        <span class="info-label">Item</span>
        <span class="info-value">{item}</span>
    </div>
    <div class="info-badge">{CATEGORY_LABELS.get(ip["category"], ip["category"])}</div>
    <div class="info-section">
        <span class="info-label">Department</span>
        <span class="info-value">{_fmt_dept(ip["dept_key"])}</span>
    </div>
    <div class="info-section">
        <span class="info-label">Item #</span>
        <span class="info-value">{ip["item_num"]}</span>
    </div>
</div>
""", unsafe_allow_html=True)

if use_kaggle:
    filtered = load_kaggle_sales_for_item(store, item, last_n_days=730)
else:
    filtered = sales[(sales["store_id"] == store) & (sales["item_id"] == item)].copy()

filtered  = filtered.sort_values("date")
price_row = prices[(prices["store_id"] == store) & (prices["item_id"] == item)]
price     = price_row["price"].iloc[0] if not price_row.empty else None
filtered["prediction"] = np.nan

if not filtered.empty:
    sales_series = filtered["sales"].astype(float).reset_index(drop=True)
    if model_choice == "Baseline":
        filtered["prediction"] = sales_series.expanding().mean()
        forecast_values = pd.Series([float(filtered["prediction"].iloc[-1])] * forecast_days)
    else:
        lgbm_model, features = _train_lightgbm_model(sales_series)
        filtered.loc[features.index, "prediction"] = lgbm_model.predict(features.drop(columns=["sales"])).astype(float)
        forecast_values = _forecast_lightgbm(lgbm_model, sales_series, forecast_days)

    last_date    = pd.to_datetime(filtered["date"].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    forecast_df  = pd.DataFrame({"date": future_dates, "sales": np.nan, "prediction": forecast_values.values})
    plot_history = filtered[["date", "sales"]].copy()
    plot_history["sales"] = plot_history["sales"].astype(float)
    plot_history["prediction"] = np.nan
    plot_data = pd.concat([plot_history, forecast_df], ignore_index=True)
else:
    plot_data = filtered[["date", "sales"]]

avg_sales    = float(filtered["sales"].mean())    if not filtered.empty else 0.0
latest_sales = float(filtered["sales"].iloc[-1]) if not filtered.empty else 0.0
mae          = float((filtered["sales"] - filtered["prediction"]).abs().mean()) if not filtered.empty else 0.0

kpi_cols = st.columns(4)
kpi_cols[0].markdown(f'<div class="kpi-card kpi-highlight"><div class="kpi-title">Price</div><div class="kpi-value">{price if price is not None else 0}</div></div>', unsafe_allow_html=True)
kpi_cols[1].markdown(f'<div class="kpi-card kpi-accent"><div class="kpi-title">Avg Sales</div><div class="kpi-value">{avg_sales:,.1f}</div></div>', unsafe_allow_html=True)
kpi_cols[2].markdown(f'<div class="kpi-card kpi-highlight"><div class="kpi-title">Latest Sales</div><div class="kpi-value">{latest_sales:,.1f}</div></div>', unsafe_allow_html=True)
kpi_cols[3].markdown(f'<div class="kpi-card kpi-success"><div class="kpi-title">Prediction MAE</div><div class="kpi-value">{mae:,.2f}</div></div>', unsafe_allow_html=True)

st.markdown("### Sales Forecast View")
history_series = filtered[["date", "sales"]].copy()
history_series["sales"] = pd.to_numeric(history_series["sales"], errors="coerce")
history_series = history_series.dropna(subset=["sales"])

if history_series.empty:
    st.warning("No non-null sales values to plot for the selected store/item.")

fig = go.Figure()
if not history_series.empty:
    fig.add_trace(go.Scatter(x=history_series["date"], y=history_series["sales"],
        mode="markers", name="sales", marker=dict(size=6, color="#1f77b4")))
fig.update_layout(title="Actual Sales", template="plotly_white", hovermode="x unified",
    legend_title_text="", xaxis_title="", yaxis_title="Sales",
    margin=dict(l=10, r=10, t=40, b=10),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,.08)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,.08)", zeroline=False)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Model Summary"):
    st.markdown("""
    **Baseline**: Expanding mean of historical sales + flat future forecast.
    **LightGBM (trained)**: Lightweight lag/rolling features trained on selected series and used for recursive forecasting.
    *Note: This is a fast demo model; replace with production training pipeline for higher accuracy.*
    """)

with st.expander("Data Snapshot"):
    st.markdown(filtered.head(10).copy().astype(str).to_html(index=False), unsafe_allow_html=True)

st.caption("MVP: Filter + Actual vs Prediction + KPIs + Summary")
