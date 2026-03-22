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

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

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
    return f"{value[:2]}***{value[-2:]}