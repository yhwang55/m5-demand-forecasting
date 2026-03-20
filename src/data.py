import os
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from .config import SAMPLE_DATA_DIR, RAW_DATA_DIR, KAGGLE_DATA_DIR

KAGGLE_DATASET = "m5-forecasting-accuracy"
KAGGLE_REQUIRED_FILES = [
    KAGGLE_DATA_DIR / "sales_train_validation.csv",
    KAGGLE_DATA_DIR / "calendar.csv",
    KAGGLE_DATA_DIR / "sell_prices.csv",
]

def load_sample_sales():
    return pd.read_csv(SAMPLE_DATA_DIR / "sales_sample.csv")

def load_sample_calendar():
    return pd.read_csv(SAMPLE_DATA_DIR / "calendar_sample.csv")

def load_sample_prices():
    return pd.read_csv(SAMPLE_DATA_DIR / "prices_sample.csv")

def _kaggle_creds_available():
    return bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))

def ensure_kaggle_dataset():
    if all(path.exists() for path in KAGGLE_REQUIRED_FILES):
        return True
    if not _kaggle_creds_available():
        return False
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=RAW_DATA_DIR, unzip=True, quiet=True)
    except Exception:
        return False
    return all(path.exists() for path in KAGGLE_REQUIRED_FILES)

def load_kaggle_sales_long(max_stores=3, max_items=20, last_n_days=365):
    sales = pd.read_csv(KAGGLE_DATA_DIR / "sales_train_validation.csv")
    store_ids = sorted(sales["store_id"].unique())[:max_stores]
    filtered = sales[sales["store_id"].isin(store_ids)]
    item_ids = sorted(filtered["item_id"].unique())[:max_items]
    filtered = filtered[filtered["item_id"].isin(item_ids)]
    d_cols = [col for col in filtered.columns if col.startswith("d_")]
    if last_n_days and last_n_days < len(d_cols):
        d_cols = d_cols[-last_n_days:]
    melted = filtered[["store_id", "item_id"] + d_cols].melt(
        id_vars=["store_id", "item_id"],
        var_name="d",
        value_name="sales",
    )
    calendar = pd.read_csv(KAGGLE_DATA_DIR / "calendar.csv", usecols=["d", "date"])
    merged = melted.merge(calendar, on="d", how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged[["store_id", "item_id", "date", "sales"]]

def load_kaggle_prices_latest():
    prices = pd.read_csv(
        KAGGLE_DATA_DIR / "sell_prices.csv",
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
    )
    latest = prices.sort_values("wm_yr_wk").groupby(["store_id", "item_id"], as_index=False).tail(1)
    latest = latest.rename(columns={"sell_price": "price"})
    return latest[["store_id", "item_id", "price"]]

def load_m5_sales(path=None):
    if path:
        return pd.read_csv(path)
    return pd.read_csv(KAGGLE_DATA_DIR / "sales_train_validation.csv")
