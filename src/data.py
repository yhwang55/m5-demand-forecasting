import os
from pathlib import Path
import zipfile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from .config import SAMPLE_DATA_DIR, RAW_DATA_DIR, KAGGLE_DATA_DIR

KAGGLE_DATASET = "m5-forecasting-accuracy"
KAGGLE_REQUIRED_FILES = [
    KAGGLE_DATA_DIR / "sales_train_validation.csv",
    KAGGLE_DATA_DIR / "calendar.csv",
    KAGGLE_DATA_DIR / "sell_prices.csv",
]

KAGGLE_LAST_ERROR = None

def load_sample_sales():
    return pd.read_csv(SAMPLE_DATA_DIR / "sales_sample.csv")

def load_sample_calendar():
    return pd.read_csv(SAMPLE_DATA_DIR / "calendar_sample.csv")

def load_sample_prices():
    return pd.read_csv(SAMPLE_DATA_DIR / "prices_sample.csv")

def _kaggle_creds_available():
    return bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))

def _unzip_if_needed(zip_path: Path, target_dir: Path):
    if not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

def ensure_kaggle_dataset():
    global KAGGLE_LAST_ERROR
    KAGGLE_LAST_ERROR = None
    if all(path.exists() for path in KAGGLE_REQUIRED_FILES):
        return True
    if not _kaggle_creds_available():
        KAGGLE_LAST_ERROR = "Missing KAGGLE_USERNAME or KAGGLE_KEY"
        return False
    KAGGLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(KAGGLE_DATASET, path=KAGGLE_DATA_DIR, quiet=True)
        zip_path = KAGGLE_DATA_DIR / f"{KAGGLE_DATASET}.zip"
        _unzip_if_needed(zip_path, KAGGLE_DATA_DIR)
    except Exception as exc:
        KAGGLE_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return False
    if not all(path.exists() for path in KAGGLE_REQUIRED_FILES):
        KAGGLE_LAST_ERROR = "Download completed but expected files are missing."
        return False
    return True

def get_kaggle_debug_status():
    return {
        "creds_available": _kaggle_creds_available(),
        "required_files_present": all(path.exists() for path in KAGGLE_REQUIRED_FILES),
        "last_error": KAGGLE_LAST_ERROR,
    }

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
