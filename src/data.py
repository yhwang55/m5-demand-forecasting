import pandas as pd
from .config import SAMPLE_DATA_DIR, RAW_DATA_DIR

def load_sample_sales():
    return pd.read_csv(SAMPLE_DATA_DIR / "sales_sample.csv")

def load_sample_calendar():
    return pd.read_csv(SAMPLE_DATA_DIR / "calendar_sample.csv")

def load_sample_prices():
    return pd.read_csv(SAMPLE_DATA_DIR / "prices_sample.csv")

def load_m5_sales(path=None):
    if path:
        return pd.read_csv(path)
    return pd.read_csv(RAW_DATA_DIR / "sales_train_validation.csv")