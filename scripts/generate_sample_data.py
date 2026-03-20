import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.config import SAMPLE_DATA_DIR

def generate_sample_data(num_stores=3, num_items=200, days=120):
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    sales_rows = []
    for store_id in range(1, num_stores + 1):
        for item_id in range(1, num_items + 1):
            base = np.random.randint(5, 50)
            trend = np.linspace(0, 5, days)
            noise = np.random.normal(0, 2, days)
            sales = np.maximum(0, base + trend + noise).round()
            for d, s in zip(dates, sales):
                sales_rows.append([store_id, item_id, d.strftime("%Y-%m-%d"), s])

    sales_df = pd.DataFrame(sales_rows, columns=["store_id", "item_id", "date", "sales"])
    sales_df.to_csv(SAMPLE_DATA_DIR / "sales_sample.csv", index=False)

    calendar_df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "event": ["None"] * len(dates)
    })
    calendar_df.to_csv(SAMPLE_DATA_DIR / "calendar_sample.csv", index=False)

    price_rows = []
    for store_id in range(1, num_stores + 1):
        for item_id in range(1, num_items + 1):
            price = np.round(np.random.uniform(1.0, 20.0), 2)
            price_rows.append([store_id, item_id, price])

    prices_df = pd.DataFrame(price_rows, columns=["store_id", "item_id", "price"])
    prices_df.to_csv(SAMPLE_DATA_DIR / "prices_sample.csv", index=False)


if __name__ == "__main__":
    generate_sample_data()