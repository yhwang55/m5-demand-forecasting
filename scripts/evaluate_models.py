import pandas as pd
from src.metrics import rmse, mae, mape

def compare_models(results):
    df = pd.DataFrame(results)
    print(df)
    return df

if __name__ == "__main__":
    sample_results = [
        {"model": "Baseline", "rmse": 0.0, "mae": 0.0, "mape": 0.0},
        {"model": "LightGBM", "rmse": 0.0, "mae": 0.0, "mape": 0.0},
        {"model": "Prophet", "rmse": 0.0, "mae": 0.0, "mape": 0.0},
    ]
    compare_models(sample_results)