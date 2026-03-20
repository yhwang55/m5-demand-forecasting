import pandas as pd
from src.pipeline import build_sample_dataset
from src.models.prophet_model import ProphetModel
from src.metrics import rmse, mae, mape

def main():
    df = build_sample_dataset()
    model = ProphetModel()
    model.fit(df, date_col="date", target_col="sales")
    preds = model.predict(df)

    print("Prophet RMSE:", rmse(df["sales"], preds))
    print("Prophet MAE:", mae(df["sales"], preds))
    print("Prophet MAPE:", mape(df["sales"], preds))


if __name__ == "__main__":
    main()