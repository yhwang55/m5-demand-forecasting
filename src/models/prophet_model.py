from prophet import Prophet
import pandas as pd


class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def fit(self, df, date_col="date", target_col="sales"):
        train_df = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
        self.model.fit(train_df)

    def predict(self, df, periods=0):
        future = df[["date"]].rename(columns={"date": "ds"})
        forecast = self.model.predict(future)
        return forecast["yhat"].values