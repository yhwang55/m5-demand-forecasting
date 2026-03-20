import pandas as pd
from sklearn.model_selection import train_test_split
from src.pipeline import build_sample_dataset
from src.models.baseline import BaselineModel
from src.metrics import rmse, mae, mape

def main():
    df = build_sample_dataset()
    X = df[["year", "month", "week", "day", "dow"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = BaselineModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Baseline RMSE:", rmse(y_test, preds))
    print("Baseline MAE:", mae(y_test, preds))
    print("Baseline MAPE:", mape(y_test, preds))


if __name__ == "__main__":
    main()