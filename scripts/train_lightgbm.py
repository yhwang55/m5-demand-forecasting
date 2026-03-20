from sklearn.model_selection import train_test_split
from src.pipeline import build_sample_dataset
from src.models.lightgbm_model import LightGBMModel
from src.metrics import rmse, mae, mape

def main():
    df = build_sample_dataset()
    X = df[["year", "month", "week", "day", "dow"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LightGBMModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("LightGBM RMSE:", rmse(y_test, preds))
    print("LightGBM MAE:", mae(y_test, preds))
    print("LightGBM MAPE:", mape(y_test, preds))


if __name__ == "__main__":
    main()