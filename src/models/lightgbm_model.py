import lightgbm as lgb


class LightGBMModel:
    def __init__(self, params=None):
        self.params = params or {"objective": "regression", "metric": "rmse"}
        self.model = None

    def fit(self, X_train, y_train):
        train_data = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(self.params, train_data, num_boost_round=100)

    def predict(self, X):
        return self.model.predict(X)