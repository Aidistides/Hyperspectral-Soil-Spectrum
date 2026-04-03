from xgboost import XGBRegressor

class XGBoostModel:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
