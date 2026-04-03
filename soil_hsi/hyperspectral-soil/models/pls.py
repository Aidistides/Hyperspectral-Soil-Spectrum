from sklearn.cross_decomposition import PLSRegression

class PLSModel:
    def __init__(self, n_components=10):
        self.model = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
