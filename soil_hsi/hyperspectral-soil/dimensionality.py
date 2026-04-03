from sklearn.decomposition import PCA

class PCAReducer:
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X):
        return self.pca.fit_transform(X)
