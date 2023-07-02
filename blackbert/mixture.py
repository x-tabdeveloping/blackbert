from sklearn.base import BaseEstimator, TransformerMixin


class MixtureTransformer(TransformerMixin):
    def __init__(self, model: BaseEstimator):
        self.model = model

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.predict_proba(X)
