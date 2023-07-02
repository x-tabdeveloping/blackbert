from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer


class ClusterTransformer(TransformerMixin):
    def __init__(self, model: ClusterMixin, n_neighbors: int = 5):
        self.model = model
        self.labeler = LabelBinarizer()
        self.neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        labels = self.model.fit_predict(X)
        if not hasattr(self.model, "predict"):
            self.neighbors.fit(X, labels)
        self.labeler.fit(labels)
        return self

    def transform(self, X):
        if hasattr(self.model, "predict"):
            labels = self.model.predict(X)
        else:
            labels = self.neighbors.predict(X)
        return self.labeler.transform(labels)
