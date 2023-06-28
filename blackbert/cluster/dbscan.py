import numpy as np
from sklearn.base import TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize


class DBSCANTransformer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        super(TransformerMixin, self).__init__()
        self.clustering = DBSCAN(*args, **kwargs)

    def fit(self, X, y=None):
        self.clustering.fit_predict(X)
        labels = self.clustering.labels_
        self.classes = np.unique(labels)
        self.classifier = KNeighborsClassifier(n_neighbors=5).fit(X, labels)
        return self

    def transform(self, X):
        labels = self.classifier.predict(X)
        return label_binarize(labels, classes=self.classes)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self):
        return self.classes
