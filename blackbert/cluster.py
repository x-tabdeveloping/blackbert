from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer


class ClusterTransformer(TransformerMixin):
    """Turns sklearn clustering algorithm into a transformer component that
    you can use in a pipeline.
    If the clustering method cannot be used for prediction
    (aka. it does not have a predict method), then a nearest neighbour
    vot will be used to infer cluster labels for unseen samples.

    Parameters
    ----------
    model: ClusterMixin
        Sklearn clustering model.
    n_neighbors: int
        Number of neighbours to use for inference.
    metric: str
        Metric to use for determining nearest neighbours.

    Attributes
    ----------
    labeler: LabelBinarizer
        Component that turns cluster labels into one-hot embeddings.
    neighbors: KNeighborsClassifier
        Classifier to use for out of sample prediction.
    """

    def __init__(
        self, model: ClusterMixin, n_neighbors: int = 5, metric: str = "cosine"
    ):
        self.model = model
        self.labeler = LabelBinarizer()
        self.neighbors = KNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric
        )

    def fit(self, X, y=None):
        """Fits the clustering algorithm and label binarizer.

        Parameters
        ----------
        X: ndarray of shape (n_observations, n_features)
            Observations to cluster.
        y: None
            Ignored, exists for compatiblity.

        Returns
        -------
        self
        """
        labels = self.model.fit_predict(X)
        if not hasattr(self.model, "predict"):
            self.neighbors.fit(X, labels)
        self.labeler.fit(labels)
        return self

    def transform(self, X):
        """Infers cluster labels for given data points.

        Parameters
        ----------
        X: ndarray of shape (n_observations, n_features)
            Observations to cluster.

        Returns
        -------
        ndarray of shape (n_observations, n_clusters)
            One-hot encoding of cluster labels.
        """
        if hasattr(self.model, "predict"):
            labels = self.model.predict(X)
        else:
            labels = self.neighbors.predict(X)
        return self.labeler.transform(labels)

    def get_feature_names_out(self):
        """Returns the cluster classes for each dimension.

        Returns
        -------
        ndarray of shape (n_clusters)
            Cluster names.
        """
        return self.labeler.classes_


def DBSCANTransformer(
    eps: float = 0.5, min_samples: int = 5, metric: str = "cosine"
) -> ClusterTransformer:
    """Convenience function for creating a DBSCAN transformer.

    Parameters
    ----------
    eps : float, default 0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default 5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : str, default 'cosine'
        The metric to use when calculating distance between instances in a
        feature array.

    Returns
    -------
    ClusterTransformer
        Sklearn transformer component that wraps DBSCAN.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    return ClusterTransformer(model, metric="cosine")


def KMeansTransformer(n_clusters: int) -> ClusterTransformer:
    """Convenience function for creating a KMeans transformer.

    Parameters
    ----------
    n_clusters: int
        Number of clusters.

    Returns
    -------
    ClusterTransformer
        Sklearn transformer component that wraps KMeans.
    """
    model = KMeans(n_clusters=n_clusters)
    return ClusterTransformer(model, metric="cosine")


def SpectralClusteringTransformer(n_clusters: int) -> ClusterTransformer:
    """Convenience function for creating a Spectral Clustering transformer.

    Parameters
    ----------
    n_clusters: int
        Number of clusters.

    Returns
    -------
    ClusterTransformer
        Sklearn transformer component that wraps SpectralClustering.
    """
    model = SpectralClustering(n_clusters=n_clusters)
    return ClusterTransformer(model, metric="cosine")
