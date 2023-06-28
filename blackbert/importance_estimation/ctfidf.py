"""Importance estimation with c-tf-idf."""
import numpy as np
import scipy.sparse as spr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from blackbert.importance_estimation._ctfidf import CTFIDFVectorizer


class CTFIDFEstimator(BaseEstimator):
    """Estimate feature importances for components with c-tf-idf.
    Divides documents to different classes based on most prominant component.

    Attributes
    ----------
    feature_importances_: ndarray of shape (n_components, n_features)
        Importance of each feature for each output feature.
    """

    def fit(self, X, y):
        """Estimates feature importances for each topic based on
        the classes assigned by selecting the highest ranking topic
        for each document.

        Parameters
        ----------
        X: sparse array of shape (n_documents, n_features)
            Bag-of-words/any other feature matrix.
        y: ndarray of shape (n_documents, n_components)
            Document-topic matrix.

        Returns
        -------
        Self
            Fitted Estimator.
        """
        n_components = y.shape[1]
        n_docs, n_features = X.shape
        topic_labels = np.argmax(y, axis=1)
        class_counts = spr.lil_array((n_components, n_features), dtype=X.dtype)
        for i_topic in range(n_components):
            documents_in_class = topic_labels == i_topic
            counts_in_class = X[documents_in_class].sum(axis=0)
            class_counts[i_topic, :] = counts_in_class
        self.feature_importances_ = CTFIDFVectorizer().fit_transform(
            class_counts.todense(), n_samples=n_docs
        )
        return self
