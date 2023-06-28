import numpy as np
import scipy.sparse as spr
from sklearn.base import TransformerMixin

from blackbert.feature_extraction.text import SparseWithText


class BlackboxTopicModel(TransformerMixin):
    """Black box topic model, that estimates document-topic distributions
    directly from text and estimates feature importances
    with the given estimation method.

    Parameters
    ----------
    model: BaseEstimator
        Model that can embed texts and output document-topic
        distributions.
    estimator: BaseEstimator
        Regression model with the feature_importances_ attribute,
        so that the importance of word features can be estimated.
        The estimator has to be clonable by sklearn's clone method.

    Attributes
    ----------
    components_: ndarray of shape (n_components, n_features)
        Feature importances for each topic.
    """

    def __init__(self, model, estimator):
        self.model = model
        self.estimator = estimator

    def fit_transform(self, X: SparseWithText, y=None):
        """Fits the model and estimates feature importances.
        Then returns document-topic importances.

        Parameters
        ----------
        X: SparseWithText
            Sparse bag-of-words matrix with a text attribute,
            that contains the underlying texts.
        y: None
            Ignored, exists for compatiblity.

        Returns
        -------
        ndarray of shape (n_documents, n_components)
            Document-topic importances.
        """
        print("Fitting topic model...")
        embeddings = self.model.fit_transform(X.texts)
        self.components_ = self.estimator.fit(
            X, embeddings
        ).feature_importances_
        if spr.issparse(self.components_):
            self.components_ = self.components_.todense()
        self.components_ = np.asarray(self.components_)
        return embeddings

    def fit(self, X: SparseWithText, y=None):
        """Fits the model and estimates feature importances.

        Parameters
        ----------
        X: SparseWithText
            Sparse bag-of-words matrix with a text attribute,
            that contains the underlying texts.
        y: None
            Ignored, exists for compatiblity.
        """
        self.fit_transform(X)
        return self

    def transform(self, X: SparseWithText):
        """Estimates topic importances for each document.

        Parameters
        ----------
        X: SparseWithText
            Sparse bag-of-words matrix with a text attribute,
            that contains the underlying texts.
        y: None
            Ignored, exists for compatiblity.

        Returns
        -------
        ndarray of shape (n_documents, n_components)
            Document-topic importances.
        """
        return self.model.transform(X.texts)
