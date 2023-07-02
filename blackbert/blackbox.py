from typing import Optional

import numpy as np
import scipy.sparse as spr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from blackbert.feature_extraction.text import SparseWithText
from blackbert.importance_estimation.trees import RandomForestEstimator


class BlackboxTopicModel(TransformerMixin):
    """Black box topic model, that estimates document-topic distributions
    directly from text and estimates feature importances
    with the given estimation method.

    Parameters
    ----------
    model: BaseEstimator
        Model that can embed texts and output document-topic
        distributions.
    estimator: BaseEstimator or None, default None
        Component with feature_importances_ attribute,
        so that the importance of word features can be estimated.
        Default is estimating feature importances with RandomForest.
        Note that for certain models the default option may be suboptimal.

    Attributes
    ----------
    components_: ndarray of shape (n_components, n_features)
        Feature importances for each topic.
    """

    def __init__(
        self,
        model: BaseEstimator,
        estimator: Optional[BaseEstimator] = None,
    ):
        self.model = model
        if estimator is None:
            estimator = RandomForestEstimator()
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


def box(
    *steps: BaseEstimator, estimator: Optional[BaseEstimator] = None
) -> BlackboxTopicModel:
    """Utility function for creating black box topic models.

    Parameters
    ----------
    *steps: BaseEstimator
        List of estimators to use for embedding texts into
        the topic space.
    estimator: BaseEstimator or None, default None
        Feature importance estimator algorithm.
        Default is Random Forest Regression.

    Returns
    -------
    BlackboxTopicModel
        Black box model following the given steps and estimating
    """
    model = make_pipeline(*steps)
    return BlackboxTopicModel(model, estimator)
