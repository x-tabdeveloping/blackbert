"""Importance estimation with decision trees and forests."""
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm


class RandomForestEstimator(BaseEstimator):
    """Estimate feature importances for components with Random Forest.

    Attributes
    ----------
    feature_importances_: ndarray of shape (n_components, n_features)
        Importance of each feature for each output feature.
    """

    def __init__(self, n_estimators: int = 10, *args, **kwargs):
        super(RandomForestEstimator, self).__init__()
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators, *args, **kwargs
        )

    def fit(self, X, y):
        """Estimates feature importances for each topic based on feature
        importances estimated by a random forest regressor

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
        self.feature_importances_ = []
        for component in tqdm(y.T, desc="Estimating feature importances"):
            estimator = clone(self.regressor).fit(X, component)
            self.feature_importances_.append(estimator.feature_importances_)
        self.feature_importances_ = np.stack(self.feature_importances_)
        return self


class DecisionTreeEstimator(BaseEstimator):
    """Estimate feature importances for components with a Decision Tree.

    Attributes
    ----------
    feature_importances_: ndarray of shape (n_components, n_features)
        Importance of each feature for each output feature.
    """

    def __init__(self, *args, **kwargs):
        super(DecisionTreeEstimator, self).__init__()
        self.regressor = DecisionTreeRegressor(*args, **kwargs)

    def fit(self, X, y):
        """Estimates feature importances for each topic based on feature
        importances estimated by a random forest regressor

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
        self.feature_importances_ = []
        for component in tqdm(y.T, desc="Estimating feature importances"):
            estimator = clone(self.regressor).fit(X, component)
            self.feature_importances_.append(estimator.feature_importances_)
        self.feature_importances_ = np.stack(self.feature_importances_)
        return self


class ExtraTreesEstimator(BaseEstimator):
    """Estimate feature importances for components with a Decision Tree.

    Attributes
    ----------
    feature_importances_: ndarray of shape (n_components, n_features)
        Importance of each feature for each output feature.
    """

    def __init__(self, n_estimators: int = 10, *args, **kwargs):
        super(ExtraTreesEstimator, self).__init__()
        self.regressor = ExtraTreesRegressor(
            n_estimators=n_estimators, *args, **kwargs
        )

    def fit(self, X, y):
        """Estimates feature importances for each topic based on feature
        importances estimated by an Extra Trees regressor.

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
        self.feature_importances_ = []
        for component in tqdm(y.T, desc="Estimating feature importances"):
            estimator = clone(self.regressor).fit(X, component)
            self.feature_importances_.append(estimator.feature_importances_)
        self.feature_importances_ = np.stack(self.feature_importances_)
        return self
