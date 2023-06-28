"""Feature importance estimation with linear models."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression


class OLSEstimator(BaseEstimator):
    """Estimate feature importances for components from Ordinary Least Squares
    linear regression coefficients.

    Attributes
    ----------
    feature_importances_: ndarray of shape (n_components, n_features)
        Importance of each feature for each output feature.
    """

    def __init__(self, *args, **kwargs):
        super(OLSEstimator, self).__init__()
        self.regressor = LinearRegression(*args, **kwargs)

    def fit(self, X, y):
        """Estimates feature importances for each topic based on
        the coefficients of an OLS linear regression model.

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
        self.feature_importances_ = self.regressor.fit(X, y).coef_
        return self


class LassoEstimator(BaseEstimator):
    """Estimate feature importances for components from Lasso
    linear regression coefficients.

    Attributes
    ----------
    feature_importances_: ndarray of shape (n_components, n_features)
        Importance of each feature for each output feature.
    """

    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        super(LassoEstimator, self).__init__()
        self.regressor = Lasso(alpha=alpha, *args, **kwargs)

    def fit(self, X, y):
        """Estimates feature importances for each topic based on
        the coefficients of an Lasso linear regression model.

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
        self.feature_importances_ = self.regressor.fit(X, y).coef_
        return self


class LogisticEstimator(BaseEstimator):
    """Estimate feature importances for components from
    logistic regression coefficients.
    Determines topic label based on most prominant component.

    Attributes
    ----------
    feature_importances_: ndarray of shape (n_components, n_features)
        Importance of each feature for each output feature.
    """

    def __init__(self, *args, **kwargs):
        super(LogisticEstimator, self).__init__()
        self.regressor = LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        """Estimates feature importances for each topic based on
        the coefficients of an Lasso linear regression model.

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
        topic_labels = np.argmax(y, axis=1)
        self.feature_importances_ = self.regressor.fit(X, topic_labels).coef_
        return self
