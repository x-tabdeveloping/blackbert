from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


class MixtureTransformer(TransformerMixin):
    """Turns sklearn mixture models into a transformer component that
    you can use in a pipeline.
    This means that the transform() method will return the probability
    of the mixture components.

    Parameters
    ----------
    model: ClusterMixin
        Sklearn mixture model.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model

    def fit(self, X, y=None):
        """Fits the mixture model.

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
        self.model.fit(X)
        return self

    def transform(self, X):
        """Returns probabilities of mixture components.

        Parameters
        ----------
        X: ndarray of shape (n_observations, n_features)
            Observations to cluster.

        Returns
        -------
        ndarray of shape (n_observations, n_clusters)
            Probabilities of mixture components for all
            observations.
        """
        return self.model.predict_proba(X)


def NormalMixture(n_components: int) -> MixtureTransformer:
    """Convenience function for using a Gaussian Mixture model
    for finding topics. Wraps the GaussianMixture class from sklearn,
    thus uses the Expectation Minimization algorithm.

    Parameters
    ----------
    n_components: int
        Number of mixture components.

    Returns
    -------
    MixtureTransformer
        Sklearn transforer wrapping GaussianMixture.
    """
    model = GaussianMixture(n_components=n_components)
    return MixtureTransformer(model)


def DirichletNormalMixture(
    n_components: int, concentration: Optional[float] = None
) -> MixtureTransformer:
    """Convenience function creating a Gaussian Mixture Model
    for estimating topic components. Uses variational inference and sets
    a Dirichlet prior over cluster weights.
    This means that this model can easily be used for empirically
    determining the number of topics as the weight
    of unimportant components vanishes.

    Parameters
    ----------
    n_components: int
        Number of mixture components. It is advisable to set this to
        a relatively high number so that the number of topics can
        be inferred from the data.
    concentration: float or None, default None
        Concentration parameter of the Dirichlet prior on the component
        weights.
        If you want the topics to be more concentrated, then set this to
        a lower number. If tou want to have more or less equally sized topics
        set this to something higher.
        Default value is 1/n_components.

    Returns
    -------
    MixtureTransformer
        Sklearn transformer wrapping BayesianGaussianMixture.
    """
    model = BayesianGaussianMixture(
        n_components=n_components, weight_concentration_prior=concentration
    )
    return MixtureTransformer(model)
