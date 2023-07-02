from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from blackbert.blackbox import BlackboxTopicModel
from blackbert.cluster import ClusterTransformer
from blackbert.importance_estimation.trees import RandomForestEstimator

try:
    from embetter.text import SentenceEncoder
except ImportError as e:
    raise ImportError(
        "If you intend to use transformer-based topic models, "
        "you should install embetter with sentence transformers"
        "\n pip install embetter[text]"
    ) from e


class BertICA(BlackboxTopicModel):
    def __init__(
        self,
        n_components: int,
        trf_model: str = "all-MiniLM-L6-v2",
    ):
        topic_model = make_pipeline(
            SentenceEncoder(trf_model),
            FastICA(n_components=n_components),
            MinMaxScaler(),
        )
        super(BertICA, self).__init__(
            model=topic_model, estimator=RandomForestEstimator()
        )


class BertDBSCAN(BlackboxTopicModel):
    def __init__(
        self,
        trf_model: str = "all-MiniLM-L6-v2",
    ):
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "If you intend to use the DBSCAN-model, you "
                "should install UMAP for dimensionality reduction."
                "\n pip install umap-learn"
            ) from e
        topic_model = make_pipeline(
            SentenceEncoder(trf_model),
            umap.UMAP(n_components=5),
            ClusterTransformer(DBSCAN()),
        )
        super(BertDBSCAN, self).__init__(
            model=topic_model, estimator=RandomForestEstimator()
        )
