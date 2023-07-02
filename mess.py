import random

import pandas as pd
import topicwizard
import umap
from embetter.text import SentenceEncoder
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from blackbert.blackbox import BlackboxTopicModel
from blackbert.cluster import ClusterTransformer
from blackbert.feature_extraction.text import LeakyCountVectorizer
from blackbert.importance_estimation.ctfidf import CTFIDFEstimator
from blackbert.importance_estimation.trees import RandomForestEstimator
from blackbert.mixture import MixtureTransformer

data = pd.read_csv("realdonaldtrump.csv")
texts = data.content.tolist()
texts = random.sample(texts, 6000)

ica_model = make_pipeline(
    SentenceEncoder("all-MiniLM-L6-v2"),
    FastICA(n_components=15),
    MinMaxScaler(),
)

dbscan_model = make_pipeline(
    SentenceEncoder("all-MiniLM-L6-v2"),
    umap.UMAP(n_components=2),
    ClusterTransformer(DBSCAN()),
)

mixture_model = make_pipeline(
    SentenceEncoder("all-MiniLM-L6-v2"),
    umap.UMAP(n_components=5),
    MixtureTransformer(
        BayesianGaussianMixture(
            n_components=20, weight_concentration_prior=0.001
        )
    ),
)

vectorizer = LeakyCountVectorizer(stop_words="english", min_df=5, max_df=0.3)
topic_model = BlackboxTopicModel(
    model=mixture_model,
    estimator=RandomForestEstimator(),
)
pipeline = make_pipeline(vectorizer, topic_model)

pipeline.fit(texts)

topicwizard.visualize(
    pipeline=pipeline,
    corpus=texts,
    exclude_pages=[
        "documents",
    ],
)
