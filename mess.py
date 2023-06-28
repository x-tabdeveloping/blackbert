import random

import topicwizard
import umap
from embetter.text import SentenceEncoder
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

from blackbert.blackbox import BlackboxTopicModel
from blackbert.cluster.dbscan import DBSCANTransformer
from blackbert.feature_extraction.text import LeakyCountVectorizer
from blackbert.importance_estimation.ctfidf import CTFIDFEstimator
from blackbert.importance_estimation.trees import RandomForestEstimator

with open("processed_sample.txt") as f:
    texts = list(f)
    texts = random.sample(texts, 4000)

ica_model = make_pipeline(
    SentenceEncoder("all-MiniLM-L6-v2"),
    FastICA(n_components=15),
    MinMaxScaler(),
)


dbscan_model = make_pipeline(
    SentenceEncoder("all-MiniLM-L6-v2"),
    umap.UMAP(n_components=2),
    DBSCANTransformer(),
)
vectorizer = LeakyCountVectorizer(stop_words="english", min_df=5, max_df=0.5)
topic_model = BlackboxTopicModel(
    model=dbscan_model,
    estimator=CTFIDFEstimator(),
)
pipeline = make_pipeline(vectorizer, topic_model)
pipeline.fit(texts)

topic_model.components_.shape

topic_model.components_

topicwizard.visualize(pipeline=pipeline, corpus=random.sample(texts, 500))
