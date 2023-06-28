from typing import Iterable, Optional

import scipy.sparse as spr
from sklearn.feature_extraction.text import CountVectorizer


class SparseWithText(spr.csr_array):
    """Compressed Sparse Row sparse array with a text attribute,
    this way the textual content of the sparse array can be
    passed down in a pipeline."""

    def __init__(self, *args, texts: Optional[Iterable[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if texts is None:
            self.texts = None
        else:
            self.texts = list(texts)


class LeakyCountVectorizer(CountVectorizer):
    """Leaky CountVectorizer class, that does essentially the exact same
    thing as scikit-learn's CountVectorizer, but returns a sparse
    array with the text attribute attached. (see SparseWithText)"""

    def fit_transform(self, raw_documents, y=None):
        res = super().fit_transform(raw_documents, y=y)
        return SparseWithText(res, texts=list(raw_documents))

    def transform(self, raw_documents):
        res = super().transform(raw_documents)
        return SparseWithText(res, texts=raw_documents)
