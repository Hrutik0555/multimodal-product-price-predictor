"""Feature pipeline: regex features + TF-IDF/SVD text + image features."""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config as C
from src.image_utils import extract_batch
from src.text_features import extract_text_features


class FeatureBuilder:
    """Fit on training data, then transform any dataframe to a dense matrix."""

    def __init__(self, use_images=True, image_dir=C.IMAGE_DIR):
        self.use_images = use_images
        self.image_dir = image_dir
        self.tfidf = TfidfVectorizer(
            max_features=C.TFIDF_MAX_FEATURES,
            ngram_range=C.TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            stop_words="english",
            dtype=np.float32,
        )
        self.svd = None

    def _text(self, df):
        return df[C.TEXT_COL].fillna("").astype(str)

    def fit(self, df):
        tfidf_matrix = self.tfidf.fit_transform(self._text(df))
        n_comp = max(1, min(C.SVD_COMPONENTS, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=C.SEED)
        self.svd.fit(tfidf_matrix)
        return self

    def transform(self, df):
        text = self._text(df)
        parts = [
            extract_text_features(text),
            self.svd.transform(self.tfidf.transform(text)),
        ]
        if self.use_images:
            parts.append(extract_batch(df[C.IMAGE_URL_COL].tolist(), self.image_dir))
        return np.hstack(parts).astype(np.float32)


def load_csv(path, require_target=False):
    df = pd.read_csv(path)
    required = [C.TEXT_COL] + ([C.TARGET_COL] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if require_target:
        df = df[df[C.TARGET_COL].notna() & (df[C.TARGET_COL] > 0)].reset_index(drop=True)
    return df
