"""Central configuration."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "price_model.joblib"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
PLOT_PATH = OUTPUT_DIR / "actual_vs_predicted.png"

# Column names
ID_COL = "sample_id"
TEXT_COL = "catalog_content"
IMAGE_URL_COL = "image_link"
TARGET_COL = "price"

# Text features
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
SVD_COMPONENTS = 64

# Image features
IMAGE_SIZE = 128
HIST_BINS = 16
DOWNLOAD_WORKERS = 16
DOWNLOAD_TIMEOUT = 10
DOWNLOAD_RETRIES = 2

# Training
VAL_SIZE = 0.15
SEED = 42
N_JOBS = -1
