"""Predict prices for a test CSV.

    python -m src.predict
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.image_utils import download_images
from src.preprocessing import load_csv

log = logging.getLogger("predict")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", default=str(C.TEST_CSV))
    p.add_argument("--model", default=str(C.MODEL_PATH))
    p.add_argument("--out", default=str(C.PREDICTIONS_CSV))
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bundle = joblib.load(args.model)
    fb, model = bundle["features"], bundle["model"]

    df = load_csv(args.test)
    if fb.use_images and not args.no_download:
        download_images(df[C.IMAGE_URL_COL].tolist())

    pred = np.clip(np.expm1(model.predict(fb.transform(df))), 0.01, None)
    ids = df[C.ID_COL] if C.ID_COL in df.columns else np.arange(len(df))
    out = pd.DataFrame({C.ID_COL: ids, C.TARGET_COL: np.round(pred, 2)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    log.info("Wrote %d predictions to %s", len(out), args.out)


if __name__ == "__main__":
    main()
