"""Train the multimodal price model.

    python -m src.train                 # text + image features
    python -m src.train --no-images     # text/tabular features only
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src import config as C
from src.evaluate import compute_metrics, plot_actual_vs_predicted, save_metrics
from src.image_utils import download_images
from src.models import build_ensemble
from src.preprocessing import FeatureBuilder, load_csv

log = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=str(C.TRAIN_CSV))
    p.add_argument("--no-images", action="store_true", help="skip image features")
    p.add_argument("--no-download", action="store_true", help="use already cached images")
    p.add_argument("--sample", type=int, default=0, help="train on a random subset (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    use_images = not args.no_images

    df = load_csv(args.train, require_target=True)
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=C.SEED).reset_index(drop=True)
    log.info("Loaded %d training rows", len(df))

    if use_images and not args.no_download:
        download_images(df[C.IMAGE_URL_COL].tolist())

    # --- validation run -----------------------------------------------------
    tr, va = train_test_split(df, test_size=C.VAL_SIZE, random_state=C.SEED)
    fb = FeatureBuilder(use_images=use_images).fit(tr)
    X_tr, X_va = fb.transform(tr), fb.transform(va)
    y_tr, y_va = tr[C.TARGET_COL].to_numpy(float), va[C.TARGET_COL].to_numpy(float)

    model = build_ensemble()
    model.fit(X_tr, np.log1p(y_tr))                      # log target stabilises skewed prices
    pred = np.clip(np.expm1(model.predict(X_va)), 0, None)

    metrics = compute_metrics(y_va, pred)
    metrics.update({"n_train": len(tr), "n_val": len(va), "use_images": use_images})
    log.info("Validation metrics: %s", metrics)
    save_metrics(metrics, C.METRICS_JSON)
    plot_actual_vs_predicted(y_va, pred, C.PLOT_PATH)

    # --- refit on all data and save ----------------------------------------
    fb_full = FeatureBuilder(use_images=use_images).fit(df)
    model_full = build_ensemble()
    model_full.fit(fb_full.transform(df), np.log1p(df[C.TARGET_COL].to_numpy(float)))

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": fb_full, "model": model_full}, C.MODEL_PATH)
    log.info("Saved model to %s", C.MODEL_PATH)


if __name__ == "__main__":
    main()
