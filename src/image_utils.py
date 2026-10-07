"""Image downloading and OpenCV feature extraction."""
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

from src import config as C

log = logging.getLogger(__name__)


def image_path(url, image_dir=C.IMAGE_DIR):
    """Deterministic cache path for a URL."""
    digest = hashlib.md5(str(url).encode("utf-8")).hexdigest()
    return Path(image_dir) / f"{digest}.jpg"


def _download_one(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return False
    for _ in range(C.DOWNLOAD_RETRIES + 1):
        try:
            r = requests.get(url, timeout=C.DOWNLOAD_TIMEOUT)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                return True
        except requests.RequestException:
            continue
    return False


def download_images(urls, image_dir=C.IMAGE_DIR, workers=C.DOWNLOAD_WORKERS):
    """Download all URLs in parallel into `image_dir`. Returns number of successes."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(u for u in urls if isinstance(u, str)))
    jobs = [(u, image_path(u, image_dir)) for u in unique]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(tqdm(ex.map(lambda j: _download_one(*j), jobs),
                            total=len(jobs), desc="Downloading images"))
    ok = int(sum(results))
    log.info("Downloaded/cached %d of %d images", ok, len(jobs))
    return ok


IMAGE_FEATURE_DIM = 3 * C.HIST_BINS + 6 + 6 + 1  # hist + mean/std + shape/edge + has_image


def _empty_features():
    return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float64)


def extract_image_features(path):
    """Colour histogram (HSV), channel stats, edge density, sharpness, aspect ratio."""
    path = Path(path)
    if not path.exists():
        return _empty_features()
    img = cv2.imread(str(path))
    if img is None:
        return _empty_features()

    h, w = img.shape[:2]
    img = cv2.resize(img, (C.IMAGE_SIZE, C.IMAGE_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = []
    for ch, rng in zip(range(3), ([0, 180], [0, 256], [0, 256])):
        hh = cv2.calcHist([hsv], [ch], None, [C.HIST_BINS], rng).flatten()
        hist.append(hh / (hh.sum() + 1e-9))
    hist = np.concatenate(hist)

    mean, std = cv2.meanStdDev(img)
    edges = cv2.Canny(gray, 100, 200)
    extras = [
        edges.mean() / 255.0,                         # edge density
        cv2.Laplacian(gray, cv2.CV_64F).var() / 1e3,  # sharpness
        w / max(h, 1),                                # aspect ratio
        float(min(h, w)) / 1000.0,                    # resolution proxy
        float((gray > 240).mean()),                   # near-white background share
        float((gray < 15).mean()),                    # near-black share
    ]
    return np.concatenate([hist, mean.flatten(), std.flatten(), extras, [1.0]])


def extract_batch(urls, image_dir=C.IMAGE_DIR):
    return np.vstack([extract_image_features(image_path(u, image_dir)) for u in urls])
