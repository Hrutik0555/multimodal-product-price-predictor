# multimodal-product-price-predictor

An end-to-end multimodal machine learning pipeline that uses tabular metadata, NLP/regex feature extraction, and computer vision to predict product list prices.

## Features

- Product price prediction (log-target regression)
- Regex extraction of value, unit, pack quantity and other metadata from text
- TF-IDF + truncated SVD text representation
- Parallel image downloading with on-disk caching
- OpenCV image features (HSV histograms, channel statistics, edges, sharpness)
- Ensemble of XGBoost, RandomForest and HistGradientBoosting
- Evaluation metrics (MAE, RMSE, R2, SMAPE) and an actual-vs-predicted plot
- CSV prediction export

## Tech stack

Python, scikit-learn, pandas, NumPy, Matplotlib, OpenCV, XGBoost, Jupyter

## Project structure

```
multimodal-product-price-predictor/
├── data/                 # train.csv, test.csv (see data/README.md)
├── models/               # saved model bundle (price_model.joblib)
├── notebooks/            # exploration notebook
├── outputs/              # metrics.json, plot, predictions.csv
├── src/
│   ├── config.py         # paths, column names, hyperparameters
│   ├── text_features.py  # regex feature extraction
│   ├── image_utils.py    # image download + OpenCV features
│   ├── preprocessing.py  # FeatureBuilder (text + image)
│   ├── models.py         # ensemble definition
│   ├── evaluate.py       # metrics and plots
│   ├── train.py          # training entrypoint
│   └── predict.py        # inference entrypoint
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Put `train.csv` and `test.csv` in `data/` (columns described in [data/README.md](data/README.md)), then:

```bash
python -m src.train                # text + image features
python -m src.train --no-images    # text/tabular only (fast)
python -m src.predict              # writes outputs/predictions.csv
```

## How it works

1. **Text/tabular:** regexes parse `Value`, `Unit` and pack quantities; units are normalised to grams / millilitres / counts. Text statistics are added, and TF-IDF is compressed to 64 dimensions with SVD.
2. **Images:** downloaded once, cached by URL hash, then summarised into colour, edge and sharpness features. Missing images give zero vectors plus a `has_image = 0` flag.
3. **Model:** features are concatenated and an ensemble is trained on `log1p(price)`; predictions are mapped back with `expm1`.
4. **Evaluation:** a 15% hold-out set gives MAE, RMSE, R2 and SMAPE; the final model is then refit on all data.

## Notes

- Column names are assumptions (`sample_id`, `catalog_content`, `image_link`, `price`); change them in `src/config.py`.
- Hand-crafted image features are lightweight; CNN embeddings would be a natural upgrade.
