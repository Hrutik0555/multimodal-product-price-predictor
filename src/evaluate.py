"""Evaluation metrics and plots."""
import json

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true, y_pred):
    """Symmetric mean absolute percentage error (in %)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def compute_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
    }


def save_metrics(metrics, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_actual_vs_predicted(y_true, y_pred, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    lim = float(max(np.max(y_true), np.max(y_pred)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4)
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title("Actual vs predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
