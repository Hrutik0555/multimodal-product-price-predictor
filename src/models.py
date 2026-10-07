"""Ensemble regressor (XGBoost + RandomForest + HistGradientBoosting)."""
import logging

from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)

from src import config as C

log = logging.getLogger(__name__)


def build_ensemble():
    members = [
        ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=C.N_JOBS, random_state=C.SEED)),
        ("hgb", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                              random_state=C.SEED)),
    ]
    weights = [1.0, 1.5]
    try:
        from xgboost import XGBRegressor
        members.append(("xgb", XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=7, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=C.N_JOBS,
            random_state=C.SEED)))
        weights.append(2.0)
    except ImportError:
        log.warning("xgboost not installed; using RandomForest + HistGradientBoosting only.")
    return VotingRegressor(members, weights=weights)
