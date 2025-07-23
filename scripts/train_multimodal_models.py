"""
Train two multimodal PHQ‑8 models with **GPU‑accelerated XGBoost ≥ 2.0**
using **CuPy** arrays for full GPU compatibility and live device reporting.

1. **Early‑fusion hyper‑parameter search** (RandomizedSearchCV)
2. **Late‑fusion stacking** (audio / text / vision base learners → Ridge meta)

This script:
- Forces CuPy onto NVIDIA GPU: prints the device ID and name.
- Uses XGBoost 2 syntax `tree_method="hist", device="cuda:0"`.
- Provides a tqdm progress bar for the late‑fusion stacking loop.

Requirements
------------
* xgboost ≥ 2.0.0  (`pip install -U xgboost`)
* cupy-cuda12x (or matching CUDA)  (`pip install cupy-cuda12x`)
* scikit-learn, pandas, numpy, tqdm
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
import cupy as cp               # CuPy for GPU arrays
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

# Force CuPy to use NVIDIA GPU 0
cp.cuda.Device(0).use()
dev = cp.cuda.Device(0)
try:
    name = dev.name
except AttributeError:
    name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode('utf-8')
print(f"Using GPU device {dev.id}: {name}")

# GPU keyword arguments for XGBoost
GPU_KW = dict(tree_method='hist', device='cuda:0', n_jobs=1)

# ------------------------------------------------------------------
# Wrapper to auto‑convert NumPy ↔ CuPy for fit & predict
# ------------------------------------------------------------------
class XGBRegressorGPU(XGBRegressor):
    """XGBRegressor that moves data to GPU for both fit() and predict()."""

    def fit(self, X, y=None, **kwargs):
        if isinstance(X, np.ndarray):
            X = cp.asarray(X)
        if isinstance(y, np.ndarray):
            y = cp.asarray(y)
        print(f"Fitting on device: {self.get_params().get('device')}")
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if isinstance(X, np.ndarray) or isinstance(X, cp.ndarray):
            X_device = cp.asarray(X)
            preds = super().predict(X_device, **kwargs)
            return cp.asnumpy(preds)
        return super().predict(X, **kwargs)

# ------------------------------------------------------------------
# Data paths and seed
# ------------------------------------------------------------------
DATA_PATH = "data/multimodal/multimodal_features.csv"
SEED      = 42

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_table(path: str = DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["participant_id", "PHQ8"])
    y = df["PHQ8"].to_numpy()
    return X.to_numpy(), y


def split_modalities(X: np.ndarray, n_text: int):
    """Split X into (audio, text, vision) blocks based on column counts."""
    # audio: first cols, text: next n_text cols, vision: rest
    audio = X[:, :- (n_text + 204)]
    text  = X[:, - (n_text + 204) : -204]
    vision= X[:, -204:]
    return audio, text, vision

# ------------------------------------------------------------------
# 1 · Early‑fusion hyper‑parameter search (GPU)
# ------------------------------------------------------------------

def run_early_fusion_hp():
    X, y = load_table()

    param_dist = {
        "n_estimators":     [300, 500, 800, 1000],
        "max_depth":        [4, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.15],
        "subsample":        [0.6, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda":       [0, 1, 5, 10],
    }

    base = XGBRegressorGPU(
        objective="reg:squarederror",
        random_state=SEED,
        **GPU_KW,
    )

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=40,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        random_state=SEED,
        verbose=1,
    )
    search.fit(X, y)
    best_mae = -search.best_score_

    print("\n=== Early‑Fusion XGB (GPU) ===")
    print("Best MAE (5‑fold CV):", round(best_mae, 3))
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

# ------------------------------------------------------------------
# 2 · Late‑fusion stacking (GPU base learners + tqdm)
# ------------------------------------------------------------------

def run_late_fusion_stack():
    X, y = load_table()
    # assume text embedding dimension 768
    audio, text, vision = split_modalities(X, n_text=768)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    meta_preds = np.zeros((len(y), 3))

    for idx, X_mod in enumerate(tqdm([audio, text, vision], desc="Modalities")):
        print(f"Training base learner {idx+1}/3 on GPU …")
        base = XGBRegressorGPU(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=SEED,
            **GPU_KW,
        )
        for tr, val in tqdm(kf.split(X_mod), total=5, desc=f"Learner {idx+1} folds"):
            base.fit(X_mod[tr], y[tr])
            preds = base.predict(X_mod[val])
            meta_preds[val, idx] = preds

    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(meta_preds, y)
    stacked_mae = mean_absolute_error(y, meta.predict(meta_preds))

    print("\n=== Late‑Fusion Stacking (GPU + CuPy) ===")
    print("Stacked MAE (5‑fold OOF):", round(stacked_mae, 3))

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_early_fusion_hp()
    run_late_fusion_stack()
