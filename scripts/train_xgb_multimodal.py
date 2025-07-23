"""
Train an XGBoost regressor on the early‑fusion multimodal feature table
`data/multimodal/multimodal_features.csv` and print MAE + RMSE.

Patched: use `np.sqrt(mean_squared_error(...))` instead of the `squared=False`
argument (compatibility with older scikit‑learn).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/multimodal/multimodal_features.csv"

df = pd.read_csv(DATA_PATH)

# Drop non‑feature columns
X = df.drop(columns=["participant_id", "PHQ8"])
y = df["PHQ8"]

# -----------------------------
# Train / validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# XGBoost regressor
# -----------------------------
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
preds = model.predict(X_val)

mae = mean_absolute_error(y_val, preds)
rmse = np.sqrt(mean_squared_error(y_val, preds))

print("MAE :", mae)
print("RMSE:", rmse)
