# scripts/evaluate_model.py

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset import DAICTextDataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "outputs/distilbert_regression/final_model"
CSV_PATH = "data/transcripts/transcripts_with_labels.csv"
BATCH_SIZE = 8

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# Load and prepare data
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df))
val_df = df[split:]

val_dataset = DAICTextDataset(val_df, tokenizer=tokenizer)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Inference
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.squeeze().cpu().numpy()
        labels = labels.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

# Clamp predictions to valid PHQ-8 range
all_preds = np.clip(all_preds, 0, 24)

# Metrics
mae = mean_absolute_error(all_labels, all_preds)
import sklearn
rmse = root_mean_squared_error(all_labels, all_preds)
corr, _ = pearsonr(all_labels, all_preds)

print("\n📊 Evaluation Metrics:")
print(f"MAE:        {mae:.2f}")
print(f"RMSE:       {rmse:.2f}")
print(f"Pearson r:  {corr:.2f}")

# Scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(all_labels, all_preds, alpha=0.7)
plt.plot([0, 24], [0, 24], linestyle='--', color='red')
plt.xlabel("True PHQ-8")
plt.ylabel("Predicted PHQ-8")
plt.title("Predicted vs Actual PHQ-8")
plt.grid(True)
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pred_vs_actual.png")
plt.show()
