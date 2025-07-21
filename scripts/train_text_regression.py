# scripts/train_distilbert_regression.py

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# -------------
# Configuration
# -------------
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/transcripts/transcripts_with_labels.csv"
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5
OUTPUT_DIR = "outputs/distilbert_regression"
SEED = 42

# -------------
# Load Dataset
# -------------
df = pd.read_csv(DATA_PATH)
df = df[["transcript_text", "phq8_score"]].dropna()
df["phq8_score"] = df["phq8_score"].astype(float)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

# Hugging Face format
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# ------------------
# Tokenization
# ------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(example["transcript_text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Hugging Face expects a "labels" column
train_dataset = train_dataset.rename_column("phq8_score", "labels")
val_dataset = val_dataset.rename_column("phq8_score", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------------
# Model (Regression)
# ---------------------
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression"
)

# ---------------------
# Metrics
# ---------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.squeeze(predictions)
    mse = mean_squared_error(labels, preds)
    return {"mse": mse}

# ---------------------
# TrainingArguments
# ---------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    greater_is_better=False,
    seed=SEED,
    report_to="none"
)

# ---------------------
# Trainer
# ---------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ---------------------
# Save final model
# ---------------------
trainer.save_model(f"{OUTPUT_DIR}/final_model")
print(f"✅ Model saved to: {OUTPUT_DIR}/final_model")
