# scripts/train_text_regression.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AdamW
from models.distilbert_regressor import DistilBERTRegressor
from utils.dataset import DAICTextDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Config
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("data/transcripts.csv")  # Columns: transcript, phq8_score
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = DAICTextDataset(train_df)
val_dataset = DAICTextDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = DistilBERTRegressor().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "outputs/checkpoints/distilbert_regressor.pt")