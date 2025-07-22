import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Load transcript CSV
INPUT_CSV = "data/transcripts/transcripts_with_labels.csv"
OUTPUT_CSV = "data/transcripts/text_features_distilbert.csv"
df = pd.read_csv(INPUT_CSV)
df["transcript_text"] = df["transcript_text"].fillna("")

# Load DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

# Helper: extract mean-pooled embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Extract embeddings
embeddings = []
participant_ids = []

print("🔍 Extracting BERT features...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    embedding = get_embedding(row["transcript_text"])
    embeddings.append(embedding)
    participant_ids.append(row["participant_id"])

# Save as CSV
embed_df = pd.DataFrame(embeddings)
embed_df["participant_id"] = participant_ids
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
embed_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved to {OUTPUT_CSV}")