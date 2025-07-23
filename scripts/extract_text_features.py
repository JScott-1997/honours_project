# scripts/extract_text_features.py

import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -------------------
# Config
# -------------------
INPUT_CSV = "data/transcripts/transcripts_with_labels.csv"
OUTPUT_CSV = "data/transcripts/text_features_mpnet.csv"
MODEL_NAME = "all-mpnet-base-v2"

# -------------------
# Load transcript data
# -------------------
df = pd.read_csv(INPUT_CSV)
df["transcript_text"] = df["transcript_text"].fillna("")

# -------------------
# Load MPNet model
# -------------------
print("📥 Loading MPNet model...")
model = SentenceTransformer(MODEL_NAME)

# -------------------
# Generate embeddings
# -------------------
print("🔍 Extracting MPNet embeddings...")
embeddings = []
participant_ids = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    text = row["transcript_text"]
    participant_id = row["participant_id"]

    emb = model.encode(text, normalize_embeddings=True)
    embeddings.append(emb)
    participant_ids.append(participant_id)

# -------------------
# Save to CSV
# -------------------
print("💾 Saving text features...")
embed_df = pd.DataFrame(embeddings)
embed_df["participant_id"] = participant_ids

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
embed_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ MPNet features saved to: {OUTPUT_CSV}")