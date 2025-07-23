import os
import pandas as pd

# ------------------ paths ------------------
TEXT_PATH   = "data/transcripts/text_features_mpnet.csv"
AUDIO_PATH  = "data/audio_features/audio_dataset_final.csv"
VISION_PATH = "data/video/visual_features_with_labels.csv"
LABEL_DIR   = "data/labels"
OUT_PATH    = "data/multimodal/multimodal_features.csv"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ------------------ helper ------------------
def load_labels() -> pd.Series:
    train = pd.read_csv(os.path.join(LABEL_DIR, "train_split_Depression_AVEC2017.csv"))
    dev   = pd.read_csv(os.path.join(LABEL_DIR, "dev_split_Depression_AVEC2017.csv"))
    df = pd.concat([train, dev], ignore_index=True)
    df["Participant_ID"] = df["Participant_ID"].apply(lambda pid: int(pid))
    return df.set_index("Participant_ID")["PHQ8_Score"]

labels = load_labels()

def clean(df: pd.DataFrame, id_col: str, label_col: str | None) -> pd.DataFrame:
    df[id_col] = df[id_col].astype(str).str.extract(r"(\d+)").astype(int)
    if label_col and label_col in df.columns:
        df = df.rename(columns={label_col: "PHQ8"})
    return df

# ------------------ load each modality ------------------
text   = pd.read_csv(TEXT_PATH)
audio  = pd.read_csv(AUDIO_PATH)
vision = pd.read_csv(VISION_PATH)

text  = clean(text,   "participant_id", None)          # label absent
audio = clean(audio,  "participant_id", "PHQ8_Score")
vision= clean(vision, "participant_id", "phq8_score")

# add PHQ8 to text from lookup
text = text.assign(PHQ8=text["participant_id"].map(labels))

print(f"text  : {text.shape[0]} rows")
print(f"audio : {audio.shape[0]} rows")
print(f"vision: {vision.shape[0]} rows")

# ------------------ merge ------------------
df = (
    vision
    .merge(audio, on=["participant_id", "PHQ8"], how="inner")
    .merge(text,  on=["participant_id", "PHQ8"], how="inner")
)

# sanity-check: one PHQ8 per participant
assert df.groupby("participant_id")["PHQ8"].nunique().eq(1).all()

print("Final multimodal shape:", df.shape)
df.to_csv(OUT_PATH, index=False)
print(f"✅ Saved to {OUT_PATH}")
