import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DAICTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer=None, max_length=512):
        self.texts = dataframe["transcript_text"].tolist()
        self.labels = dataframe["phq8_score"].tolist()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }
