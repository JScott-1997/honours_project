import torch
import torch.nn as nn
from transformers import DistilBertModel

class DistilBERTRegressor(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.3):
        super(DistilBERTRegressor, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)  # Predicts PHQ-8 score

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        dropped = self.dropout(cls_token)
        return self.regressor(dropped).squeeze(1)
