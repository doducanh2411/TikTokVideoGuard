import torch.nn as nn
from transformers import VivitForVideoClassification, BertModel, AutoTokenizer


class AttentionMultiModalViViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.vivit = VivitForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.vivit.classifier = nn.Linear(
            in_features=768, out_features=768, bias=True)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, batch_first=True)

        self.fc = nn.Linear(768, num_classes)

    def forward(self, x_3d, title):
        video_output = self.vivit(x_3d)
        video_features = video_output.logits.unsqueeze(
            1)  # (batch_size, 1, 768)

        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True)
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        # (batch_size, seq_len, 768)
        text_features = text_output.last_hidden_state

        attn_output, _ = self.cross_attention(
            query=text_features, key=video_features, value=video_features)

        x = attn_output.mean(dim=1)  # (batch_size, 768)

        x = self.fc(x)

        return x
