import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from transformers import BertModel, AutoTokenizer


class AttentionMultiModalEarlyFusion(nn.Module):
    def __init__(self, num_classes=4, num_input_channels=32):
        super().__init__()

        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.cnn.features[0][0] = nn.Conv2d(num_input_channels * 3, 32, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.cnn.classifier[1] = nn.Linear(
            self.cnn.classifier[1].in_features, 768)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.video_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, batch_first=True)
        self.text_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, batch_first=True)

        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d, title):
        # (bs, T*C, H, W)
        x_3d = x_3d.view(x_3d.size(0), x_3d.size(
            1) * x_3d.size(2), x_3d.size(3), x_3d.size(4))

        video_features = self.cnn(x_3d).unsqueeze(1)  # (batch_size, 1, 768)

        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True)
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        # (batch_size, seq_len, 768)
        text_features = text_output.last_hidden_state

        attention_video, _ = self.video_attention(
            video_features, video_features, video_features)

        attention_text, _ = self.text_attention(
            text_features, text_features, text_features)

        combined_features = torch.cat((attention_video, attention_text), dim=1)
        pooled_features = combined_features.mean(dim=1)  # (batch_size, 768)

        x = self.fc1(pooled_features)
        x = F.relu(x)
        x = self.fc2(x)

        return x
