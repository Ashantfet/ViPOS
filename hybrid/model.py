import torch
import torch.nn as nn
import config

class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()

        # CNN Encoder (local features)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Transformer Encoder (global context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=config.DROPOUT, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Multi-head classifiers
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(256),
                nn.Dropout(config.DROPOUT),
                nn.Linear(256, config.NUM_BINS)
            ) for _ in range(config.NUM_OUTPUT_HEADS)
        ])

    def forward(self, x):
        features = self.cnn_encoder(x)  # [B, C, H, W]
        b, c, h, w = features.size()
        features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, N, C]

        encoded = self.transformer_encoder(features)  # [B, N, 256]
        pooled = encoded.mean(dim=1)  # [B, 256]

        outputs = [head(pooled) for head in self.heads]
        return outputs
