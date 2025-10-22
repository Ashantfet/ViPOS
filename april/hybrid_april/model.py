# #april/tmp/apriltag_6dof_hybrid_output
# import torch
# import torch.nn as nn
# import config

# class PoseEstimationModel(nn.Module):
#     def __init__(self):
#         super(PoseEstimationModel, self).__init__()

#         # CNN Encoder (local features)
#         self.cnn_encoder = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#         )

#         # Transformer Encoder (global context)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=256, nhead=8, dim_feedforward=512, dropout=config.DROPOUT, batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

#         # Multi-head classifiers
#         self.heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.LayerNorm(256),
#                 nn.Dropout(config.DROPOUT),
#                 nn.Linear(256, config.NUM_BINS)
#             ) for _ in range(config.NUM_OUTPUT_HEADS)
#         ])

#     def forward(self, x):
#         features = self.cnn_encoder(x)  # [B, C, H, W]
#         b, c, h, w = features.size()
#         features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, N, C]

#         encoded = self.transformer_encoder(features)  # [B, N, 256]
#         pooled = encoded.mean(dim=1)  # [B, 256]

#         outputs = [head(pooled) for head in self.heads]
#         return outputs

# # april/tmp/apriltag_6dof_hybrid_output_8lcnn
# import torch
# import torch.nn as nn
# import config


# class PoseEstimationModel(nn.Module):
#     def __init__(self):
#         super(PoseEstimationModel, self).__init__()

#         # ===== CNN Encoder (8 layers) =====
#         self.cnn_encoder = nn.Sequential(
#             # Layer 1
#             nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             # Layer 2
#             nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             # Layer 3
#             nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             # Layer 4
#             nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             # Layer 5
#             nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             # Layer 6
#             nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),

#             # Layer 7
#             nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             # Layer 8
#             nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
#         )

#         # ===== Transformer Encoder =====
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=256,
#             nhead=8,
#             dim_feedforward=512,
#             dropout=config.DROPOUT,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

#         # ===== Multi-head Classifiers =====
#         self.heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.LayerNorm(256),
#                 nn.Dropout(config.DROPOUT),
#                 nn.Linear(256, config.NUM_BINS)
#             )
#             for _ in range(config.NUM_OUTPUT_HEADS)
#         ])

#     def forward(self, x):
#         features = self.cnn_encoder(x)  # [B, C, H, W]
#         b, c, h, w = features.size()
#         features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, N, C]

#         encoded = self.transformer_encoder(features)  # [B, N, 256]
#         pooled = encoded.mean(dim=1)  # [B, 256]

#         outputs = [head(pooled) for head in self.heads]
#         return outputs


# # ===== Optional Test Run =====
# if __name__ == "__main__":
#     model = PoseEstimationModel()
#     dummy = torch.randn(1, 3, 256, 256)
#     out = model(dummy)
#     for i, o in enumerate(out):
#         print(f"Head {i} output shape: {o.shape}")
#     print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")




# #april/tmp/apriltag_6dof_hybrid_output_5L
# import torch
# import torch.nn as nn
# import config


# # ===== Basic Conv Block =====
# class ConvBlock(nn.Module):
#     """Conv → BN → ReLU"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.block(x)


# # ===== Pose Estimation Model =====
# class PoseEstimationModel(nn.Module):
#     def __init__(self):
#         super(PoseEstimationModel, self).__init__()

#         # -----------------------------
#         # CNN ENCODER (Progressive 64→512)
#         # -----------------------------
#         self.cnn_encoder = nn.Sequential(
#             # Stage 1
#             ConvBlock(3, 64),
#             ConvBlock(64, 64),
#             nn.MaxPool2d(2, 2),

#             # Stage 2
#             ConvBlock(64, 128),
#             ConvBlock(128, 128),
#             nn.MaxPool2d(2, 2),

#             # Stage 3
#             ConvBlock(128, 256),
#             ConvBlock(256, 256),
#             nn.MaxPool2d(2, 2),

#             # Stage 4
#             ConvBlock(256, 512),
#             ConvBlock(512, 512),
#             nn.MaxPool2d(2, 2),

#             # Extra Refinement Layers
#             ConvBlock(512, 512),
#             ConvBlock(512, 512),
#         )

#         # -----------------------------
#         # TRANSFORMER ENCODER
#         # -----------------------------
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=512,  # Match final CNN feature depth
#             nhead=8,
#             dim_feedforward=1024,
#             dropout=config.DROPOUT,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

#         # -----------------------------
#         # MULTI-HEAD CLASSIFIERS
#         # -----------------------------
#         self.heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.LayerNorm(512),
#                 nn.Dropout(config.DROPOUT),
#                 nn.Linear(512, config.NUM_BINS)
#             ) for _ in range(config.NUM_OUTPUT_HEADS)
#         ])

#     def forward(self, x):
#         # CNN feature extraction
#         features = self.cnn_encoder(x)  # [B, 512, H', W']
#         b, c, h, w = features.size()

#         # Flatten spatial dimensions for transformer
#         features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, N, 512]

#         # Transformer encoder for global reasoning
#         encoded = self.transformer_encoder(features)  # [B, N, 512]
#         pooled = encoded.mean(dim=1)  # [B, 512]

#         # Predict 6-DoF (each output head handles one component)
#         outputs = [head(pooled) for head in self.heads]
#         return outputs


# # ===== Test Run (Optional) =====
# if __name__ == "__main__":
#     import torch
#     dummy = torch.randn(2, 3, 256, 256)
#     model = PoseEstimationModel()
#     outputs = model(dummy)
#     print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
#     for i, o in enumerate(outputs):
#         print(f"Head {i}: {o.shape}")


# april/tmp/apriltag_6dof_hybrid_output_v2
import torch
import torch.nn as nn
import math
import config


# ---------------------------------------------------
# 🔹 Conv Block (Conv → GroupNorm → ReLU)
# ---------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------
# 🔹 Robust 2D Positional Encoding (supports non-square)
# ---------------------------------------------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D positional encoding")

    def forward(self, x):
        """
        x: [B, N, C] where N = H*W
        Works for non-square H, W.
        """
        b, n, c = x.shape

        # Try to infer H, W automatically
        h = w = int(math.sqrt(n))
        if h * w != n:
            # handle non-square features (fallback)
            w = int(math.sqrt(n * 4 / 3))  # small correction
            h = n // w

        if h * w != n:
            raise ValueError(f"Cannot reshape feature tokens ({n}) into 2D grid. Got h={h}, w={w}")

        # Create meshgrid for positions
        y_pos = torch.arange(h, device=x.device).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w, device=x.device).unsqueeze(0).repeat(h, 1)

        div_term = torch.exp(
            torch.arange(0, c // 2, 2, device=x.device).float() * (-math.log(10000.0) / (c // 2))
        )

        # Build sine/cosine patterns
        pe = torch.zeros(1, c, h, w, device=x.device)
        pe[0, 0::4, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.unsqueeze(-1).unsqueeze(-1))
        pe[0, 1::4, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.unsqueeze(-1).unsqueeze(-1))
        pe[0, 2::4, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.unsqueeze(-1).unsqueeze(-1))
        pe[0, 3::4, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.unsqueeze(-1).unsqueeze(-1))

        # Flatten back to [B, N, C]
        pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, c)
        return x + pe


# ---------------------------------------------------
# 🔹 Attention Pooling (instead of mean)
# ---------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        b, n, c = x.shape
        query = self.query.expand(b, -1, -1)  # [B, 1, C]
        pooled, _ = self.attn(query, x, x)
        return pooled.squeeze(1)  # [B, C]


# ---------------------------------------------------
# 🔹 Pose Estimation Hybrid Model
# ---------------------------------------------------
class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()

        # -----------------------------
        # CNN Encoder (8 layers)
        # -----------------------------
        self.cnn_encoder = nn.Sequential(
            # Stage 1
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),

            # Stage 2
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),

            # Stage 3
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),

            # Stage 4
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
        )

        # -----------------------------
        # Transformer Encoder
        # -----------------------------
        d_model = 256
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.pos_encoding = PositionalEncoding2D(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # -----------------------------
        # Attention Pooling for global context
        # -----------------------------
        self.attn_pool = AttentionPooling(d_model, heads=4)

        # -----------------------------
        # Multi-Head Regression Heads
        # -----------------------------
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(config.DROPOUT),
                nn.Linear(d_model, config.NUM_BINS)
            )
            for _ in range(config.NUM_OUTPUT_HEADS)
        ])

    def forward(self, x):
        # 1. CNN feature extraction
        features = self.cnn_encoder(x)  # [B, 256, H', W']
        b, c, h, w = features.size()

        # 2. Flatten + add positional encodings
        features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        features = self.pos_encoding(features)

        # 3. Transformer for global reasoning
        encoded = self.transformer_encoder(features)

        # 4. Attention pooling instead of mean
        pooled = self.attn_pool(encoded)

        # 5. Predict with multi-heads
        outputs = [head(pooled) for head in self.heads]
        return outputs


# ---------------------------------------------------
# 🔹 Helper Function
# ---------------------------------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total


# ---------------------------------------------------
# 🔹 Optional Sanity Check
# ---------------------------------------------------
if __name__ == "__main__":
    dummy = torch.randn(2, 3, 256, 256)
    model = PoseEstimationModel()
    outputs = model(dummy)
    count_parameters(model)
    for i, o in enumerate(outputs):
        print(f"Head {i} → {o.shape}")
