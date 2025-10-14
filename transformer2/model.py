import torch.nn as nn
from torchvision import models
import config

class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()
        # Train ViT from scratch by not using pre-trained weights
        self.backbone = models.vit_b_16(weights=None)
        
        # All parameters are trainable from the start
        for param in self.backbone.parameters():
            param.requires_grad = True

        num_ftrs = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity()

        # self.common_head_features = nn.Sequential(
        #     nn.Linear(num_ftrs, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        self.common_head_features = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            #nn.Dropout(0.5),
            # Comment out or replace Dropout layers with nn.Identity()
            nn.Identity(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            #nn.Dropout(0.5)
            # Comment out or replace Dropout layers with nn.Identity()
            nn.Identity()
        )
        self.heads = nn.ModuleList([
            nn.Linear(512, config.NUM_BINS) for _ in range(len(config.DIMENSION_NAMES))
        ])

    def forward(self, x):
        features = self.backbone(x)
        common_features = self.common_head_features(features)
        outputs = [head(common_features) for head in self.heads]
        return outputs
