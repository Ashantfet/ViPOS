import torch
import torch.nn as nn
from torchvision import models

# Import config from the classification project directory
import config

class PoseEstimationModel(nn.Module):
    """
    Deep learning model for multi-head camera pose classification.
    Uses a pre-trained Vision Transformer (ViT) as a backbone and adds 5 separate classification heads.
    """
    def __init__(self, num_layers_to_unfreeze=3):
        """
        Initializes the PoseEstimationModel for multi-head classification.
        Each head predicts one of config.NUM_BINS classes.

        Args:
            num_layers_to_unfreeze (int): The number of final ViT encoder layers to unfreeze for fine-tuning.
        """
        super(PoseEstimationModel, self).__init__()
        
        # Load a pre-trained Vision Transformer (ViT) as the backbone.
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Freeze all parameters in the backbone initially.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- Finetuning: Unfreeze a specific number of final encoder blocks ---
        # A common practice is to unfreeze the final layers, as they are the most
        # abstract and task-specific. The ViT encoder has 12 layers.
        total_layers = len(self.backbone.encoder.layers)
        num_layers_to_unfreeze = min(num_layers_to_unfreeze, total_layers) # Safety check

        for i in range(total_layers - num_layers_to_unfreeze, total_layers):
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = True
        
        # The ViT's output feature size is 768. We get this from the final 'head' block.
        num_ftrs = self.backbone.heads.head.in_features
        
        # We replace the original final classifier with an identity module to get the features.
        self.backbone.heads.head = nn.Identity()

        # Define the common feature extractor part of the head.
        self.common_head_features = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Comment out or replace Dropout layers with nn.Identity()
            #nn.Identity(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
            # Comment out or replace Dropout layers with nn.Identity()
            #nn.Identity()
        )

        # Create separate classification heads for each pose dimension.
        self.heads = nn.ModuleList([
            nn.Linear(512, config.NUM_BINS) for _ in range(config.NUM_OUTPUT_HEADS)
        ])

    def forward(self, x):
        """
        Forward pass through the model.
        Returns:
            list: A list of torch.Tensor, where each tensor contains the logits
                  for one of the pose dimensions.
        """
        # Pass input through the ViT backbone to extract features.
        features = self.backbone(x)
        
        # Pass features through the common head layers.
        common_features = self.common_head_features(features)

        # Pass common features through each specific head.
        outputs = [head(common_features) for head in self.heads]
        
        return outputs

# This block allows you to test the model module independently.
if __name__ == '__main__':
    print("Testing classification/model.py module with ViT backbone...")
    
    # Create a dummy input tensor
    dummy_input = torch.randn(2, 3, 224, 224).to(config.DEVICE)
    
    # Instantiate the model with a fine-tuning strategy (e.g., unfreeze the last 3 layers)
    model = PoseEstimationModel(num_layers_to_unfreeze=3).to(config.DEVICE)
    print(f"Model initialized and moved to {config.DEVICE}")
    
    try:
        output_logits = model(dummy_input)
        print(f"Forward pass successful. Model returned {len(output_logits)} output tensors.")
        
        for i, logits in enumerate(output_logits):
            expected_shape = (dummy_input.shape[0], config.NUM_BINS)
            print(f"  Head {i} (for {config.DIMENSION_NAMES[i]}): Output shape: {logits.shape}, Expected: {expected_shape}")
            assert logits.shape == expected_shape, f"Head {i} output shape mismatch!"
        
        print(f"All output shapes are correct.")
        
        print("\nVerifying requires_grad status...")
        # Check a parameter in the ViT's encoder layers
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if "backbone.encoder.layers" in name and param.requires_grad:
                print(f"  {name}: {param.requires_grad}")
                unfrozen_count += 1
        
        print(f"\nFound {unfrozen_count} trainable parameters in ViT encoder layers.")
        print(f"Note: This confirms that the layers are correctly unfrozen for fine-tuning.")

    except Exception as e:
        print(f"Error during model testing: {e}")
