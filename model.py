import torch
import torch.nn as nn
from torchvision import models

# Assuming config.py is in the same directory or accessible via PYTHONPATH
import config

class PoseEstimationModel(nn.Module):
    """
    Deep learning model for camera pose estimation.
    Uses a pre-trained ResNet as a backbone and adds a custom regression head.
    """
    def __init__(self, num_output_features=5):
        """
        Initializes the PoseEstimationModel.

        Args:
            num_output_features (int): The number of output features for the pose (e.g., 5 for x, y, z, azimuth, elevation).
        """
        super(PoseEstimationModel, self).__init__()
        
        # Load a pre-trained ResNet50 model
        # Using ResNet50_Weights.IMAGENET1K_V1 for best available weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all parameters in the backbone initially.
        # These can be unfrozen later during fine-tuning.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the number of features output by the backbone's final pooling layer
        # before its original classification head.
        num_ftrs = self.backbone.fc.in_features
        
        # Replace the original fully connected layer with an identity module.
        # This effectively removes the classification head, allowing us to use
        # the backbone for feature extraction.
        self.backbone.fc = nn.Identity()

        # Define the regression head: a series of linear layers to map
        # the extracted features to the desired pose outputs.
        self.regression_head = nn.Sequential(
            nn.Linear(num_ftrs, 1024), # First linear layer
            nn.ReLU(),                 # ReLU activation for non-linearity
            nn.Dropout(0.5),           # Dropout for regularization to prevent overfitting
            nn.Linear(1024, 512),      # Second linear layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_output_features) # Final linear layer outputting the 5 pose values
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predicted pose values.
        """
        # Pass the input through the ResNet backbone to extract features.
        # Gradients will only be computed for layers whose requires_grad is True.
        features = self.backbone(x)
        
        # Pass the extracted features through the custom regression head.
        output = self.regression_head(features)
        return output

# This block allows you to test the model module independently.
# It will only run when model.py is executed directly.
if __name__ == '__main__':
    print("Testing model.py module...")
    
    # Create a dummy input tensor (batch_size, channels, height, width)
    # 224x224 is the expected input size for ResNet50 after transforms.
    dummy_input = torch.randn(1, 3, 224, 224).to(config.DEVICE)
    
    # Initialize the model and move it to the configured device
    model = PoseEstimationModel(num_output_features=5).to(config.DEVICE)
    print(f"Model initialized and moved to {config.DEVICE}")
    
    # Print model summary (optional, requires torchsummary)
    # from torchsummary import summary
    # summary(model, (3, 224, 224))

    # Test a forward pass
    try:
        output = model(dummy_input)
        print(f"Forward pass successful.")
        print(f"Output tensor shape: {output.shape}")
        print(f"Output values: {output.cpu().detach().numpy()}")
    except Exception as e:
        print(f"Error during forward pass: {e}")

    # Test freezing/unfreezing logic (manual check)
    print("\nInitial requires_grad status:")
    for name, param in model.named_parameters():
        if "backbone.fc" not in name: # Exclude identity layer
            print(f"{name}: {param.requires_grad}")
    
    print("\nAttempting to unfreeze layer4 and layer3:")
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    for param in model.backbone.layer3.parameters():
        param.requires_grad = True

    print("\nRequires_grad status after unfreezing:")
    for name, param in model.named_parameters():
        if "backbone.fc" not in name:
            print(f"{name}: {param.requires_grad}")

