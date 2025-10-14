import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import math

# ==================== CONFIGURATION ====================
# This config is self-contained for this script.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "tmp", "blender_data_final_render")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
# Path to your saved deep learning model checkpoint
MODEL_CHECKPOINT_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "tmp", "classification_model_outputs_ViT", "model_best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_BINS = 100
DIMENSION_NAMES = ['norm_x', 'norm_y', 'norm_z', 'norm_azimuth', 'norm_elevation']
DIMENSION_RANGES = {dim: {'min': -1.0, 'max': 1.0} for dim in DIMENSION_NAMES}
BIN_EDGES = {dim: np.linspace(dim_range['min'], dim_range['max'], NUM_BINS + 1)
             for dim, dim_range in DIMENSION_RANGES.items()}
for dim in BIN_EDGES:
    BIN_EDGES[dim][-1] += 1e-6

# Transformations for feature extraction
FEATURE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== DEEP LEARNING MODEL ARCHITECTURE ====================
# We need to define the model architecture here to be able to load its state_dict
class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()
        # NEW: Use Vision Transformer Backbone
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # ViT's feature dimension is 768, not 2048 like ResNet
        num_ftrs = self.backbone.heads.head.in_features
        # Replace the original final classifier
        self.backbone.heads.head = nn.Identity()

        self.common_head_features = nn.Sequential(
            nn.Linear(num_ftrs, 1024), # This will now be Linear(768, 1024)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.heads = nn.ModuleList([
            nn.Linear(512, NUM_BINS) for _ in range(len(DIMENSION_NAMES))
        ])

    def forward(self, x):
        features = self.backbone(x)
        common_features = self.common_head_features(features)
        outputs = [head(common_features) for head in self.heads]
        return outputs

def load_fine_tuned_feature_extractor(model_checkpoint_path, device):
    """Loads the fine-tuned ViT backbone from a saved model checkpoint."""
    print("Loading fine-tuned deep learning model...")
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_checkpoint_path}")
        
    model = PoseEstimationModel().to(device)
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    feature_extractor = nn.Sequential(
        model.backbone,
        model.common_head_features
    )
    
    print("Fine-tuned feature extractor loaded successfully.")
    return feature_extractor

# ==================== DATASET UTILITIES ====================
def _get_bin_index(value, dim_name, bin_edges):
    bin_index = np.digitize(value, bin_edges[dim_name]) - 1
    return np.clip(bin_index, 0, NUM_BINS - 1).item()

def load_data_and_extract_features(metadata_file, data_dir, device, feature_extractor, num_images_to_process=None):
    """
    Loads data, extracts features from images using the provided feature_extractor,
    and discretizes the labels into bin indices.
    """
    print("Loading metadata...")
    metadata_frame = pd.read_csv(metadata_file)
    if num_images_to_process:
        print(f"Processing a subset of {num_images_to_process} images.")
        metadata_frame = metadata_frame.iloc[:min(num_images_to_process, len(metadata_frame))]

    all_features = []
    all_labels = []

    print("Extracting features and preparing labels...")
    with torch.no_grad():
        for idx in tqdm(range(len(metadata_frame)), desc="Processing Images"):
            img_name = metadata_frame.iloc[idx]['image_path']
            image_path = os.path.join(data_dir, os.path.basename(img_name))
            
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = FEATURE_TRANSFORMS(image).unsqueeze(0).to(device)
                
                features_tensor = feature_extractor(image_tensor).squeeze(0).cpu()
                all_features.append(features_tensor.numpy())
                
                continuous_pose_values = metadata_frame.iloc[idx][DIMENSION_NAMES].values.astype(float)
                binned_labels = [
                    _get_bin_index(continuous_pose_values[i], dim_name, BIN_EDGES)
                    for i, dim_name in enumerate(DIMENSION_NAMES)
                ]
                all_labels.append(binned_labels)
            except Exception as e:
                print(f"Warning: Skipping image {image_path} due to error: {e}")
                
    return np.array(all_features), np.array(all_labels)

# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    # 1. Load the fine-tuned feature extractor
    fine_tuned_extractor = load_fine_tuned_feature_extractor(MODEL_CHECKPOINT_PATH, DEVICE)
    
    # 2. Load data and extract features using the fine-tuned model
    features, labels = load_data_and_extract_features(
        METADATA_FILE, DATA_DIR, DEVICE, fine_tuned_extractor
    )
    
    if features.size == 0 or labels.size == 0:
        print("Error: No data to train on after feature extraction. Exiting.")
    else:
        # 3. Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.4, random_state=42
        )
        print(f"\nTraining with {len(X_train)} samples, testing with {len(X_test)} samples.")
        
        # 4. Train a separate Random Forest model for each dimension
        models = {}
        for i, dim_name in enumerate(DIMENSION_NAMES):
            print(f"\n--- Training Random Forest for '{dim_name}' ---")
            y_train_dim = y_train[:, i]
            y_test_dim = y_test[:, i]
            
            model = RandomForestClassifier(n_estimators=500, max_depth=25, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train_dim)
            models[dim_name] = model
            
            # 5. Evaluate the model
            y_pred_dim = model.predict(X_test)
            accuracy = accuracy_score(y_test_dim, y_pred_dim)
            print(f"Accuracy for '{dim_name}': {accuracy:.4f} (or {accuracy*100:.2f}%)")

        # 6. Overall evaluation
        print("\n--- Overall Performance on Test Set ---")
        all_preds = []
        for dim_name in DIMENSION_NAMES:
            model = models[dim_name]
            y_pred_dim = model.predict(X_test)
            all_preds.append(y_pred_dim)
        
        all_preds = np.stack(all_preds, axis=1)
        
        average_accuracy = np.mean([accuracy_score(y_test[:, i], all_preds[:, i]) for i in range(5)])
        print(f"Overall Average Accuracy: {average_accuracy:.4f} (or {average_accuracy*100:.2f}%)")
        
        exact_matches = np.all(all_preds == y_test, axis=1)
        exact_match_accuracy = np.mean(exact_matches)
        print(f"Exact Match Accuracy (All 5 Dimensions Correct): {exact_match_accuracy:.4f} (or {exact_match_accuracy*100:.2f}%)")
