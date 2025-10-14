import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

# ==================== CONFIGURATION ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "tmp", "blender_data_final_render")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
# Path to your saved deep learning model checkpoint
MODEL_CHECKPOINT_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "tmp", "classification_model_outputs_ViT", "model_best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

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
class FeatureExtractor(nn.Module):
    """
    A frozen ViT backbone to extract features.
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.heads.head = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

class MLPHeads(nn.Module):
    """
    The trainable MLP classifier heads.
    """
    def __init__(self, num_input_features):
        super(MLPHeads, self).__init__()
        self.common_head_features = nn.Sequential(
            nn.Linear(num_input_features, 1024),
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
        common_features = self.common_head_features(x)
        outputs = [head(common_features) for head in self.heads]
        return outputs

# ==================== DATASET UTILITIES ====================
class FeatureDataset(Dataset):
    """Dataset for loading pre-extracted features and labels."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def _get_bin_index(value, dim_name, bin_edges):
    bin_index = np.digitize(value, bin_edges[dim_name]) - 1
    return np.clip(bin_index, 0, NUM_BINS - 1).item()

def load_data_and_extract_features(metadata_file, data_dir, device, num_images_to_process=None):
    """
    Loads data and extracts features using a frozen ViT backbone.
    """
    print("Loading metadata...")
    metadata_frame = pd.read_csv(metadata_file)
    if num_images_to_process:
        print(f"Processing a subset of {num_images_to_process} images.")
        metadata_frame = metadata_frame.iloc[:min(num_images_to_process, len(metadata_frame))]

    extractor = FeatureExtractor().to(device)
    extractor.eval()

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
                
                features_tensor = extractor(image_tensor).squeeze(0).cpu()
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
    # Hyperparameters for the MLP training
    MLP_LR = 1e-3
    MLP_EPOCHS = 250 # A small number of epochs should be enough
    
    # Define a path to save the trained MLP model
    # We will save it in a separate directory to avoid overwrites
    MLP_OUTPUT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "tmp", "mlp_baseline_outputs")
    os.makedirs(MLP_OUTPUT_DIR, exist_ok=True)
    MLP_MODEL_SAVE_PATH = os.path.join(MLP_OUTPUT_DIR, "mlp_model.pth")
    
    # 1. Extract features and prepare labels from the full dataset
    features, labels = load_data_and_extract_features(METADATA_FILE, DATA_DIR, DEVICE)
    
    if features.size == 0 or labels.size == 0:
        print("Error: No data to train on after feature extraction. Exiting.")
        exit()

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.4, random_state=RANDOM_SEED
    )
    print(f"\nTraining with {len(X_train)} samples, testing with {len(X_test)} samples.")
    
    # 3. Create datasets and data loaders for the MLP
    train_dataset = FeatureDataset(X_train, y_train)
    test_dataset = FeatureDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 4. Initialize and train the MLP heads
    mlp_model = MLPHeads(num_input_features=768).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=MLP_LR)
    
    print("\n--- Training MLP Heads ---")
    for epoch in range(MLP_EPOCHS):
        mlp_model.train()
        running_loss = 0.0
        for feats, labs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MLP_EPOCHS} Training"):
            feats = feats.to(DEVICE)
            labs = labs.to(DEVICE).T
            
            optimizer.zero_grad()
            outputs = mlp_model(feats)
            
            loss = 0.0
            for i in range(len(DIMENSION_NAMES)):
                loss += criterion(outputs[i], labs[i])
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1} Training Loss: {running_loss / len(train_loader):.4f}")
        
    print("MLP training complete.")
    
    # 5. Evaluate the trained MLP on the test set
    print("\n--- Evaluating MLP on Test Set ---")
    mlp_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for feats, labs in tqdm(test_loader, desc="Evaluating MLP"):
            feats = feats.to(DEVICE)
            outputs = mlp_model(feats)
            
            batch_preds = []
            for i in range(len(DIMENSION_NAMES)):
                _, preds = torch.max(outputs[i], 1)
                batch_preds.append(preds.cpu())
            
            all_preds.append(torch.stack(batch_preds, dim=1).numpy())
            all_labels.append(labs.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    average_accuracy = np.mean([accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(len(DIMENSION_NAMES))])
    print(f"Overall Average Accuracy: {average_accuracy:.4f} (or {average_accuracy*100:.2f}%)")
    
    exact_matches = np.all(all_preds == all_labels, axis=1)
    exact_match_accuracy = np.mean(exact_matches)
    print(f"Exact Match Accuracy (All 5 Dimensions Correct): {exact_match_accuracy:.4f} (or {exact_match_accuracy*100:.2f}%)")
    
    # Save the trained MLP model
    print(f"\nSaving MLP model to {MLP_MODEL_SAVE_PATH}...")
    torch.save(mlp_model.state_dict(), MLP_MODEL_SAVE_PATH)
    print("MLP model saved successfully.")
