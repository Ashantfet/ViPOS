import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models

# ==================== CONFIGURATION ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "tmp", "blender_data_final_render")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "tmp", "vit_last_layer_finetune_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

NUM_BINS = 100
DIMENSION_NAMES = ['norm_x', 'norm_y', 'norm_z', 'norm_azimuth', 'norm_elevation']
DIMENSION_RANGES = {dim: {'min': -1.0, 'max': 1.0} for dim in DIMENSION_NAMES}
BIN_EDGES = {dim: np.linspace(dim_range['min'], dim_range['max'], NUM_BINS + 1)
             for dim, dim_range in DIMENSION_RANGES.items()}
for dim in BIN_EDGES:
    BIN_EDGES[dim][-1] += 1e-6

BATCH_SIZE = 64
NUM_WORKERS = 4
TRAIN_SPLIT = 0.6
MLP_LR = 1e-3
MLP_EPOCHS = 50
CHECKPOINT_INTERVAL = 5

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
VAL_TRANSFORMS = TRAIN_TRANSFORMS

# ==================== MODEL ARCHITECTURE ====================
class PoseEstimationModel(nn.Module):
    def __init__(self, num_layers_to_unfreeze=1):
        super(PoseEstimationModel, self).__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1) #loading pre-trained ViT model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze only the last few encoder layers for fine-tuning
        total_layers = len(self.backbone.encoder.layers)
        num_layers_to_unfreeze = min(num_layers_to_unfreeze, total_layers)
        for i in range(total_layers - num_layers_to_unfreeze, total_layers):
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = True

        num_ftrs = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity()

        self.common_head_features = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
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

# ==================== DATASET UTILITIES ====================
class PoseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        metadata_file_path = os.path.join(self.data_dir, "metadata.csv")
        self.metadata_frame = pd.read_csv(metadata_file_path)
        valid_indices = []
        for idx in tqdm(range(len(self.metadata_frame)), desc="Validating Images"):
            img_full_path = os.path.join(self.data_dir, os.path.basename(self.metadata_frame.iloc[idx]['image_path']))
            if os.path.exists(img_full_path):
                valid_indices.append(idx)
        self.metadata_frame = self.metadata_frame.iloc[valid_indices].reset_index(drop=True)
        if len(self.metadata_frame) == 0:
            raise RuntimeError(f"No valid image files found in '{self.data_dir}'. Dataset is empty.")
        self.bin_edges = {dim: np.linspace(dim_range['min'], dim_range['max'], NUM_BINS + 1) for dim, dim_range in DIMENSION_RANGES.items()}
        for dim in self.bin_edges: self.bin_edges[dim][-1] += 1e-6
    def __len__(self): return len(self.metadata_frame)
    def _get_bin_index(self, value, dim_name):
        bin_index = np.digitize(value, self.bin_edges[dim_name]) - 1
        return np.clip(bin_index, 0, NUM_BINS - 1).item()
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        img_full_path = os.path.join(self.data_dir, os.path.basename(self.metadata_frame.iloc[idx]['image_path']))
        image = Image.open(img_full_path).convert("RGB")
        if self.transform: image = self.transform(image)
        continuous_pose_values = self.metadata_frame.iloc[idx][DIMENSION_NAMES].values.astype(float)
        binned_labels = [self._get_bin_index(continuous_pose_values[i], dim_name) for i, dim_name in enumerate(DIMENSION_NAMES)]
        binned_labels_tensor = torch.tensor(binned_labels, dtype=torch.long)
        return image, binned_labels_tensor

# ==================== TRAINING FUNCTION ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        all_train_preds, all_train_labels = [], []
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")
        for images, binned_labels in train_loader_tqdm:
            images, binned_labels = images.to(DEVICE), binned_labels.to(DEVICE).T
            optimizer.zero_grad()
            outputs = model(images)
            loss = sum(criterion(outputs[i], binned_labels[i]) for i in range(len(DIMENSION_NAMES)))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

            preds = [torch.argmax(out, dim=1).cpu() for out in outputs]
            all_train_preds.append(torch.stack(preds, dim=1).numpy())
            all_train_labels.append(binned_labels.T.cpu().numpy())
        
        train_losses.append(running_loss / len(train_loader.dataset))
        
        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)
        
        avg_train_acc = np.mean([accuracy_score(all_train_labels[:, i], all_train_preds[:, i]) for i in range(len(DIMENSION_NAMES))]) * 100
        train_accuracies.append(avg_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_loss = 0.0
        all_val_preds, all_val_labels = [], []
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation")
        with torch.no_grad():
            for images, binned_labels in val_loader_tqdm:
                images, binned_labels = images.to(DEVICE), binned_labels.to(DEVICE).T
                outputs = model(images)
                loss = sum(criterion(outputs[i], binned_labels[i]) for i in range(len(DIMENSION_NAMES)))
                running_loss += loss.item() * images.size(0)
                val_loader_tqdm.set_postfix(loss=loss.item())
                
                preds = [torch.argmax(out, dim=1).cpu() for out in outputs]
                all_val_preds.append(torch.stack(preds, dim=1).numpy())
                all_val_labels.append(binned_labels.T.cpu().numpy())

        val_losses.append(running_loss / len(val_loader.dataset))
        
        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_labels = np.concatenate(all_val_labels, axis=0)
        
        avg_val_acc = np.mean([accuracy_score(all_val_labels[:, i], all_val_preds[:, i]) for i in range(len(DIMENSION_NAMES))]) * 100
        val_accuracies.append(avg_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f} | Train Acc: {avg_train_acc:.2f}%, Val Acc: {avg_val_acc:.2f}%")
        
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = [torch.argmax(out, dim=1).cpu() for out in outputs]
            all_preds.append(torch.stack(preds, dim=1).numpy())
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_accuracy = np.mean([accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(len(DIMENSION_NAMES))]) * 100
    exact_match_accuracy = np.mean(np.all(all_preds == all_labels, axis=1)) * 100
    return avg_accuracy, exact_match_accuracy

# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)

    # 1. Data
    full_dataset = PoseDataset(DATA_DIR, transform=TRAIN_TRANSFORMS)
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    train_subset, val_subset = random_split(full_dataset, [train_size, len(full_dataset)-train_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 2. Model, Loss, Optimizer
    model = PoseEstimationModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR)
    
    # 3. Training
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, MLP_EPOCHS)
    
    # 4. Evaluation
    avg_acc, exact_acc = evaluate_model(model, val_loader)
    print(f"Overall Average Accuracy: {avg_acc:.2f}%")
    print(f"Exact Match Accuracy: {exact_acc:.2f}%")
