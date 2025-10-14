import os
import torch
from torchvision import transforms

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "tmp", "apriltag_6dof_dataset_rendered")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tmp", "apriltag_6dof_hybrid_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Device & Seed ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- Splits ---
TRAIN_SPLIT = 0.6  # 60% train / 40% val

# --- Epochs & Checkpointing ---
PREVIOUS_COMPLETED_EPOCHS = 0
EPOCHS_TO_TRAIN_NOW = 1
TOTAL_EPOCHS = PREVIOUS_COMPLETED_EPOCHS + EPOCHS_TO_TRAIN_NOW
CHECKPOINT_INTERVAL = 5

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4   # smaller, avoids divergence with transformers
UNFREEZE_EPOCH = 2
FINE_TUNE_LR_FACTOR = 0.1

# --- Scheduler ---
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6

# --- Classification ---
NUM_BINS = 100  ##try with 200,500,50
DIMENSION_NAMES = ["norm_x", "norm_y", "norm_z", "norm_azimuth", "norm_elevation"]
DIMENSION_RANGES = {dim: {"min": -1.0, "max": 1.0} for dim in DIMENSION_NAMES}
NUM_OUTPUT_HEADS = len(DIMENSION_NAMES)

# --- Regularization ---
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3

# --- Image preprocessing ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(3),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

NUM_WORKERS = 4
