import os
import torch
from torchvision import transforms

# --- Project Paths ---
# Get the absolute path of the directory containing this config.py file
CLASSIFICATION_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# The DATA_DIR (blender_data_final_render) is one level up from 'classification/'
PROJECT_PARENT_DIR = os.path.dirname(CLASSIFICATION_PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_PARENT_DIR, "tmp", "blender_data_final_render")

# Output directory for classification results (inside project/tmp/classification_model_outputs_ViT)
# This is a new, unique directory for the ViT model's results.
OUTPUT_DIR = os.path.join(PROJECT_PARENT_DIR, "tmp", "classification_model_outputs_ViT")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True) # For saving checkpoints

# --- Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
RANDOM_SEED = 42 # For reproducibility
BATCH_SIZE = 64
NUM_WORKERS = 4 # Number of data loading workers (adjust based on your CPU cores)

# Epochs and Training Schedule
PREVIOUS_COMPLETED_EPOCHS = 0 # <-- Set to 0 to start from scratch for the ViT model
EPOCHS_TO_TRAIN_NOW = 1    # Number of new epochs to train in this run
TOTAL_EPOCHS = PREVIOUS_COMPLETED_EPOCHS + EPOCHS_TO_TRAIN_NOW # Total epochs to aim for

UNFREEZE_EPOCH = 2 # Epoch (1-indexed) at which to unfreeze the ViT layers
LEARNING_RATE = 1e-3 # Initial learning rate for the classifier heads
FINE_TUNE_LR_FACTOR = 0.1 # Factor to reduce LR by when fine-tuning

# Learning Rate Scheduler Parameters
SCHEDULER_PATIENCE = 10 # Number of epochs with no improvement after which learning rate will be reduced
SCHEDULER_FACTOR = 0.1  # Factor by which the learning rate will be reduced
SCHEDULER_MIN_LR = 1e-6 # Minimum learning rate to avoid very small steps

# Checkpointing
CHECKPOINT_INTERVAL = 5 # Save a checkpoint every N epochs, in addition to the best model

# Data Split
TRAIN_SPLIT = 0.6 # 60% for training, 40% for validation

# --- Classification Specific Parameters ---
NUM_BINS = 100 # <-- AS PER PROFESSOR'S INSTRUCTION
DIMENSION_RANGES = {
    'norm_x': {'min': -1.0, 'max': 1.0},
    'norm_y': {'min': -1.0, 'max': 1.0},
    'norm_z': {'min': -1.0, 'max': 1.0},
    'norm_azimuth': {'min': -1.0, 'max': 1.0},
    'norm_elevation': {'min': -1.0, 'max': 1.0},
}
DIMENSION_NAMES = ['norm_x', 'norm_y', 'norm_z', 'norm_azimuth', 'norm_elevation']
NUM_OUTPUT_HEADS = len(DIMENSION_NAMES)
NUM_CLASSES_PER_HEAD = NUM_BINS

# --- Image Transforms ---
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
