import os
import torch
from torchvision import transforms # Import transforms here for specific validation transforms if needed

# --- Paths ---
# PROJECT_ROOT should point to the '/scratch/kalidas_1/project/' directory.
# Since main.py is directly in 'project/', this will correctly resolve.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct DATA_DIR and OUTPUT_DIR relative to PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "tmp", "blender_data_final_render")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tmp", "vit_scratch_outputs_unaugmented")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42 # For reproducibility of data split and model initialization
TRAIN_SPLIT = 0.6  # 60% for training, 40% for validation

# Epochs and Checkpointing
PREVIOUS_COMPLETED_EPOCHS = 0 # Set to 0 for a fresh start, or N if resuming after N epochs
EPOCHS_TO_TRAIN_NOW = 1 # Number of epochs to train in this specific run
TOTAL_EPOCHS = PREVIOUS_COMPLETED_EPOCHS + EPOCHS_TO_TRAIN_NOW # Total epochs to aim for (e.g., 0 + 50 = 50)
CHECKPOINT_INTERVAL = 5       # Save model checkpoint every N epochs ###updated to 5foir testing

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # Initial learning rate for the regression head
UNFREEZE_EPOCH = 2    # Epoch (1-indexed) at which to unfreeze backbone layers (e.g., at epoch 5, layers become trainable)
FINE_TUNE_LR_FACTOR = 0.1 # Factor to reduce LR by for fine-tuning (e.g., if LR=1e-3, fine-tune LR becomes 1e-4)

# Learning Rate Scheduler
SCHEDULER_PATIENCE = 5 # Number of epochs with no improvement before LR is reduced
SCHEDULER_FACTOR = 0.1 # Factor by which to reduce the LR
SCHEDULER_MIN_LR = 1e-6 # Minimum learning rate

# Classification Specific Parameters
NUM_BINS = 100
DIMENSION_NAMES = ['norm_x', 'norm_y', 'norm_z', 'norm_azimuth', 'norm_elevation']
DIMENSION_RANGES = {dim: {'min': -1.0, 'max': 1.0} for dim in DIMENSION_NAMES}
NUM_OUTPUT_HEADS = len(DIMENSION_NAMES) # Number of heads for our multi-head classifier

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transformations for feature extraction
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomRotation(3),
    #transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
NUM_WORKERS = 4 