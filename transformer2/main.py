import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import argparse

import config
from dataset import PoseDataset
from model import PoseEstimationModel
from train import train_model
from utils import load_latest_checkpoint, plot_losses, manual_check, evaluate_model

import os
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_manual_check_only(model_path=None, num_samples=10):
    print("Running manual check only...")
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    print("Setting up validation data loader...")
    full_dataset = PoseDataset(config.DATA_DIR, transform=config.VAL_TRANSFORMS)
    val_subset = full_dataset # For now, use the full dataset for checking
    
    print("Initializing model...")
    model = PoseEstimationModel().to(device)

    if model_path:
        load_path = model_path
    else:
        load_path = os.path.join(config.OUTPUT_DIR, "model_best.pth")

    print(f"Loading model from: {load_path}")
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    else:
        print(f"Error: Model not found at {load_path}.")
        return

    model.eval()

    print(f"\n--- Manual Prediction Check ({num_samples} samples) ---")
    manual_check(model, val_subset, num_samples=num_samples)
    print("\nManual check complete.")


def main():
    # print("Starting Pose Estimation Model Training...")
    # set_seed(config.RANDOM_SEED)
    # print(f"Using device: {config.DEVICE}")

    # print("\n1. Setting up data loaders...")
    # full_dataset = PoseDataset(config.DATA_DIR, transform=None)
    # train_size = int(config.TRAIN_SPLIT * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size],
    #                                           generator=torch.Generator().manual_seed(config.RANDOM_SEED))
    
    # train_dataset = PoseDataset(config.DATA_DIR, transform=config.TRAIN_TRANSFORMS)
    # val_dataset = PoseDataset(config.DATA_DIR, transform=config.VAL_TRANSFORMS)
    # train_subset = Subset(train_dataset, train_indices)
    # val_subset = Subset(val_dataset, val_indices)
    
    # train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    # val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # print(f"Total dataset size: {len(full_dataset)}")
    # print(f"Training set size: {len(train_subset)}")
    # print(f"Validation set size: {len(val_subset)}")
    print("Starting Pose Estimation Model Training...")
    set_seed(config.RANDOM_SEED)

    print(f"Using device: {config.DEVICE}")

    print("\n1. Setting up data loaders...")
    
    base_full_dataset = PoseDataset(config.DATA_DIR, transform=None) 

    # --- Data Splitting Logic ---
    # Uncomment the desired split type and comment out the other.
    
    # --- Option A: Custom Interleaved Split ---
    total_samples = len(base_full_dataset)
    train_indices = [i for i in range(total_samples) if i % 5 < 3]
    val_indices = [i for i in range(total_samples) if i % 5 >= 3]
    train_size = len(train_indices)
    val_size = len(val_indices)
    print(f"Using custom 3:2 interleaved split.")

    # # --- Option B: Reproducible Random Split (Active by default) ---
    # train_size = int(config.TRAIN_SPLIT * len(base_full_dataset))
    # val_size = len(base_full_dataset) - train_size
    # indices = list(range(len(base_full_dataset)))
    # train_indices, val_indices = random_split(indices, [train_size, val_size],
    #                                           generator=torch.Generator().manual_seed(config.RANDOM_SEED))
    # print(f"Using reproducible random split.")
    # # --- End Data Splitting Logic ---

    train_dataset_with_transforms = PoseDataset(config.DATA_DIR, transform=config.TRAIN_TRANSFORMS)
    val_dataset_with_transforms = PoseDataset(config.DATA_DIR, transform=config.VAL_TRANSFORMS)

    train_subset = Subset(train_dataset_with_transforms, train_indices)
    val_subset = Subset(val_dataset_with_transforms, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"Total dataset size: {len(base_full_dataset)}")
    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")

    print("\n2. Initializing model...")
    model = PoseEstimationModel().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR
    )
    
    # --- Code to inspect the model's structure ---
    print("--- Model Architecture ---")
    print(model)
    print("--------------------------")
    # ----------------------------------------------

    print("\n3. Checking for existing checkpoints...")
    start_epoch_index, best_val_loss = load_latest_checkpoint(model, optimizer, scheduler, config.OUTPUT_DIR, config.DEVICE)
    config.TOTAL_EPOCHS = start_epoch_index + config.EPOCHS_TO_TRAIN_NOW
    
    if start_epoch_index > 0:
        print(f"Resuming training from Epoch {start_epoch_index + 1}.")
    else:
        print("No checkpoint found. Starting training from Epoch 1.")

    print("\n4. Starting model training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        config.TOTAL_EPOCHS, config.LEARNING_RATE, config.UNFREEZE_EPOCH, config.FINE_TUNE_LR_FACTOR,
        config.OUTPUT_DIR, start_epoch_index, best_val_loss
    )
    print("\nFinished Training!")

    print("\n5. Final Evaluation on Validation Set...")
    final_val_loss, final_avg_acc, final_exact_acc = evaluate_model(model, val_loader, criterion)
    print(f"Final Validation Loss: {final_val_loss:.4f} | Final Validation Accuracy: {final_avg_acc:.2f}%")
    print(f"Exact Match Accuracy: {final_exact_acc:.2f}%")

    print("\n--- Manual Prediction Check (10 samples) ---")
    manual_check(model, val_subset, num_samples=10)
    
    print(f"\nTraining completed. Best model saved to: {os.path.join(config.OUTPUT_DIR, 'model_best.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a camera pose estimation model.")
    parser.add_argument("--check-only", action="store_true", 
                        help="Run only the manual prediction check using the best saved model.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to a specific model checkpoint to load for manual check.")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to manually check.")
    args = parser.parse_args()

    if args.check_only:
        run_manual_check_only(model_path=args.model_path, num_samples=args.num_samples)
    else:
        main()
