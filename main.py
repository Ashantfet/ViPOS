import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset # Corrected imports
from tqdm import tqdm # Corrected import
import argparse # Corrected import
import random
# Import modules from our project
import config
from dataset import PoseDataset # Only PoseDataset is imported from dataset.py
from model import PoseEstimationModel
from train import train_model # train_model function
from utils import plot_losses, manual_check, load_latest_checkpoint # All necessary functions from utils

import os
import random
import numpy as np

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_manual_check_only(model_path=None, num_samples=10):
    """
    Initializes model, loads a checkpoint, and runs only the manual check.
    """
    print("Running manual check only...")
    set_seed(config.RANDOM_SEED)

    # 1. Device Setup
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Setting up data loaders (only validation needed for manual check)
    print("Setting up validation data loader...")
    # Initialize PoseDataset with no transform initially for random_split to get indices
    base_full_dataset = PoseDataset(config.DATA_DIR, transform=None) 
    
    # Perform random split to get validation indices
    train_size = int(config.TRAIN_SPLIT * len(base_full_dataset))
    #val_size = len(base_full_dataset) - train_size # val_size is derived from split
    
    indices = list(range(len(base_full_dataset)))
    # Use torch.Generator for reproducibility in random_split
    _, val_indices = random_split(indices, [train_size, len(base_full_dataset) - train_size], # Pass actual sizes
                                              generator=torch.Generator().manual_seed(config.RANDOM_SEED))

    # Create a new PoseDataset instance for validation with the correct VAL_TRANSFORMS
    val_dataset_with_transforms = PoseDataset(config.DATA_DIR, transform=config.VAL_TRANSFORMS)
    # Create a Subset using the validation indices
    val_subset = Subset(val_dataset_with_transforms, val_indices)

    # 3. Initializing model
    print("Initializing model...")
    model = PoseEstimationModel(num_output_features=5).to(device)

    # 4. Load specified checkpoint or best model
    if model_path:
        load_path = model_path
    else:
        # Default to loading the best model if no specific path is given
        load_path = os.path.join(config.OUTPUT_DIR, "model_best.pth") 
    
    print(f"Loading model from: {load_path}")
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    else:
        print(f"Error: Model not found at {load_path}. Please provide a valid model path or train a model first.")
        return

    model.eval() # Set model to evaluation mode

    # 5. Perform manual check
    print(f"\n--- Manual Prediction Check ({num_samples} samples) ---")
    manual_check(model, val_subset, num_samples=num_samples) # Pass val_subset, not val_loader
    print("\nManual check complete.")


def main():
    print("Starting Pose Estimation Model Training...")
    set_seed(config.RANDOM_SEED) # Set seed for reproducibility

    # 1. Device Setup
    print(f"Using device: {config.DEVICE}")

    # 2. Setting up data loaders
    print("\n1. Setting up data loaders...")
    
    # Initialize PoseDataset with no transform initially for random_split to get indices
    base_full_dataset = PoseDataset(config.DATA_DIR, transform=None) 

    # Split dataset into training and validation indices
    train_size = int(config.TRAIN_SPLIT * len(base_full_dataset))
    #val_size = len(base_full_dataset) - train_size # val_size is derived from split
    
    indices = list(range(len(base_full_dataset)))
    # Use torch.Generator for reproducibility in random_split
    train_indices, val_indices = random_split(indices, [train_size, len(base_full_dataset) - train_size],
                                              generator=torch.Generator().manual_seed(config.RANDOM_SEED))

    # Create new PoseDataset instances with specific transforms for train and val
    train_dataset_with_transforms = PoseDataset(config.DATA_DIR, transform=config.TRAIN_TRANSFORMS)
    val_dataset_with_transforms = PoseDataset(config.DATA_DIR, transform=config.VAL_TRANSFORMS)

    # Create Subsets to apply transforms to the correct indices after splitting
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
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    # 3. Initializing model
    print("\n2. Initializing model...")
    model = PoseEstimationModel(num_output_features=5).to(config.DEVICE)
    criterion = nn.MSELoss()

    # A dummy optimizer for loading checkpoint, actual optimizer is set in train_model
    dummy_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE) 

    # 4. Checking for existing checkpoints
    print("\n3. Checking for existing checkpoints...")
    start_epoch_index, best_val_loss = load_latest_checkpoint(model, dummy_optimizer, config.OUTPUT_DIR, config.DEVICE) 
    
    # Update TOTAL_EPOCHS based on potential resume
    # This ensures that 'total_epochs' passed to train_model reflects the full run
    config.TOTAL_EPOCHS = start_epoch_index + config.EPOCHS_TO_TRAIN_NOW 
    
    if start_epoch_index > 0:
        print(f"Resuming training from Epoch {start_epoch_index + 1}.")
    else:
        print("No checkpoint found. Starting training from Epoch 1.")


    # 5. Starting model training
    print("\n4. Starting model training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        total_epochs=config.TOTAL_EPOCHS,
        initial_lr=config.LEARNING_RATE,
        unfreeze_epoch=config.UNFREEZE_EPOCH,
        fine_tune_lr_factor=config.FINE_TUNE_LR_FACTOR,
        output_dir=config.OUTPUT_DIR,
        start_epoch_index=start_epoch_index,
        best_val_loss=best_val_loss,
        # Pass scheduler parameters from config
        scheduler_patience=config.SCHEDULER_PATIENCE,
        scheduler_factor=config.SCHEDULER_FACTOR,
        scheduler_min_lr=config.SCHEDULER_MIN_LR
    )

    # 6. Plotting losses
    print("\n5. Plotting losses...")
    # Corrected argument order for plot_losses
    plot_losses(train_losses, val_losses, config.OUTPUT_DIR, config.TOTAL_EPOCHS, start_epoch_index)
    print(f"Loss plot saved to: {os.path.join(config.OUTPUT_DIR, f'loss_plot_ep{config.TOTAL_EPOCHS}.png')}")


    # 7. Final Evaluation on Validation Set (with tqdm progress)
    print("\n6. Evaluating model on validation set...")
    model.eval() # Set model to evaluation mode
    running_val_loss = 0.0
    all_preds = []
    all_targets = []

    # Wrap val_loader with tqdm for progress
    val_loader_tqdm_eval = tqdm(val_loader, desc="Final Evaluation", unit="batch")

    with torch.no_grad():
        for images, poses in val_loader_tqdm_eval:
            images = images.to(config.DEVICE)
            poses = poses.to(config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, poses)
            running_val_loss += loss.item() * images.size(0)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(poses.cpu().numpy())
            val_loader_tqdm_eval.set_postfix(loss=loss.item()) # Update progress bar with current batch loss

    final_val_mse = running_val_loss / len(val_loader.dataset)
    print(f"Validation MSE Loss: {final_val_mse:.4f}")

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # No need for tqdm for MAE calc, as it's on already accumulated numpy arrays
    mae_x = np.mean(np.abs(all_preds[:, 0] - all_targets[:, 0]))
    mae_y = np.mean(np.abs(all_preds[:, 1] - all_targets[:, 1]))
    mae_z = np.mean(np.abs(all_preds[:, 2] - all_targets[:, 2]))
    mae_azimuth = np.mean(np.abs(all_preds[:, 3] - all_targets[:, 3]))
    mae_elevation = np.mean(np.abs(all_preds[:, 4] - all_targets[:, 4]))

    print("\nMean Absolute Errors (MAE) on Validation Set:")
    print(f"  MAE_x: {mae_x:.4f}")
    print(f"  MAE_y: {mae_y:.4f}")
    print(f"  MAE_z: {mae_z:.4f}")
    print(f"  MAE_azimuth: {mae_azimuth:.4f}")
    print(f"  MAE_elevation: {mae_elevation:.4f}")

    # 8. Manual Prediction Check
    print("\n--- Manual Prediction Check (10 samples) ---")
    manual_check(model, val_subset, num_samples=10) # Correctly pass val_subset, not val_loader

    # Final model saving (best model is already saved periodically by train_model)
    print(f"\nTraining completed. Best model saved to: {os.path.join(config.OUTPUT_DIR, 'model_best.pth')}")


if __name__ == "__main__":
    # Use argparse for robust command-line argument parsing
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
