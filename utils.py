import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset # Used for type hinting and clarity
from tqdm import tqdm # For progress bars
import random # For manual check sampling
import config # Import config for all hyperparameters
# --- Helper functions for Checkpointing ---
def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, val_loss, output_dir, is_best, device):
    """
    Saves the model's state, optimizer state, and scheduler state to a checkpoint file.
    Also saves a 'best' model checkpoint if the current model performs best.
    
    Args:
        epoch (int): The current epoch number (1-indexed).
        model_state_dict (dict): The state dictionary of the model.
        optimizer_state_dict (dict): The state dictionary of the optimizer.
        scheduler_state_dict (dict): The state dictionary of the scheduler.
        val_loss (float): The validation loss at the current epoch.
        output_dir (str): The base directory to save checkpoints.
        is_best (bool): True if this is the best model seen so far.
        device (torch.device): The device the model is on (for loading/saving consistency).
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_name = f"model_epoch_{epoch:03d}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    best_model_path = os.path.join(output_dir, "model_best.pth") # Best model saved directly in output_dir

    state = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict, # NEW: Include scheduler state
        'val_loss': val_loss,
        'device': str(device) # Save device as string for compatibility
    }

    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved for epoch {epoch} to {checkpoint_path}")

    if is_best:
        torch.save(state, best_model_path)
        print(f"Best model saved (validation loss: {val_loss:.4f}) to {best_model_path}")


def load_latest_checkpoint(model, optimizer, output_dir, device):
    """
    Loads the latest (highest epoch) checkpoint from the specified directory.
    If no checkpoints are found, returns starting values for epoch and best_val_loss.
    
    Args:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        output_dir (str): The base directory where checkpoints are saved.
        device (torch.device): The device to load the checkpoint onto.
        
    Returns:
        tuple: (start_epoch_index, best_val_loss)
               start_epoch_index is 0-indexed and indicates the epoch to start training from.
               best_val_loss is the lowest validation loss found in checkpoints.
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    latest_checkpoint_epoch = 0
    best_val_loss = float('inf')
    latest_checkpoint_path = None

    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
        
        if checkpoints:
            epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
            if epochs:
                latest_checkpoint_epoch = max(epochs)
                latest_checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{latest_checkpoint_epoch:03d}.pth")
                
                print(f"Loading checkpoint from: {latest_checkpoint_path}")
                checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Check for and load scheduler state if it exists in the checkpoint
                if 'scheduler_state_dict' in checkpoint:
                    # Note: The scheduler itself needs to be initialized in train_model before loading its state
                    # This function just loads the state for train_model to pick up.
                    # We store it in a dummy variable or pass it back if train_model needs it.
                    # For now, train_model handles scheduler init and state loading internally.
                    pass # Handled in train_model
                    
                best_val_loss = checkpoint['val_loss']
                start_epoch_index = checkpoint['epoch'] # This is 1-indexed epoch from checkpoint, used for resuming.
                
                print(f"Resuming training from Epoch {start_epoch_index + 1}.")
                return start_epoch_index, best_val_loss
    
    print("No checkpoint found. Starting training from Epoch 1.")
    return 0, float('inf') # Return 0-indexed epoch and infinity for best_val_loss


# --- Helper function for plotting ---
def plot_losses(train_losses, val_losses, output_dir, total_epochs, start_epoch_index):
    """
    Plots training and validation loss over epochs and saves the plot.
    
    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        output_dir (str): Directory to save the plot.
        total_epochs (int): Total number of epochs for the x-axis range.
        start_epoch_index (int): 0-indexed epoch from which training started (for plotting x-axis).
    """
    plt.figure(figsize=(10, 6))
    
    # Adjust x-axis for plotting if resuming training
    epochs_trained_in_run = len(train_losses)
    # The x-axis should start from start_epoch_index (0-indexed)
    epochs_range = range(start_epoch_index, start_epoch_index + epochs_trained_in_run)

    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Training and Validation Loss Over Epochs (Total: {total_epochs} epochs)")
    plt.legend()
    plt.grid(True)
    
    # Ensure plot directory exists within output_dir
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_save_path = os.path.join(plot_dir, f"loss_plot_ep{total_epochs}.png")
    plt.savefig(plot_save_path)
    plt.close() # Close the plot to free memory


# --- Helper function for manual prediction check ---
def manual_check(model, val_dataset, num_samples=10):
    """
    Performs a manual check of model predictions vs. ground truth for a few random samples.
    
    Args:
        model (nn.Module): The trained model.
        val_dataset (torch.utils.data.Dataset): The validation dataset (Subset object).
        num_samples (int): Number of random samples to check.
    """
    model.eval() # Set model to evaluation mode
    
    # Ensure num_samples does not exceed dataset size
    if num_samples > len(val_dataset):
        num_samples = len(val_dataset)
        print(f"Warning: num_samples adjusted to {num_samples} as it exceeds dataset size.")

    random_indices = random.sample(range(len(val_dataset)), num_samples)

    with torch.no_grad(): # Disable gradient calculation
        for i, idx in enumerate(random_indices):
            image, gt_pose = val_dataset[idx] # Get image and ground truth pose from the Dataset
            
            # Add batch dimension to image (C, H, W) -> (1, C, H, W)
            image_input = image.unsqueeze(0).to(config.DEVICE)
            
            predicted_pose = model(image_input).squeeze(0).cpu() # Get prediction, remove batch dim, move to CPU

            # Print results
            # val_dataset.indices[idx] maps the subset index back to the original full dataset index
            print(f"\nSample {i+1} (Original Index: {val_dataset.indices[idx]}):")
            print(f"  Ground Truth Pose (norm): {gt_pose.numpy()}")
            print(f"  Predicted Pose (norm):    {predicted_pose.numpy()}")
            print(f"  Absolute Difference:      {torch.abs(gt_pose - predicted_pose).numpy()}")
    model.train() # Set model back to training mode if it was in train mode before the call (important if you call manual_check mid-training)

