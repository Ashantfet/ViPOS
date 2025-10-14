import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import glob # For listing files
import re   # For regular expressions to parse epoch numbers

# Import config from the classification project directory
import config

# --- Helper functions for Checkpointing ---
def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, val_loss, output_dir, is_best, device):
    """
    Saves the model's state, optimizer state, and scheduler state to a checkpoint file.
    Also saves a separate 'best' model checkpoint if the current validation loss is the lowest.
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
        'scheduler_state_dict': scheduler_state_dict,
        'val_loss': val_loss,
        'device': str(device)
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
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    latest_checkpoint_epoch = 0
    best_val_loss = float('inf')
    latest_checkpoint_path = None

    best_model_path = os.path.join(output_dir, "model_best.pth")
    if os.path.exists(best_model_path):
        try:
            # FIX: Added weights_only=True to follow PyTorch recommendation
            best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
            best_val_loss = best_checkpoint['val_loss']
        except Exception as e:
            print(f"Warning: Could not load best_val_loss from {best_model_path}: {e}")

    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
        
        if checkpoints:
            epochs = [int(re.search(r'model_epoch_(\d+)\.pth', f).group(1)) for f in checkpoints if re.search(r'model_epoch_(\d+)\.pth', f)]
            
            if epochs:
                latest_checkpoint_epoch = max(epochs)
                latest_checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{latest_checkpoint_epoch:03d}.pth")
                
                print(f"Loading checkpoint from: {latest_checkpoint_path}")
                # FIX: Added weights_only=True to follow PyTorch recommendation
                checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=True)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("DEBUG: Optimizer state loaded from checkpoint.")
                except ValueError as e:
                    print(f"WARNING: Could not load optimizer state from checkpoint due to parameter group mismatch: {e}")
                    print("Optimizer state will be re-initialized from scratch. This might affect training stability.")
                
                best_val_loss = min(checkpoint['val_loss'], best_val_loss)
                
                start_epoch_index = checkpoint['epoch']
                
                print(f"Resuming training from Epoch {start_epoch_index + 1}.")
                return start_epoch_index, best_val_loss
    
    print("No checkpoint found. Starting training from Epoch 1.")
    return config.PREVIOUS_COMPLETED_EPOCHS, best_val_loss


# --- Helper function for plotting ---
def plot_losses(train_losses, val_losses, output_dir, total_epochs, start_epoch_index):
    """
    Plots training and validation loss over epochs and saves the plot.
    """
    plt.figure(figsize=(10, 6))
    
    epochs_trained_in_run = len(train_losses)
    epochs_range = range(start_epoch_index, start_epoch_index + epochs_trained_in_run)

    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (CrossEntropy)")
    plt.title(f"Training and Validation Loss Over Epochs (Total: {total_epochs} epochs)")
    plt.legend()
    plt.grid(True)
    
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_save_path = os.path.join(plot_dir, f"loss_plot_ep{total_epochs}.png")
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Loss plot saved to: {plot_save_path}")


# --- Helper function for manual prediction check ---
def manual_check(model, val_dataset, num_samples=10):
    """
    Performs a manual check of model predictions vs. ground truth for a few random samples.
    """
    model.eval()
    
    if num_samples > len(val_dataset):
        num_samples = len(val_dataset)
        print(f"Warning: num_samples adjusted to {num_samples} as it exceeds dataset size.")

    random_indices = random.sample(range(len(val_dataset)), num_samples)

    print("\n--- Manual Prediction Check (10 samples) ---")
    print("Format: Dimension | Ground Truth Bin | Predicted Bin | Correct?")

    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            image, gt_binned_labels = val_dataset[idx]
            
            image_input = image.unsqueeze(0).to(config.DEVICE)
            
            predicted_logits_list = model(image_input)
            
            print(f"\nSample {i+1} (Original Index: {val_dataset.indices[idx]}):")
            
            all_correct = True
            for dim_idx, dim_name in enumerate(config.DIMENSION_NAMES):
                gt_bin = gt_binned_labels[dim_idx].item()
                
                predicted_bin = torch.argmax(predicted_logits_list[dim_idx], dim=1).item()
                
                is_correct = (gt_bin == predicted_bin)
                if not is_correct:
                    all_correct = False
                
                print(f"  {dim_name:<15} | {gt_bin:<16} | {predicted_bin:<13} | {is_correct}")
            
            if all_correct:
                print("  --> All dimensions predicted correctly for this sample!")
            else:
                print("  --> Some dimensions misclassified for this sample.")

    model.train()
