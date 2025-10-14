import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # For progress bars
import os
# Import modules from our project
import config # Import config for all hyperparameters
from utils import save_checkpoint # Only import save_checkpoint as plot_losses/manual_check are used in main

def train_model(model, train_loader, val_loader, criterion, 
                total_epochs, initial_lr, unfreeze_epoch, fine_tune_lr_factor, output_dir,
                start_epoch_index=0, best_val_loss=float('inf'),
                # NEW: Learning Rate Scheduler Parameters passed as arguments
                scheduler_patience=None, scheduler_factor=None, scheduler_min_lr=None): # Use None defaults for clarity
    """
    Trains and validates the model over multiple epochs.
    Integrates learning rate scheduling and handles unfreezing layers.
    Saves model checkpoints periodically and the best performing model.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): The loss function (e.g., nn.MSELoss).
        total_epochs (int): Total number of epochs to train for.
        initial_lr (float): Initial learning rate for the optimizer.
        unfreeze_epoch (int): The 1-indexed epoch at which to unfreeze backbone layers.
        fine_tune_lr_factor (float): Factor to reduce LR by when fine-tuning.
        output_dir (str): Directory to save model checkpoints and plots.
        start_epoch_index (int): 0-indexed epoch to start training from (for resuming).
        best_val_loss (float): The best validation loss achieved so far (for checkpointing).
        scheduler_patience (int, optional): Patience for ReduceLROnPlateau. Defaults to None, expects config.
        scheduler_factor (float, optional): Factor for ReduceLROnPlateau. Defaults to None, expects config.
        scheduler_min_lr (float, optional): Minimum LR for ReduceLROnPlateau. Defaults to None, expects config.
    
    Returns:
        tuple: A tuple containing lists of training losses and validation losses per epoch.
    """
    
    train_losses = []
    val_losses = []

    # Initialize optimizer based on current training phase (frozen vs. fine-tuning)
    # This also handles setting up the optimizer correctly if resuming training.
    if start_epoch_index >= unfreeze_epoch - 1: # -1 because unfreeze_epoch is 1-indexed, start_epoch_index is 0-indexed
        # If resuming after the unfreeze point, or starting directly in fine-tune phase,
        # ensure backbone is unfrozen
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        for param in model.backbone.layer3.parameters():
            param.requires_grad = True
        current_lr = initial_lr * fine_tune_lr_factor
        optimizer = optim.Adam(model.parameters(), lr=current_lr)
        print(f"Resuming training in fine-tuning phase (Epoch {start_epoch_index+1}). Initial LR for this run: {current_lr:.6f}")
    else:
        # If starting from scratch or before the unfreeze point, only head is trainable
        current_lr = initial_lr
        optimizer = optim.Adam(model.regression_head.parameters(), lr=current_lr)
        print(f"Starting training with frozen backbone (Epoch {start_epoch_index+1}). Initial LR: {current_lr:.6f}")

    # Initialize the learning rate scheduler using parameters (preferably from config)
    # Use parameters passed as arguments, falling back to config if not provided (though main.py should provide them)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         # Monitor validation loss (minimize)
        factor=scheduler_factor if scheduler_factor is not None else config.SCHEDULER_FACTOR, # Use passed arg or config
        patience=scheduler_patience if scheduler_patience is not None else config.SCHEDULER_PATIENCE, # Use passed arg or config
        min_lr=scheduler_min_lr if scheduler_min_lr is not None else config.SCHEDULER_MIN_LR  # Use passed arg or config
        # Removed 'verbose=True' as it might not be supported in user's PyTorch version
    )

    # If resuming, load optimizer and scheduler states from the checkpoint
    if start_epoch_index > 0:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_epoch_{start_epoch_index:03d}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("DEBUG: Optimizer state loaded from checkpoint.")
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("DEBUG: Scheduler state loaded from checkpoint.")
            # Print current LR after loading to confirm
            for param_group in optimizer.param_groups:
                print(f"DEBUG: Optimizer current LR after load: {param_group['lr']:.6f}")
        else:
            print(f"WARNING: Checkpoint for epoch {start_epoch_index} not found at {checkpoint_path}. Optimizer and scheduler states not loaded.")


    print(f"Starting training from Epoch {start_epoch_index + 1} out of {total_epochs} total epochs.")

    for epoch in range(start_epoch_index, total_epochs):
        current_epoch_1_indexed = epoch + 1 # Use 1-indexed epoch for display and unfreeze logic

        print(f"\n--- Epoch {current_epoch_1_indexed}/{total_epochs} ---")

        # --- Training Phase ---
        model.train() # Set model to training mode
        running_train_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {current_epoch_1_indexed} Training", unit="batch")

        for images, poses in train_loader_tqdm:
            images = images.to(config.DEVICE)
            poses = poses.to(config.DEVICE)

            optimizer.zero_grad() # Zero the gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, poses) # Calculate loss
            loss.backward() # Backward pass (calculate gradients)
            optimizer.step() # Update weights

            running_train_loss += loss.item() * images.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item()) # Update progress bar with current batch loss

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        print(f"Epoch {current_epoch_1_indexed} Training Loss: {epoch_train_loss:.4f}")

        # Unfreezing logic
        # current_epoch_1_indexed == unfreeze_epoch means unfreeze after this epoch's training/validation
        # (effectively for the *next* epoch's training)
        if current_epoch_1_indexed == unfreeze_epoch:
            # Check if layers are already trainable to prevent redundant re-initialization
            if not any(p.requires_grad for p in model.backbone.layer4.parameters()):
                print(f"--- Unfreezing backbone layers from Epoch {current_epoch_1_indexed+1} ---")
                for param in model.backbone.layer4.parameters():
                    param.requires_grad = True
                for param in model.backbone.layer3.parameters():
                    param.requires_grad = True
                # Re-initialize optimizer and scheduler with new parameters (including unfrozen layers)
                optimizer = optim.Adam(model.parameters(), lr=initial_lr * fine_tune_lr_factor)
                print(f"New Learning Rate for fine-tuning: {initial_lr * fine_tune_lr_factor:.6f}")
                
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE,
                    min_lr=config.SCHEDULER_MIN_LR # Removed verbose=True here too
                )

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {current_epoch_1_indexed} Validation", unit="batch")

        with torch.no_grad(): # Disable gradient calculation during validation
            for images, poses in val_loader_tqdm:
                images = images.to(config.DEVICE)
                poses = poses.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, poses)
                running_val_loss += loss.item() * images.size(0)
                val_loader_tqdm.set_postfix(loss=loss.item())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        print(f"Epoch {current_epoch_1_indexed} Validation Loss: {epoch_val_loss:.4f}")
        
        # Step the learning rate scheduler (after validation loss is computed)
        scheduler.step(epoch_val_loss)
        for param_group in optimizer.param_groups:
            print(f"Epoch {current_epoch_1_indexed} Learning Rate: {param_group['lr']:.6f}")


        # --- Checkpointing Logic ---
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint periodically (every CHECKPOINT_INTERVAL epochs) OR if it's the best model found so far
        if (current_epoch_1_indexed % config.CHECKPOINT_INTERVAL == 0) or is_best:
            print(f"Saving checkpoint for Epoch {current_epoch_1_indexed}...")
            save_checkpoint(
                epoch=current_epoch_1_indexed, # Save 1-indexed epoch for naming consistency
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(), # Include scheduler state in checkpoint
                val_loss=epoch_val_loss,
                output_dir=output_dir,
                is_best=is_best,
                device=config.DEVICE # Pass the device for saving
            )

    return train_losses, val_losses
