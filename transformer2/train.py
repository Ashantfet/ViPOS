import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import numpy as np

import config
from utils import save_checkpoint

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                total_epochs, initial_lr, unfreeze_epoch, fine_tune_lr_factor, output_dir,
                start_epoch_index=0, best_val_loss=float('inf')):
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    # Unfreeze logic to handle resuming
    if start_epoch_index >= unfreeze_epoch:
        total_layers = len(model.backbone.encoder.layers)
        num_layers_to_unfreeze = 3
        for i in range(total_layers - num_layers_to_unfreeze, total_layers):
            for param in model.backbone.encoder.layers[i].parameters():
                param.requires_grad = True
        
        optimizer = optim.Adam(model.parameters(), lr=initial_lr * fine_tune_lr_factor)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE,
            min_lr=config.SCHEDULER_MIN_LR
        )
    
    for epoch in range(start_epoch_index, total_epochs):
        current_epoch_1_indexed = epoch + 1
        print(f"\n--- Epoch {current_epoch_1_indexed}/{total_epochs} ---")

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_all_preds, train_all_labels = [], []
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {current_epoch_1_indexed} Training")
        for images, binned_labels in train_loader_tqdm:
            images, binned_labels = images.to(config.DEVICE), binned_labels.to(config.DEVICE).T
            optimizer.zero_grad()
            outputs = model(images)
            loss = sum(criterion(outputs[i], binned_labels[i]) for i in range(len(config.DIMENSION_NAMES)))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

            preds = [torch.argmax(out, dim=1).cpu() for out in outputs]
            train_all_preds.append(torch.stack(preds, dim=1).numpy())
            train_all_labels.append(binned_labels.T.cpu().numpy())
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        avg_train_acc = np.mean([accuracy_score(np.concatenate(train_all_labels, axis=0)[:, i], np.concatenate(train_all_preds, axis=0)[:, i]) for i in range(len(config.DIMENSION_NAMES))]) * 100
        train_accuracies.append(avg_train_acc)

        print(f"Epoch {current_epoch_1_indexed} Training Loss: {epoch_train_loss:.4f} | Avg. Training Accuracy: {avg_train_acc:.2f}%")

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {current_epoch_1_indexed} Validation")
        with torch.no_grad():
            for images, binned_labels in val_loader_tqdm:
                images, binned_labels = images.to(config.DEVICE), binned_labels.to(config.DEVICE).T
                outputs = model(images)
                loss = sum(criterion(outputs[i], binned_labels[i]) for i in range(len(config.DIMENSION_NAMES)))
                running_val_loss += loss.item() * images.size(0)
                val_loader_tqdm.set_postfix(loss=loss.item())
                
                preds = [torch.argmax(out, dim=1).cpu() for out in outputs]
                all_val_preds.append(torch.stack(preds, dim=1).numpy())
                all_val_labels.append(binned_labels.T.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        avg_val_acc = np.mean([accuracy_score(np.concatenate(all_val_labels, axis=0)[:, i], np.concatenate(all_val_preds, axis=0)[:, i]) for i in range(len(config.DIMENSION_NAMES))]) * 100
        val_accuracies.append(avg_val_acc)
        
        print(f"Epoch {current_epoch_1_indexed} Validation Loss: {epoch_val_loss:.4f} | Avg. Validation Accuracy: {avg_val_acc:.2f}%")
        
        scheduler.step(epoch_val_loss)
        
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            save_checkpoint(
                epoch=current_epoch_1_indexed,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                val_loss=epoch_val_loss,
                output_dir=output_dir,
                is_best=True,
                device=config.DEVICE
            )
        else:
             if (current_epoch_1_indexed % config.CHECKPOINT_INTERVAL == 0):
                print(f"Saving checkpoint for Epoch {current_epoch_1_indexed}...")
                save_checkpoint(
                    epoch=current_epoch_1_indexed,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    val_loss=epoch_val_loss,
                    output_dir=output_dir,
                    is_best=False,
                    device=config.DEVICE
                )

    return train_losses, val_losses, train_accuracies, val_accuracies
