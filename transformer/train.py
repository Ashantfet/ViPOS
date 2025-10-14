import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import numpy as np

import config
from utils import save_checkpoint

def train_model(
    model, train_loader, val_loader, criterion,
    total_epochs, initial_lr, unfreeze_epoch, fine_tune_lr_factor, output_dir,
    start_epoch_index=0, best_val_loss=float('inf'),
    optimizer=None, scheduler=None
):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    if optimizer is None or scheduler is None:
        raise ValueError("Optimizer and scheduler must be provided.")

    # Set LR based on whether we're fine-tuning or training from scratch
    is_finetune = start_epoch_index >= unfreeze_epoch
    base_lr = initial_lr * fine_tune_lr_factor if is_finetune else initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr

    print(f"Starting from epoch {start_epoch_index + 1}/{total_epochs} with LR = {base_lr:.6f}")

    for epoch in range(start_epoch_index, total_epochs):
        epoch_idx = epoch + 1
        print(f"\n=== Epoch {epoch_idx}/{total_epochs} ===")

        # Unfreezing logic
        if epoch_idx == unfreeze_epoch:
            print(f"Unfreezing last few ViT layers at epoch {epoch_idx}")
            total_layers = len(model.backbone.encoder.layers)
            for i in range(total_layers - 3, total_layers):
                for p in model.backbone.encoder.layers[i].parameters():
                    p.requires_grad = True

            # Reduce learning rate for fine-tuning backbone
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * fine_tune_lr_factor
            print(f"Backbone LR updated to {initial_lr * fine_tune_lr_factor:.6f}")

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_preds, train_labels = [], []

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch_idx} Train", unit="batch")
        for images, labels in train_iter:
            images = images.to(config.DEVICE)
            labels = labels.T.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            # Compute average loss across heads
            per_head_losses = [
                criterion(outputs[i], labels[i]) for i in range(config.NUM_OUTPUT_HEADS)
            ]
            loss = sum(per_head_losses) / config.NUM_OUTPUT_HEADS
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            train_iter.set_postfix(avg_loss=loss.item())

            # Collect predictions and labels
            batch_preds = torch.stack([torch.argmax(o, dim=1).cpu() for o in outputs], dim=1)
            train_preds.append(batch_preds)
            train_labels.append(labels.T.cpu())

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        train_preds = torch.cat(train_preds).numpy()
        train_labels = torch.cat(train_labels).numpy()

        per_dim_acc = [
            accuracy_score(train_labels[:, i], train_preds[:, i])
            for i in range(config.NUM_OUTPUT_HEADS)
        ]
        train_acc = np.mean(per_dim_acc) * 100
        train_accuracies.append(train_acc)

        print(f"Train Loss: {epoch_train_loss:.4f} | Avg Acc: {train_acc:.2f}%")
        print("Per-head train losses:", ["{:.4f}".format(l.item()) for l in per_head_losses])

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_preds, val_labels = [], []

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch_idx} Val", unit="batch")
        with torch.no_grad():
            for images, labels in val_iter:
                images = images.to(config.DEVICE)
                labels = labels.T.to(config.DEVICE)

                outputs = model(images)
                per_head_losses = [
                    criterion(outputs[i], labels[i]) for i in range(config.NUM_OUTPUT_HEADS)
                ]
                loss = sum(per_head_losses) / config.NUM_OUTPUT_HEADS

                running_val_loss += loss.item() * images.size(0)
                val_iter.set_postfix(avg_loss=loss.item())

                batch_preds = torch.stack([torch.argmax(o, dim=1).cpu() for o in outputs], dim=1)
                val_preds.append(batch_preds)
                val_labels.append(labels.T.cpu())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()

        per_dim_val_acc = [
            accuracy_score(val_labels[:, i], val_preds[:, i])
            for i in range(config.NUM_OUTPUT_HEADS)
        ]
        val_acc = np.mean(per_dim_val_acc) * 100
        val_accuracies.append(val_acc)

        print(f"Val Loss: {epoch_val_loss:.4f} | Avg Acc: {val_acc:.2f}%")
        print("Per-head val losses:", ["{:.4f}".format(l.item()) for l in per_head_losses])

        # Step scheduler
        scheduler.step(epoch_val_loss)
        print("LRs:", [f"{pg['lr']:.6e}" for pg in optimizer.param_groups])

        # Checkpointing
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            print(f"New best val loss: {best_val_loss:.4f}")

        if is_best or (epoch_idx % config.CHECKPOINT_INTERVAL == 0):
            print("Saving checkpoint...")
            save_checkpoint(
                epoch=epoch_idx,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                val_loss=epoch_val_loss,
                output_dir=output_dir,
                is_best=is_best,
                device=config.DEVICE
            )

    return train_losses, val_losses, train_accuracies, val_accuracies
