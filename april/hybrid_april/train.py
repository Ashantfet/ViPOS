import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import config
from utils import save_checkpoint

# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------
def per_head_accuracy(preds, labels):
    return [(pred.argmax(1) == labels[:, i]).float().mean().item()
            for i, pred in enumerate(preds)]

def tolerance_accuracy(preds, labels, tol=2):
    return [((pred.argmax(1) - labels[:, i]).abs() <= tol).float().mean().item()
            for i, pred in enumerate(preds)]

# ----------------------------------------------------------
# Training loop (with CSV logging per epoch)
# ----------------------------------------------------------
def train_model(model, train_loader, val_loader, total_epochs, output_dir,
                start_epoch_index=0, best_val_loss=float("inf")):
    """
    Trains model and logs per-epoch metrics into epoch_metrics.csv in output_dir.
    Returns nothing (checkpoints and CSV are saved to disk).
    """
    device = config.DEVICE

    criterion = nn.CrossEntropyLoss(label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.0))
    # optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=config.SCHEDULER_FACTOR,
    #     patience=config.SCHEDULER_PATIENCE, min_lr=config.SCHEDULER_MIN_LR
    # )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # --- Scheduler setup ---
    if getattr(config, "SCHEDULER_TYPE", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=config.SCHEDULER_MIN_LR
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE, min_lr=config.SCHEDULER_MIN_LR
        )
        # Optional: Warmup scheduler for first few epochs
    def warmup_lr_lambda(current_epoch):
        warmup_epochs = getattr(config, "WARMUP_EPOCHS", 0)
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / max(1, warmup_epochs)
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

    # Prepare CSV logging
    csv_path = os.path.join(output_dir, "epoch_metrics.csv")
    header = ["epoch", "train_loss", "val_loss"]
    for d in config.DIMENSION_NAMES:
        header += [f"train_acc_{d}", f"val_acc_{d}", f"val_tol_acc_{d}"]
    if start_epoch_index == 0:
        # write header (overwrite old CSV if any)
        os.makedirs(output_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    for epoch in range(start_epoch_index, total_epochs):
        current_epoch_1_indexed = epoch + 1
        print(f"\n--- Epoch {current_epoch_1_indexed}/{total_epochs} ---")

        # ------------------ Training ------------------
        model.train()
        running_train_loss = 0.0
        train_accs_sum = np.zeros(config.NUM_OUTPUT_HEADS)
        train_tol_accs_sum = np.zeros(config.NUM_OUTPUT_HEADS)
        num_train_batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {current_epoch_1_indexed} Training"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = sum(criterion(outputs[i], labels[:, i]) for i in range(config.NUM_OUTPUT_HEADS))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            batch_accs = per_head_accuracy(outputs, labels)
            batch_tol_accs = tolerance_accuracy(outputs, labels)
            train_accs_sum += np.array(batch_accs)
            train_tol_accs_sum += np.array(batch_tol_accs)
            num_train_batches += 1

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        avg_train_accs = (train_accs_sum / max(1, num_train_batches)).tolist()
        avg_train_tol_accs = (train_tol_accs_sum / max(1, num_train_batches)).tolist()

        # ------------------ Validation ------------------
        model.eval()
        running_val_loss = 0.0
        val_accs_sum = np.zeros(config.NUM_OUTPUT_HEADS)
        val_tol_accs_sum = np.zeros(config.NUM_OUTPUT_HEADS)
        num_val_batches = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {current_epoch_1_indexed} Validation"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = sum(criterion(outputs[i], labels[:, i]) for i in range(config.NUM_OUTPUT_HEADS))
                running_val_loss += loss.item() * images.size(0)

                batch_accs = per_head_accuracy(outputs, labels)
                batch_tol_accs = tolerance_accuracy(outputs, labels)
                val_accs_sum += np.array(batch_accs)
                val_tol_accs_sum += np.array(batch_tol_accs)
                num_val_batches += 1

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        avg_val_accs = (val_accs_sum / max(1, num_val_batches)).tolist()
        avg_val_tol_accs = (val_tol_accs_sum / max(1, num_val_batches)).tolist()

        # ------------------ Logging to console ------------------
        # ---------------- Logging ----------------
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        for i, dim_name in enumerate(config.DIMENSION_NAMES):
            print(f"  {dim_name:<15} | "
                  f"Train Exact: {avg_train_accs[i]*100:.2f}% | "
                  f"Val Exact: {avg_val_accs[i]*100:.2f}% | "
                  f"Val Tol ±2: {avg_val_tol_accs[i]*100:.2f}%")

        # overall averages across all dimensions
        overall_train_exact = np.mean(avg_train_accs) * 100
        overall_train_tol   = np.mean(avg_train_tol_accs) * 100
        overall_val_exact   = np.mean(avg_val_accs) * 100
        overall_val_tol     = np.mean(avg_val_tol_accs) * 100

        print(f"Overall Train Exact Acc: {overall_train_exact:.2f}% | Train Tol Acc (±2): {overall_train_tol:.2f}%")
        print(f"Overall Val   Exact Acc: {overall_val_exact:.2f}% | Val Tol Acc (±2): {overall_val_tol:.2f}%")


        # ------------------ Save to CSV ------------------
        row = [current_epoch_1_indexed, avg_train_loss, avg_val_loss]
        for i in range(config.NUM_OUTPUT_HEADS):
            row += [avg_train_accs[i], avg_val_accs[i], avg_val_tol_accs[i]]
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # ------------------ Scheduler & Checkpoint ------------------
        # scheduler.step(avg_val_loss)
        # Step learning rate schedulers properly
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # Apply warmup for first few epochs
        if epoch < getattr(config, "WARMUP_EPOCHS", 0):
            warmup_scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("New best model saved.")
            save_checkpoint(current_epoch_1_indexed, model.state_dict(), optimizer.state_dict(),
                            scheduler.state_dict(), avg_val_loss, output_dir,
                            is_best=True, device=config.DEVICE)
        elif (current_epoch_1_indexed % config.CHECKPOINT_INTERVAL == 0):
            save_checkpoint(current_epoch_1_indexed, model.state_dict(), optimizer.state_dict(),
                            scheduler.state_dict(), avg_val_loss, output_dir,
                            is_best=False, device=config.DEVICE)

    return
