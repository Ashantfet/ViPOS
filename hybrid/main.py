import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
from dataset import PoseDataset
from model import PoseEstimationModel
from train import train_model
from utils import load_latest_checkpoint, manual_check, evaluate_model, plot_epoch_metrics

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print("Starting Pose Estimation Model Training...")
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # ------------------- Data loaders -------------------
    print("\n1. Setting up data loaders...")
    base_dataset = PoseDataset(config.DATA_DIR, transform=None)
    total_samples = len(base_dataset)
    train_indices = [i for i in range(total_samples) if i % 5 < 3]
    val_indices = [i for i in range(total_samples) if i % 5 >= 3]

    train_dataset = PoseDataset(config.DATA_DIR, transform=config.TRAIN_TRANSFORMS)
    val_dataset = PoseDataset(config.DATA_DIR, transform=config.VAL_TRANSFORMS)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    print(f"Dataset: total={total_samples}, train={len(train_subset)}, val={len(val_subset)}")

    # ------------------- Model -------------------
    print("\n2. Initializing model...")
    model = PoseEstimationModel().to(device)
    print(model)

    # ------------------- Checkpoint resume -------------------
    print("\n3. Checking for checkpoints...")
    # load_latest_checkpoint returns (epoch, val_loss)
    start_epoch_index, best_val_loss = load_latest_checkpoint(model, None, None, config.OUTPUT_DIR, device)
    total_epochs = start_epoch_index + config.EPOCHS_TO_TRAIN_NOW
    print(f"Starting from epoch index: {start_epoch_index} (0 means fresh). Target total epochs: {total_epochs}")

    # ------------------- Train -------------------
    print("\n4. Starting training...")
    train_model(model, train_loader, val_loader, total_epochs, config.OUTPUT_DIR,
                start_epoch_index=start_epoch_index, best_val_loss=best_val_loss)
    print("\nTraining finished.")

    # ------------------- Final Evaluation -------------------
    print("\n5. Final Evaluation on Validation Set...")
    criterion = nn.CrossEntropyLoss(label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.0))
    final_val_loss, per_dim_exact, per_dim_tol, overall_exact, overall_tol, exact_match_acc = evaluate_model(
        model, val_loader, criterion, tol=2)

    print(f"\nFinal Validation Loss: {final_val_loss:.4f}")
    for i, dim_name in enumerate(config.DIMENSION_NAMES):
        print(f"{dim_name:<15} | Exact Acc: {per_dim_exact[i]:.2f}% | Tolerance Acc (±2): {per_dim_tol[i]:.2f}%")
    print(f"\nOverall Per-dim Exact Accuracy: {overall_exact:.2f}%")
    print(f"Overall Per-dim Tolerance Accuracy (±2): {overall_tol:.2f}%")
    print(f"Exact Match Accuracy (all 5 correct): {exact_match_acc:.2f}%")

    # ------------------- Manual Check -------------------
    print("\n6. Manual Prediction Check (10 samples)")
    manual_check(model, val_subset, num_samples=10, tol=2)

    # ------------------- Plotting -------------------
    csv_path = os.path.join(config.OUTPUT_DIR, "epoch_metrics.csv")
    try:
        print("\n7. Creating metric plots...")
        plot_epoch_metrics(csv_path, config.OUTPUT_DIR)
        print("Plots saved to:", config.OUTPUT_DIR)
    except Exception as e:
        print("Could not create plots:", e)

    print("\nAll done. Best model (if saved) is in:", os.path.join(config.OUTPUT_DIR, "model_best.pth"))

if __name__ == "__main__":
    main()
