import os
import re
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import config

# -------------------------
# Checkpointing
# -------------------------
def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict,
                    val_loss, output_dir, is_best, device):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth")

    state = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "val_loss": val_loss,
    }
    torch.save(state, checkpoint_path)

    if is_best:
        best_model_path = os.path.join(output_dir, "model_best.pth")
        torch.save(state, best_model_path)

    print(f"Checkpoint saved (epoch {epoch}) → {checkpoint_path}")


def load_latest_checkpoint(model, optimizer, scheduler, output_dir, device):
    """
    Load the latest checkpoint if present. Uses weights_only=True in torch.load
    to avoid unpickling arbitrary objects (safer and future-proof).
    Returns: (latest_epoch_int, best_val_loss_float)
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return 0, float("inf")

    latest_epoch, latest_checkpoint_path = 0, None
    for filename in os.listdir(checkpoint_dir):
        match = re.match(r"model_epoch_(\d{3})\.pth", filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint_path = os.path.join(checkpoint_dir, filename)

    if latest_checkpoint_path:
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        # Use weights_only=True to avoid untrusted unpickling of arbitrary objects
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=True)

        # old checkpoints may store e.g. model_state_dict under different keys — expect 'model_state_dict'
        # Our saving format uses 'model_state_dict' etc.
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("model")))

        if optimizer is not None and scheduler is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("Optimizer and scheduler states loaded.")
            except Exception:
                print("Warning: could not load optimizer/scheduler states (possible mismatch).")

        return checkpoint.get("epoch", latest_epoch), checkpoint.get("val_loss", float("inf"))

    return 0, float("inf")


# -------------------------
# Plotting utilities (reads epoch_metrics.csv)
# -------------------------
def _read_epoch_csv(csv_path):
    """Return header list and data dict with numpy arrays. Expects epoch_metrics.csv format."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    data = {h: [] for h in header}
    for r in rows:
        for h, v in zip(header, r):
            try:
                data[h].append(float(v))
            except:
                # fallback: try parse empty -> nan
                data[h].append(float("nan"))

    for k in list(data.keys()):
        data[k] = np.array(data[k])
    return header, data


def _ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def plot_losses(header, data, output_dir):
    epochs = np.arange(1, len(data["epoch"]) + 1)
    train_loss = data.get("train_loss")
    val_loss = data.get("val_loss")

    plt.figure(figsize=(8, 5))
    if train_loss is not None:
        plt.plot(epochs, train_loss, label="Train Loss")
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.grid(True)
    plt.legend()
    out = os.path.join(output_dir, "loss_vs_epoch.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_per_dim_accuracy(header, data, output_dir, dim_names, show_train=True):
    epochs = np.arange(1, len(data["epoch"]) + 1)
    for dim in dim_names:
        train_col = f"train_acc_{dim}"
        val_col = f"val_acc_{dim}"
        tol_col = f"val_tol_acc_{dim}"

        plt.figure(figsize=(8, 5))
        if show_train and train_col in data:
            plt.plot(epochs, data[train_col] * 100, label="Train Exact Acc")
        if val_col in data:
            plt.plot(epochs, data[val_col] * 100, label="Val Exact Acc")
        if tol_col in data:
            plt.plot(epochs, data[tol_col] * 100, label="Val Tol Acc (±2)")

        plt.title(f"Accuracy vs Epoch — {dim}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.legend()
        out = os.path.join(output_dir, f"accuracy_{dim}.png")
        plt.savefig(out)
        plt.close()
        print(f"Saved: {out}")


def plot_overview(header, data, output_dir, dim_names):
    epochs = np.arange(1, len(data["epoch"]) + 1)
    n = len(dim_names)
    # decide grid
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(-1)

    for i, dim in enumerate(dim_names):
        ax = axes[i]
        val_col = f"val_acc_{dim}"
        tol_col = f"val_tol_acc_{dim}"
        if val_col in data:
            ax.plot(epochs, data[val_col] * 100, label="Val Exact")
        if tol_col in data:
            ax.plot(epochs, data[tol_col] * 100, label="Val Tol ±2")
        ax.set_title(dim)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True)
        ax.legend()
    # hide extra axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    out = os.path.join(output_dir, "accuracy_overview.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def plot_epoch_metrics(csv_path, output_dir, dim_names=None):
    """
    Top-level function to read epoch CSV and create plots.
    - csv_path: path to epoch_metrics.csv
    - output_dir: directory to save PNGs
    - dim_names: list of DIMENSION_NAMES (optional)
    """
    header, data = _read_epoch_csv(csv_path)
    if dim_names is None:
        dim_names = []
        for col in header:
            if col.startswith("train_acc_"):
                dim_names.append(col.replace("train_acc_", ""))
        if not dim_names:
            for col in header:
                if col.startswith("val_acc_"):
                    dim_names.append(col.replace("val_acc_", ""))
    _ensure_output_dir(output_dir)
    plot_losses(header, data, output_dir)
    plot_per_dim_accuracy(header, data, output_dir, dim_names)
    plot_overview(header, data, output_dir, dim_names)


# -------------------------
# Manual check (unchanged)
# -------------------------
def manual_check(model, val_dataset, num_samples=10, tol=2):
    model.eval()
    random_indices = torch.randint(0, len(val_dataset), (num_samples,))
    correct_counts = np.zeros(len(config.DIMENSION_NAMES))
    tol_counts = np.zeros(len(config.DIMENSION_NAMES))

    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            image, gt_bins = val_dataset[idx]
            image = image.unsqueeze(0).to(config.DEVICE)
            outputs = model(image)
            preds = [torch.argmax(out, dim=1).cpu().item() for out in outputs]

            print(f"\n--- Sample {i+1} (Index {idx}) ---")
            for j, dim_name in enumerate(config.DIMENSION_NAMES):
                gt, pred = gt_bins[j].item(), preds[j]
                is_correct = (gt == pred)
                within_tol = abs(gt - pred) <= tol

                if is_correct:
                    correct_counts[j] += 1
                if within_tol:
                    tol_counts[j] += 1

                print(f"  {dim_name:<15} | GT: {gt:<3} | Pred: {pred:<3} | "
                      f"Exact: {is_correct} | Within ±{tol}: {within_tol}")

    print("\n--- Manual Check Summary ---")
    for j, dim_name in enumerate(config.DIMENSION_NAMES):
        exact_acc = correct_counts[j] / num_samples * 100
        tol_acc = tol_counts[j] / num_samples * 100
        print(f"{dim_name:<15}: Exact Acc = {exact_acc:.1f}% | Tolerance Acc (±{tol}) = {tol_acc:.1f}%")

    model.train()


# -------------------------
# Evaluation (unchanged but returns more details)
# -------------------------
def evaluate_model(model, data_loader, criterion, tol=2):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)

            loss = sum(criterion(outputs[i], labels[:, i]) for i in range(config.NUM_OUTPUT_HEADS))
            running_loss += loss.item() * images.size(0)

            batch_preds = [out.argmax(1).cpu().numpy() for out in outputs]
            all_preds.append(np.stack(batch_preds, axis=1))
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = running_loss / len(data_loader.dataset)

    # Per-dim exact & tolerance accuracy
    per_dim_exact = np.mean(all_preds == all_labels, axis=0) * 100
    per_dim_tol = np.mean(np.abs(all_preds - all_labels) <= tol, axis=0) * 100

    overall_exact = np.mean(per_dim_exact)
    overall_tol = np.mean(per_dim_tol)
    exact_match_acc = np.mean(np.all(all_preds == all_labels, axis=1)) * 100

    return avg_loss, per_dim_exact, per_dim_tol, overall_exact, overall_tol, exact_match_acc
