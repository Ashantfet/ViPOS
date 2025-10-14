#!/usr/bin/env python3
import os, random, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# ===============================================================
# CONFIGURATION
# ===============================================================
BASE_DIR = "/scratch/kalidas_1/project/tmp/apriltag_6dof_dataset_rendered/final_dataset_raw"
CSV_PATH = os.path.join(BASE_DIR, "deeptag_vs_blender_normalized.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "hybrid_vit_binned_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
NUM_BINS = 100  # Number of bins per 6D component (classification targets)

# Pose columns
POSE_COLS = [
    "norm_gt_cam_tx","norm_gt_cam_ty","norm_gt_cam_tz",
    "norm_gt_cam_roll","norm_gt_cam_pitch","norm_gt_cam_yaw"
]

# ===============================================================
# SEED
# ===============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(RANDOM_SEED)

# ===============================================================
# DATASET
# ===============================================================
class PoseBinnedDataset(Dataset):
    def __init__(self, csv_path, base_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=POSE_COLS)
        self.base_dir = base_dir
        self.transform = transform
        self.bin_edges = np.linspace(-1, 1, NUM_BINS + 1)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, r["crop_path"])
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        y_cont = r[POSE_COLS].values.astype(np.float32)
        y_bins = np.digitize(y_cont, self.bin_edges) - 1
        y_bins = np.clip(y_bins, 0, NUM_BINS - 1)
        return img, torch.tensor(y_bins, dtype=torch.long)

# ===============================================================
# MODEL — ViT backbone + 6 output heads (binned classification)
# ===============================================================
class VitPoseBinned(nn.Module):
    def __init__(self, num_bins=NUM_BINS):
        super().__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        emb_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(emb_dim, num_bins) for _ in range(6)])
    def forward(self, x):
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]

# ===============================================================
# ACCURACY FUNCTION
# ===============================================================
def compute_accuracy(preds, targets, tol=2):
    preds = torch.stack([torch.argmax(p, dim=1) for p in preds], dim=1)
    diffs = torch.abs(preds - targets)
    exact = (diffs == 0).sum().item()
    tol_ok = (diffs <= tol).sum().item()
    total = diffs.numel()
    return 100.0 * exact / total, 100.0 * tol_ok / total

# ===============================================================
# TRAIN FUNCTION
# ===============================================================
def train():
    print(f"🚀 Training on {DEVICE}")

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.ToTensor()
    ])
    tf_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    ds = PoseBinnedDataset(CSV_PATH, BASE_DIR, tf_train)
    total = len(ds)
    train_size = int(0.7 * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    val_ds.dataset.transform = tf_val

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"✅ Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = VitPoseBinned().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    start_epoch = 0
    ckpts = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("checkpoint_epoch_")]
    if ckpts:
        latest = sorted(ckpts)[-1]
        ck = torch.load(os.path.join(OUTPUT_DIR, latest), map_location=DEVICE)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val_loss = ck.get("best_val_loss", best_val_loss)
        print(f"🔄 Resumed from checkpoint {latest} (epoch {start_epoch})")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss, correct, tol_correct, total = 0, 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, y_bins in pbar:
            imgs, y_bins = imgs.to(DEVICE), y_bins.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outs = model(imgs)
                loss = sum(criterion(outs[i], y_bins[:, i]) for i in range(6)) / 6
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            exact, tol_ok = compute_accuracy(outs, y_bins)
            correct += exact; tol_correct += tol_ok; total += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{exact:.2f}%"})

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_tol_acc = tol_correct / total

        # Validation
        model.eval()
        val_loss, v_correct, v_tol_correct, v_total = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, y_bins in val_loader:
                imgs, y_bins = imgs.to(DEVICE), y_bins.to(DEVICE)
                with torch.cuda.amp.autocast():
                    outs = model(imgs)
                    loss = sum(criterion(outs[i], y_bins[:, i]) for i in range(6)) / 6
                val_loss += loss.item()
                v_exact, v_tol_ok = compute_accuracy(outs, y_bins)
                v_correct += v_exact; v_tol_correct += v_tol_ok; v_total += 1

        avg_val_loss = val_loss / len(val_loader)
        val_acc = v_correct / v_total
        val_tol_acc = v_tol_correct / v_total

        print(f"Epoch {epoch+1} | TrainLoss: {avg_train_loss:.4f} | "
              f"TrainAcc: {train_acc:.2f}% | ±2Acc: {train_tol_acc:.2f}% | "
              f"ValLoss: {avg_val_loss:.4f} | ValAcc: {val_acc:.2f}% | ±2Acc: {val_tol_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pth"))
            print("🏆 Saved best model")

        if (epoch + 1) % 10 == 0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
            print(f"💾 Saved checkpoint at epoch {epoch+1}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_final.pth"))
    print("✅ Training complete.")

if __name__ == "__main__":
    train()
