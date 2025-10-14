#!/usr/bin/env python3
"""
Single-file training script:
- Loads CSV produced earlier (supports flexible rvec/roll naming)
- Builds a ResNet -> Transformer hybrid (ViT-like) in one file (no external HF deps)
- Trains with AdamW, label smoothing, ReduceLROnPlateau scheduler
- Mixed precision enabled by default
- Checkpointing + resume
- Logs epoch metrics to CSV
"""

import os
import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm

# -------------------------
# CONFIG (edit paths / params here)
# -------------------------
BASE_DIR = "/scratch/kalidas_1/project/tmp/apriltag_6dof_dataset_rendered/final_dataset_raw"
CSV_PATH = os.path.join(BASE_DIR, "deeptag_vs_blender_normalized.csv")
DATA_DIR = BASE_DIR  # crop paths stored relative to this (e.g. "crops/..._crop.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "hybrid_vit_resnet_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model / data params
NUM_BINS = 50             # reduce bins from 100 -> 50 for better generalization (tweakable)
NUM_OUTPUT_HEADS = 6        # tx, ty, tz, roll, pitch, yaw
TAG_INPUT_DIMS = 6
IMG_SIZE = 224              # input image size
D_MODEL = 512               # transformer hidden dim (match ResNet channels after projection)
TRANSFORMER_LAYERS = 6
TRANSFORMER_HEADS = 8

# Training params
BATCH_SIZE = 32
EPOCHS = 260
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
NUM_WORKERS = 4             # set 0 if you see dataloader worker errors
CKPT_EVERY = 10             # save checkpoint every N epochs
MIXED_PRECISION = True      # use torch.cuda.amp

# Other
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH_METRICS_CSV = os.path.join(OUTPUT_DIR, "epoch_metrics.csv")
BEST_MODEL_FN = os.path.join(OUTPUT_DIR, "model_best.pth")
FINAL_MODEL_FN = os.path.join(OUTPUT_DIR, "model_final.pth")

# Prefixes (normalized columns)
DEEPTAG_TAG_PREFIX = "norm_deeptag_"
GT_CAM_PREFIX = "norm_gt_cam_"

# -------------------------
# Utilities
# -------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Dataset (inline, robust path resolution + flexible column naming)
# -------------------------
class PoseDatasetBinnedSingleFile(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, num_bins=NUM_BINS):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
        self.num_bins = num_bins

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(self.df)} rows from {csv_path}")

        # define possible dimension names (two styles)
        style_rvec = ["tx", "ty", "tz", "rvecx", "rvecy", "rvecz"]
        style_rpy =  ["tx", "ty", "tz", "roll", "pitch", "yaw"]

        def resolve_cols(prefix):
            s1 = [f"{prefix}{x}" for x in style_rvec]
            s2 = [f"{prefix}{x}" for x in style_rpy]
            has1 = all(c in self.df.columns for c in s1)
            has2 = all(c in self.df.columns for c in s2)
            if has1:
                print(f"✅ Using rvec style for prefix {prefix}")
                return s1
            elif has2:
                print(f"✅ Using roll/pitch/yaw style for prefix {prefix}")
                return s2
            else:
                raise ValueError(f"Missing expected columns for prefix {prefix}. Need either rvec or roll style.")

        self.tag_cols = resolve_cols(DEEPTAG_TAG_PREFIX)
        self.cam_cols = resolve_cols(GT_CAM_PREFIX)

        # Accept image identification columns: crop_path preferred else image_id
        self.has_crop_path = "crop_path" in self.df.columns
        self.has_image_id = "image_id" in self.df.columns
        if not (self.has_crop_path or self.has_image_id):
            raise ValueError("CSV must contain either 'crop_path' or 'image_id' column")

        # Coerce numeric for important cols and drop rows with NaN
        important = self.tag_cols + self.cam_cols
        self.df[important] = self.df[important].apply(pd.to_numeric, errors="coerce")
        before = len(self.df)
        self.df = self.df.dropna(subset=important).reset_index(drop=True)
        print(f"Cleaned NaNs: removed {before - len(self.df)} rows. Remaining: {len(self.df)}")

        # precompute bin edges
        self.bin_edges = np.linspace(-1.0, 1.0, self.num_bins + 1)
        self.image_root = os.path.join(self.data_dir, "crops")

    def quantize_value(self, val):
        idx = np.digitize([val], self.bin_edges)[0] - 1
        return int(np.clip(idx, 0, self.num_bins - 1))

    def _resolve_img_path(self, row):
        if self.has_crop_path and pd.notna(row.get("crop_path")):
            p = row.get("crop_path")
            p = os.path.join(self.data_dir, p) if not os.path.isabs(p) else p
            if os.path.exists(p):
                return p
        if self.has_image_id and pd.notna(row.get("image_id")):
            imgid = str(row.get("image_id"))
            cand = os.path.join(self.image_root, imgid.replace(".png", "_crop.png"))
            if os.path.exists(cand):
                return cand
            cand2 = os.path.join(self.image_root, imgid)
            if os.path.exists(cand2):
                return cand2
        if "image" in row and pd.notna(row.get("image")):
            cand = row.get("image")
            cand = os.path.join(self.data_dir, cand) if not os.path.isabs(cand) else cand
            if os.path.exists(cand):
                return cand
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_img_path(row)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for row {idx}; tried crop_path / image_id in {self.data_dir}")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        tag_pose = torch.tensor(row[self.tag_cols].values.astype(np.float32), dtype=torch.float32)

        cam_vals = row[self.cam_cols].values.astype(np.float32)
        cam_bins = [self.quantize_value(float(v)) for v in cam_vals]
        cam_bins = torch.tensor(cam_bins, dtype=torch.long)

        return img, tag_pose, cam_bins

# -------------------------
# Model: ResNet -> ViT-style Transformer hybrid (single-file ViT using nn.TransformerEncoder)
# -------------------------
class HybridResNetViT(nn.Module):
    def __init__(self, num_bins=NUM_BINS, num_heads=NUM_OUTPUT_HEADS,
                 d_model=D_MODEL, n_layers=TRANSFORMER_LAYERS, n_heads=TRANSFORMER_HEADS):
        super().__init__()
        # ResNet34 backbone pretrained on ImageNet (remove classifier)
        resnet = models.resnet34(weights=None)  # set None if no internet; change to pretrained weights name if available
        # Keep conv1..layer3 (to reduce memory), optionally keep layer4 if you have memory
        # We'll use features from penultimate conv block (before avgpool)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3  # output channels 256
        )
        backbone_out_channels = 256

        # Project backbone channels to d_model for transformer tokens
        self.proj = nn.Linear(backbone_out_channels, d_model)

        # Positional embeddings (learned) for up to H*W tokens. We will create dynamically.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb = None  # created on forward based on token length

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Optional small MLP to fuse tag pose (6D)
        self.tag_embed = nn.Sequential(
            nn.Linear(TAG_INPUT_DIMS, d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4, d_model//2),
            nn.ReLU()
        )

        # Final fusion + heads
        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model//2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Output heads: classification per DoF (num_bins classes)
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_bins)) for _ in range(num_heads)])

        # init parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
        for m in self.heads:
            nn.init.xavier_uniform_(m[1].weight)
            nn.init.constant_(m[1].bias, 0.0)

    def forward(self, image, tag_pose):
        """
        image: [B, 3, H, W]
        tag_pose: [B, 6]
        returns: list of NUM_OUTPUT_HEADS tensors [B, NUM_BINS]
        """
        B = image.size(0)
        x = self.backbone(image)               # [B, C=256, h, w]
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)      # [B, N=h*w, C]
        x = self.proj(x)                       # [B, N, d_model]

        # prepare cls token + positional embeddings
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        seq = torch.cat([cls, x], dim=1)        # [B, N+1, d_model]
        Np1 = seq.size(1)

        if (self.pos_emb is None) or (self.pos_emb.size(1) < Np1):
            # create/resize pos_emb
            # pos_emb shape [1, max_len, d_model]
            self.pos_emb = nn.Parameter(torch.randn(1, Np1, seq.size(2), device=seq.device) * 0.02)

        seq = seq + self.pos_emb[:, :Np1, :]

        # transformer
        seq = self.transformer(seq)   # [B, N+1, d_model]
        cls_out = seq[:, 0, :]        # [B, d_model]

        # embed tag pose
        tag_feat = self.tag_embed(tag_pose)  # [B, d_model//2]

        # fuse
        fused = torch.cat([cls_out, tag_feat], dim=1)  # [B, d_model + d_model//2]
        fused = self.fusion(fused)                    # [B, d_model]

        outs = [head(fused) for head in self.heads]   # list of [B, num_bins]
        return outs

# -------------------------
# Training utilities
# -------------------------
def save_checkpoint(model, optimizer, epoch, best_val_loss, path):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }
    torch.save(ckpt, path)
    print(f"💾 Saved checkpoint → {path}")

def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"🔄 Resumed from checkpoint {path} (epoch {start_epoch})")
    return model, optimizer, start_epoch, best_val_loss

def compute_accuracy(preds, targets, tol=2):
    """
    preds: np array [N, 6]
    targets: np array [N, 6]
    returns exact_acc, tol_acc (percent)
    """
    diffs = np.abs(preds - targets)
    exact = (diffs == 0).sum()
    tol_ok = (diffs <= tol).sum()
    total = diffs.size
    return 100.0 * exact / total, 100.0 * tol_ok / total

# -------------------------
# Main train function
# -------------------------
def train_all():
    set_seed(RANDOM_SEED)
    device = DEVICE
    print(f"🚀 Using device: {device}")

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor()
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # dataset + loaders
    ds = PoseDatasetBinnedSingleFile(CSV_PATH, DATA_DIR, transform=train_tf, num_bins=NUM_BINS)
    total = len(ds)
    train_n = int(0.6 * total)
    val_n = total - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])
    # override val transform for val split
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"✅ Dataset split -> train: {len(train_ds)} | val: {len(val_ds)}")

    # model, criterion, optimizer, scheduler
    model = HybridResNetViT(num_bins=NUM_BINS).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # resume
    best_val_loss = float("inf")
    start_epoch = 0
    ckpts = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("checkpoint_epoch_")])
    if ckpts:
        latest = os.path.join(OUTPUT_DIR, ckpts[-1])
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(latest, model, optimizer, device)

    # metrics CSV
    if not os.path.exists(EPOCH_METRICS_CSV):
        pd.DataFrame(columns=[
            "epoch","train_loss","train_acc","train_tol_acc",
            "val_loss","val_acc","val_tol_acc","lr"
        ]).to_csv(EPOCH_METRICS_CSV, index=False)

    scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PRECISION and device.startswith("cuda")))

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        total_train = 0
        correct_train = 0
        tol_correct_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, tag_pose, cam_bins in pbar:
            imgs = imgs.to(device, non_blocking=True)
            tag_pose = tag_pose.to(device, non_blocking=True)
            cam_bins = cam_bins.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(MIXED_PRECISION and device.startswith("cuda"))):
                outs = model(imgs, tag_pose)   # list of [B, num_bins]
                loss = 0.0
                for i in range(NUM_OUTPUT_HEADS):
                    loss = loss + criterion(outs[i], cam_bins[:, i])
                loss = loss / NUM_OUTPUT_HEADS

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # compute batch preds for train accuracy
            preds = torch.stack([torch.argmax(o, dim=1) for o in outs], dim=1)  # [B, 6]
            diffs = torch.abs(preds - cam_bins)
            exact = (diffs == 0).sum().item()
            tol_ok = (diffs <= 2).sum().item()
            total_train += diffs.numel()
            correct_train += exact
            tol_correct_train += tol_ok

            cur_acc = 100.0 * correct_train / total_train
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{cur_acc:.2f}%"})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train
        train_tol_acc = 100.0 * tol_correct_train / total_train

        # ---------- validation ----------
        model.eval()
        val_loss = 0.0
        total_val = 0
        correct_val = 0
        tol_correct_val = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, tag_pose, cam_bins in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                tag_pose = tag_pose.to(device, non_blocking=True)
                cam_bins = cam_bins.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(MIXED_PRECISION and device.startswith("cuda"))):
                    outs = model(imgs, tag_pose)
                    loss = 0.0
                    for i in range(NUM_OUTPUT_HEADS):
                        loss = loss + criterion(outs[i], cam_bins[:, i])
                    loss = loss / NUM_OUTPUT_HEADS
                val_loss += loss.item()

                preds = torch.stack([torch.argmax(o, dim=1) for o in outs], dim=1)
                diffs = torch.abs(preds - cam_bins)
                exact = (diffs == 0).sum().item()
                tol_ok = (diffs <= 2).sum().item()
                total_val += diffs.numel()
                correct_val += exact
                tol_correct_val += tol_ok

                all_preds.append(preds.cpu().numpy())
                all_targets.append(cam_bins.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct_val / total_val
        val_tol_acc = 100.0 * tol_correct_val / total_val

        # scheduler step
        scheduler.step(avg_val_loss)
        lr_now = optimizer.param_groups[0]['lr']

        # print epoch summary
        print(f"Epoch {epoch+1} | TrainLoss: {avg_train_loss:.4f} | TrainAcc: {train_acc:.2f}% | ±2Acc: {train_tol_acc:.2f}% | "
              f"ValLoss: {avg_val_loss:.4f} | ValAcc: {val_acc:.2f}% | ±2Acc: {val_tol_acc:.2f}% | LR: {lr_now:.6f}")

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_FN)
            print(f"🏆 Saved new best model → {BEST_MODEL_FN}")

        # write metrics to CSV
        row = {
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "train_tol_acc": train_tol_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_tol_acc": val_tol_acc,
            "lr": lr_now
        }
        pd.DataFrame([row]).to_csv(EPOCH_METRICS_CSV, index=False, mode='a', header=False)

        # checkpoint every N epochs
        if (epoch + 1) % CKPT_EVERY == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, best_val_loss, ckpt_path)

    # final save + manual inspection
    torch.save(model.state_dict(), FINAL_MODEL_FN)
    print(f"✅ Final model saved → {FINAL_MODEL_FN}")

    # manual check samples printed
    print("\n🔍 Manual check (first val batch):")
    model.eval()
    with torch.no_grad():
        for imgs, tag_pose, cam_bins in val_loader:
            imgs = imgs.to(device); tag_pose = tag_pose.to(device); cam_bins = cam_bins.to(device)
            outs = model(imgs, tag_pose)
            preds = torch.stack([torch.argmax(o, dim=1) for o in outs], dim=1).cpu().numpy()
            gts = cam_bins.cpu().numpy()
            for i in range(min(10, preds.shape[0])):
                print(f"Pred: {preds[i]} | GT: {gts[i]} | Diff: {np.abs(preds[i]-gts[i])}")
            break

if __name__ == "__main__":
    train_all()
