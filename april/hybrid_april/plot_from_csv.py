#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

# --- helper utilities ---
POSSIBLE_EPOCH_COLS = ["epoch", "Epoch", "ep", "index", "idx", "step"]

def read_csv_infer(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # convert to floats where possible (empty => nan)
    data = {h: [] for h in header}
    for r in rows:
        for h, v in zip(header, r):
            try:
                data[h].append(float(v))
            except:
                try:
                    # try removing % and cast
                    if isinstance(v, str) and v.strip().endswith("%"):
                        data[h].append(float(v.strip().strip("%")))
                    else:
                        data[h].append(float("nan"))
                except:
                    data[h].append(float("nan"))
    for k in list(data.keys()):
        data[k] = np.array(data[k], dtype=np.float64)
    return header, data

def find_epoch_col(header):
    for c in POSSIBLE_EPOCH_COLS:
        if c in header:
            return c
    return None

def maybe_scale_to_pct(arr):
    # If values appear to be in [0,1], scale to [0,100]
    if np.isnan(arr).all():
        return arr
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx <= 1.01:  # likely 0..1
        return arr * 100.0
    return arr

def ensure_output_dir(outdir):
    os.makedirs(outdir, exist_ok=True)

# --- plotting functions ---
def plot_loss(epochs, train_loss, val_loss, outdir):
    plt.figure(figsize=(8,5))
    if train_loss is not None:
        plt.plot(epochs, train_loss, label="Train Loss")
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.grid(True)
    plt.legend()
    outpath = os.path.join(outdir, "loss_vs_epoch.png")
    plt.savefig(outpath); plt.close()
    print("Saved:", outpath)

def plot_overall(epochs, train_exact=None, val_exact=None, val_tol=None, outdir=None):
    plt.figure(figsize=(8,5))
    plotted = False
    if train_exact is not None:
        plt.plot(epochs, train_exact, label="Train Exact")
        plotted = True
    if val_exact is not None:
        plt.plot(epochs, val_exact, label="Val Exact")
        plotted = True
    if val_tol is not None:
        plt.plot(epochs, val_tol, label="Val Tol ±2")
        plotted = True
    if not plotted:
        return
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title("Overall Accuracies vs Epoch")
    plt.grid(True); plt.legend()
    outpath = os.path.join(outdir, "overall_acc_vs_epoch.png")
    plt.savefig(outpath); plt.close()
    print("Saved:", outpath)

def plot_per_dim(epochs, data, dim_names, outdir):
    for dim in dim_names:
        plt.figure(figsize=(8,5))
        plotted = False
        tcol = f"train_acc_{dim}"
        vcol = f"val_acc_{dim}"
        vtcol = f"val_tol_acc_{dim}"
        if tcol in data:
            plt.plot(epochs, data[tcol], label="Train Exact")
            plotted = True
        if vcol in data:
            plt.plot(epochs, data[vcol], label="Val Exact")
            plotted = True
        if vtcol in data:
            plt.plot(epochs, data[vtcol], label="Val Tol ±2")
            plotted = True
        if not plotted:
            print(f"Skipping {dim}: no columns found among {tcol},{vcol},{vtcol}")
            plt.close()
            continue
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracies vs Epoch — {dim}")
        plt.grid(True); plt.legend()
        outpath = os.path.join(outdir, f"accuracy_{dim}.png")
        plt.savefig(outpath); plt.close()
        print("Saved:", outpath)

# --- main entrypoint ---
def main():
    parser = argparse.ArgumentParser(description="Robust plotter for epoch_metrics.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to epoch_metrics.csv")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots (defaults to csv folder)")
    parser.add_argument("--dims", type=str, default=None, help="Comma-separated dim names (defaults to infer from CSV header train_acc_*)")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    outdir = args.out if args.out else os.path.dirname(csv_path)
    ensure_output_dir(outdir)

    header, rawdata = read_csv_infer(csv_path)
    print("CSV header columns detected:")
    print(header)

    # determine epoch vector
    epoch_col = find_epoch_col(header)
    if epoch_col:
        epochs = rawdata[epoch_col]
        # if epochs are floats like 1.0, convert to int for plotting ticks
        try:
            epochs = epochs.astype(int)
        except:
            pass
    else:
        # fallback: use row index starting at 1
        print("No explicit epoch column found in header. Using row numbers as epoch indices.")
        n = len(next(iter(rawdata.values())))
        epochs = np.arange(1, n+1)

    # Try to find/prepare loss arrays
    train_loss = rawdata.get("train_loss") if "train_loss" in rawdata else None
    val_loss = rawdata.get("val_loss") if "val_loss" in rawdata else None

    # Try overall accuracies (they might be 0..1 or 0..100)
    overall_train_exact = None
    overall_val_exact = None
    overall_val_tol = None

    for key in ["train_exact_overall", "train_exact_overall".lower(), "train_exact_overall".upper()]:
        if key in rawdata:
            overall_train_exact = maybe_scale_to_pct(rawdata[key])
            break
    # some CSVs use 'train_exact' names; try alternatives
    if overall_train_exact is None:
        for key in ["train_exact", "train_exact_overall", "train_exact_overall".lower()]:
            if key in rawdata:
                overall_train_exact = maybe_scale_to_pct(rawdata[key])
                break

    for key in ["val_exact_overall", "val_exact", "val_exact_overall".lower()]:
        if key in rawdata:
            overall_val_exact = maybe_scale_to_pct(rawdata[key])
            break

    for key in ["val_tol_overall", "val_tol", "val_tol_overall".lower()]:
        if key in rawdata:
            overall_val_tol = maybe_scale_to_pct(rawdata[key])
            break

    # If overall arrays present but in 0..1 scale, scale to percent
    if overall_train_exact is not None and np.nanmax(overall_train_exact) <= 1.01:
        overall_train_exact = overall_train_exact * 100.0
    if overall_val_exact is not None and np.nanmax(overall_val_exact) <= 1.01:
        overall_val_exact = overall_val_exact * 100.0
    if overall_val_tol is not None and np.nanmax(overall_val_tol) <= 1.01:
        overall_val_tol = overall_val_tol * 100.0

    # For per-dim columns: detect dimension names automatically if not provided
    if args.dims:
        dim_names = args.dims.split(",")
    else:
        dim_names = []
        for h in header:
            if h.startswith("train_acc_"):
                dim_names.append(h.replace("train_acc_", ""))
        # fallback to val_acc_ if none found
        if not dim_names:
            for h in header:
                if h.startswith("val_acc_"):
                    dim_names.append(h.replace("val_acc_", ""))
    if not dim_names:
        print("Could not infer any dimension names from CSV header. Per-dim plots will be skipped.")
    else:
        print("Inferred dimension names:", dim_names)

    # convert per-dim columns to percent if needed
    data = {}
    for k, arr in rawdata.items():
        # if the column likely represents accuracy in 0..1 range, convert to percent for plotting convenience
        if k.startswith("train_acc_") or k.startswith("val_acc_") or k.startswith("val_tol_acc_"):
            if np.nanmax(arr) <= 1.01:
                data[k] = arr * 100.0
            else:
                data[k] = arr
        else:
            data[k] = arr

    # Plotting
    try:
        plot_loss(epochs, train_loss, val_loss, outdir)
    except Exception as e:
        print("Could not plot loss:", e)

    try:
        plot_overall(epochs, overall_train_exact, overall_val_exact, overall_val_tol, outdir)
    except Exception as e:
        print("Could not plot overall accuracies:", e)

    try:
        plot_per_dim(epochs, data, dim_names, outdir)
    except Exception as e:
        print("Could not plot per-dim accuracies:", e)

    print("Done. Check output directory:", outdir)

if __name__ == "__main__":
    main()
