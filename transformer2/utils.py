import torch
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import config  # Make sure config.py exists in your project directory

def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, val_loss, output_dir, is_best, device):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth")
    
    state = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'val_loss': val_loss,
    }
    
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_model_path = os.path.join(output_dir, "model_best.pth")
        torch.save(state, best_model_path)
    print(f"Checkpoint saved for epoch {epoch} to {checkpoint_path}")

def load_latest_checkpoint(model, optimizer, scheduler, output_dir, device):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return 0, float('inf')
    
    latest_epoch = 0
    latest_checkpoint_path = None
    for filename in os.listdir(checkpoint_dir):
        match = re.match(r"model_epoch_(\d{3})\.pth", filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint_path = os.path.join(checkpoint_dir, filename)

    if latest_checkpoint_path:
        print(f"Loading checkpoint from: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Optimizer and scheduler states loaded.")
        except ValueError:
            print("Warning: Optimizer and scheduler states could not be loaded due to a mismatch.")

        return checkpoint['epoch'], checkpoint['val_loss']
    
    return 0, float('inf')


def plot_losses(train_losses, val_losses, output_dir, total_epochs, start_epoch_index):
    epochs_range = range(start_epoch_index, total_epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CrossEntropy)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(output_dir, "plots", f"loss_plot_ep{total_epochs}.png")
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to: {plot_path}")


def manual_check(model, val_dataset, num_samples=10):
    model.eval()
    random_indices = torch.randint(0, len(val_dataset), (num_samples,))
    
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            image, gt_binned_labels = val_dataset[idx]
            
            image_input = image.unsqueeze(0).to(config.DEVICE)
            
            outputs = model(image_input)
            predicted_bins = [torch.argmax(out, dim=1).cpu().item() for out in outputs]
            
            gt_bins = gt_binned_labels.numpy()
            
            print(f"\n--- Sample {i+1} (Original Index: {val_dataset.indices[idx]}) ---")
            all_correct = True
            
            for dim_idx, dim_name in enumerate(config.DIMENSION_NAMES):
                predicted_bin = predicted_bins[dim_idx]
                gt_bin = gt_bins[dim_idx]
                is_correct = (predicted_bin == gt_bin)
                if not is_correct:
                    all_correct = False
                print(f"  {dim_name:<15} | Ground Truth: {gt_bin:<5} | Predicted: {predicted_bin:<5} | Correct? {is_correct}")

            if all_correct:
                print("  --> All dimensions predicted correctly for this sample.")
            else:
                print("  --> Some dimensions misclassified for this sample.")

    model.train()

def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, binned_labels_list in tqdm(data_loader, desc="Evaluating"):
            images = images.to(config.DEVICE)
            target_labels_per_head = binned_labels_list.T.to(config.DEVICE)
            outputs_list = model(images)
            
            total_loss = 0.0
            batch_predictions = []
            for head_idx in range(config.NUM_OUTPUT_HEADS):
                head_outputs = outputs_list[head_idx]
                total_loss += criterion(head_outputs, target_labels_per_head[head_idx])
                
                _, predicted_bins = torch.max(head_outputs, 1)
                batch_predictions.append(predicted_bins.cpu())
            
            running_loss += total_loss.item() * images.size(0)
            
            all_preds.append(torch.stack(batch_predictions, dim=1).numpy())
            all_labels.append(binned_labels_list.numpy())

    final_val_loss_avg = running_loss / len(data_loader.dataset)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    correct_predictions_per_head = np.sum(all_preds == all_labels, axis=0)
    avg_accuracy_per_dim = (correct_predictions_per_head / len(all_labels)) * 100
    
    overall_avg_accuracy = np.mean(avg_accuracy_per_dim)
    exact_match_accuracy = np.mean(np.all(all_preds == all_labels, axis=1)) * 100
    
    return final_val_loss_avg, overall_avg_accuracy, exact_match_accuracy
