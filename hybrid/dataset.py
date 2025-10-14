import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import config

class PoseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        metadata_file_path = os.path.join(self.data_dir, "metadata.csv")
        self.metadata_frame = pd.read_csv(metadata_file_path)
        
        valid_indices = []
        for idx in tqdm(range(len(self.metadata_frame)), desc="Validating Images"):
            img_full_path = os.path.join(self.data_dir, os.path.basename(self.metadata_frame.iloc[idx]['image_path']))
            if os.path.exists(img_full_path):
                valid_indices.append(idx)
        self.metadata_frame = self.metadata_frame.iloc[valid_indices].reset_index(drop=True)
        if len(self.metadata_frame) == 0:
            raise RuntimeError(f"No valid image files found in '{self.data_dir}'. Dataset is empty.")

        self.bin_edges = {dim: np.linspace(dim_range['min'], dim_range['max'], config.NUM_BINS + 1)
                          for dim, dim_range in config.DIMENSION_RANGES.items()}
        for dim in self.bin_edges:
            self.bin_edges[dim][-1] += 1e-6

    def __len__(self):
        return len(self.metadata_frame)

    def _get_bin_index(self, value, dim_name):
        bin_index = np.digitize(value, self.bin_edges[dim_name]) - 1
        return np.clip(bin_index, 0, config.NUM_BINS - 1).item()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_full_path = os.path.join(self.data_dir, os.path.basename(self.metadata_frame.iloc[idx]['image_path']))
        image = Image.open(img_full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        continuous_pose_values = self.metadata_frame.iloc[idx][config.DIMENSION_NAMES].values.astype(float)
        binned_labels = [self._get_bin_index(continuous_pose_values[i], dim_name)
                         for i, dim_name in enumerate(config.DIMENSION_NAMES)]
        binned_labels_tensor = torch.tensor(binned_labels, dtype=torch.long)
        
        return image, binned_labels_tensor
