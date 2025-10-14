import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# Import config from the classification project directory
import config

class PoseDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images and their corresponding camera pose data.
    This version discretizes continuous pose values into bin indices for classification.
    It now filters out invalid entries in __init__ to prevent recursion errors.
    """
    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset.
        Args:
            data_dir (str): Path to the directory containing images and metadata.csv.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Construct full path to metadata.csv
        metadata_file_path = os.path.join(self.data_dir, "metadata.csv")
        
        print(f"DEBUG (transformer/dataset.py): Attempting to load metadata from: '{metadata_file_path}'")
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(
                f"ERROR: Metadata file not found at '{metadata_file_path}'. "
            )

        self.metadata_frame = pd.read_csv(metadata_file_path)
        
        # --- NEW: Filter out rows with missing image files in __init__ ---
        valid_indices = []
        for idx in tqdm(range(len(self.metadata_frame)), desc="Validating Images"):
            img_name_in_csv = self.metadata_frame.iloc[idx]['image_path']
            img_full_path = os.path.join(self.data_dir, os.path.basename(img_name_in_csv))
            if os.path.exists(img_full_path):
                valid_indices.append(idx)
            # else:
            #     print(f"Warning: Skipping missing image file at index {idx}: {img_full_path}")
        
        self.metadata_frame = self.metadata_frame.iloc[valid_indices].reset_index(drop=True)
        # --- END NEW ---

        if len(self.metadata_frame) == 0:
            raise RuntimeError(f"No valid image files found in '{self.data_dir}'. Dataset is empty.")

        print(f"DEBUG (transformer/dataset.py): Successfully loaded metadata. Found {len(self.metadata_frame)} valid entries.")

        # Pre-calculate bin edges for efficiency
        self.bin_edges = {}
        for dim_name, dim_range in config.DIMENSION_RANGES.items():
            self.bin_edges[dim_name] = np.linspace(
                dim_range['min'], dim_range['max'], config.NUM_BINS + 1
            )
            self.bin_edges[dim_name][-1] += 1e-6

    def __len__(self):
        """Returns the total number of valid samples in the dataset."""
        return len(self.metadata_frame)

    def _get_bin_index(self, value, dim_name):
        """Maps a continuous value to its corresponding bin index (0 to NUM_BINS-1)."""
        if dim_name not in self.bin_edges:
            raise ValueError(f"Dimension '{dim_name}' not found in DIMENSION_RANGES.")
        
        bin_index = np.digitize(value, self.bin_edges[dim_name]) - 1
        return np.clip(bin_index, 0, config.NUM_BINS - 1).item()

    def __getitem__(self, idx):
        """
        Retrieves an image and its binned pose labels by index.
        Note: This is now guaranteed to work for a valid index due to filtering in __init__.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_csv = self.metadata_frame.iloc[idx]['image_path']
        img_full_path = os.path.join(self.data_dir, os.path.basename(img_name_in_csv)) 
        
        try:
            image = Image.open(img_full_path).convert("RGB")
        except Exception as e:
            # This should ideally not be hit if filtering in __init__ is correct
            raise RuntimeError(f"Failed to open or process image at path {img_full_path} after validation: {e}")

        if self.transform:
            image = self.transform(image)

        continuous_pose_values = self.metadata_frame.iloc[idx][config.DIMENSION_NAMES].values.astype(float)
        
        binned_labels = []
        for i, dim_name in enumerate(config.DIMENSION_NAMES):
            bin_index = self._get_bin_index(continuous_pose_values[i], dim_name)
            binned_labels.append(bin_index)
        
        binned_labels_tensor = torch.tensor(binned_labels, dtype=torch.long)

        return image, binned_labels_tensor

if __name__ == '__main__':
    # This block is for testing the dataset module independently
    print("Testing transformer/dataset.py module...")
    
    test_data_dir = config.DATA_DIR

    try:
        dataset = PoseDataset(test_data_dir, transform=config.VAL_TRANSFORMS)
        print(f"Dataset size: {len(dataset)}")
        
        sample_image, sample_binned_labels = dataset[0]
        print(f"Successfully loaded sample 0.")
        print(f"Image tensor shape: {sample_image.shape}")
        print(f"Binned labels (Tensor): {sample_binned_labels}")
    except Exception as e:
        print(f"An error occurred during dataset testing: {e}")

