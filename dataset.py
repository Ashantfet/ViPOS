import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Import config (assuming config.py is in the project root or accessible)
import config

# Define common image transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet input
    transforms.ToTensor(),          # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PoseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Construct full path to metadata.csv
        metadata_file_path = os.path.join(self.data_dir, "metadata.csv")
        
        # --- ENHANCED DEBUGGING AND ERROR CHECKING ---
        print(f"DEBUG: Data directory received: '{self.data_dir}'")
        print(f"DEBUG: Constructed metadata file path: '{metadata_file_path}'")
        
        # Check if the constructed path exists at all
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(
                f"ERROR: No file or directory found at the constructed path: '{metadata_file_path}'. "
                f"Please ensure 'metadata.csv' exists inside '{self.data_dir}'."
            )
        
        # Explicitly check if it's a file or a directory based on os module
        if os.path.isfile(metadata_file_path):
            print(f"DEBUG: os.path.isfile confirms '{metadata_file_path}' IS a file.")
        else:
            print(f"DEBUG: os.path.isfile says '{metadata_file_path}' IS NOT a file.")

        if os.path.isdir(metadata_file_path):
            print(f"DEBUG: os.path.isdir confirms '{metadata_file_path}' IS a directory.")
            # If it's confirmed as a directory here, then pandas will correctly throw IsADirectoryError.
            # This is the scenario we need to understand.
            raise IsADirectoryError(
                f"CRITICAL ERROR: Python's os.path.isdir confirms '{metadata_file_path}' is a directory. "
                f"It should be the 'metadata.csv' file. Please check your file system entry."
            )
        # --- END ENHANCED DEBUGGING ---

        # If we reach here, os.path.isfile should have returned True.
        self.metadata_frame = pd.read_csv(metadata_file_path)
        print(f"DEBUG: Successfully loaded metadata. Found {len(self.metadata_frame)} entries.")


    def __len__(self):
        return len(self.metadata_frame)

    def __getitem__(self, idx):
        # Get image path
        # Assuming image_path is the second column (index 1) in your metadata.csv
        img_name_in_csv = self.metadata_frame.iloc[idx, 1]
        img_full_path = os.path.join(self.data_dir, img_name_in_csv) 
        
        # Add a quick check here too, in case image files are missing
        if not os.path.exists(img_full_path):
            print(f"WARNING: Image file not found for index {idx}: '{img_full_path}'. Skipping this sample.")
            # You might want to handle this more robustly, e.g., return a placeholder or raise an error
            # For now, this warning will help identify missing images.
            # For actual robust training, you'd filter out invalid entries from metadata_frame in __init__
            return self.__getitem__((idx + 1) % len(self)) # Simple fallback: try next image, might loop if many missing

        try:
            image = Image.open(img_full_path).convert("RGB") # Ensure 3 channels
        except Exception as e:
            print(f"ERROR: Could not open image '{img_full_path}' for index {idx}. Error: {e}. Skipping this sample.")
            return self.__getitem__((idx + 1) % len(self)) # Simple fallback

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get pose values (normalized x, y, z, azimuth, elevation)
        # Ensure these columns are read as numeric (float)
        pose = self.metadata_frame.iloc[idx][['norm_x', 'norm_y', 'norm_z', 'norm_azimuth', 'norm_elevation']].values.astype(float)
        pose = torch.tensor(pose, dtype=torch.float32)

        return image, pose

# This block allows you to test the dataset module independently.
if __name__ == '__main__':
    print("Testing dataset.py module independently...")
    
    # You MUST set this to your actual DATA_DIR for this test block to work correctly.
    # For example:
    # test_data_dir = "/home/ashant/Desktop/project/tmp/blender_data_final_render"
    # Ensure this path is correct AND that metadata.csv is inside it.
    
    # Fallback/Placeholder if config.DATA_DIR isn't set (unlikely in main project context)
    if not hasattr(config, 'DATA_DIR'):
        print("Error: config.DATA_DIR not found. Please ensure config.py is accessible and correct.")
        exit() # Exit if config is not properly loaded
        
    test_data_dir = config.DATA_DIR # Use the DATA_DIR from your config.py

    print(f"Using DATA_DIR for test: {test_data_dir}")

    try:
        dataset = PoseDataset(test_data_dir, transform=image_transforms)
        print(f"Dataset size: {len(dataset)}")
        
        # Try to fetch one item
        img_tensor, pose_tensor = dataset[0]
        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Pose tensor: {pose_tensor}")
        print("Dataset loaded and sample fetched successfully.")
    except Exception as e:
        print(f"Error during dataset testing: {e}")

