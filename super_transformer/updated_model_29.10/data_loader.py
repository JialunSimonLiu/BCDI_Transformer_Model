# Jialun Liu UCL/LCN 24.10.2024
# This supervised transformer model has the input: diffraction pattern (x, y, amplitude), the model Output: predicted crystal in the form of (x, y, amplitude, phase)
# Test full information from a diffraction pattern
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader, random_split

batch_size = 32
img_size = 32
N_seed = 6
# Dataset loading

class CrystalDataset(Dataset):
    def __init__(self, dp_file, seeds_file, img_size=img_size):
        # Open H5 files once during initialization
        self.dp_data = h5py.File(dp_file, 'r')
        self.seeds_data = h5py.File(seeds_file, 'r')
        self.num_samples = len(self.dp_data.keys())  # Assuming both files have the same number of groups
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate keys for diffraction pattern and seed data
        dp_group_key = f'full_dp_group_{idx + 1}'
        seed_group_key = f'seeds_group_{idx + 1}'

        # Extract diffraction pattern (x, y, amplitude)
        dp_array = np.array(self.dp_data[dp_group_key])
        dp_tensor = torch.tensor(dp_array, dtype=torch.float32)

        # Extract seeds (x, y, amplitude, phase)
        seeds_array = np.array(self.seeds_data[seed_group_key])
        seeds_tensor = torch.tensor(seeds_array, dtype=torch.float32)

        # Reformat the tensors for the transformer
        # Reshape diffraction pattern to (batch_size, num_tokens, 3)
        dp_tensor = dp_tensor.view(self.img_size * self.img_size, 3)  # Flattened grid of (x, y, amplitude)

        # Reshape seeds (crystal target) to (batch_size, num_tokens, 4)
        seeds_tensor = seeds_tensor.view(N_seed, 4)  # Flattened grid of (x, y, amplitude, phase)

        return dp_tensor, seeds_tensor


# Function to create a DataLoader
def create_data_loader(batch_size, dp_file_path, seeds_file_path):
    dataset = CrystalDataset(dp_file_path, seeds_file_path)
    # Split the dataset (assuming you have a dataset with 50 samples)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader

