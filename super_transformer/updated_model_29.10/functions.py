import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import torch.fft
import matplotlib.pyplot as plt
import torch.fft as fft
import torch.nn.functional as F
from transformer_model import *

# Crystal parameters
grid_size = 16  # Size of the crystal
radius = grid_size // 2
sigma = 1.5
img_size = 32

############################################# Functions #############################################
# Function to create density from amplitude and phase
def create_density_function(amplitude, phase):
    real = amplitude * torch.cos(phase)  # Phase is already in radians (-pi to pi)
    imag = amplitude * torch.sin(phase)
    return torch.complex(real, imag)


# Generate a diffraction pattern (only amplitude) from a real-space crystal
def generate_diffraction_pattern(crystal, img_size):
    # Perform Fourier transform on the crystal to get the diffraction pattern
    real_space = torch.fft.fftshift(torch.tensor(crystal, dtype=torch.complex64), dim=(-2, -1))
    diffraction_pattern = torch.fft.fft2(real_space)
    diffraction_pattern = torch.fft.fftshift(diffraction_pattern, dim=(-2, -1))

    # Amplitude is the square root of intensity
    amplitude = torch.sqrt(torch.abs(diffraction_pattern))

    # Normalise amplitude between 0 and 1
    amplitude_norm = amplitude / (torch.max(amplitude) + 1e-8)

    # Generate (x, y, amplitude) input for the model
    coords = [(i / img_size, j / img_size, amplitude_norm[i, j].item()) for i in range(img_size) for j in
              range(img_size)]
    coords = torch.tensor(coords, dtype=torch.float32)

    return coords

# Helper to compute amplitude and phase FFT for each crystal
def compute_fft_dp(crystal):
    amplitude = crystal[..., 2].view(img_size, img_size)  # Amplitude is at index 2
    phase = crystal[..., 3].view(img_size, img_size)  # Phase is at index 3
    real_space = create_density_function(amplitude, phase)

    # Apply FFT and get the magnitude of the diffraction pattern
    real_space = torch.fft.fftshift(real_space)
    dp_complex = torch.fft.fft2(real_space)
    dp_magnitude = torch.abs(torch.sqrt(torch.fft.fftshift(dp_complex)))
    return dp_magnitude

def calculate_chi_square(predicted_dp, target_dp):

    predicted_norm = predicted_dp / (torch.max(predicted_dp) + 1e-8)
    target_norm = target_dp / (torch.max(target_dp) + 1e-8)

    # Calculate chi-square similarity
    chi_square = torch.sum((predicted_norm - target_norm) ** 2) / (
            torch.sqrt(torch.sum(predicted_norm ** 2) * torch.sum(target_norm ** 2)) + 1e-8)
    return chi_square

def loss_function(predicted, target, img_size=img_size, grid_size=grid_size):
    batch_size = predicted.shape[0]

    # Preprocess target to transform it to crystal image format
    target_processed = []
    for i in range(batch_size):
        # Add batch dimension to `unnormalised_target` to match expected input shape for `discrete_voronoi`
        unnormalised_target = unnormalise_positions(target[i].unsqueeze(0), grid_size=grid_size)
        nearest_seed_index = discrete_voronoi(unnormalised_target, grid_size=grid_size)

        # Pass `unnormalised_target` directly without indexing to keep the batch dimension
        crystal_image_target = assign_phase_values(nearest_seed_index, unnormalised_target, grid_size=grid_size,
                                                   pad_size=img_size)
        target_processed.append(crystal_image_target[0])  # Extract the crystal image from the batch dimension

    target_processed = torch.stack(target_processed, dim=0)  # Shape: (batch_size, img_size*img_size, 4)

    # Pre-allocate space for predicted and target diffraction patterns
    predicted_dp = torch.zeros(batch_size, img_size, img_size, dtype=torch.float32, device=predicted.device)
    target_dp = torch.zeros(batch_size, img_size, img_size, dtype=torch.float32, device=target.device)

    # Calculate diffraction pattern for each crystal in predicted and processed target
    for batch_idx in range(batch_size):
        predicted_dp[batch_idx] = compute_fft_dp(predicted[batch_idx])
        target_dp[batch_idx] = compute_fft_dp(target_processed[batch_idx])

    # Chi square loss
    loss = calculate_chi_square(predicted_dp, target_dp)

    return loss

