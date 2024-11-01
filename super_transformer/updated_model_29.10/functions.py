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
grid_size = 16  # Size of the crystal (for Voronoi diagram)
img_size = 32  # Padded size (output image size)
radius = grid_size // 2
sigma = 1.5
N_seed = 6
min_distance = 4
phase_diff = torch.pi / 2
std = 0.05

############################################# Loss Function #############################################
def loss_function(predicted, target, img_size=img_size, grid_size=grid_size,min_distance=min_distance):
    batch_size = predicted.shape[0]

    # Convert each set of target seeds into a crystal image
    target_processed = []
    for i in range(batch_size):
        # Unnormalise target seeds, use Voronoi diagram to convert it back to a crystal
        unnormalised_target = unnormalise_positions(target[i].unsqueeze(0), grid_size=grid_size)
        nearest_seed_index = discrete_voronoi(unnormalised_target, grid_size=grid_size)
        # Assign phase and amplitude to the Voronoi diagram to create a crystal image
        crystal_image_target = assign_phase_values(nearest_seed_index, unnormalised_target, grid_size=grid_size,
                                                   pad_size=img_size, use_base_phase_diff=False)
        target_processed.append(crystal_image_target[0])  # Remove batch dimension

    # Stack target crystals to form a batch
    target_processed = torch.stack(target_processed, dim=0)  # Shape: (batch_size, img_size*img_size, 4)

    # Initialise tensors for predicted and target diffraction patterns
    predicted_dp = torch.zeros(batch_size, img_size, img_size, dtype=torch.float32, device=predicted.device)
    target_dp = torch.zeros(batch_size, img_size, img_size, dtype=torch.float32, device=target.device)

    # Compute diffraction pattern for each crystal image in both predicted and target
    for batch_idx in range(batch_size):
        predicted_dp[batch_idx] = compute_fft_dp(predicted[batch_idx])  # Fourier transform of predicted crystal
        target_dp[batch_idx] = compute_fft_dp(target_processed[batch_idx])  # Fourier transform of target crystal

    # Calculate chi-square loss between predicted and target diffraction patterns
    loss = calculate_chi_square(predicted_dp, target_dp)

    # Add overlap penalty based on minimum distance constraint
    overlap_penalty = 0.0
    for i in range(batch_size):
        for j in range(N_seed):
            for k in range(j + 1, N_seed):
                dist = torch.norm(predicted[i, j, :2] - predicted[i, k, :2], dim=-1)
                overlap_penalty += torch.clamp(min_distance - dist, min=0)

    total_loss = loss #+ overlap_penalty * 0.1  # Adjust weight of penalty as needed

    return loss


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


def unnormalise_positions(seeds, grid_size=grid_size, min_distance=min_distance):
    """Unnormalises (x, y) positions from [0, 1] to [0, grid_size-1] and maintains minimum distance."""
    unnormalised_seeds = seeds.clone()
    unnormalised_seeds[..., :2] = (seeds[..., :2] * (grid_size - 1))

    """# Ensure minimum distance between seeds
    for i in range(seeds.shape[1]):
        for j in range(i + 1, seeds.shape[1]):
            # A clean way to use sqrt to find the distance between 2 points
            dist = torch.norm(unnormalised_seeds[:, i, :2] - unnormalised_seeds[:, j, :2], dim=-1)
            mask = dist < min_distance
            unnormalised_seeds[:, j, :2][mask] += min_distance  # Adjust position if overlap detected"""
    for i in range(seeds.shape[1]):
        for j in range(i + 1, seeds.shape[1]):
            while True:
                dist = torch.norm(unnormalised_seeds[:, i, :2] - unnormalised_seeds[:, j, :2], dim=-1)
                if torch.all(dist >= min_distance):
                    break  # No further adjustments needed if distances are satisfied
                # Apply a small random shift to separate overlapping seeds
                shift = (torch.rand_like(unnormalised_seeds[:, j, :2]) - 0.5) * 2 * min_distance
                unnormalised_seeds[:, j, :2] += shift * (dist < min_distance).float().unsqueeze(-1)

    return unnormalised_seeds


def discrete_voronoi(seeds, grid_size=grid_size):
    # Batch handling for Voronoi computation
    batch_size, num_seeds, _ = seeds.size()
    device = seeds.device

    # Creating a meshgrid for the coordinates and adjusting for batch compatibility
    x = torch.arange(0, grid_size, device=device)
    y = torch.arange(0, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Expanding dimensions to match batch and seed dimensions
    xx = xx.unsqueeze(0).unsqueeze(-1).expand(batch_size, grid_size, grid_size, num_seeds)
    yy = yy.unsqueeze(0).unsqueeze(-1).expand(batch_size, grid_size, grid_size, num_seeds)

    # Calculate distances from each grid point to each seed in the batch
    distances = torch.sqrt((xx - seeds[:, :, 0].view(batch_size, 1, 1, num_seeds)) ** 2 +
                           (yy - seeds[:, :, 1].view(batch_size, 1, 1, num_seeds)) ** 2)

    # Find the nearest seed index for each point in the grid
    nearest_seed_index = torch.argmin(distances, dim=3)
    return nearest_seed_index

def phase_with_min_difference(seeds, batch_size, num_seeds, device, use_base_phase_diff, base_phase_diff = phase_diff):
    if use_base_phase_diff:
        # Generate phases with a minimum separation of base_phase_diff
        base_phases = torch.arange(num_seeds, device=device).float() * base_phase_diff
        # Wrap each base phase to stay within [-π, π]
        base_phases = torch.tanh(base_phases ) * torch.pi #((base_phases + torch.pi) % (2 * torch.pi)) - torch.pi
        base_phases = base_phases.unsqueeze(0).repeat(batch_size, 1)
        # Add jitter to base phases while maintaining minimum separation
        jitter = torch.normal(mean=0, std=std, size=base_phases.shape, device=device)
        pred_phases = base_phases + jitter
    else:
        # No base phase difference, initialize phases close to zero
        pred_phases = seeds[:, :, 3]

    # Ensure all phases remain within [-π, π] after adding jitter
    pred_phases = torch.tanh(pred_phases) * torch.pi #((pred_phases + torch.pi) % (2 * torch.pi)) - torch.pi
    return pred_phases

def assign_phase_values(nearest_seed_index, seeds, grid_size=grid_size, pad_size=img_size, use_base_phase_diff=True):
    """Assigns amplitude and random phases with minimum separation to Voronoi regions."""
    device = seeds.device
    batch_size, num_seeds, _ = seeds.size()

    # Extract amplitude values
    amplitude_values = seeds[:, :, 2]

    # Generate phases with minimum separation based on `use_base_phase_diff`
    pred_phases = phase_with_min_difference(seeds, batch_size, num_seeds, device, use_base_phase_diff=use_base_phase_diff)

    # Assign amplitude and phase values based on Voronoi regions
    amplitude_assigned = amplitude_values.gather(1, nearest_seed_index.view(batch_size, -1))
    phase_assigned = pred_phases.gather(1, nearest_seed_index.view(batch_size, -1))

    # Reshape to (grid_size, grid_size) and apply circular mask within grid_size before padding
    amplitude_assigned = amplitude_assigned.view(batch_size, grid_size, grid_size)
    phase_assigned = phase_assigned.view(batch_size, grid_size, grid_size)

    # Create circular mask within grid_size, consistent with seed_3dp_info.py
    radius = grid_size // 2
    x = torch.arange(-radius, radius, device=device)
    y = torch.arange(-radius, radius, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    mask = (xx ** 2 + yy ** 2) <= radius ** 2

    # Apply mask to amplitude and phase within grid_size
    amplitude_assigned = amplitude_assigned.masked_fill(~mask, 0)
    phase_assigned = phase_assigned.masked_fill(~mask, 0)

    # Pad amplitude and phase to (pad_size, pad_size)
    pad_width = (pad_size - grid_size) // 2
    amplitude_assigned = F.pad(amplitude_assigned, (pad_width, pad_width, pad_width, pad_width), mode='constant',
                               value=0)
    phase_assigned = F.pad(phase_assigned, (pad_width, pad_width, pad_width, pad_width), mode='constant', value=0)

    # Stack for final output
    x_positions = torch.arange(pad_size, device=device).repeat(pad_size, 1).T.flatten().unsqueeze(0).repeat(batch_size,
                                                                                                            1)
    y_positions = torch.arange(pad_size, device=device).repeat(pad_size, 1).flatten().unsqueeze(0).repeat(batch_size, 1)
    crystal_image = torch.stack(
        (x_positions, y_positions, amplitude_assigned.view(batch_size, -1), phase_assigned.view(batch_size, -1)),
        dim=-1)

    return crystal_image

def seed_to_crystal(seed_predictions, grid_size=grid_size, pad_size=img_size):
    nearest_seed_index = discrete_voronoi(seed_predictions, grid_size=grid_size)

    # Assign amplitude and phase values based on Voronoi regions and create the final padded crystal image
    crystal_image = assign_phase_values(
        nearest_seed_index,
        seed_predictions,
        grid_size=grid_size,
        pad_size=pad_size,
        use_base_phase_diff=True)

    return crystal_image

# Convert seed predictions to a crystal image
def seed_to_crystal_true(seed_predictions, grid_size=grid_size, pad_size=img_size):
    nearest_seed_index = discrete_voronoi(seed_predictions, grid_size=grid_size)

    # Assign amplitude and phase values based on Voronoi regions and create the final padded crystal image
    crystal_image = assign_phase_values(
        nearest_seed_index,
        seed_predictions,
        grid_size=grid_size,
        pad_size=pad_size,
        use_base_phase_diff=False
    )

    return crystal_image