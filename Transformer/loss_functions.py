import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import torch.fft
import matplotlib.pyplot as plt
"""
def loss_function(pred_seed_info, true_seed_info, source_diffraction_pattern, voronoi_model):
    # Generate crystal diagram using Voronoi model
    pred_crystal_diagram = voronoi_model(pred_seed_info)
    true_crystal_diagram = voronoi_model(true_seed_info)

    # Apply Fourier transform
    pred_diffraction_pattern = torch.fft.fft2(pred_crystal_diagram)
    true_diffraction_pattern = torch.fft.fft2(true_crystal_diagram)

    # Calculate loss
    loss = nn.MSELoss()(pred_diffraction_pattern, source_diffraction_pattern)

    return loss
"""



def generate_voronoi_diagram(seeds, grid_size=64):

    num_seeds = seeds.shape[0]
    phase_values = seeds[:, 2]

    # Creating a meshgrid for the coordinates
    x = np.arange(0, grid_size, 1)
    y = np.arange(0, grid_size, 1)
    xx, yy = np.meshgrid(x, y)

    # Calculate the distance from each point in the grid to each seed
    distances = np.sqrt((xx[..., np.newaxis] - seeds[:, 0]) ** 2 + (yy[..., np.newaxis] - seeds[:, 1]) ** 2)

    # Find the nearest seed for each point in the grid
    nearest_seed_index = np.argmin(distances, axis=2)
    phase_assigned = phase_values[nearest_seed_index]

    return phase_assigned


def create_amplitude_function(grid_size):

    amplitude_grid = np.zeros((grid_size, grid_size))

    # Sphere parameters
    radius = grid_size / 2
    center = grid_size / 2

    # Create the 2D projection of the sphere
    for i in range(grid_size):
        for j in range(grid_size):
            distance_from_center = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance_from_center <= radius:
                amplitude_grid[i, j] = np.sqrt(radius ** 2 - distance_from_center ** 2)

    amplitude_grid /= np.max(amplitude_grid)
    return amplitude_grid


def create_density_function(amplitude, phase):

    return amplitude * np.exp(1j * phase)


def fourier_transform(image):

    ft = np.fft.fft2(image)
    ft_shifted = np.fft.fftshift(ft)
    magnitude = np.abs(ft_shifted)
    return magnitude, ft_shifted


def process_batch(batch_seeds, grid_size=64):

    results = []
    amplitude = create_amplitude_function(grid_size)

    for seeds in batch_seeds:
        phase = generate_voronoi_diagram(seeds, grid_size)
        density = create_density_function(amplitude, phase)
        magnitude, _ = fourier_transform(density)
        results.append((phase, magnitude))

    return results


# Example usage
batch_seeds = np.array([
    # Example batch of seeds
    [[10, 10, 0.5], [30, 30, 1.0], [50, 50, 1.5]],  # Crystal 1
    [[15, 15, 0.3], [35, 35, 0.9], [55, 55, 1.2]]  # Crystal 2
])

results = process_batch(batch_seeds, grid_size=64)

# Plotting results for visualization (optional)
for i, (phase, magnitude) in enumerate(results):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f'Phase Diagram {i + 1}')
    plt.imshow(phase, cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(f'Diffraction Pattern {i + 1}')
    plt.imshow(magnitude, cmap='inferno')
    plt.colorbar()

    plt.show()

