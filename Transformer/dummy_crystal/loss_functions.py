import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import torch.fft
import matplotlib.pyplot as plt

# Crystal parameters
grid_size = 16  # Size of the crystal
radius = grid_size // 2
sigma = 1.5
pad_size = 32

############################################# Loss function #############################################
# Projection function
def project_onto_circle(x, y, center_x, center_y, radius):
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    if distance <= radius:
        return x, y
    else:
        return grid_size,grid_size

# Function to create diffraction pattern based on seeds
def discrete_voronoi(seeds, grid_size):
    #Generates a discrete Voronoi diagram on a grid and pads it to a specified size.

    # Creating a meshgrid for the coordinates
    x = np.arange(0, grid_size, 1)
    y = np.arange(0, grid_size, 1)
    xx, yy = np.meshgrid(x, y)
    # Calculate the distance from each point in the grid to each seed
    distances = np.sqrt((xx[..., np.newaxis] - seeds[:, 0])**2 + (yy[..., np.newaxis] - seeds[:, 1])**2)
    # Find the nearest seed for each point in the grid
    nearest_seed_index = np.argmin(distances, axis=2)
    return nearest_seed_index

def create_diffraction_pattern(seeds, grid_size, sigma, height, width):
    device = seeds.device
    batch_size = seeds.size(0)
    dp = torch.zeros((batch_size, height, width), device=device)

    for batch_idx in range(batch_size):
        for seed in seeds[batch_idx]:
            x, y, amp = seed
            x, y = int(x), int(y)  # Ensure x and y are integers
            for i in range(width):
                for j in range(height):
                    dp[batch_idx, i, j] += amp * torch.exp(-((i - x) ** 2 + (j - y) ** 2) / (2 * sigma ** 2))

    return dp


def create_circular_voronoi_diagram(seeds, grid_size, pad_size):
    voronoi_diagram = discrete_voronoi(seeds, grid_size)
    radius = grid_size // 2
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x ** 2 + y ** 2 <= radius ** 2
    voronoi_diagram[~mask] = 0
    padded_voronoi = pad_voronoi_diagram(voronoi_diagram, grid_size, pad_size)
    return padded_voronoi


def pad_voronoi_diagram(nearest_seed_index, grid_size, pad_size):
    pad_width = (pad_size - grid_size) // 2
    padded_voronoi = np.pad(nearest_seed_index, pad_width, mode='constant', constant_values=0)
    return padded_voronoi


def assign_phase_values(nearest_seed_index, seeds, grid_size, pad_size):
    num_seeds = seeds.shape[0]
    phase_values = seeds[:, 2]  # Phase values from the seeds array
    phase_assigned = phase_values[nearest_seed_index]
    labels = nearest_seed_index

    radius = grid_size // 2
    circle_center = pad_size // 2

    for i in range(pad_size):
        for j in range(pad_size):
            if (i - circle_center) ** 2 + (j - circle_center) ** 2 > radius ** 2:
                phase_assigned[i, j] = 0

    return phase_assigned, labels, phase_values


def create_density_function(amplitude, phase):
    two_pi = 2.0 * torch.pi
    real = amplitude * torch.cos(two_pi * phase)
    imag = amplitude * torch.sin(two_pi * phase)
    complex = torch.complex(real, imag)
    return complex


def create_diffraction_pattern(patterns, grid_size, sigma):

    device = patterns.device
    batch_size = patterns.size(0)
    height = grid_size * 2
    width = grid_size * 2
    dp = torch.zeros(batch_size, height, width).to(device)

    # Generate a grid of coordinates
    x_coords = torch.arange(width, device=device).view(1, 1, -1)
    y_coords = torch.arange(height, device=device).view(1, -1, 1)

    for batch_idx in range(batch_size):
        pattern = patterns[batch_idx]
        x = pattern[:, 0].view(-1, 1, 1)
        y = pattern[:, 1].view(-1, 1, 1)
        amp = pattern[:, 2].view(-1, 1, 1)

        # Calculate the squared distances and apply the Gaussian function
        dist_squared = (x_coords - x) ** 2 + (y_coords - y) ** 2
        gauss = amp * torch.exp(-dist_squared / (2 * sigma ** 2))

        # Sum the contributions from all patterns
        dp[batch_idx] = gauss.sum(dim=0)

    return dp

def calculate_chi_square(predicted, target):
    return torch.sum((predicted - target) ** 2) / torch.sum(target)

def support(output, circle_center, radius):
    """circular support"""
    dist_from_center = (output[:, :, 0] - circle_center) ** 2 + (output[:, :, 1] - circle_center) ** 2
    outside_circle = dist_from_center > radius ** 2
    # Padding
    pad_size = radius
    output[:, :, 0][outside_circle] = circle_center + radius * (output[:, :, 0][outside_circle] - circle_center) / torch.sqrt(dist_from_center[outside_circle])
    output[:, :, 1][outside_circle] = circle_center + radius * (output[:, :, 1][outside_circle] - circle_center) / torch.sqrt(dist_from_center[outside_circle])
    return output