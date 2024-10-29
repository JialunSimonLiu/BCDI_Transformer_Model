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

# Crystal parameters
grid_size = 16  # Size of the crystal
radius = grid_size // 2
sigma = 1.5
pad_size = 32

############################################# Functions #############################################
def discrete_voronoi(seeds, grid_size):
    device = seeds.device
    # Creating a meshgrid for the coordinates
    x = torch.arange(0, grid_size, device=device)
    y = torch.arange(0, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    # Expanding the dimensions for broadcasting
    xx = xx.unsqueeze(-1)
    yy = yy.unsqueeze(-1)
    # Calculate the distance from each point in the grid to each seed
    distances = torch.sqrt((xx - seeds[:, 0]) ** 2 + (yy - seeds[:, 1]) ** 2)
    # Find the nearest seed for each point in the grid
    nearest_seed_index = torch.argmin(distances, dim=2)
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


def pad_voronoi_diagram(voronoi_diagram, grid_size, pad_size):
    # Padding width calculated as before
    pad_width = (pad_size - grid_size) // 2
    # Apply padding using torch's F.pad() function
    padded_voronoi = F.pad(voronoi_diagram, (pad_width, pad_width, pad_width, pad_width), mode='constant', value=0)
    return padded_voronoi


def assign_phase_values(nearest_seed_index, seeds, grid_size, pad_size):
    device = seeds.device
    num_seeds = seeds.shape[0]
    phase_values = seeds[:, 2]  # Phase values from the seeds array
    # Wrap phases to the range [-pi, pi]
    pred_phases = ((phase_values + torch.pi) % (2 * torch.pi)) - torch.pi
    # Assign phase values based on the nearest seed index
    phase_assigned = phase_values[nearest_seed_index]
    labels = nearest_seed_index
    # Calculate radius and circle center
    radius = grid_size // 2
    circle_center = pad_size // 2
    # Create a grid for the padded size
    x = torch.arange(0, pad_size, device=device)
    y = torch.arange(0, pad_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    # Calculate the distance from the center for each point in the padded grid
    mask = (xx - circle_center) ** 2 + (yy - circle_center) ** 2 > radius ** 2
    phase_assigned[mask] = 0
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
    x_coords = torch.arange(width, device=device).reshape(1, 1, -1)
    y_coords = torch.arange(height, device=device).reshape(1, -1, 1)

    for batch_idx in range(batch_size):
        pattern = patterns[batch_idx]
        x = pattern[:, 0].reshape(-1, 1, 1)
        y = pattern[:, 1].reshape(-1, 1, 1)
        amp = pattern[:, 2].reshape(-1, 1, 1)

        # Calculate the squared distances and apply the Gaussian function
        dist_squared = (x_coords - x) ** 2 + (y_coords - y) ** 2
        gauss = amp * torch.exp(-dist_squared / (2 * sigma ** 2))

        # Sum the contributions from all patterns
        dp[batch_idx] = gauss.sum(dim=0)

    return dp

def calculate_chi_square(predicted, target):
    predicted_norm = predicted / torch.max(predicted)
    target_norm = target / torch.max(target)
    chi_squared = torch.sum((predicted_norm - target_norm) ** 2) / torch.sqrt(torch.sum(predicted_norm ** 2) * torch.sum(target_norm ** 2))
    return chi_squared

# Projection function
def project_onto_circle(x, y, center_x, center_y, radius):
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    if distance <= radius:
        return x, y
    else:
        return grid_size,grid_size


def avoid_seed_overlap(seeds, min_distance=1.0):
    num_seeds = seeds.size(0)

    for i in range(num_seeds):
        for j in range(i + 1, num_seeds):
            dist = torch.norm(seeds[i, :2] - seeds[j, :2], p=2)
            if dist < min_distance:
                # Compute the direction vector and normalize it
                direction = (seeds[j, :2] - seeds[i, :2])
                direction = direction / torch.norm(direction, p=2)  # Normalize direction
                # Calculate perturbation and update the second seed
                perturbation = direction * (min_distance - dist) / 2
                seeds[j, :2] += perturbation

                # Recheck for overlap after perturbation
                # If the seeds still overlap, we could repeat the process but let's break here for now
                break

    return seeds


def support(output, circle_center, radius):
    """circular support"""
    dist_from_center = (output[:, :, 0] - circle_center) ** 2 + (output[:, :, 1] - circle_center) ** 2
    outside_circle = dist_from_center > radius ** 2
    # Padding
    pad_size = radius
    output[:, :, 0][outside_circle] = circle_center + radius * (output[:, :, 0][outside_circle] - circle_center) / torch.sqrt(dist_from_center[outside_circle])
    output[:, :, 1][outside_circle] = circle_center + radius * (output[:, :, 1][outside_circle] - circle_center) / torch.sqrt(dist_from_center[outside_circle])
    return output

############################################# Loss function #############################################
class CustomLoss(nn.Module):
    def __init__(self, grid_size, radius, sigma, pad_size):
        super(CustomLoss, self).__init__()
        self.grid_size = grid_size
        self.radius = radius
        self.sigma = sigma
        self.pad_size = pad_size

    def forward(self, output, targets):

        device = output.device
        batch_size = output.size(0)

        # Reshape output and targets to (batch_size, num_seeds, 3)
        output = output.reshape(batch_size, -1, 3)
        targets = targets.reshape(batch_size, -1, 3)

        """# Ensure x, y in output are integers within bounds and avoid seed overlap
        output[:, :, 0:2] = torch.clamp(output[:, :, 0:2].round().int(), min=0, max=grid_size - 1)

        for batch_idx in range(batch_size):
            output[batch_idx] = avoid_seed_overlap(output[batch_idx])  # Efficient overlap check

        # Ensure x, y in output are within the supporting circle
        circle_center = grid_size // 2
        output = support(output, circle_center, radius)"""

        # Create the predicted real space directly in a tensor without list accumulation
        predicted_real_space = torch.zeros((batch_size, pad_size, pad_size), dtype=torch.complex64, device=device)


        for batch_idx in range(batch_size):
            seeds_pred = output[batch_idx]
            # Generate the Voronoi diagram and assign phase values
            padded_voronoi_pred = create_circular_voronoi_diagram(seeds_pred, grid_size, pad_size)
            phase_pred, _, _ = assign_phase_values(padded_voronoi_pred, seeds_pred, grid_size, pad_size)
            # Create the density function and fill into predicted real space tensor
            predicted_real_space[batch_idx] = create_density_function(1.0, phase_pred)

        # Fourier transform to get diffraction patterns
        predicted_real_space = torch.fft.fftshift(predicted_real_space, dim=(-2, -1))
        predicted_dp = torch.fft.fft2(predicted_real_space)
        predicted_dp = torch.fft.fftshift(predicted_dp, dim=(-2, -1)).abs()

        # Create target diffraction pattern from ground truth
        target_dp = create_diffraction_pattern(targets, grid_size, sigma)

        # Calculate chi-square loss between predicted and target diffraction patterns
        loss = calculate_chi_square(predicted_dp, target_dp)

        return loss


############################################# Plots #############################################
def plot_diffraction_patterns(predicted_dp, target_dp):
    """
    Plots the predicted and target diffraction patterns side-by-side.
    """
    plt.figure(figsize=(12, 6))

    # Predicted Diffraction Pattern
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_dp.cpu().numpy()[0], cmap='hot', interpolation='nearest')
    plt.title("Predicted Diffraction Pattern")
    plt.colorbar()

    # Target Diffraction Pattern
    plt.subplot(1, 2, 2)
    plt.imshow(target_dp.cpu().numpy()[0], cmap='hot', interpolation='nearest')
    plt.title("Target Diffraction Pattern")
    plt.colorbar()

    plt.show()


def plot_voronoi_crystal(seeds_pred, seeds_true, grid_size, pad_size):
    """
    Plots the predicted and target Voronoi crystals side-by-side.
    """
    # Generate Voronoi diagrams for both predicted and true seeds
    predicted_voronoi = create_circular_voronoi_diagram(seeds_pred, grid_size, pad_size)
    target_voronoi = create_circular_voronoi_diagram(seeds_true, grid_size, pad_size)

    # Move tensors to CPU and convert to NumPy for plotting
    predicted_voronoi = predicted_voronoi.cpu().numpy()
    target_voronoi = target_voronoi.cpu().numpy()

    plt.figure(figsize=(12, 6))

    # Predicted Voronoi Crystal
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_voronoi, cmap='cool', interpolation='nearest')
    plt.title("Predicted Voronoi Crystal")

    # Target Voronoi Crystal
    plt.subplot(1, 2, 2)
    plt.imshow(target_voronoi, cmap='cool', interpolation='nearest')
    plt.title("Target Voronoi Crystal")

    plt.show()


def visualise_final_results(model, test_loader, device, grid_size, pad_size, sigma):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data in test_loader:
            patterns, seeds_true = data
            patterns = patterns.to(device)
            seeds_true = seeds_true.to(device)

            # Forward pass through the model to get predictions
            output = model(patterns)
            # Print the shapes before reshaping
            print(f"Original output shape: {output.shape}")
            print(f"Original seeds_true shape: {seeds_true.shape}")

            # Reshape the output and ground truth for comparison
            output = output.reshape(output.size(0), -1, 3)  # Use reshape instead of view
            seeds_true = seeds_true.reshape(seeds_true.size(0), -1, 3)
            # Print the shapes after reshaping
            print(f"Reshaped output shape: {output.shape}")
            print(f"Reshaped seeds_true shape: {seeds_true.shape}")

            # Take the first batch for visualization (can be changed to any batch)
            predicted_seeds = output[0]  # Model prediction
            true_seeds = seeds_true[0]  # Ground truth

            # Create predicted diffraction pattern
            predicted_real_space = torch.zeros((1, pad_size, pad_size), dtype=torch.complex64, device=device)
            padded_voronoi_pred = create_circular_voronoi_diagram(predicted_seeds, grid_size, pad_size)
            phase_pred, _, _ = assign_phase_values(padded_voronoi_pred, predicted_seeds, grid_size, pad_size)
            predicted_real_space[0] = create_density_function(1.0, phase_pred)

            # Fourier transform to get diffraction pattern
            predicted_real_space = torch.fft.fftshift(predicted_real_space, dim=(-2, -1))
            predicted_dp = torch.fft.fft2(predicted_real_space)
            predicted_dp = torch.fft.fftshift(predicted_dp, dim=(-2, -1)).abs()

            # Create target diffraction pattern from ground truth seeds
            target_dp = predicted_dp#create_diffraction_pattern(true_seeds, grid_size, sigma)

            # Plot the diffraction patterns and Voronoi crystals
            plot_diffraction_patterns(predicted_dp, target_dp)
            plot_voronoi_crystal(predicted_seeds, true_seeds, grid_size, pad_size)

            # After visualizing one batch, break the loop
            break
