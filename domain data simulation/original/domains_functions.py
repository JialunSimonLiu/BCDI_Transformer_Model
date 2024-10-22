################################################################################
# multi_domains_simulation.py
# Simulations of multi-domains crystals with Voronoi partitions,
# and corresponding diffraction patterns
# - Jialun Liu, LCN, UCL, 11.01.2024, jialun.liu.17@ucl.ac.uk

################################################################################
import numpy as np
import matplotlib.pyplot as plt
import math
import tifffile as tiff
from PIL import Image

#################### Functions ####################
def generate_circular_seeds(num_seeds, radius):
    #np.random.seed(0)
    seeds = []
    while len(seeds) < num_seeds:
        x, y = np.random.rand(2) * 2 - 1  # Generate points
        if x ** 2 + y ** 2 <= 1:  # Check if the point is inside the circle
            new_seed = np.array([x * radius + radius, y * radius + radius])
            # Check for overlapping with existing seeds
            overlap = False
            for seed in seeds:
                if np.linalg.norm(new_seed - seed) < 3:
                    overlap = True
                    break
            # Add the new seed if it doesn't overlap
            if not overlap:
                seeds.append(new_seed)
            # Break if the loop is stuck
            if len(seeds) > 10 * num_seeds:
                print(
                    "Unable to place all seeds without overlap.")
                break

    return np.array(seeds)

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

def create_circular_voronoi_diagram(seeds, grid_size, pad_size):
    # Creates and pads a circular Voronoi diagram.

    voronoi_diagram = discrete_voronoi(seeds, grid_size)

    # Apply circular mask
    radius = grid_size // 2
    y, x = np.ogrid[-radius:radius, -radius:radius]
    mask = x**2 + y**2 <= radius**2
    voronoi_diagram[~mask] = 0 # Mark outside of the circle

    # Pad the diagram
    padded_voronoi = pad_voronoi_diagram(voronoi_diagram, grid_size, pad_size)
    return padded_voronoi

def pad_voronoi_diagram(nearest_seed_index, grid_size, pad_size):
    # Pad the result to the desired size
    pad_width = (pad_size - grid_size) // 2
    padded_voronoi = np.pad(nearest_seed_index, pad_width, mode='constant', constant_values=0)
    return padded_voronoi


def assign_phase_values(nearest_seed_index, num_seeds, grid_size, pad_size):
    """
    Assigns random phase values to each region in the Voronoi diagram.
    """
    # Generate random phase values for each seed
    phase_values = np.random.uniform(-np.pi, np.pi, num_seeds)
    # Map the phase values to the Voronoi regions
    phase_assigned = phase_values[nearest_seed_index]
    labels = nearest_seed_index
    # Setting phase values to 0 for padding
    # Circular

    radius = grid_size // 2
    circle_center = pad_size/2

    for i in range(pad_size):
        for j in range(pad_size):
            if (i - circle_center) ** 2 + (j - circle_center) ** 2 > radius ** 2:
                phase_assigned[i, j] = 0  # Set phase value to 0 outside the circle

    return phase_assigned, labels, phase_values

def create_amplitude_function(grid_size, pad_size):
    """
    Creates a 2D projection of a sphere to use as an amplitude function.
    """
    # Initialise the grid
    amplitude_grid = np.zeros((grid_size, grid_size))

    # Sphere parameters
    radius = grid_size / 2
    center = grid_size / 2

    # Create the 2D projection of the sphere
    for i in range(grid_size):
        for j in range(grid_size):
            distance_from_center = np.sqrt((i - center)**2 + (j - center)**2)
            if distance_from_center <= radius:
                amplitude_grid[i, j] = np.sqrt(radius**2 - distance_from_center**2)

    # Pad the amplitude grid
    pad_width = (pad_size - grid_size) // 2
    padded_amplitude = np.pad(amplitude_grid, pad_width, mode='constant', constant_values=0)
    padded_amplitude_norm = padded_amplitude/ np.max(padded_amplitude)
    return padded_amplitude_norm

def create_density_function(amplitude, phase):
    """
    Combines the amplitude and phase to form a complex-valued function.
    """
    return amplitude * np.exp(1j * phase)

def fourier_transform(image):
    """
    Computes the Fourier transform of an image and returns its magnitude (diffraction pattern).
    """
    # Compute the Fourier transform
    ft = np.fft.fft2(image)
    ft_shifted = np.fft.fftshift(ft) # Contains full information of the DP

    # Compute the magnitude (diffraction pattern)
    magnitude = np.abs(ft_shifted)

    return magnitude, ft_shifted

