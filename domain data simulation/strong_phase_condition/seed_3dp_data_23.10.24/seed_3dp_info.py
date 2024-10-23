
################################################################################
# multi_domains_simulation.py
# Simulations of multi-domains crystals with Voronoi partitions,
# and corresponding diffraction patterns
# - Jialun Liu, LCN, UCL, 11.01.2024, jialun.liu.17@ucl.ac.uk
# output movie of series of domains for UMinn test  IKR 9/2024
# option to switch to list of input (x,y,ph) values instead of random
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import math
import tifffile as tiff
import h5py
from scipy.fft import fft2, fftshift
from skimage.feature import peak_local_max
from skimage import img_as_float

#################### Parameters ####################
N = 50 # Number of crystals
num_seeds = 6  # Number of domains
grid_size = 16  # Size of the crystal
pad_size = 32  # Size of the sample
zoom_size = 32  # Size of the zoomed diffraction pattern
clash = 4  # minimum overlap distance of seeds (originally 3)
cut = 0.5  # radians minimum separation of allowed phase values
N_maxima = 100 # Fixed number of DP maxima for the batch training
prominence = 0.001  # Adjust the prominence value as needed between 0 and 1
threshold = 0.01 # Remove the pixels below the threshold amplitude

#################### Functions ####################

def generate_circular_seeds(num_seeds, radius, seed):
    np.random.seed(seed)
    seeds = []
    while len(seeds) < num_seeds:
        x, y = np.random.rand(2) * 2 - 1  # Generate points
        if x ** 2 + y ** 2 <= 1:  # Check if the point is inside the circle
            new_seed = np.array([x * radius + radius, y * radius + radius])
            # Check for overlapping with existing seeds
            overlap = False
            for seed in seeds:
                if np.linalg.norm(new_seed - seed[:2]) < clash:
                    overlap = True
                    break
            if not overlap:
                amplitude = 1.0
                phase = np.random.uniform(-np.pi, np.pi)
                seeds.append([new_seed[0], new_seed[1], amplitude, phase])
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
    # do not reset random number seed from choosing positions
    # force phase values to be sufficiently different
    phase_values = np.zeros(num_seeds) + 99
    for i in range(num_seeds):
        phase = np.random.uniform(-np.pi, np.pi-cut)  # reduce phase range to avoid close values near pi
        # keep repeating until sufficiently different phase found
        while np.min(np.abs(phase_values-phase)) < cut :
            phase = np.random.uniform(-np.pi, np.pi-cut)
        phase_values[i] = phase

    #print(phase_values)
    # Map the phase values to the Voronoi regions
    phase_assigned = phase_values[nearest_seed_index]
    labels = nearest_seed_index

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
    intensity = np.abs(ft_shifted)
    magnitude = np.sqrt(intensity)

#    return magnitude, ft_shifted
    return magnitude

def remove_zero_intensity(diffraction_pattern, threshold = threshold):
    return np.where(diffraction_pattern > threshold, diffraction_pattern, 0)

def find_maxima_with_prominence(image, prominence):
    """ImageJ find maximum method to extract the DP feature"""
    # Normalise the image to [0,1], my data was normalised
    image = img_as_float(image)
    # Find all local maxima for feature detection and image segmentation
    coordinates = peak_local_max(image, min_distance=1, threshold_abs=0)
    maxima = []
    for coord in coordinates:
        x, y = coord
        local_max = image[x, y]
        # 3 by 3 grid
        surrounding = image[max(0, x-1):x+2, max(0, y-1):y+2]
        #plt.imshow(surrounding),plt.colorbar(),plt.show()
        if local_max - np.max(surrounding[surrounding != local_max]) >= prominence:
            maxima.append([x, y, round(local_max, 4)])
    maxima_pad = np.pad(maxima, ((0, N_maxima - len(maxima)), (0, 0)), mode='constant',
                   constant_values=0)

    return maxima_pad

def save_as_tiff(data, filename):
    tiff.imsave(f'{filename}.tif', data)


def save_data(seeds, diffraction_patterns, prefix):
    """Save seeds and diffraction patterns to both txt and h5 formats."""

    # Function to save seeds
    def save_seeds(seeds, txt_path, h5_path):
        # Save seeds to txt
        with open(txt_path, 'w') as seed_file:
            seed_file.write('x y amplitude phase\n')
            for i, seed in enumerate(seeds):
                seed_file.write(f'Seeds Group {i + 1}:\n')
                np.savetxt(seed_file, seed, fmt='%f')
                seed_file.write('\n')
        # Save seeds to h5
        with h5py.File(h5_path, 'w') as f:
            for i, seed in enumerate(seeds):
                f.create_dataset(f'seeds_group_{i + 1}', data=seed)

    # Function to save diffraction patterns
    def save_diffraction_patterns(dp_patterns, prefix):
        for dp_type, dp_set in enumerate(dp_patterns):  # Loop over DP types
            txt_path = f'{prefix}_dp_{dp_type}.txt'
            h5_path = f'{prefix}_dp_{dp_type}.h5'

            # Save diffraction patterns to txt
            with open(txt_path, 'w') as dp_file:
                dp_file.write('Flattened Diffraction Pattern\n')
                for i, dp_frame in enumerate(dp_set):
                    dp_file.write(f'Diffraction Pattern Group {i + 1}:\n')
                    np.savetxt(dp_file, dp_frame.flatten().reshape(1, -1), fmt='%f')
                    dp_file.write('\n')

            # Save diffraction patterns to h5
            with h5py.File(h5_path, 'w') as f:
                for i, dp_frame in enumerate(dp_set):
                    f.create_dataset(f'dp_group_{i + 1}', data=dp_frame)

    # Call saving functions for seeds and diffraction patterns
    save_seeds(seeds, f'{prefix}_seeds.txt', f'{prefix}_seeds.h5')
    save_diffraction_patterns(diffraction_patterns, prefix)


#################### Simulation and Output ####################

file_lab = 'train_simulated_'
all_seeds = []
all_full_diffraction = []
all_thresholded_diffraction = []
all_maxima_diffraction = []

# Loop over each frame to simulate and plot
for j in range(0, N):  # Simulating for 50 frames
    seeds = generate_circular_seeds(num_seeds, grid_size // 2, j)
    all_seeds.append(seeds)

    padded_voronoi_diagram = create_circular_voronoi_diagram(seeds, grid_size, pad_size)
    phase_assigned_voronoi, labels, phase_values = assign_phase_values(padded_voronoi_diagram, num_seeds, grid_size, pad_size)

    amplitude = create_amplitude_function(grid_size, pad_size)
    density_function = create_density_function(amplitude, phase_assigned_voronoi)
    diffraction_pattern = fourier_transform(density_function)
    diffraction_pattern_norm = diffraction_pattern / np.max(diffraction_pattern)

    # Generate three versions of the diffraction pattern
    full_diffraction = diffraction_pattern_norm
    thresholded_diffraction = remove_zero_intensity(full_diffraction)
    maxima_diffraction = find_maxima_with_prominence(full_diffraction, prominence)

    all_full_diffraction.append(full_diffraction)
    all_thresholded_diffraction.append(thresholded_diffraction)
    all_maxima_diffraction.append(maxima_diffraction)

"""    # Plot the amplitude of the crystal
    plt.figure()
    plt.imshow(amplitude, cmap='viridis')
    plt.colorbar()
    plt.title(f'Crystal Amplitude (Frame {j})')
    plt.show()

    # Plot the phase of the crystal
    plt.figure()
    plt.imshow(phase_assigned_voronoi, cmap='twilight')
    plt.colorbar()
    plt.title(f'Crystal Phase (Frame {j})')
    plt.show()

    # Plot the full diffraction pattern
    plt.figure()
    plt.imshow(full_diffraction, cmap='inferno')
    plt.colorbar()
    plt.title(f'Full Diffraction Pattern (Frame {j})')
    plt.show()

    # Plot the thresholded diffraction pattern (with 0 removed)
    plt.figure()
    plt.imshow(thresholded_diffraction, cmap='plasma')
    plt.colorbar()
    plt.title(f'Thresholded Diffraction Pattern (Frame {j})')
    plt.show()

    # Plot the diffraction pattern maxima
    plt.figure()
    plt.imshow(maxima_diffraction, cmap='cool')
    plt.colorbar()
    plt.title(f'Diffraction Pattern Maxima (Frame {j})')
    plt.show()"""

# Convert all the lists to arrays for saving
all_seeds = np.concatenate(all_seeds, axis=0)
all_full_diffraction = np.array(all_full_diffraction)
all_thresholded_diffraction = np.array(all_thresholded_diffraction)
all_maxima_diffraction = np.array(all_maxima_diffraction)

# Save seeds and diffraction patterns in txt and h5 formats (4 files total)
save_data(all_seeds, [all_full_diffraction, all_thresholded_diffraction, all_maxima_diffraction], file_lab)


