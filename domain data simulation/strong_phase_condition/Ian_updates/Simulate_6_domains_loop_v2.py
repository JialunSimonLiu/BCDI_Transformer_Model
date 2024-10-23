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
#from PIL import Image

#################### Parameters ####################
num_seeds = 6 # Number of domains
grid_size = 16 # Size of the crystal
pad_size = 32 # Size of the sample
zoom_size = 32 # Size of the zoomed diffraction pattern
clash = 4 # minimum overlap distance of seeds (originally 3)
cut = 0.5  # radians minimum separation of allowed phase alues

#################### Functions ####################
def generate_circular_seeds(num_seeds, radius, s):
    np.random.seed(s)
    seeds = []
    while len(seeds) < num_seeds:
        x, y = np.random.rand(2) * 2 - 1  # Generate points
        if x ** 2 + y ** 2 <= 1:  # Check if the point is inside the circle
            new_seed = np.array([x * radius + radius, y * radius + radius])
            # Check for overlapping with existing seeds
            overlap = False
            for seed in seeds:
                if np.linalg.norm(new_seed - seed) < clash:
                    overlap = True
                    break
            # Add the new seed if it doesn't overlap
            if not overlap:
                #print('[%10.5f, %10.5f],' %(x,y))  # check they are the same
                #print('[%10.5f, %10.5f],' %(xy[len(seeds),0],xy[len(seeds),1]))
                #new_seed2 = np.array([xy[len(seeds),0] * radius + radius,
                #   xy[len(seeds),1] * radius + radius])
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
    # Setting phase values to 0 for padding
    # Cubic
    """
    pad_width = (pad_size - grid_size) // 2
    phase_assigned[:pad_width, :] = 0
    phase_assigned[-pad_width:, :] = 0
    phase_assigned[:, :pad_width] = 0
    phase_assigned[:, -pad_width:] = 0
    """
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
    intensity = magnitude**2

#    return magnitude, ft_shifted
    return intensity, ft_shifted

#################### Plotting and image processing ####################
def zoom_into_center(image, zoom_size):
    """
    Zooms into the center of an image.
    """
    center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
    half_zoom = zoom_size // 2
    return image[center_x - half_zoom:center_x + half_zoom, center_y - half_zoom:center_y + half_zoom]

#################### save files ####################
def save_as_tiff(data, filename, switch):
    data = data.astype(np.float32)
    if switch == 0:     # first time creates file
        return tiff.imwrite(filename, data, metadata=None)
    else:
        return tiff.imwrite(filename, data, append=True, metadata=None)

#################### Calculations ####################
# ********** executable code starts here *************
# declare position and phase input lists to override random numbers
# take values from version with random nubmers and seed = 0 to generate same image
# IKR 6/2024 additions
# IKR 9/2024 version to generate ranom sequence of frames
# ovoid overlap of seeds; avoid phases too close
file_lab = ('50-frames-v2')
ff = open((f'Seed_Phase_Pos_{num_seeds}domains_' + file_lab + '.txt'), "w")
ff.write(f"Frame Seed Phase [ PosX PosY ] \n")
for j in range(0,50):
# seed random numbers with iterate j
    seeds = generate_circular_seeds(num_seeds, grid_size // 2, j)
    #print(seeds)
#### Calculate Density functions (Amplitudes, phases), Diffraction patterns
# Assign phase values to each region
    padded_voronoi_diagram = create_circular_voronoi_diagram(seeds, grid_size, pad_size)
    phase_assigned_voronoi, labels, phase_values = assign_phase_values(padded_voronoi_diagram, num_seeds, grid_size, pad_size)
# Print seeds and corresponding phases and save them in a file too
    scaled_origin = (pad_size - grid_size) // 2
    for i, (seed, phase) in enumerate(zip(seeds, phase_values)):
        print(f"Seed {i}: Phase = {phase}, Position = {scaled_origin + seed}")
        ff.write(f"{j} {i} {phase} {scaled_origin + seed} \n")

    amplitude = create_amplitude_function(grid_size, pad_size) # Amplitude is normallised
    density_function = create_density_function(amplitude, phase_assigned_voronoi)
# Fourier transform to get the diffraction pattern
    diffraction_pattern = fourier_transform(density_function)[0]
    diffraction_pattern_norm = diffraction_pattern/np.max(diffraction_pattern)
# Zoom into the center of the Fourier Transform
    zoomed_df_norm = zoom_into_center(diffraction_pattern_norm, zoom_size)
    phase_scaled = (phase_assigned_voronoi + np.pi)
    save_as_tiff(amplitude, (f'amp_{num_seeds}domains_' + file_lab + '.tif'), j)
    save_as_tiff(phase_scaled, (f'phase_{num_seeds}domains_' + file_lab + '.tif'), j)
    save_as_tiff(zoomed_df_norm, (f'2d_df_{num_seeds}domains_' + file_lab + '.tif'), j)
    print("Files saved successfully.")

ff.close()


