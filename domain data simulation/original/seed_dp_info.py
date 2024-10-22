import numpy as np
import matplotlib.pyplot as plt
import math
import tifffile as tiff
from PIL import Image
from skimage.feature import peak_local_max
from skimage import img_as_float
from domains_functions import *
import h5py

#################### User-Defined Parameters ####################
N = 100 # Number of crystals to generate
plot_crystal_indices = [0]

#################### Parameters ####################
grid_size = 16  # Size of the crystal
pad_size = 32  # Size of the sample
zoom_size = 32  # Size of the zoomed diffraction pattern
prominence = 0.001  # Adjust the prominence value as needed between 0 and 1
N_maxima = 100 # Fixed number of DP maxima for the batch training

#################### Initializing Numpy Arrays for Amplitude and Phase Data ####################
real_amp_2d = np.zeros((N, pad_size, pad_size))
real_pha_2d = np.zeros((N, pad_size, pad_size))
real_density_2d = np.zeros((N, pad_size, pad_size)).astype(np.complex128)
dp_2d = np.zeros((N, pad_size, pad_size))
dp_norm_2d = np.zeros((N, pad_size, pad_size))
seeds_dp= []

#################### Functions ####################
def transform_diffraction_pattern(dp_pattern):

    # Flatten the 64x64 diffraction pattern
    dp_flat = dp_pattern.flatten()  # Changes shape from (64, 64) to (4096,)

    # Generate x, y coordinates for a 64x64 grid
    x, y = np.meshgrid(np.arange(pad_size), np.arange(pad_size), indexing='xy')
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Initialize the final array
    final_dp = np.zeros((pad_size*pad_size, 3))
    final_dp[:, 0] = x_flat  # Set x coordinates
    final_dp[:, 1] = y_flat  # Set y coordinates
    final_dp[:, 2] = dp_flat # Set intensity values

    return final_dp

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

#################### Main Simulation Loop ####################
# Seed data is (X_i,y_i,phase_i)
N_max = 5
for crystal_index in range(N):
    num_seeds = np.random.randint(1, N_max)  # Randomly choosing the number of domains between 1 and 5
    # Generating seeds and Voronoi diagram
    seeds = np.round(generate_circular_seeds(num_seeds, grid_size // 2))
    voronoi_diagram = create_circular_voronoi_diagram(seeds, grid_size, pad_size)
    seeds_pad_pos = seeds + grid_size// 2
    print("seeds_pad_pos", seeds_pad_pos)

    # Assigning phase values and initializing amplitude
    real_pha_2d[crystal_index], labels, phase = assign_phase_values(voronoi_diagram, num_seeds, grid_size, pad_size)
    real_amp_2d[crystal_index] = create_amplitude_function(grid_size, pad_size)
    #print("phase", phase)

    # Combine position and phase information for seeds
    seeds_pos_phase = np.round(np.hstack((seeds_pad_pos, phase[:, np.newaxis])),4)
    # ! important padding for the transformer model (fixed text length)
    seeds_final = np.pad(seeds_pos_phase, ((0, N_max - num_seeds), (0, 0)), mode='constant',
                         constant_values=0)  # Pad with 0
    #print("seeds_final", seeds_final)

    # Calculate a density function and the magnitude of the diffraction pattern
    real_density_2d[crystal_index] = create_density_function(real_amp_2d[crystal_index], real_pha_2d[crystal_index])
    dp_2d[crystal_index] = fourier_transform(real_density_2d[crystal_index])[0]
    dp_norm_2d[crystal_index] = dp_2d[crystal_index]/np.max(dp_2d[crystal_index])
    # Method I: flatten to 64 by 64
    #diffraction_pattern = transform_diffraction_pattern(dp_norm_2d[crystal_index])
    # Method II: extract the speckle features using the ImageJ find Maximum method
    diffraction_pattern = find_maxima_with_prominence(dp_norm_2d[crystal_index], prominence)

    seeds_dp.append((seeds_final, diffraction_pattern))

    # Plotting the specified crystal
    if crystal_index in plot_crystal_indices:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(real_amp_2d[crystal_index], cmap='bwr')
        plt.title('Amplitude')

        plt.subplot(1, 3, 2)
        plt.imshow(real_pha_2d[crystal_index], cmap='bwr', vmin=-np.pi, vmax=np.pi)
        plt.title('Phase')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        # Plot the diffraction pattern here
        plt.imshow(np.log(1 + dp_norm_2d[crystal_index]), cmap='jet')
        plt.title('Diffraction Pattern')

        plt.show()

# Saving all amplitudes and phases as .npy files
#np.save('real_amp_2d.npy', real_amp_2d)
#np.save('real_pha_2d.npy', real_pha_2d)
#np.save('dp_norm_2d.npy', dp_norm_2d)

def save_data(crystals, file_path='data_train.h5'):
    with h5py.File(file_path, 'w') as f:
        for i, (seeds, diffraction_patterns) in enumerate(crystals):
            group = f.create_group(f'seeds_dp_{i+1}')
            group.create_dataset('seeds', data=seeds)
            group.create_dataset('diffraction_patterns', data=diffraction_patterns)

#print(seeds_dp)
save_data(seeds_dp)

def save_data_as_text(crystals, file_path='data_train.txt'):
    with open(file_path, 'w') as file:
        for i, (seeds, diffraction_patterns) in enumerate(crystals):
            file.write(f'Seeds and Diffraction Patterns Group {i+1}\n')
            file.write('Seeds:\n')
            file.write(' '.join(map(str, seeds)) + '\n')
            file.write('Diffraction Patterns:\n')
            #for pattern in diffraction_patterns:
            file.write(' '.join(map(str, diffraction_patterns)) + '\n')
            file.write('\n')

save_data_as_text(seeds_dp)
