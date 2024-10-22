import numpy as np
import tifffile as tiff
from skimage.feature import peak_local_max
from skimage import img_as_float
import matplotlib.pyplot as plt

def find_maxima_with_prominence(image_path, prominence):
    # Load the image
    image = tiff.imread(image_path)
    # Normalise the image to [0,1], my data was normalised
    image = img_as_float(image)
    # Find all local maxima for feature detection and image segmentation
    coordinates = peak_local_max(image, min_distance=1, threshold_abs=0)
    # Filter maxima based on prominence
    maxima = []
    for coord in coordinates:
        x, y = coord
        local_max = image[x, y]
        # 2 by 2 grid
        surrounding = image[max(0, x-1):x+1, max(0, y-1):y+1]
        #plt.imshow(surrounding),plt.colorbar(),plt.show()
        if local_max - np.max(surrounding[surrounding != local_max]) >= prominence:
            maxima.append((x, y, local_max))

    return maxima, image

def plot_image_with_maxima(image, maxima, image_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='hot', vmin=np.min(image), vmax=np.max(image))
    plt.colorbar()
    if maxima:
        x, y, _ = zip(*maxima)  # Unzip the list of tuples into three lists
        plt.scatter(y, x, color='cyan', s=40, marker='o', label='Maxima')
    plt.title('Local Maxima in Image')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

# Example usage
image_path = '2d_df_5domains.tif'
prominence = 0.001  # Adjust the prominence value as needed
maxima_points, image = find_maxima_with_prominence(image_path, prominence)
print("Maxima coordinates:", maxima_points)
plot_image_with_maxima(image, maxima_points, image_path)

#do it in amplitude not intensity
# Parameters
width = 32
height = 32
sigma = 1.5

# Create a blank canvas
dp = np.zeros((height, width))

# Create Gaussian distributions for each point and add to the canvas
for x, y, amp in maxima_points:
    for i in range(width):
        for j in range(height):
            dp[i, j] += amp * np.exp(-((i - x) ** 2 + (j - y) ** 2) / (2 * sigma ** 2))

# Plot the resulting diffraction pattern
plt.imshow(dp, cmap='hot', origin='lower')
plt.colorbar()
if maxima_points:
    x, y, _ = zip(*maxima_points)  # Unzip the list of tuples into three lists
    plt.scatter(y, x, color='cyan', s=40, marker='o', label='Maxima')
plt.title('Diffraction Pattern with Gaussian Distributions')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.show()

# Function to calculate chi-square difference
def calculate_chi_square(original, expected):
    chi_square = np.sum(((original - expected) ** 2)) / np.sum(expected)
    return chi_square

# Calculate chi-square difference
chi_square_value = calculate_chi_square(image, dp)
print(f"Chi-Square Difference: {chi_square_value}")