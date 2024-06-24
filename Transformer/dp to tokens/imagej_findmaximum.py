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
        # 3 by 3 grid
        surrounding = image[max(0, x-1):x+2, max(0, y-1):y+2]
        #plt.imshow(surrounding),plt.colorbar(),plt.show()
        if local_max - np.max(surrounding[surrounding != local_max]) >= prominence:
            maxima.append((x, y, local_max))

    return maxima, image

def plot_image_with_maxima(image, maxima, image_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='hot', vmin=np.min(image), vmax=np.max(image) * 0.5)
    if maxima:
        x, y, _ = zip(*maxima)  # Unzip the list of tuples into three lists
        plt.scatter(y, x, color='cyan', s=40, marker='o', label='Maxima')
    plt.title('Local Maxima in Image')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

# Example usage
image_path = '2d_df_10domains.tif'
prominence = 0.001  # Adjust the prominence value as needed
maxima_points, image = find_maxima_with_prominence(image_path, prominence)
print("Maxima coordinates:", maxima_points)
plot_image_with_maxima(image, maxima_points, image_path)

#do it in amplitude not intensity
plt.imshow(maxima_points)
plt.show()