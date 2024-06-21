"""import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import peak_local_max
from skimage import exposure

# Load image and convert to grayscale
def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')  # Convert to grayscale
    return np.array(img)


# Detect peaks in a specific central region
def detect_peaks(image_array, num_peaks, region_size=20, visualize=True):
    # Apply adaptive histogram equalization for contrast enhancement
    image_eq = np.array(image_array)

    # Define the central region
    center_x, center_y = image_eq.shape[0] // 2, image_eq.shape[1] // 2
    central_region = image_eq[center_x - region_size:center_x + region_size,
                              center_y - region_size:center_y + region_size]

    # Detect peaks
    coordinates = peak_local_max(central_region, num_peaks=num_peaks, min_distance=5, threshold_rel=0.5)

    # Adjust coordinates relative to the full image
    coordinates[:, 0] += center_x - region_size
    coordinates[:, 1] += center_y - region_size

    if visualize:
        # Plot the image and peaks
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap='gray')
        ax.scatter(coordinates[:, 1], coordinates[:, 0], s=10, color='red', label='Detected Peaks')
        ax.legend()
        plt.title('Diffraction Pattern with Highlighted Peaks')
        plt.show()

    return coordinates

# Main execution
image_path = '2d_df_10domains.tif'
image = load_image(image_path)
num_peaks = 1
coordinates = detect_peaks(image, num_peaks)

plt.imshow(image)
plt.colorbar()
plt.show()"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import exposure

def load_image(image_path):
    # Load the image and convert to grayscale
    with Image.open(image_path) as img:
        img = img.convert('L')
    return np.array(img)

def enhance_image(image_array):
    # Apply adaptive histogram equalization to enhance contrast
    return exposure.equalize_adapthist(image_array)

def detect_peaks(image_eq, num_peaks=10, min_distance=5, threshold_rel=0.15):
    # Detect peaks in the image
    coordinates = peak_local_max(image_eq, num_peaks=num_peaks, min_distance=min_distance, threshold_rel=threshold_rel)
    return coordinates

def plot_peaks(image_eq, coordinates):
    # Plot the image with detected peaks highlighted
    plt.figure(figsize=(10, 10))
    plt.imshow(image_eq, cmap='gray')
    plt.scatter(coordinates[:, 1], coordinates[:, 0], s=100, color='red', edgecolor='r', label='Detected Peaks')
    plt.title('Diffraction Pattern with Highlighted Peaks')
    plt.axis('off')  # Hide axes
    plt.colorbar()
    plt.show()

# Main execution block
image_path = r'C:\Users\Ian\Documents\LJL\Machine Learning\ml\Transformer\2d_df_5domains.tif'
 # Adjust this to the correct path where you saved the file
image_array = load_image(image_path)
image_eq = enhance_image(image_array)
num_peaks = int(input("Enter the desired number of peaks to detect: "))  # Allow user to input the number of peaks to detect
coordinates = detect_peaks(image_eq, num_peaks=num_peaks)
plot_peaks(image_array, coordinates)

#watershed imageJ