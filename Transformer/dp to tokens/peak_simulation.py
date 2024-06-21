import numpy as np
import matplotlib.pyplot as plt

def simulate_peaks(num_peaks):
    actual_positions = np.linspace(10, 90, num_peaks)
    actual_intensities = np.random.uniform(low=50, high=100, size=num_peaks)
    actual_widths = np.random.uniform(low=5, high=10, size=num_peaks)

    predicted_positions = actual_positions + np.random.normal(loc=0, scale=2.5, size=num_peaks)
    predicted_intensities = actual_intensities * np.random.uniform(0.8, 1.2, size=num_peaks)
    predicted_widths = actual_widths * np.random.uniform(0.8, 1.2, size=num_peaks)

    return (actual_positions, actual_intensities, actual_widths,
            predicted_positions, predicted_intensities, predicted_widths)

def plot_peaks(actual_positions, actual_intensities, predicted_positions, predicted_intensities):
    plt.figure(figsize=(10, 5))
    plt.stem(actual_positions, actual_intensities, linefmt='b-', markerfmt='bo', basefmt=" ", label='Actual Peaks')
    plt.stem(predicted_positions, predicted_intensities, linefmt='r--', markerfmt='rx', basefmt=" ", label='Predicted Peaks')
    plt.xlabel('Position')
    plt.ylabel('Intensity')
    plt.title('Actual vs Predicted Peaks')
    plt.legend()
    plt.show()

def gaussian_similarity(x1, x2, amp1, amp2, sigma_space=5, sigma_amp=5, alpha=0.5):
    #return np.exp(-np.square(x1[:, None] - x2) / (2 * sigma**2))
    pos1 = x1
    pos2 = x2
    pos_diff = np.square(pos1 - pos2)
    amp_diff = np.square(amp1 - amp2)
    combined_similarity = np.exp(
        -((alpha * pos_diff / (2 * sigma_space ** 2)) + ((1 - alpha) * amp_diff / (2 * sigma_amp ** 2))))
    print(combined_similarity)
    return combined_similarity

def plot_assignments(predicted_positions, actual_positions, predicted_intensities, actual_intensities):
    assignments = gaussian_similarity(predicted_positions, actual_positions, predicted_intensities, actual_intensities)
    #print(assignments)
    plt.figure(figsize=(8, 6))
    plt.plot(assignments)
    plt.title('Guassian Similarity')
    plt.xlabel('Peak Index')
    plt.ylabel('Similarity')
    plt.show()

def main():
    num_peaks = 10
    actual_positions, actual_intensities, actual_widths, predicted_positions, predicted_intensities, predicted_widths = simulate_peaks(num_peaks)
    plot_peaks(actual_positions, actual_intensities, predicted_positions, predicted_intensities)
    plot_assignments(predicted_positions, actual_positions, predicted_intensities, actual_intensities)

if __name__ == "__main__":
    main()
