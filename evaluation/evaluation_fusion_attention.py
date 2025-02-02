# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from scipy.stats import skew, kurtosis, entropy

def plot_cdf(image_path, output_dir):
    """Plot cumulative distribution function (CDF) for each color channel of an image and save as an image."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return

    # Extract the base name of the image file to use in the title and output filename
    base_name = os.path.basename(image_path)
    base_name_no_ext = os.path.splitext(base_name)[0]

    # Create adaptive title and output filename
    title = f'CDF for {base_name_no_ext}'
    output_filename = f'cdf_{base_name_no_ext}.png'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    color = ('b', 'g', 'r')
    plt.figure()
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        plt.plot(cdf_normalized, color=col)
        plt.xlim([0, 256])
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Cumulative Frequency')

    # Save the CDF plot
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    print(f"CDF plot saved to {output_path}")

def calculate_statistics(image):
    """Calculate mean, standard deviation, skewness, kurtosis, and entropy for each color channel of an image."""
    stats = {}
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        channel = image[:, :, i]
        hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
        stats[col] = {
            'mean': np.mean(channel),
            'std_dev': np.std(channel),
            'skewness': skew(channel.flatten()),
            'kurtosis': kurtosis(channel.flatten()),
            'entropy': entropy(hist, base=2)
        }
    return stats

def save_statistics(stats, title, filename):
    """Save statistical metrics for each color channel of an image to a file."""
    with open(filename, 'w') as f:
        f.write(f'{title}\n')
        for color, stat in stats.items():
            f.write(f'Channel: {color.upper()}\n')
            for key, value in stat.items():
                f.write(f'  {key.capitalize()}: {value}\n')

# Example usage
image_path = './attention-test/test1/proposed-dual-attention_mnt_d_Seaport_satellite_images_lng_4.100000_lat_52.000000_sentinel_2_rgb.png'  # Update the path as needed
output_dir = './attention-test/output_plots'

# Plot CDF and save
plot_cdf(image_path, output_dir)

# Calculate and save statistics
image = cv2.imread(image_path)
base_name = os.path.basename(image_path)
base_name_no_ext = os.path.splitext(base_name)[0]
title = f'Statistics for {base_name_no_ext}'
stats_filename = f'stats_{base_name_no_ext}.txt'

statistics = calculate_statistics(image)
save_statistics(statistics, title, os.path.join(output_dir, stats_filename))
