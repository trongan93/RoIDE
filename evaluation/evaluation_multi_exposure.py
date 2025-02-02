# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw


import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

# Load multi-exposure images
image_paths = glob.glob('./multi-exposure-generated/*.png')  # Update the path and file extension
if not image_paths:
    raise FileNotFoundError("No images found in the specified directory.")

# Sort image paths by the numerical part of the filename (e.g., 'x0', 'x1', ..., 'x16')
def extract_number(filename):
    match = re.search(r'x(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

image_paths.sort(key=extract_number)

# Calculate histograms for each image and each color channel
colors = ('b', 'g', 'r')  # OpenCV loads images in BGR order
histograms = {color: [] for color in colors}

for img_path in image_paths:
    img = cv2.imread(img_path)
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        histograms[color].append(hist)

# Create output directory if it doesn't exist
output_dir = './multi-exposure-generated-histogram'
os.makedirs(output_dir, exist_ok=True)

# Save images and histograms
for i, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Display the image
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[0].set_title(img_name)

    # Plot combined histograms for each color channel
    for color in colors:
        ax[1].plot(histograms[color][i], color=color)
    ax[1].set_xlim([0, 256])
    ax[1].set_title(f'RGB Histograms for {img_name}')
    ax[1].set_xlabel('Pixel Value')
    ax[1].set_ylabel('Frequency')

    # Save the figure
    output_path = os.path.join(output_dir, f'image_with_histogram_{i+1}.png')
    plt.savefig(output_path)
    plt.close(fig)

print(f"Images with histograms saved to {output_dir}")
