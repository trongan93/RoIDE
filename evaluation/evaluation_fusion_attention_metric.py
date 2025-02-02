# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw

import matplotlib.pyplot as plt
import numpy as np

# Define the metrics for each image version
metrics = {
    "Raw LDR": {
        "B": [25.22, 9.37, -1.82, 3.83, 3.13],
        "G": [27.13, 10.06, -1.98, 3.62, 2.95],
        "R": [21.68, 8.79, -0.66, 5.59, 3.29]
    },
    "No Spatial Attention": {
        "B": [176.30, 62.43, -2.42, 4.02, 4.11],
        "G": [163.48, 58.66, -2.39, 3.85, 3.92],
        "R": [133.19, 47.26, -2.27, 3.87, 4.42]
    },
    "No Channel Attention": {
        "B": [145.48, 51.38, -2.44, 4.09, 3.83],
        "G": [131.32, 47.13, -2.39, 3.85, 3.72],
        "R": [128.25, 45.04, -2.38, 4.09, 4.38]
    },
    "Proposed Dual Attention": {
        "B": [86.74, 30.77, -2.39, 3.99, 3.36],
        "G": [105.20, 37.76, -2.38, 3.85, 3.10],
        "R": [85.03, 30.35, -2.20, 3.78, 4.23]
    }
}

metric_names = ["Mean", "Std Dev", "Skewness", "Kurtosis", "Entropy"]
channels = ["B", "G", "R"]
colors = ["blue", "green", "red"]
image_versions = list(metrics.keys())

# Create bar charts for each metric
for i, metric in enumerate(metric_names):
    fig, ax = plt.subplots()
    for j, (channel, color) in enumerate(zip(channels, colors)):
        values = [metrics[img][channel][i] for img in image_versions]
        ax.bar(np.arange(len(image_versions)) + j * 0.25, values, width=0.25, label=f"{channel} Channel", color=color)
    ax.set_xticks(np.arange(len(image_versions)) + 0.25)
    ax.set_xticklabels(image_versions)
    ax.set_title(f"{metric} Comparison")
    ax.legend()
    plt.show()
