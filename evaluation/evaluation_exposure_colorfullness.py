# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw

import cv2
import numpy as np

def colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_root = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
    mean_root = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
    return std_root + (0.3 * mean_root)

# Example usage
image_path = './test-color-exposure/maxima-exposure-controller.png'
image = cv2.imread(image_path)
colorfulness_value = colorfulness(image)
print(f'Colorfulness: {colorfulness_value}')