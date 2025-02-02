# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from scipy.stats import skew, kurtosis
import imageio.v2 as imageio
import os

# Load images
low_exposure_image_path = '/home/trongan93/Desktop/hdr-net-paper-test-case/case9-well-exposure/hdr_mnt_d_Seaport_satellite_images_lng_3.450000_lat_47.240000_sentinel_2_rgb.png'  # Ensure these paths are correct
normal_image_path = './multi-exposure-generated-test-color/normal.png'
color_distortion_image_path = './multi-exposure-generated-test-color/color-distorition.png'

low_exposure_image = cv2.imread(low_exposure_image_path)
normal_image = cv2.imread(normal_image_path)
color_distortion_image = cv2.imread(color_distortion_image_path)

if low_exposure_image is None or normal_image is None or color_distortion_image is None:
    print("Error: One or more images not found. Please check the file paths.")
else:
    # Resize images to match the dimensions of the normal image
    if low_exposure_image.shape != normal_image.shape:
        low_exposure_image = cv2.resize(low_exposure_image, (normal_image.shape[1], normal_image.shape[0]))
    if color_distortion_image.shape != normal_image.shape:
        color_distortion_image = cv2.resize(color_distortion_image, (normal_image.shape[1], normal_image.shape[0]))


    def plot_histograms(image, title, filename):
        """Plot histograms for each color channel of an image and save as an image."""
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.title(title)
        plt.savefig(filename)
        plt.close()


    def plot_cdf(image, title, filename):
        """Plot cumulative distribution function (CDF) for each color channel of an image and save as an image."""
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            plt.plot(cdf_normalized, color=col)
            plt.xlim([0, 256])
        plt.title(title)
        plt.savefig(filename)
        plt.close()


    def calculate_metrics(reference_image, test_image):
        """Calculate and print MSE, PSNR, and SSIM between two images."""
        mse = mean_squared_error(reference_image, test_image)
        psnr = peak_signal_noise_ratio(reference_image, test_image)
        ssim = structural_similarity(reference_image, test_image, multichannel=True, channel_axis=2, win_size=5)
        return mse, psnr, ssim


    def plot_rgb_distribution(image, title, gif_filename):
        """Plot 3D RGB distribution of an image and save as a rotating GIF."""
        r, g, b = cv2.split(image)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(r.flatten(), g.flatten(), b.flatten(),
                   c=np.vstack((r.flatten(), g.flatten(), b.flatten())).T / 255.0, marker='.')
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        plt.title(title)

        # Create rotation GIF
        with imageio.get_writer(gif_filename, mode='I', duration=0.1) as writer:
            for angle in range(0, 360, 10):
                ax.view_init(30, angle)
                filename = f"temp_{angle}.png"
                plt.savefig(filename)
                writer.append_data(imageio.imread(filename))
                os.remove(filename)
        plt.close()


    def calculate_statistics(image):
        """Calculate mean, variance, skewness, and kurtosis for each color channel of an image."""
        stats = {}
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            channel = image[:, :, i]
            stats[col] = {
                'mean': np.mean(channel),
                'variance': np.var(channel),
                'skewness': skew(channel.flatten()),
                'kurtosis': kurtosis(channel.flatten())
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


    # Create output directory if it doesn't exist
    output_dir = './multi-exposure-generated-test-color/output_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save histograms
    plot_histograms(low_exposure_image, 'Histogram for Low Exposure Image',
                    os.path.join(output_dir, 'histogram_low_exposure.png'))
    # plot_histograms(normal_image, 'Histogram for Normal Image', os.path.join(output_dir, 'histogram_normal.png'))
    # plot_histograms(color_distortion_image, 'Histogram for Color Distortion Image',
    #                 os.path.join(output_dir, 'histogram_color_distortion.png'))

    # Plot and save CDFs
    plot_cdf(low_exposure_image, 'CDF for Low Exposure Image', os.path.join(output_dir, 'cdf_low_exposure.png'))
    # plot_cdf(normal_image, 'CDF for Normal Image', os.path.join(output_dir, 'cdf_normal.png'))
    # plot_cdf(color_distortion_image, 'CDF for Color Distortion Image',
    #          os.path.join(output_dir, 'cdf_color_distortion.png'))

    # # Calculate and print metrics
    # mse_low, psnr_low, ssim_low = calculate_metrics(normal_image, low_exposure_image)
    # mse_color, psnr_color, ssim_color = calculate_metrics(normal_image, color_distortion_image)

    # Save metrics to file
    # with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
    #     f.write('Low Exposure Image vs Normal Image\n')
    #     f.write(f'MSE: {mse_low}\n')
    #     f.write(f'PSNR: {psnr_low}\n')
    #     f.write(f'SSIM: {ssim_low}\n\n')
    #     f.write('Color Distortion Image vs Normal Image\n')
    #     f.write(f'MSE: {mse_color}\n')
    #     f.write(f'PSNR: {psnr_color}\n')
    #     f.write(f'SSIM: {ssim_color}\n')

    # Plot and save RGB distribution as GIF
    # plot_rgb_distribution(low_exposure_image, 'RGB Distribution for Low Exposure Image',
    #                       os.path.join(output_dir, 'rgb_distribution_low_exposure.gif'))
    # plot_rgb_distribution(normal_image, 'RGB Distribution for Normal Image',
    #                       os.path.join(output_dir, 'rgb_distribution_normal.gif'))
    # plot_rgb_distribution(color_distortion_image, 'RGB Distribution for Color Distortion Image',
    #                       os.path.join(output_dir, 'rgb_distribution_color_distortion.gif'))

    # Calculate and save statistics
    # low_exposure_stats = calculate_statistics(low_exposure_image)
    # normal_stats = calculate_statistics(normal_image)
    # color_distortion_stats = calculate_statistics(color_distortion_image)
    #
    # save_statistics(low_exposure_stats, 'Low Exposure Image Statistics',
    #                 os.path.join(output_dir, 'low_exposure_image_statistics.txt'))
    # save_statistics(normal_stats, 'Normal Image Statistics', os.path.join(output_dir, 'normal_image_statistics.txt'))
    # save_statistics(color_distortion_stats, 'Color Distortion Image Statistics',
    #                 os.path.join(output_dir, 'color_distortion_image_statistics.txt'))
