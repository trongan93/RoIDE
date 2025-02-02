# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw


import numpy as np
from skimage.metrics import structural_similarity as ssim

def pu_ssim(img1, img2):
    # Pyramid decomposition levels
    levels = 3 #or 5
    # Default weights for PU-SSIM
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    ssim_vals = [_pu_ssim(img1, img2, level) for level in range(levels)]
    # Add a small epsilon value to SSIM values to avoid raising negative values to a power
    epsilon = 1e-10
    ssim_vals = [ssim_val + epsilon for ssim_val in ssim_vals]
    msssim = np.prod([ssim_vals[level] ** weights[level] for level in range(levels)])
    return msssim

def _pu_ssim(img1, img2, level):
    downsample_factor = 2 ** level
    img1_downsampled = img1[::downsample_factor, ::downsample_factor, :]
    img2_downsampled = img2[::downsample_factor, ::downsample_factor, :]
    data_range = img2_downsampled.max() - img2_downsampled.min()
    min_size = min(img1_downsampled.shape[0], img1_downsampled.shape[1])
    win_size = min(3, min_size)
    ssim_val = ssim(img1_downsampled, img2_downsampled, multichannel=True, data_range=data_range, win_size=win_size)
    return ssim_val

import torch
from lpips import LPIPS


def lpips_score(img1, img2):
    # Convert images to PyTorch tensors
    img1_tensor = torch.tensor(img1.transpose(2, 0, 1)).unsqueeze(0).float()  # Assuming img1 is in HWC format
    img2_tensor = torch.tensor(img2.transpose(2, 0, 1)).unsqueeze(0).float()  # Assuming img2 is in HWC format

    # Initialize LPIPS model
    # lpips_model = LPIPS(net='alex', spatial=True)
    lpips_model = LPIPS(net='alex', spatial=True, lpips=True)

    # Compute LPIPS score
    lpips_value = lpips_model(img1_tensor, img2_tensor)

    return lpips_value.mean().item()

# # Example usage
# img1 = np.random.rand(512, 512, 3)
# img2 = np.random.rand(512, 512, 3)
#
# # PU-MS-SSIM stands for Pyramid-Weighted Multi-Scale Structural Similarity Index Measure. It combines the pyramid decomposition from Pu-SSIM with the multi-scale approach from MS-SSIM.
# pu_ssim_score = pu_ssim(img1, img2)
# print("PU-SSIM:", pu_ssim_score)
#
# # LPIPS (Learned Perceptual Image Patch Similarity): LPIPS is a metric that measures the similarity between two images based on features extracted from a pre-trained deep neural network. It is designed to capture perceptual similarity more accurately than traditional metrics like SSIM.
# lpips_value = lpips_score(img1, img2)
# print("LPIPS:", lpips_value)


import cv2
def reinhard_tone_mapping(hdr_image, intensity=0.18, light_adapt=0.6):
    # Convert the HDR image to the LDR image using Reinhard tone mapping
    ldr_image = np.zeros_like(hdr_image, dtype=np.float32)

    # Calculate the luminance of the HDR image
    luminance = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2GRAY)
    luminance_mean = np.exp(np.mean(np.log(1e-6 + luminance)))

    # Perform Reinhard tone mapping
    ldr_image = hdr_image / (1 + hdr_image / (intensity * luminance_mean))

    # Apply light adaptation
    ldr_image = light_adapt * (ldr_image * (1 / luminance_mean))

    # Clip the LDR image to the [0, 1] range
    ldr_image = np.clip(ldr_image, 0, 1)

    # Convert the LDR image to uint8 format
    ldr_image = (255 * ldr_image).astype(np.uint8)

    return ldr_image

from skimage.metrics import peak_signal_noise_ratio as psnr
def pu_psnr(img1, img2):
    # Convert images to LAB color space
    psnr_val = psnr(img1, img2)
    return psnr_val



import glob
import pandas as pd
if __name__ == '__main__':

    generated_hdr_path = './test_output_seaport/set3_HDRNetFusion_seaport_rgb_dataset_x8'
    ldr_path = "/mnt/d/SeaportDataset/test_set_3"
    reinhard_tone_mapping_path = './test_output_seaport/set3_HDRNetFusion_seaport_rgb_dataset_x8_tone_mapped'

    sum_time = 0
    generated_hdr_path_list = glob.glob(generated_hdr_path + "/*")

    score_results = []
    for full_path_generated_hdr_test_file in generated_hdr_path_list:
        print(full_path_generated_hdr_test_file)
        generated_hdr_test_file = full_path_generated_hdr_test_file.split('/')[-1]
        # print(generated_hdr_test_file)
        full_path_ldr_test_file = ldr_path + '/' + generated_hdr_test_file
        print(full_path_ldr_test_file)

        hdr_image_data = cv2.imread(full_path_generated_hdr_test_file, cv2.IMREAD_COLOR).astype('uint8')
        hdr_image_data = reinhard_tone_mapping(hdr_image_data)
        cv2.imwrite(reinhard_tone_mapping_path + '/' + generated_hdr_test_file, hdr_image_data)
        ldr_image_data = cv2.imread(full_path_ldr_test_file, cv2.IMREAD_COLOR).astype('uint8')

        # hdr_image_data_norm = cv2.normalize(hdr_image_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # ldr_image_data_norm = cv2.normalize(ldr_image_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        pu_ssim_score = pu_ssim(hdr_image_data,ldr_image_data)
        print(pu_ssim_score)

        pu_psnr_score = pu_psnr(hdr_image_data,ldr_image_data)
        print(pu_psnr_score)

        lpips_score_val = lpips_score(hdr_image_data, ldr_image_data)
        print(lpips_score_val)

        # score_results.append('PU SSIM: {}, LPIPS: {}'.format(pu_ssim_score, pu_ssim_score))
        score_results.append({'full_path_ldr_test_file': full_path_ldr_test_file, 'full_path_generated_hdr_test_file': full_path_generated_hdr_test_file, 'pu_ssim_score': pu_ssim_score, 'pu_psnr_score': pu_psnr_score, 'lpips_score': lpips_score_val})


        # break

    # Create Data frame from the score_results
    df = pd.DataFrame(score_results)

    # Write the DataFrame to Excel file
    df.to_excel('./score_results.xlsx', index=False)