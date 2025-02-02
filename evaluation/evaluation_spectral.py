# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw


import torch
import piq
from skimage.io import imread
from PIL import Image
import spectrum
import numpy as np
from spectrum.datasets import marple_data
from pylab import legend, ylim
import matplotlib.pyplot as plt
from math import log10, sqrt

@torch.no_grad()
def validattion_dataset_non_reference_factor(_test_img_path):
    x = torch.tensor(imread(_test_img_path)).permute(2, 0, 1)[None, ...] / 255.

    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        x = x.cuda()

    # To compute BRISQUE score as a measure, use lower case function from the library
    brisque_index: torch.Tensor = piq.brisque(x, data_range=1., reduction='none')
    print(f"BRISQUE index: {brisque_index.item():0.4f}")

    # # To compute CLIP-IQA score as a measure, use PyTorch module from the library
    # # clip_iqa_index: torch.Tensor = piq.CLIPIQA(data_range=1.).to(x.device)(x)
    # clipiqa = piq.CLIPIQA()
    # score = clipiqa(x)
    # print(f"CLIP-IQA: {score:0.4f}")

    # To compute TV as a measure, use lower case function from the library:
    tv_index: torch.Tensor = piq.total_variation(x)
    print(f"TV index: {tv_index.item():0.4f}")
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    rmse = sqrt(mse)
    return psnr, mse, rmse

def spectralFidelity(_lowlight_img_path, _enhanced_img_path):
    # Ref https://pyspectrum.readthedocs.io/en/latest/tutorial_front_image.html
    norm = True
    sides = 'onesided' # ['onesided', 'twosided', 'centerdc', 'default']
    # low light image reader
    data_lowlight = Image.open(_lowlight_img_path)
    data_lowlight = data_lowlight.resize((256, 256), Image.Resampling.LANCZOS)
    data_lowlight = (np.asarray(data_lowlight) / 256.0)
    r_ll_img = data_lowlight[:,:,0]
    g_ll_img = data_lowlight[:, :, 1]
    b_llimg = data_lowlight[:, :, 2]

    # burg estimated method
    r_ll_p = spectrum.pburg(np.mean(r_ll_img, axis=1), order=15, NFFT=4096)

    r_ll_p.plot(label='low light image: red channel', norm=norm, sides=sides, color="red", linestyle='--')
    g_ll_p = spectrum.pburg(np.mean(g_ll_img, axis=1), order=15, NFFT=4096)
    g_ll_p.plot(label='low light image: green channel', norm=norm, sides=sides, color="green", linestyle='--')
    b_ll_p = spectrum.pburg(np.mean(b_llimg, axis=1), order=15, NFFT=4096)
    b_ll_p.plot(label='low light image: blue channel', norm=norm, sides=sides, color="blue", linestyle='--')

    # enhanced image reader
    data_enhanced = Image.open(_enhanced_img_path)
    data_enhanced = data_enhanced.resize((256, 256), Image.Resampling.LANCZOS)
    data_enhanced = (np.asarray(data_enhanced) / 256.0)
    r_enhanced_img = data_enhanced[:, :, 0]
    g_enhanced_img = data_enhanced[:, :, 1]
    b_enhanced_img = data_enhanced[:, :, 2]

    # burg estimated method
    r_enhanced_p = spectrum.pburg(np.mean(r_enhanced_img, axis=1), order=15, NFFT=4096)
    r_enhanced_p.plot(label='enhanced image: red channel', norm=norm, sides=sides, color="red")
    g_enhanced_p = spectrum.pburg(np.mean(g_enhanced_img, axis=1), order=15, NFFT=4096)
    g_enhanced_p.plot(label='enhanced image: green channel', norm=norm, sides=sides, color="green")
    b_enhanced_p = spectrum.pburg(np.mean(b_enhanced_img, axis=1), order=15, NFFT=4096)
    b_enhanced_p.plot(label='enhanced image: blue channel', norm=norm, sides=sides, color="blue")

    legend(loc='upper right', prop={'size': 8}, ncol=1)
    # ylim([-100, 10])

    # processing on SPD data
    from spectrum import tools as stools

    enhanced_power = 10 * stools.log10(b_enhanced_p.psd / max(b_enhanced_p.psd))
    ll_power = 10 * stools.log10(b_ll_p.psd / max(b_ll_p.psd))
    psnr_ll_enhanced_power, mse_ll_enhanced_power,  rmse_ll_enhanced_power = PSNR(enhanced_power,ll_power)
    print('psnr: ', psnr_ll_enhanced_power)
    print('mse: ', mse_ll_enhanced_power)
    print('rmse: ', rmse_ll_enhanced_power)


if __name__ == '__main__':
    # lowlight_img_path = '/mnt/d/SeaportDataset/test_set_3/_mnt_d_Seaport_satellite_images_lng_3.110000_lat_50.560000_sentinel_2_rgb.png'
    # enhanced_img_path = '/home/trongan93/Projects/github/Enhance-Low-Light-Satellite-Image/test_output_seaport/set3_KD_seaport_rgb_dataset/_mnt_d_Seaport_satellite_images_lng_3.110000_lat_50.560000_sentinel_2_rgb.png'

    # lowlight_img_path = '/mnt/d/SeaportDataset/test_set_3/_mnt_d_Seaport_satellite_images_lng_4.240000_lat_51.520000_sentinel_2_rgb.png'
    # enhanced_img_path = '/home/trongan93/Projects/github/Enhance-Low-Light-Satellite-Image/test_output_seaport/set3_KD_seaport_rgb_dataset/_mnt_d_Seaport_satellite_images_lng_4.240000_lat_51.520000_sentinel_2_rgb.png'


    lowlight_img_path = '/mnt/d/SeaportDataset/test_set_4/_mnt_d_Seaport_satellite_images_lng_5.270000_lat_52.590000_landsat_8_rgb.png'
    enhanced_img_path = '/home/trongan93/Projects/github/Enhance-Low-Light-Satellite-Image/test_output_seaport/set4/_mnt_d_Seaport_satellite_images_lng_5.270000_lat_52.590000_landsat_8_rgb.png'

    ground_truth_path = ''
    # validattion_dataset_non_reference_factor(enhanced_img_path)
    spectralFidelity(lowlight_img_path,enhanced_img_path)
    plt.xlabel("Frequency (cycle/pixel)")
    plt.ylabel("Power Spectral Densities (dB/cycle)")
    plt.show()
    # norm = True
    # sides = 'centerdc'

    # # MA method
    # p = spectrum.pma(marple_data, 15, 30, NFFT=4096)
    # p.plot(label='MA (15, 30)', norm=norm, sides=sides)
    # plt.show()


