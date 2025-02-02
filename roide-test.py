import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # server config 1
import torch
import torch.nn as nn
import torchvision
import torch.optim
import fusenet
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import wandb
import warnings

warnings.filterwarnings('ignore')

# Start a new wandb run to track this script
wandb.init(
    project="HDRNet-Test",
    mode="online",
    config={
        "architecture": "HDRNet-testing",
    }
)


def hdrgeneration(image_path, dest_path, src_path):
    scale_factor = 1
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    HDRNet = model.HDRNet(scale_factor).cuda()
    HDRNet.load_state_dict(torch.load(
        "snapshots_weight_trongan93_final/snapshots_weight_trongan93_HDRNet_medium_exposureHDR_net_Epoch250.pth"))

    Fusion_net = fusenet.Fusion_module(channels=3, r=2).cuda().eval()
    Fusion_net.load_state_dict(torch.load(
        "snapshots_weight_trongan93_final/snapshots_weight_trongan93_HDRNet_medium_exposureFusion_net_Epoch250.pth"))

    start = time.time()
    x1, x2, x3, x4, x5, x6, x7, x8, x16, x_r = HDRNet(data_lowlight)
    fusion_net_result = Fusion_net(x1, x8)

    images_fusion_net_result = wandb.Image(fusion_net_result)
    wandb.log({"images_fusion by FusionNet": images_fusion_net_result})

    end_time = time.time() - start

    print(end_time)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save fusion result
    fusion_result_path = os.path.join(dest_path, 'fused_image', base_name + '_fused.png')
    os.makedirs(os.path.dirname(fusion_result_path), exist_ok=True)
    torchvision.utils.save_image(fusion_net_result, fusion_result_path)

    # Save multi-exposure images
    multi_exposure_path = os.path.join(dest_path, 'multi_exposure_images')
    os.makedirs(multi_exposure_path, exist_ok=True)

    exposures = [x1, x2, x3, x4, x5, x6, x7, x8, x16]
    for i, exposure in enumerate(exposures, start=1):
        exposure_result_path = os.path.join(multi_exposure_path, f'{base_name}_x{i}.png')
        torchvision.utils.save_image(exposure, exposure_result_path)

    return end_time


if __name__ == '__main__':
    with torch.no_grad():
        filePath = "/mnt/d/SeaportDataset/test_set_6"
        target_path = 'test_output_seaport/set_6_HDR_final/'
        os.makedirs(target_path, exist_ok=True)
        sum_time = 0

        for image in os.listdir(filePath):
            image_path = os.path.join(filePath, image)
            print(image_path)
            sum_time += hdrgeneration(image_path, target_path, filePath)

        print(f"Total processing time: {sum_time} seconds")
