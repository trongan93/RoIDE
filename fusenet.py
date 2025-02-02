# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw

import torch
import torch.nn as nn
import wandb


class Fusion_module(nn.Module):

    def __init__(self, channels=3, r=2):
        super(Fusion_module, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * channels),
            nn.Sigmoid(),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input  ## First perform one-step self-correction on the features
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim=1)
        agg_input = self.channel_agg(recal_input)  ## Perform feature compression because only the weight of one feature is calculated
        local_w = self.local_att(agg_input)  ## Partial attention is spatial attention
        # local_w_one = torch.ones_like(local_w) # for testing no spatial attention
        global_w = self.global_att(agg_input)  ## Global attention is channel attention
        # global_w_one = torch.ones_like(global_w) # for testing no channel attention
        w = self.sigmoid(local_w * global_w)  ## Calculate the weight of feature x1
        # # test only for w without local_w
        # w = self.sigmoid(local_w_one * global_w) #remove after test
        # test only for w without global_w
        # w = self.sigmoid(local_w * global_w_one)  # remove after test

        xo = w * x1 + (1 - w) * x2  ## fusion results ## Feature aggregation
        # # Log fused images
        # images_fused_image = wandb.Image(
        #     xo,
        # )
        # wandb.log({"images_fused_image by Fusion module": images_fused_image})
        return xo


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
# Fusion_Module reference from: https://github.com/Linfeng-Tang/PSFusion/blob/main/PSF.py#L348


class UNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x_intergrated, x_enhanced):
        conv1 = self.dconv_down1(x_intergrated)
        x_intergrated = self.maxpool(conv1)

        conv2 = self.dconv_down2(x_intergrated)
        x_intergrated = self.maxpool(conv2)

        conv3 = self.dconv_down3(x_intergrated)
        x_intergrated = self.maxpool(conv3)

        x_intergrated = self.dconv_down4(x_intergrated)

        x_intergrated = nn.functional.interpolate(x_intergrated, scale_factor=2, mode='bilinear', align_corners=True)
        x_intergrated = torch.cat([x_intergrated, conv3], dim=1)

        x_intergrated = self.dconv_up3(x_intergrated)
        x_intergrated = nn.functional.interpolate(x_intergrated, scale_factor=2, mode='bilinear', align_corners=True)
        x_intergrated = torch.cat([x_intergrated, conv2], dim=1)

        x_intergrated = self.dconv_up2(x_intergrated)
        x_intergrated = nn.functional.interpolate(x_intergrated, scale_factor=2, mode='bilinear', align_corners=True)
        x_intergrated = torch.cat([x_intergrated, conv1], dim=1)

        x_intergrated = self.dconv_up1(x_intergrated)

        out = self.conv_last(x_intergrated)

        # # Log unet output images
        # images_unet_output = wandb.Image(
        #     out,
        # )
        # wandb.log({"images_unet_output by UNet": images_unet_output})

        # out = out*x_enhanced
        enhanced_3_channels_by_unet = out + x_enhanced * (
                    torch.pow(out, 2) - out)  # follow the idea of ZeroDCE integration

        # # Log enhanced images
        # enhanced_3_channels_by_unet = wandb.Image(
        #     enhanced_3_channels_by_unet,
        # )
        # wandb.log({"enhanced_3_channels by UNet": enhanced_3_channels_by_unet})

        return out
