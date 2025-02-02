# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # remember to set 1 when runing on server 38

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim

import sys
import argparse
import time
import dataloader
import model
import fusenet
import Myloss
import fusion_loss

import numpy as np
from torchvision import transforms


# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="RoIDE-Net",
    mode="online", # mode: online, disabled
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "RoIDE-Net",
        "dataset": "SDR-satellite",
        "epochs": 5,
    }
)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    # Define the training model
    scale_factor = config.scale_factor
    HDR_net = model.HDRNet(scale_factor).cuda()  # HDR Net
    Fusion_net = fusenet.Fusion_module(channels=3,r=2).cuda()

    # Define the Data loader
    if config.load_pretrain == True:
        HDR_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    print(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    # Define the loss function
    L_color = Myloss.L_color(8)
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16)
    L_TV = Myloss.L_TV()
    L_kl = Myloss.L_KDL("mean")

    # Define data batch and amount
    dataAmount = train_dataset.__len__()
    batchSize = config.train_batch_size
    
    # Define optimizer
    optimizer_hdr_net = torch.optim.Adam(HDR_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer_fusion_net = torch.optim.Adam(Fusion_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Define train model
    HDR_net.train()
    Fusion_net.train()

    # torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

    for epoch in range(config.num_epochs):
        completeSum = 0
        for iteration, img_lowlight in enumerate(train_loader):
            # loading model to GPU and define the config of exposure
            img_lowlight = img_lowlight.cuda()
            b, _, _, _ = img_lowlight.size()
            E_min = 0.4 # test case 1: 0.8
            E_max = 0.6 # test case 1: 0.2

            # Zero the gradients for both optimizers at the beginning of each iteration.
            optimizer_hdr_net.zero_grad()
            optimizer_fusion_net.zero_grad()

            # TRAIN HDR NET MODEL
            x1, x2, x3, x4, x5, x6, x7, x8, x16, x_r = HDR_net(img_lowlight)

            x_min_integrated = x1
            x_medium_integrated = x4
            x_max_integrated = x8

            loss_tv_noweight = L_TV(x_r)
            Loss_TV = 7000 * loss_tv_noweight
            wandb.log({"loss_tv_noweight": loss_tv_noweight, 'epoch': epoch})

            loss_spa_noweight = torch.mean(L_spa(x_max_integrated, img_lowlight))
            loss_spa = 100 * loss_spa_noweight
            wandb.log({"loss_spa_noweight": loss_spa_noweight, 'epoch': epoch})

            # loss_exp_noweight = (torch.mean(L_exp(x_max_integrated,E_max)) + torch.mean(L_exp(x_min_integrated, E_min)))/2
            loss_exp_noweight = torch.mean(L_exp(x_max_integrated, E_max))

            loss_exp = 50*loss_exp_noweight
            wandb.log({"loss_exp": loss_exp, 'epoch': epoch})

            loss_col = torch.mean(L_color(x_max_integrated))

            loss_kl_rb, loss_kl_rg, loss_kl_gb, loss_kl = L_kl(img_lowlight, x_max_integrated)
            wandb.log({"loss_kl_rb": loss_kl_rb, 'epoch': epoch})
            wandb.log({"loss_kl_rg": loss_kl_rg, 'epoch': epoch})
            wandb.log({"loss_kl_gb": loss_kl_gb, 'epoch': epoch})
            wandb.log({"loss_kl_noweight": loss_kl, 'epoch': epoch})
            loss_kl = 5 * loss_kl

            loss_hdr_net = Loss_TV + loss_spa + loss_exp + loss_col + loss_kl
            # loss_hdr_net = Loss_TV + loss_spa + loss_exp + loss_col
            wandb.log({"loss_hdr_net": loss_hdr_net, 'epoch': epoch})

            loss_hdr_net.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(HDR_net.parameters(),config.grad_clip_norm) # prevent gradient explode
            optimizer_hdr_net.step()

            # TRAIN FUSION NET MODEL
            x_integrated_min_deatach = x_min_integrated.detach()
            x_integrated_medium_deatach = x_medium_integrated.detach()
            x_integrated_max_deatach = x_max_integrated.detach()

            fused_result = Fusion_net(x_integrated_min_deatach, x_integrated_max_deatach)
            # loss_col_2 = torch.mean(L_color(fusion_net_result))
            loss_total, loss_intensity, loss_grad, loss_ssim= fusion_loss.Fusion_loss(x_integrated_min_deatach,x_integrated_max_deatach,fused_result)

            wandb.log({"loss_fusion_net_intensity": loss_intensity, 'epoch': epoch})
            wandb.log({"loss_fusion_net_grad": loss_grad, 'epoch': epoch})
            wandb.log({"loss_fusion_net_ssim": loss_ssim, 'epoch': epoch})
            wandb.log({"loss_fusion_net_total": loss_total, 'epoch': epoch})

            loss_fusion_net = loss_total




            wandb.log({"loss_fusion_net": loss_fusion_net, 'epoch': epoch})

            loss_fusion_net.backward()
            torch.nn.utils.clip_grad_norm(Fusion_net.parameters(), config.grad_clip_norm)  # prevent gradient explode
            optimizer_fusion_net.step()




            
            completeSum += b 
            pComplete = int(completeSum / dataAmount * 100) // 2
            pUndo = int((1 - (completeSum / dataAmount)) * 100) // 2 
            
            if ((iteration+1) % config.display_iter) == 0:
                print("Epoch : "+ str(epoch + 1) +  "  [" + "-"*pComplete + ">" + " "*pUndo + "] - loss_hdr_net: " + str(loss_hdr_net.item()) + " - loss_fusion_net: " + str(loss_fusion_net.item()), "\r", end='')


        # torch.autograd.set_detect_anomaly(False)  # Disable anomaly detection after training

        if ((epoch + 1) % 10) == 0:
            print("Saving Model " + config.snapshots_folder + "RoIDE-Net_HDR_net_Epoch" + str((epoch + 1)) + '.pth' + " with Epoch " + str((epoch + 1)))
            torch.save(HDR_net.state_dict(), config.snapshots_folder + "RoIDE-Net_HDR_net_Epoch" + str((epoch + 1)) + '.pth')

            print("Saving Model " + config.snapshots_folder + "RoIDE-Net_Fusion_net_Epoch" + str(
                (epoch + 1)) + '.pth' + " with Epoch " + str((epoch + 1)))
            torch.save(Fusion_net.state_dict(), config.snapshots_folder + "RoIDE-Net_Fusion_net_Epoch" + str((epoch + 1)) + '.pth')

            print("Model Saved\n\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--ldr_images_path', type=str, default="/satellite_ldr_imgs/")
    # local trongan lab pc: /mnt/d/ZeroDCEDataSet/ZeroDCE/satellite_ldr_imgs/
    # server 38 : /mnt/d/satellite_ldr_imgs/
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots_weight_trongan93_RoIDE-Net_8_inter/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    # parser.add_argument('--pretrain_dir', type=str, default= "./snapshots_weight_trongan93/Epoch99.pth") #Need change the model path
    # parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    config = parser.parse_args()

    # gpu_devices = ','.join([str(id) for id in config.gpu_devices])
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    # print arguments
    for arg in vars(config):
        print(arg, getattr(config, arg))

    train(config)
