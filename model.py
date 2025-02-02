import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import wandb


class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class mobileEnhanceNet(nn.Module):

    def __init__(self,scale_factor):
        super(mobileEnhanceNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

    #   zerodce DWC + p-shared
        self.e_conv1 = CSDN_Tem(3,number_f) 
        self.e_conv2 = CSDN_Tem(number_f,number_f) 
        self.e_conv3 = CSDN_Tem(number_f,number_f) 
        self.e_conv4 = CSDN_Tem(number_f,number_f) 
        self.e_conv5 = CSDN_Tem(number_f*2,number_f) 
        self.e_conv6 = CSDN_Tem(number_f*2,number_f) 
        self.e_conv7 = CSDN_Tem(number_f*2,3) 

    def enhance(self, x,x_r):

        x = x + x_r*(torch.pow(x,2)-x)
        x = x + x_r*(torch.pow(x,2)-x)
        x = x + x_r*(torch.pow(x,2)-x)
        enhance_image_1 = x + x_r*(torch.pow(x,2)-x)		
        x = enhance_image_1 + x_r*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + x_r*(torch.pow(x,2)-x)	
        x = x + x_r*(torch.pow(x,2)-x)
        enhance_image = x + x_r*(torch.pow(x,2)-x)	

        return enhance_image
        
    def forward(self, x):
        if self.scale_factor==1:
            x_down = x
        else:
            x_down = F.interpolate(x,scale_factor=1/self.scale_factor, mode='bilinear')

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        if self.scale_factor==1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
        enhance_image = self.enhance(x,x_r)
        return enhance_image,x_r


class EnhanceNet(nn.Module):

    def __init__(self, scale_factor):
        super(EnhanceNet, self).__init__()

        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,3,3,1,1,bias=True) 

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x,scale_factor=1/self.scale_factor, mode='bilinear')    
    
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        
        if self.scale_factor==1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
        
        x = x + x_r*(torch.pow(x,2)-x)
        x = x + x_r*(torch.pow(x,2)-x)
        x = x + x_r*(torch.pow(x,2)-x)
        enhance_image_1 = x + x_r*(torch.pow(x,2)-x)		
        x = enhance_image_1 + x_r*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + x_r*(torch.pow(x,2)-x)	
        x = x + x_r*(torch.pow(x,2)-x)
        enhance_image = x + x_r*(torch.pow(x,2)-x)	
        
        return enhance_image,x_r


class EnhanceNet_1st_minimal(nn.Module):

    def __init__(self, scale_factor):
        super(EnhanceNet_1st_minimal, self).__init__()

        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        # self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        # self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear')

        x1 = self.relu(self.e_conv1(x_down))
        # x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x1))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x5], 1)))

        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)

        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)

        return enhance_image, x_r


class simpleEnhanceNet(nn.Module): # Use this one

    def __init__(self, scale_factor):
        super(simpleEnhanceNet, self).__init__()

        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners= False)
        #self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        #self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,3,3,1,1,bias=True) 

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x,scale_factor=1/self.scale_factor, mode='nearest')    
            #x_down = F.interpolate(x,scale_factor=1/self.scale_factor)
            
    
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x_r = F.sigmoid(self.e_conv3(x2+x1)) * 2 - 1
        #x_r = F.tanh(self.e_conv3(x2+x1))
        
        if self.scale_factor==1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
        
        x = x + x_r*(torch.pow(x,2)-x)
        x = x + x_r*(torch.pow(x,2)-x)
        x = x + x_r*(torch.pow(x,2)-x)
        enhance_image_1 = x + x_r*(torch.pow(x,2)-x)		
        x = enhance_image_1 + x_r*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + x_r*(torch.pow(x,2)-x)	
        x = x + x_r*(torch.pow(x,2)-x)
        enhance_image = x + x_r*(torch.pow(x,2)-x)	
        
        return enhance_image,x_r

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
class simpleEnhanceNet_CSDN(nn.Module):
    def __init__(self,scale_factor) -> None:
        super(simpleEnhanceNet_CSDN, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, 3)

    def enhance(self, x, x_r):

        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)

        # Reduce the curve intergration
        # x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        # x = x + x_r * (torch.pow(x, 2) - x)
        # x = x + x_r * (torch.pow(x, 2) - x)
        # enhance_image = x + x_r * (torch.pow(x, 2) - x)

        # return enhance_image
        return enhance_image_1

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='nearest')

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x_r = F.sigmoid(self.e_conv3(x2 + x1)) * 2 - 1

        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)

        enhance_image = self.enhance(x,x_r)

        return enhance_image, x_r

class HDRNet(nn.Module):

    def __init__(self, scale_factor):
        super(HDRNet, self).__init__()

        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear')

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)

        # # Log the last layer of HDRNet
        # last_layer_of_hdrnet_images = wandb.Image(
        #     x_r,
        # )
        # wandb.log({"last_layer_of_hdrnet_images": last_layer_of_hdrnet_images})

        x = x + x_r * (torch.pow(x, 2) - x)
        x1 = x
        # x_integrated = x
        x = x + x_r * (torch.pow(x, 2) - x)
        x2 = x
        # x_integrated = torch.cat((x_integrated,x), dim=1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x3 = x
        # x_integrated = torch.cat((x_integrated,x), dim=1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x4 = x
        # x_integrated = torch.cat((x_integrated,x), dim=1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x5 = x
        # x_integrated = torch.cat((x_integrated,x), dim=1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x6 = x
        # x_integrated = torch.cat((x_integrated,x), dim=1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x7 = x
        # x_integrated = torch.cat((x_integrated,x), dim=1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x8 = x
        # x_integrated = torch.cat((x_integrated,x), dim=1)

        # # Log integrated images of HDRNet
        # images_x1 = wandb.Image(
        #     x1,
        # )
        # wandb.log({"images_x1 by HDR": images_x1})

        # images_x2 = wandb.Image(
        #     x2,
        # )
        # wandb.log({"images_x2 by HDR": images_x2})
        #
        # images_x3 = wandb.Image(
        #     x3,
        # )
        # wandb.log({"images_x3 by HDR": images_x3})
        #
        # images_x4 = wandb.Image(
        #     x4,
        # )
        # wandb.log({"images_x4 by HDR": images_x4})
        #
        # images_x5 = wandb.Image(
        #     x5,
        # )
        # wandb.log({"images_x5 by HDR": images_x5})
        #
        # images_x6 = wandb.Image(
        #     x6,
        # )
        # wandb.log({"images_x6 by HDR": images_x6})
        #
        # images_x7 = wandb.Image(
        #     x7,
        # )
        # wandb.log({"images_x7 by HDR": images_x7})
        #
        # images_x8 = wandb.Image(
        #     x8,
        # )
        # wandb.log({"images_x8 by HDR": images_x8})

        # images_x_integrated = wandb.Image(
        #     x_integrated,
        # )
        # wandb.log({"x_integrated by HDR": images_x_integrated})
        x = x + x_r * (torch.pow(x, 2) - x) #x9
        x = x + x_r * (torch.pow(x, 2) - x) #x10
        x = x + x_r * (torch.pow(x, 2) - x) #x11
        x = x + x_r * (torch.pow(x, 2) - x) #x12
        x = x + x_r * (torch.pow(x, 2) - x) #x13
        x = x + x_r * (torch.pow(x, 2) - x) #x14
        x = x + x_r * (torch.pow(x, 2) - x) #x15
        x = x + x_r * (torch.pow(x, 2) - x) #x16
        x16 = x

        # images_x16 = wandb.Image(
        #     x16,
        # )
        # wandb.log({"images_x16 by HDR": images_x16})

        return x1, x2, x3, x4, x5, x6, x7, x8, x16, x_r