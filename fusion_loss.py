import torch
import torch.nn.functional as F
import torch.nn as nn
from math import exp
import torchvision.transforms as transforms

# Ref: https://github.com/Linfeng-Tang/PSFusion/blob/main/Fusion_losses.py

def Fusion_loss(rgb_1, rgb_2, fu, weights=[1, 10, 10]):

    # Original Sobel function
    rgb_1_Y, rgb_1_Cb, rgb_1_Cr = RGB2YCrCb(rgb_1)
    rgb_2_Y, rgb_2_Cb, rgb_2_Cr = RGB2YCrCb(rgb_2)
    fu_Y, fu_Cb, fu_Cr = RGB2YCrCb(fu)

    transform = transforms.Grayscale()


    sobelconv = Sobelxy()
    # vi_grad_x, vi_grad_y = sobelconv(rgb_2_Y)
    # ir_grad_x, ir_grad_y = sobelconv(rgb_1_Y)
    # fu_grad_x, fu_grad_y = sobelconv(fu_Y)

    vi_grad_x, vi_grad_y = sobelconv(transform(rgb_1))
    ir_grad_x, ir_grad_y = sobelconv(transform(rgb_2))
    fu_grad_x, fu_grad_y = sobelconv(transform(fu))

    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)

    # # Sobel Function v2
    # sobelconv = Sobel()
    # transform = transforms.Grayscale()
    # grad_rgb1 = sobelconv(transform(rgb_1))
    # grad_rgb2 = sobelconv(transform(rgb_2))
    # grad_fused = sobelconv(transform(fu))
    #
    # grad_joint = torch.max(grad_rgb1,grad_rgb2)
    # loss_grad = F.l1_loss(grad_joint,grad_fused)


    loss_ssim = corr_loss(rgb_1, rgb_2, fu)

    loss_intensity = final_mse1(rgb_1, rgb_2, fu) + 0 * F.l1_loss(fu, torch.max(rgb_1, rgb_2))
    loss_total = weights[0] * loss_ssim + weights[1] * loss_grad + weights[2] * loss_intensity
    return loss_total, loss_intensity, loss_grad, loss_ssim

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]

        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0).cuda()
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0).cuda()
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)

class Sobel(nn.Module):
    # Ref: https://github.com/chaddy1004/sobel-operator-pytorch/tree/master
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

def corr_loss(image_ir, img_vis, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss

class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)

def final_mse1(img_ir, img_vis, img_fuse):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    # std_ir = std(img_ir)
    # std_vi = std(img_vis)
    # # std_ir = sum(img_ir)
    # # std_vi = sum(img_vis)
    #
    # zero = torch.zeros_like(std_ir)
    # one = torch.ones_like(std_vi)
    #
    # m = torch.mean(img_ir)
    # map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    # # map2 = torch.where((std_ir - std_vi) >= 0, zero, one)
    # map_ir=torch.where(map1+mask>0, one, zero)
    # map_vi= 1 - map_ir
    #
    # res = map_ir * mse_ir + map_vi * mse_vi
    # res = res * w_vi
    res = mse_ir + mse_vi
    return res.mean()

# Variance calculation
def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def RGB2YCrCb(rgb_image):
    """
    Convert RGB format to YCrCb format
    Used in color space conversion of intermediate results, because the default size of rgb_image is [B, C, H, W]
    :param rgb_image: image data in RGB format
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr