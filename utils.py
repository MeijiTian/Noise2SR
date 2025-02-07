import torch
import numpy as np
import SimpleITK as sitk
import math
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import os


class Sub_sampler():
    def __init__(self,scale_factor,device):
        self.select_idx_ = torch.stack(
            [torch.randperm(scale_factor * scale_factor) for _ in range(1024*1024)], dim = 0
        ).to(device)
        self.scale = scale_factor

    def sample_img(self,img, sample_out_channels):

        img_unshuffle = F.pixel_unshuffle(img, self.scale)
        n, c, h, w = img_unshuffle.shape
        img_unshuffle = img_unshuffle.permute(0, 2, 3, 1).reshape(-1, self.scale * self.scale)
        mask = torch.ones_like(img_unshuffle,device= img.device)
        shuffled_indices = torch.randperm(n * h * w).to(img.device)
        select_idx = torch.index_select(self.select_idx_[:n* h * w], dim=0, index=shuffled_indices)

        subsampled_img = torch.gather(img_unshuffle, dim=1, index=select_idx[:, :sample_out_channels])
        mask.scatter_(1, select_idx[:, :sample_out_channels], 0)
        subsampled_img = subsampled_img.reshape(n, h, w, sample_out_channels).permute(0, 3, 1, 2)
        mask = mask.reshape(n, h, w, c).permute(0, 3, 1, 2)
        mask = F.pixel_shuffle(mask, upscale_factor=self.scale)

        return subsampled_img, mask


def SSIM(img, ground_truth):
    N = img.shape[0]
    res = []
    for k in range(N):
        data_range = np.max(ground_truth[k]) - np.min(ground_truth[k])
        res.append(structural_similarity(img[k], ground_truth[k], data_range=data_range))
    res = np.array(res)
    return res

def PSNR(img, ground_truth):
    N = img.shape[0]
    res = []
    for k in range(N):
        mse = np.mean((img[k] - ground_truth[k]) ** 2)
        if mse == 0.:
            return float('inf')
        data_range = np.max(ground_truth[k]) - np.min(ground_truth[k])
        res.append(20 * np.log10(data_range) - 10 * np.log10(mse))
    return np.array(res)

def makedir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('make a new folder_'+ path)
    else:
        print(path + ' already exsits')
















