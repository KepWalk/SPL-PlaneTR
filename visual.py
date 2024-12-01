import scipy.io as sio
import os
import cv2
import time
import random
import pickle
import numpy as np
from PIL import Image
import yaml
import sys
import cv2

import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as tf
from torchvision.transforms import transforms

from models.Mp3dDataset import Mp3dDataset, ToTensor
from models.S2D3DSDataset import S2d3dsDataset, ToTensor
from utils.utils import Set_Config, Set_Logger, Set_Ckpt_Code_Debug_Dir
from torch.utils.data import DataLoader

from models.planeTR_HRNet import PlaneTR_HRNet as PlaneTR
from models.ScanNetV1_PlaneDataset import scannetv1_PlaneDataset

from utils.misc import AverageMeter, get_optimizer, get_coordinate_map

from utils.metric import eval_plane_recall_depth, eval_plane_recall_normal, evaluateMasks

from utils.disp import plot_depth_recall_curve, plot_normal_recall_curve, visualizationBatch

from utils.visual_tools import map_masks_to_colors


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--mode', default='eval', type=str,
                    help='train / eval')
parser.add_argument('--backbone', default='hrnet', type=str,
                    help='only support hrnet now')
parser.add_argument('--cfg_path', default='configs/config_planeTR_eval_other.yaml', type=str,
                    help='full path of the config file')
args = parser.parse_args()

NUM_GPUS = 1

torch.backends.cudnn.benchmark = True


class AddGaussianNoise(object):
    def __init__(self, std=50):
        self.std = std

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img = np.array(img)
        noise = np.random.randn(*img.shape) * self.std
        noise_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        noise_img = Image.fromarray(noise_img)

        return noise_img

def load_dataset(cfg, args):
    transforms = tf.Compose([
        AddGaussianNoise(std=30),
        tf.ToTensor(),
    ])

    assert NUM_GPUS > 0

    if args.mode == 'train':
        subset = 'train'
    else:
        subset = 'val'

    if NUM_GPUS > 1:
        is_shuffle = False
    else:
        is_shuffle = subset == 'train'

    if cfg.dataset.name == 'scannet':
        dataset = scannetv1_PlaneDataset
    else:
        # todo: add support for nyu dataset
        print("undefined dataset!")
        exit()

    predict_center = cfg.model.if_predict_center

    if NUM_GPUS > 1:
        assert args.mode == 'train'
        dataset_plane = dataset(subset=subset, transform=transforms, root_dir=cfg.dataset.root_dir, predict_center=predict_center)
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset_plane)
        loaders = torch.utils.data.DataLoader(dataset_plane, batch_size=cfg.dataset.batch_size, shuffle=is_shuffle,
                                                   num_workers=cfg.dataset.num_workers, pin_memory=True, sampler=data_sampler)
    else:
        loaders = data.DataLoader(
            dataset(subset=subset, transform=transforms, root_dir=cfg.dataset.root_dir, predict_center=predict_center),
            batch_size=cfg.dataset.batch_size, shuffle=is_shuffle, num_workers=cfg.dataset.num_workers, pin_memory=True
        )
        data_sampler = None

    return loaders, data_sampler


def eval(cfg, logger):

    # ScanNet
    data_loader, _ = load_dataset(cfg, args)

    # val_data = Mp3dDataset(subset="val",
    #                         transform=tf.Compose([
    #                             ToTensor()]),
    #                         datafolder='../mp3d-plane'
    #                         )
    # data_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8,
    #                         pin_memory=False)

    # val_data = S2d3dsDataset(subset="val",
    #                          transform=transforms.Compose([
    #                              ToTensor()]),
    #                          datafolder="../S2D3DS-plane"
    #                          )
    # data_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8,
    #                         pin_memory=False)

    dataset_name = 'Matterport3D'

    with torch.no_grad():
        for iter, sample in enumerate(data_loader):

            # if int(iter) == 501:
            #     exit(0)

            print("processing image %d"%(iter))
            image = sample['image']
            gt_seg = sample['instance']
            num_planes = sample['num_planes']
            depth = sample['depth']

            # gt_rgb
            image = image.squeeze(0).numpy()
            image *= 255
            image = image.astype(np.uint8).transpose(1, 2, 0)
            image = np.clip(image, 0, 255)
            # gt
            gt_seg = gt_seg.squeeze(0)
            gt_seg = gt_seg[:num_planes]
            gt_seg = gt_seg.numpy().astype(np.uint8)
            gt_seg = map_masks_to_colors(gt_seg)
            # depth
            depth = depth.squeeze(0).squeeze(0).numpy()
            depth = (depth * 255 / (depth.max())).astype(np.uint8)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

            Image.fromarray(image).save("../Noisy/RGB/RGB%d.png" % (iter))
            Image.fromarray(depth).save("../Noisy/Depth/Depth%d.png" % (iter))
            Image.fromarray(gt_seg).save("../Noisy/GT/GT%d.png" % (iter))


if __name__ == '__main__':
    cfg = Set_Config(args)

    # ------------------------------------------- set distribution
    if args.mode == 'train' and NUM_GPUS > 1:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        print('initialize DDP successfully... ')

    # ------------------------------------------ set logger
    logger = Set_Logger(args, cfg)

    # ------------------------------------------ main
    if args.mode == 'eval':
        eval(cfg, logger)
    else:
        exit()
