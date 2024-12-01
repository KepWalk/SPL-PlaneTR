import os
import time
import random
import numpy as np


import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as tf
from torchvision.transforms import transforms

from models.Mp3dDataset import Mp3dDataset, ToTensor
from models.S2D3DSDataset import S2d3dsDataset, ToTensor
from utils.utils import Set_Config, Set_Logger
from torch.utils.data import DataLoader

from models.planeTR_HRNet import PlaneTR_HRNet as PlaneTR
from models.ScanNetV1_PlaneDataset import scannetv1_PlaneDataset

from utils.misc import get_coordinate_map

from utils.metric import evaluateMasks

import matplotlib

matplotlib.use('Agg')
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


def load_dataset(cfg, args):
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        dataset_plane = dataset(subset=subset, transform=transforms, root_dir=cfg.dataset.root_dir,
                                predict_center=predict_center)
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset_plane)
        loaders = torch.utils.data.DataLoader(dataset_plane, batch_size=cfg.dataset.batch_size, shuffle=is_shuffle,
                                              num_workers=cfg.dataset.num_workers, pin_memory=True,
                                              sampler=data_sampler)
    else:
        loaders = data.DataLoader(
            dataset(subset=subset, transform=transforms, root_dir=cfg.dataset.root_dir, predict_center=predict_center),
            batch_size=cfg.dataset.batch_size, shuffle=is_shuffle, num_workers=cfg.dataset.num_workers, pin_memory=True
        )
        data_sampler = None

    return loaders, data_sampler


def eval(cfg, logger):
    logger.info('*' * 40)
    localtime = time.asctime(time.localtime(time.time()))
    logger.info(localtime)
    logger.info('start evaluating......')
    logger.info('*' * 40)

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build network
    network = PlaneTR(cfg)

    # load nets into gpu
    network = network.to(device)

    # create debug dir
    if cfg.if_debug:
        debug_dir = 'debug_eval/'
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

    # load pretrained weights if existed
    if not (cfg.resume_dir == 'None'):
        loc = 'cuda:{}'.format(args.local_rank)
        # model_dict = torch.load(cfg.resume_dir, map_location=loc)
        model_dict = torch.load(cfg.resume_dir)
        model_dict_ = {}
        if NUM_GPUS > 1:
            for k, v in model_dict.items():
                k_ = 'module.' + k
                model_dict_[k_] = v
            network.load_state_dict(model_dict_)
        else:
            network.load_state_dict(model_dict)

    # build data loader
    if cfg.dataset.name == "mp3d" or cfg.dataset.name == "synthetic":
        val_data = Mp3dDataset(subset="val",
                               transform=tf.Compose([
                                   ToTensor()]),
                               datafolder=cfg.dataset.root_dir
                               )
        data_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8,
                                 pin_memory=False)
    elif cfg.dataset.name == "s2d3ds":
        val_data = S2d3dsDataset(subset="val",
                                 transform=transforms.Compose([
                                     ToTensor()]),
                                 datafolder=cfg.dataset.root_dir
                                 )
        data_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8,
                                 pin_memory=False)
    else:
        raise "Dataset not found"

    # set network state
    use_lines = cfg.model.use_lines
    network.eval()

    k_inv_dot_xy1 = get_coordinate_map(device)
    num_queries = cfg.model.num_queries
    embedding_dist_threshold = cfg.model.embedding_dist_threshold

    plane_Seg_Metric = np.zeros((3))

    RI = 0.
    SC = 0.
    VoI = 0.
    cnt = 0
    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            # print("processing image %d"%(iter))
            image = sample['image'].to(device)
            # instance = sample['instance'].to(device)
            gt_seg = sample['segmentation'].numpy()
            gt_depth = sample['depth'].to(device)

            if use_lines:
                num_lines = sample['num_lines']
                lines = sample['lines'].to(device)  # 200, 4
            else:
                num_lines = None
                lines = None

            bs, _, h, w = image.shape
            assert bs == 1, "batch size should be 1 when testing!"
            # assert h == 192 and w == 256

            sp_t_s = time.time()

            # forward pass
            outputs = network(image, lines, num_lines)

            # decompose outputs
            pred_logits = outputs['pred_logits'][0]  # num_queries, 3
            pred_param = outputs['pred_param'][0]  # num_queries, 3
            pred_plane_embedding = outputs['pred_plane_embedding'][0]  # num_queries, 2
            pred_pixel_embedding = outputs['pixel_embedding'][0]  # 2, h, w

            assert 'pixel_depth' in outputs.keys()
            pred_pixel_depth = outputs['pixel_depth'][0, 0]  # h, w

            # remove non-plane instance
            pred_prob = F.softmax(pred_logits, dim=-1)  # num_queries, 3
            score, labels = pred_prob.max(dim=-1)
            labels[labels != 1] = 0
            label_mask = labels > 0
            if sum(label_mask) == 0:
                _, max_pro_idx = pred_prob[:, 1].max(dim=0)
                label_mask[max_pro_idx] = 1
            valid_param = pred_param[label_mask, :]  # valid_plane_num, 3
            valid_plane_embedding = pred_plane_embedding[label_mask, :]  # valid_plane_num, c_embedding
            valid_plane_num = valid_plane_embedding.shape[0]
            valid_plane_prob = score[label_mask]  # valid_plane_num
            assert valid_plane_num <= num_queries

            # calculate dist map
            c_embedding = pred_plane_embedding.shape[-1]
            flat_pixel_embedding = pred_pixel_embedding.view(c_embedding, -1).t()  # hw, c_embedding
            dist_map_pixel2planes = torch.cdist(flat_pixel_embedding, valid_plane_embedding, p=2)  # hw, valid_plane_num
            dist_pixel2onePlane, planeIdx_pixel2onePlane = dist_map_pixel2planes.min(-1)  # [hw,]
            dist_pixel2onePlane = dist_pixel2onePlane.view(h, w)  # h, w
            planeIdx_pixel2onePlane = planeIdx_pixel2onePlane.view(h, w)  # h, w
            mask_pixelOnPlane = dist_pixel2onePlane <= embedding_dist_threshold  # h, w

            # get plane segmentation
            gt_seg = gt_seg.reshape(h, w)  # h, w
            predict_segmentation = planeIdx_pixel2onePlane.cpu().numpy()  # h, w
            if int(mask_pixelOnPlane.sum()) < (h * w):  # set plane idx of non-plane pixels as num_queries + 1
                predict_segmentation[mask_pixelOnPlane.cpu().numpy() == 0] = num_queries + 1
            predict_segmentation = predict_segmentation.reshape(h,
                                                                w)  # h, w (0~num_queries-1:plane idx, num_queries+1:non-plane)

            # 3 evaluation: plane segmentation
            plane_Seg_Statistics = evaluateMasks(predict_segmentation, gt_seg, device,
                                                 pred_non_plane_idx=num_queries + 1)
            plane_Seg_Metric += np.array(plane_Seg_Statistics)

            # ------------------------------------ log info

            print(
                f"RI(+):{plane_Seg_Statistics[0]:.3f} | VI(-):{plane_Seg_Statistics[1]:.3f} | SC(+):{plane_Seg_Statistics[2]:.3f}")
            RI += plane_Seg_Statistics[0]
            VoI += plane_Seg_Statistics[1]
            SC += plane_Seg_Statistics[2]
            cnt += 1

        RI /= cnt
        VoI /= cnt
        SC /= cnt
        print(f"mRI(+):{RI:.3f} | mVoI(-):{VoI:.3f} | mSC(+):{SC:.3f}")


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
