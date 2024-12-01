import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import torch.nn.functional as F

image_w = 256
image_h = 192


class S2d3dsDataset(Dataset):
    def __init__(self, datafolder, subset='train', transform=None):
        self.datafolder = datafolder
        self.subset = subset
        self.transform = transform
        assert subset in ['train', 'val']
        if subset == 'train':
            self.coco = COCO(annotation_file=os.path.join(datafolder, 's2d3ds_train.json'))
        else:
            self.coco = COCO(annotation_file=os.path.join(datafolder, 's2d3ds_val.json'))
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    '''
        return:
            image: [3, H, W]
            depth: [H, W]
            segmentation: [H, W]
            instance: [num_planes, H, W]
    '''

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        if self.subset == 'train':
            image_path = os.path.join(self.datafolder, 'images', self.coco.loadImgs(image_id)[0]['file_name'])
        else:
            image_path = os.path.join(self.datafolder, 'images_val', self.coco.loadImgs(image_id)[0]['file_name'])
        depth_path = image_path.replace('images', 'depths').replace('rgb', 'depth').replace('.jpg', '.png')
        image = np.array(Image.open(image_path), dtype=np.uint8)
        depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0

        segmentation = np.zeros(image.shape[:2], dtype=np.uint8)
        annotation_id = self.coco.getAnnIds(imgIds=image_id)
        annotation = self.coco.loadAnns(annotation_id)
        for idx, i in enumerate(annotation):
            segmentation[self.coco.annToMask(i) > 0] = idx + 1
            idx += 1

        sample = {}
        if self.transform:
            sample = self.transform({
                'image': image,
                'depth': depth,
                'segmentation': segmentation
            })
            image = sample['image']
            depth = sample['depth']
            segmentation = sample['segmentation']

            # resize 192*256
            h, w = image.shape[1:]
            image = F.interpolate(image.unsqueeze(0), size=(192, 256), mode="bilinear", align_corners=False).squeeze(0)
            depth = F.interpolate(depth.unsqueeze(0), size=(192, 256), mode="bilinear", align_corners=False).squeeze(0)
            segmentation = F.interpolate(segmentation.unsqueeze(0).unsqueeze(0), size=(192, 256), mode="nearest").squeeze(0).squeeze(0)

        mask = []
        unique_idx = torch.unique(segmentation)
        unique_idx = [x for x in unique_idx if x]
        for i in unique_idx:
            mask.append((segmentation == i).float())
        mask = torch.stack(mask)
        bbox = self.masks_to_bboxes(mask)
        num_planes = len(unique_idx)

        masks = torch.zeros(30, 192, 256, dtype=torch.float32)
        masks[:num_planes] = mask

        lines_file_path = image_path.replace('.jpg', '.txt').replace('images', 'lines')
        lines, num_lines = self.get_lines(lines_file_path)  # 200, 4
        lines[..., 0:3:2] *= 256 / w
        lines[..., 1:4:2] *= 192 / h

        sample.update({
            "image": image,
            "depth": depth,
            "segmentation": segmentation,
            'instance': masks,
            'num_planes': num_planes,
            'num_lines': num_lines,
            'lines': torch.FloatTensor(lines),
        })

        return sample

    def masks_to_bboxes(self, masks):
        batch_size, height, width = masks.size()
        bounding_boxes = torch.zeros((batch_size, 4), dtype=torch.float32)

        for b in range(batch_size):
            mask = masks[b]
            nonzero_indices = torch.nonzero(mask)

            if nonzero_indices.size(0) == 0:
                assert "no mask!"
            else:
                ymin = torch.min(nonzero_indices[:, 0])
                xmin = torch.min(nonzero_indices[:, 1])
                ymax = torch.max(nonzero_indices[:, 0])
                xmax = torch.max(nonzero_indices[:, 1])
                bounding_boxes[b] = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

        return bounding_boxes

    def get_lines(self, line_path):
        lines = np.loadtxt(line_path, dtype=np.float32, delimiter=',')
        lines = lines.reshape(-1, 4)

        lines_pad = np.zeros([192, 4], dtype=np.float32)
        line_num = lines.shape[0]
        if line_num == 0:
            pass
        elif line_num > 192:
            lines_pad = lines[0:192, :]
            line_num = 192
        else:
            lines_pad[0:line_num, :] = lines

        if line_num == 0:
            line_num = 1
        return lines_pad, line_num


class ToTensor(object):
    def __call__(self, sample):
        image, depth, segmentation = sample['image'], sample['depth'], sample['segmentation']
        # [H, W, C] -> [C, H, W]
        image = transforms.ToTensor()(image)
        # [1, H, W]
        depth = transforms.ToTensor()(depth)
        return {
            'image': image,
            'depth': depth,
            'segmentation': torch.from_numpy(segmentation.astype(np.int16)).float()
        }

class RandomFlip(object):
    def __call__(self, sample):
        image, depth, segmentation = sample['image'], sample['depth'], sample['segmentation']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            segmentation = np.fliplr(segmentation).copy()

        return {
            'image': image,
            'depth': depth,
            'segmentation': segmentation
        }