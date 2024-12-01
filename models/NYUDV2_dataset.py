import os
import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as tf
import numpy as np
import cv2
from PIL import Image
import json
import scipy.io as sio

class nyudv2_PlaneDataset(data.Dataset):
    def __init__(self, subset='train', transform=None, root_dir=None, predict_center=False):
        assert subset in ['train', 'val']
        if subset == 'val':
            subset = 'test'
        self.subset = subset
        self.transform = transform
        self.predict_center = predict_center

        self.dir_anno = os.path.join(root_dir, subset + '_annotations.json')
        with open(self.dir_anno, 'r') as load_f:
            self.anno = json.load(load_f)
        self.dir_rgb_depth = os.path.join(root_dir, self.anno[0]['dir_AB'])
        data_rgb_depth = sio.loadmat(self.dir_rgb_depth)
        self.data_rgbs = data_rgb_depth['rgbs']
        self.data_depths = data_rgb_depth['depths']
        self.name_rgbs = [self.anno[i]['rgb_path'] for i in range(len(self.anno))]

        self.plane_dir = os.path.join(root_dir, 'data_plane')

        self.precompute_K_inv_dot_xy_1()

    def get_plane_parameters(self, plane, plane_nums, segmentation):
        valid_region = segmentation != 20
        plane = plane[:plane_nums]
        h, w = segmentation.shape
        plane_parameters2 = np.ones((3, h, w))
        for i in range(plane_nums):
            plane_mask = segmentation == i
            plane_mask = plane_mask.astype(np.float32)
            cur_plane_param_map = np.ones((3, h, w)) * plane[i, :].reshape(3, 1, 1)
            plane_parameters2 = plane_parameters2 * (1-plane_mask) + cur_plane_param_map * plane_mask
        # plane_instance parameter, padding zero to fix size
        plane_instance_parameter = np.concatenate((plane, np.zeros((20 - plane.shape[0], 3))), axis=0)
        return plane_parameters2, valid_region, plane_instance_parameter

    def precompute_K_inv_dot_xy_1(self, h=192, w=256):
        focal_length = 517.97
        offset_x = 320
        offset_y = 240

        K = [[focal_length, 0, offset_x],
             [0, focal_length, offset_y],
             [0, 0, 1]]

        K_inv = np.linalg.inv(np.array(K))
        self.K_inv = K_inv

        K_inv_dot_xy_1 = np.zeros((3, h, w))
        xy_map = np.zeros((2, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640

                ray = np.dot(self.K_inv,
                             np.array([xx, yy, 1]).reshape(3, 1))
                K_inv_dot_xy_1[:, y, x] = ray[:, 0]
                xy_map[0, y, x] = float(x) / w
                xy_map[1, y, x] = float(y) / h

        # precompute to speed up processing
        self.K_inv_dot_xy_1 = K_inv_dot_xy_1
        self.xy_map = xy_map

    def plane2depth(self, plane_parameters, num_planes, segmentation, gt_depth, h=192, w=256):

        depth_map = 1. / np.sum(self.K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
        depth_map = depth_map.reshape(h, w)

        # replace non planer region depth using sensor depth map
        depth_map[segmentation == 20] = gt_depth[segmentation == 20]
        return depth_map

    def get_lines(self, line_path):
        lines = np.loadtxt(line_path, dtype=np.float32, delimiter=',')
        lines = lines.reshape(-1, 4)

        lines_pad = np.zeros([200, 4], dtype=np.float32)
        line_num = lines.shape[0]
        if line_num == 0:
            pass
        elif line_num > 200:
            lines_pad = lines[0:200, :]
            line_num = 200
        else:
            lines_pad[0:line_num, :] = lines

        # return lines_pad, line_num
        if line_num == 0:
            line_num = 1
        return lines_pad, line_num

    def __getitem__(self, index):
        if self.subset == 'train':
            image_idx = int(self.name_rgbs[index].split('_')[2]) + 1
        else:
            image_idx = int(self.name_rgbs[index].split('_')[1]) + 1
        plane_data_path = os.path.join(self.plane_dir, 'plane_instance_' + str(image_idx) + '.npz')

        image_seg_map_path = os.path.join(self.plane_dir, 'plane_instance_' + str(image_idx) + '.png')
        image_seg_map = cv2.imread(image_seg_map_path)
        image = image_seg_map[:, :256, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        gt_depth = np.zeros((192, 256), dtype=np.float32)

        data = np.load(plane_data_path)

        plane = data['plane_param']
        num_planes = plane.shape[0]

        gt_segmentation = data['plane_instance']
        gt_segmentation = gt_segmentation - 1
        seg_mak = (gt_segmentation >= 0).astype(np.float32)
        gt_segmentation = gt_segmentation * seg_mak + (1-seg_mak) * 20

        gt_segmentation = gt_segmentation.reshape((192, 256))
        segmentation = np.zeros([21, 192, 256], dtype=np.uint8)

        # get segmentation: 21, h ,w
        _, h, w = segmentation.shape
        for i in range(num_planes + 1):
            # deal with backgroud
            if i == num_planes:
                seg = gt_segmentation == 20
            else:
                seg = gt_segmentation == i
            segmentation[i, :, :] = seg.reshape(h, w)

        # get plane center
        gt_plane_instance_centers = np.zeros([21, 2])
        gt_plane_pixel_centers = np.zeros([2, 192, 256], dtype=np.float)  # 2, 192, 256
        if self.predict_center:
            for i in range(num_planes):
                plane_mask = gt_segmentation == i
                pixel_num = plane_mask.sum()
                plane_mask = plane_mask.astype(np.float)
                x_map = self.xy_map[0] * plane_mask
                y_map = self.xy_map[1] * plane_mask
                x_sum = x_map.sum()
                y_sum = y_map.sum()
                plane_x = x_sum / pixel_num
                plane_y = y_sum / pixel_num
                gt_plane_instance_centers[i, 0] = plane_x
                gt_plane_instance_centers[i, 1] = plane_y

                center_map = np.zeros([2, 192, 256], dtype=np.float)  # 2, 192, 256
                center_map[0, :, :] = plane_x
                center_map[1, :, :] = plane_y
                center_map = center_map * plane_mask  # 2, 192, 256

                gt_plane_pixel_centers = gt_plane_pixel_centers * (1 - plane_mask) + center_map

        # surface plane parameters
        plane_parameters, valid_region, plane_instance_parameter = \
            self.get_plane_parameters(plane, num_planes, gt_segmentation)

        # since some depth is missing, we use plane to recover those depth following PlaneNet
        gt_depth = gt_depth.reshape(192, 256)
        depth = self.plane2depth(plane_parameters, num_planes, gt_segmentation, gt_depth).reshape(1, 192, 256)

        # get line segments
        lines_file_path = plane_data_path.replace('data_plane', 'data_lines')
        lines_file_path = lines_file_path.replace('npz', 'txt')
        lines, num_lines = self.get_lines(lines_file_path)  # 200, 4

        sample = {
            'image': image,
            'num_planes': num_planes,
            'instance': torch.ByteTensor(segmentation),
            # one for planar and zero for non-planar
            'semantic': 1 - torch.FloatTensor(segmentation[num_planes, :, :]).unsqueeze(0),
            'gt_seg': torch.LongTensor(gt_segmentation), # single channel
            'depth': torch.FloatTensor(depth),
            'plane_parameters': torch.FloatTensor(plane_parameters),
            'valid_region': torch.ByteTensor(valid_region.astype(np.uint8)).unsqueeze(0),
            'plane_instance_parameter': torch.FloatTensor(plane_instance_parameter),
            'gt_plane_instance_centers': torch.FloatTensor(gt_plane_instance_centers),
            'gt_plane_pixel_centers': torch.FloatTensor(gt_plane_pixel_centers),
            'image_name_idx': image_idx,
            'num_lines': num_lines,
            'lines': torch.FloatTensor(lines),
            'data_path': plane_data_path
        }
        return sample

    def __len__(self):
        return len(self.anno)

    def check_lines(self):
        for i in range(len(self.data_list)):
            if self.subset == 'train':
                data_path = self.data_list[i]
            else:
                data_path = str(i) + '.npz'
            data_path = os.path.join(self.root_dir, data_path)

            lines_file_path = data_path.replace(self.subset, self.subset + '_img')
            lines_file_path = lines_file_path.replace('npz', 'txt')

            lines = np.loadtxt(lines_file_path, dtype=np.float32, delimiter=',')
            lines = lines.reshape(-1, 4)
            line_num = lines.shape[0]
            if line_num <= 1:
                lineinfo = np.zeros([1, 4], dtype=np.float32)
                print(lines_file_path)
                # np.savetxt(lines_file_path, lineinfo, fmt='%.3f', delimiter=',')

Tensor_to_Image = tf.Compose([
    tf.Normalize([0.0, 0.0, 0.0], [1.0/0.229, 1.0/0.224, 1.0/0.225]),
    tf.Normalize([-0.485, -0.456, -0.406], [1.0, 1.0, 1.0]),
    tf.ToPILImage()
])

def tensor_to_image(image):
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

if __name__ == '__main__':
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    root_dir = 'dataset/nyu_plane/'

    loader = data.DataLoader(
            nyudv2_PlaneDataset(subset='val', transform=transforms, root_dir=root_dir, predict_center=False),
            batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )

    for iter, sample in enumerate(loader):
        image = sample['image']
        depth = sample['depth']
        image_name_idx = sample['image_name_idx']
        seg_map = sample['gt_seg'].numpy()

        image = tensor_to_image(image[0].cpu())
        cv2.imwrite('image_%d.png'%(image_name_idx), image)

        print(np.unique(seg_map))
        print(seg_map[0, 0, 0])

        if iter > 1:
            break